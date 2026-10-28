"""把批量顶点/边数组写进 Blender（依赖 bpy）。

关键点：mesh 用 vertices.add + foreach_set 一次性写入，
而不是 from_pydata 的逐点 Python 列表转换，更不是每个实体一个对象。

另外两件不属于"几何"但也必须在这里做的事：
1. 文字：DXF 的文字没有几何，只有字符串和定位。这里借 Blender 的字体引擎
   把字符串描成**字形轮廓折线**，并进同一个网格——所以导入结果仍然只有
   一个对象，导出 OBJ/在仿真里也看得见字的形状。
2. 线框转实体面：Isaac Sim 这类引擎的 OBJ 导入器只认多边形面，会直接丢掉
   `l` 线段。把每条边扩成一条极细的四边形带，就能变成真正的面。
"""

from __future__ import annotations

import math
import os

import numpy as np

import bpy

# DXF $INSUNITS -> 米
INSUNITS_TO_METER = {
    0: None,       # 未指定
    1: 0.0254,     # 英寸
    2: 0.3048,     # 英尺
    3: 1609.344,   # 英里
    4: 0.001,      # 毫米
    5: 0.01,       # 厘米
    6: 1.0,        # 米
    7: 1000.0,     # 千米
    8: 2.54e-5,    # 微英寸
    9: 2.54e-2,    # 密耳
    10: 0.9144,    # 码
    11: 1e-10,     # 埃
    12: 1e-9,      # 纳米
    13: 1e-6,      # 微米
    14: 0.1,       # 分米
    15: 10.0,      # 十米
    16: 100.0,     # 百米
}

# 字体候选：要求中英文都有。文件名带 Fallback 的纯 CJK 回退字体排最后——
# 它没有拉丁字形，英文会渲染成一片空心方框。
FONT_CANDIDATES = (
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    # 兜底（无拉丁字形，只在没有别的字体时用）
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)

_MT_ALIGN_Y = {
    "TOP": "TOP",
    "CENTER": "CENTER",
    "BOTTOM": "BOTTOM",
}


def mesh_from_arrays(name, verts: np.ndarray, edges: np.ndarray):
    """用 numpy 数组直接建 mesh，返回对象。"""
    me = bpy.data.meshes.new(name)
    nv = len(verts)
    ne = len(edges)
    if nv:
        me.vertices.add(nv)
        me.vertices.foreach_set(
            "co", np.ascontiguousarray(verts, dtype=np.float32).ravel()
        )
    if ne:
        me.edges.add(ne)
        me.edges.foreach_set(
            "vertices", np.ascontiguousarray(edges, dtype=np.int32).ravel()
        )
    me.update(calc_edges=False)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def mesh_from_faces(name, verts: np.ndarray, faces: np.ndarray):
    """用四边形面建 mesh（线框转实体面时用）。"""
    me = bpy.data.meshes.new(name)
    nv = len(verts)
    if nv:
        me.vertices.add(nv)
        me.vertices.foreach_set(
            "co", np.ascontiguousarray(verts, dtype=np.float32).ravel()
        )
    nf = len(faces)
    if nf:
        me.loops.add(nf * 4)
        me.loops.foreach_set(
            "vertex_index", np.ascontiguousarray(faces, dtype=np.int32).ravel()
        )
        me.polygons.add(nf)
        me.polygons.foreach_set(
            "loop_start", np.arange(0, nf * 4, 4, dtype=np.int32)
        )
    me.update(calc_edges=False)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


# ------------------------------------------------------------------ 文字
def _outline(font, body, align_x="LEFT", align_y="BOTTOM_BASELINE", resolution=6):
    """把一个字符串描成折线轮廓，返回 (verts[N,2], edges[M,2])，单位为 em。

    fill_mode='NONE' 时字体曲线不填充，to_mesh 出来的就是纯轮廓边——
    正好是我们想要的"可看的字形线框"。
    """
    cu = bpy.data.curves.new("fx_tmp_text", type="FONT")
    try:
        cu.body = body
        cu.font = font
        cu.size = 1.0
        cu.align_x = align_x
        cu.align_y = align_y
        cu.resolution_u = max(1, int(resolution))
        cu.fill_mode = "NONE"
        ob = bpy.data.objects.new("fx_tmp_text", cu)
        bpy.context.scene.collection.objects.link(ob)
        me = None
        try:
            me = ob.to_mesh()
            nv = len(me.vertices)
            ne = len(me.edges)
            if nv == 0 or ne == 0:
                return None
            co = np.empty(nv * 3, np.float32)
            me.vertices.foreach_get("co", co)
            v = co.reshape(nv, 3)[:, :2].astype(np.float64)
            e = np.empty(ne * 2, np.int32)
            me.edges.foreach_get("vertices", e)
            return v, e.reshape(ne, 2).astype(np.int64)
        finally:
            if me is not None:
                ob.to_mesh_clear()
            bpy.data.objects.remove(ob, do_unlink=True)
    finally:
        bpy.data.curves.remove(cu)


def _probe(font):
    """实测这个字体能不能同时画好中文和英文。

    返回 (拉丁字形顶点数, 中文字形顶点数)。拉丁数为 0 或极小，说明它只有
    CJK 字形（Droid Sans Fallback 就是这样），英文会画成空心方框。
    """
    lat = _outline(font, "ABC", resolution=4)
    cjk = _outline(font, "\u56fd", resolution=4)
    return (
        len(lat[0]) if lat else 0,
        len(cjk[0]) if cjk else 0,
    )


def pick_font(explicit=None):
    """挑一个能画中英文的字体。返回 (font, 说明)。"""
    for path in ([explicit] if explicit else []) + list(FONT_CANDIDATES):
        if not path or not os.path.exists(path):
            continue
        try:
            font = bpy.data.fonts.load(path)
        except Exception:  # noqa: BLE001 - 字体损坏/格式不支持都跳过
            continue
        lat, cjk = _probe(font)
        if lat >= 60 and cjk > 0:
            return font, "%s（拉丁 %d / 中文 %d 顶点）" % (
                os.path.basename(path), lat, cjk
            )
        if lat >= 60 and cjk == 0:
            # 只有英文：还能用，但中文会是方框，继续往后找
            bpy.data.fonts.remove(font, do_unlink=True)
            continue
        bpy.data.fonts.remove(font, do_unlink=True)
    return None, ""


def _cap_ratio(font, resolution=6):
    """字号 1.0 时，大写字母的实际高度（em 比例）。

    DXF 的组码 40 是"字高"，含义接近大写字母高度；而 Blender 的 size 是 em。
    不同字体的比例差得很远（思源 0.26、文鼎 0.69），写死一个数就会大小差一倍，
    所以这里实测一次反推。
    """
    for ch in ("H", "\u56fd"):
        out = _outline(font, ch, resolution=resolution)
        if out is not None:
            v = out[0]
            h = float(v[:, 1].max() - v[:, 1].min())
            if h > 1e-6:
                return h
    return 0.7


class _GlyphCache:
    """同一个字符串只描一次。

    图纸里"门""窗""1""A"这类字会重复上千次，缓存能省掉绝大部分 to_mesh 调用。
    """

    def __init__(self, font, resolution):
        self.font = font
        self.resolution = resolution
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, body, align_x, align_y):
        key = (body, align_x, align_y)
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        ax = align_x if align_x in ("LEFT", "CENTER", "RIGHT") else "LEFT"
        ay = align_y if align_y in (
            "TOP", "TOP_BASELINE", "CENTER", "BOTTOM_BASELINE", "BOTTOM"
        ) else "BOTTOM_BASELINE"
        out = _outline(self.font, body, ax, ay, self.resolution)
        self._cache[key] = out
        return out


def text_arrays(records, font_path=None, resolution=6):
    """把文字记录描成 (verts, edges, layers)。

    records: [(TextItem, 4x4 矩阵, 图层 id), ...]，矩阵已含 OCS/块变换。
    """
    if not records:
        return (
            np.empty((0, 3), np.float64),
            np.empty((0, 2), np.int64),
            np.empty(0, np.int32),
        )
    font, _desc = pick_font(font_path)
    if font is None:
        return (
            np.empty((0, 3), np.float64),
            np.empty((0, 2), np.int64),
            np.empty(0, np.int32),
        )

    ratio = _cap_ratio(font, resolution)
    cache = _GlyphCache(font, resolution)
    out_v, out_e, out_l = [], [], []
    base = 0
    for item, mat, layer in records:
        body = item.body
        if not body or not body.strip():
            continue
        lines = body.split("\n")
        # 字高换算：dxf 的 height 是大写字高，em 缩放要除以实测比例
        hs = item.height / ratio if ratio > 1e-6 else item.height
        step_em = (5.0 / 3.0) * float(item.line_spacing or 1.0) * ratio
        shear = math.tan(math.radians(item.oblique)) if item.oblique else 0.0
        wf = item.width if 0.01 <= item.width <= 100.0 else 1.0
        for li, line in enumerate(lines):
            if not line:
                continue
            glyph = cache.get(line, item.align_x, item.align_y)
            if glyph is None:
                continue
            gv, ge = glyph
            p = gv.copy()
            if wf != 1.0:
                p[:, 0] *= wf
            if shear:
                p[:, 0] += shear * p[:, 1]
            p *= hs
            if li:
                p[:, 1] -= li * step_em * hs
            p3 = np.empty((len(p), 3), np.float64)
            p3[:, :2] = p
            p3[:, 2] = 0.0
            p3 = p3 @ mat[:3, :3].T + mat[:3, 3]
            out_v.append(p3)
            out_e.append(ge + base)
            out_l.append(np.full(len(ge), layer, np.int32))
            base += len(p)

    if not out_v:
        return (
            np.empty((0, 3), np.float64),
            np.empty((0, 2), np.int64),
            np.empty(0, np.int32),
        )
    return (
        np.vstack(out_v),
        np.vstack(out_e),
        np.concatenate(out_l),
    )


# ------------------------------------------------------------------ 转面
def robust_span(verts):
    """用分位数估计图纸的实际跨度。

    直接取 min/max 会被极远处的游离图块带偏（实测一张图跨度 8000 多米，
    99% 的几何其实挤在 30 米内）。这里按 1%~99% 分位算，异常值不参与。
    """
    if len(verts) == 0:
        return 1.0
    lo = np.percentile(verts, 1.0, axis=0)
    hi = np.percentile(verts, 99.0, axis=0)
    span = float(np.max(hi - lo))
    return span if span > 1e-9 else 1.0


def ribbon(verts, edges, width):
    """把每条边扩成一条细四边形带，返回 (verts, faces)。"""
    if len(edges) == 0:
        return verts, np.empty((0, 4), np.int64)
    p0 = verts[edges[:, 0]]
    p1 = verts[edges[:, 1]]
    d = p1 - p0
    length = np.hypot(d[:, 0], d[:, 1])
    ok = length > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, length, 1.0), 0.0)
    # 边的法向（XY 平面内）；竖直边没有 XY 方向，用 X 轴当参考
    ux = np.where(ok, -d[:, 1] * inv, 1.0)
    uy = np.where(ok, d[:, 0] * inv, 0.0)
    h = 0.5 * width
    ox, oy = ux * h, uy * h

    n = len(edges)
    v = np.empty((4 * n, 3), np.float64)
    v[0::4, 0] = p0[:, 0] + ox
    v[0::4, 1] = p0[:, 1] + oy
    v[0::4, 2] = p0[:, 2]
    v[1::4, 0] = p1[:, 0] + ox
    v[1::4, 1] = p1[:, 1] + oy
    v[1::4, 2] = p1[:, 2]
    v[2::4, 0] = p1[:, 0] - ox
    v[2::4, 1] = p1[:, 1] - oy
    v[2::4, 2] = p1[:, 2]
    v[3::4, 0] = p0[:, 0] - ox
    v[3::4, 1] = p0[:, 1] - oy
    v[3::4, 2] = p0[:, 2]
    faces = np.arange(4 * n, dtype=np.int64).reshape(n, 4)
    return v, faces


def _make_object(name, verts, edges, solid_faces, width):
    if solid_faces and len(edges):
        rv, rf = ribbon(verts, edges, width)
        return mesh_from_faces(name, rv, rf)
    return mesh_from_arrays(name, verts, edges)


def build_objects(
    acc,
    layer_names,
    base_name="DXF",
    scale=1.0,
    by_layer=False,
    center=False,
    text_records=None,
    font_path=None,
    text_resolution=6,
    solid_faces=False,
    line_width=0.0,
):
    """把累加器（+ 文字）转成对象列表。返回 (objects, stats)。"""
    verts, edges, elayers = acc.arrays()
    n_text = 0
    text_stats = {}

    if text_records:
        tv, te, tl = text_arrays(text_records, font_path, text_resolution)
        n_text = len(te)
        if len(tv):
            offset = len(verts)
            verts = np.vstack([verts, tv]) if len(verts) else tv
            edges = np.vstack([edges, te + offset]) if len(edges) else te + offset
            elayers = np.concatenate([elayers, tl]) if len(elayers) else tl
            text_stats["text_verts"] = int(len(tv))

    if len(verts) == 0:
        return [], {"objects": 0, "verts": 0, "edges": 0, "texts": 0}

    if scale != 1.0:
        verts = verts * scale

    offset = None
    if center:
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        offset = (lo + hi) / 2.0
        offset[2] = lo[2]  # Z 只落到地面，不居中

    if offset is not None:
        verts = verts - offset

    width = float(line_width or 0.0)
    if solid_faces and width <= 0.0:
        width = robust_span(verts) * 0.0015

    objects = []
    if not by_layer or len(layer_names) <= 1:
        ob = _make_object(base_name, verts, edges, solid_faces, width)
        objects.append(ob)
    else:
        for lid, lname in enumerate(layer_names):
            sel = elayers == lid
            if not np.any(sel):
                continue
            sub_e = edges[sel]
            used = np.unique(sub_e)
            remap = np.full(len(verts), -1, dtype=np.int64)
            remap[used] = np.arange(len(used), dtype=np.int64)
            ob = _make_object(
                "%s_%s" % (base_name, _safe_name(lname)),
                verts[used],
                remap[sub_e],
                solid_faces,
                width,
            )
            objects.append(ob)

    for ob in objects:
        ob.show_wire = not solid_faces
        ob.show_in_front = False

    stats = {
        "objects": len(objects),
        # 转面后每条边变成 4 个顶点 + 1 个四边形，网格顶点数要按实际写进 mesh 的算
        "verts": int(4 * len(edges)) if solid_faces and len(edges) else int(len(verts)),
        "edges": int(len(edges)),
        "faces": int(len(edges)) if solid_faces else 0,
        "texts": n_text,
        "line_width": float(width) if solid_faces else 0.0,
    }
    stats.update(text_stats)
    return objects, stats


def _safe_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(name))
    return out.strip() or "layer"
