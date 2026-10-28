"""把 DXF 实体转成批量顶点/边数组（不依赖 bpy）。

所有几何统一表示为 (verts[N,3], edges[M,2])，最后一次性喂给 Blender。
任何"逐实体创建对象"的做法都被避开——那是原插件慢的根源。
"""

from __future__ import annotations

import math

import numpy as np

from . import ocs, text as textmod
from .dxf import (
    Segment,
    expand_ranges,
    first_per_owner,
    gather_floats,
    gather_ints,
    gather_str,
)

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0

SUPPORTED = (
    "LINE", "LWPOLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "INSERT", "POLYLINE", "HATCH", "TEXT", "MTEXT", "ATTRIB",
)

# 被其他实体消费的子记录，不算"跳过"
CONSUMED = {"VERTEX", "SEQEND"}

SKIPPED_TYPES = {
    "DIMENSION", "POINT", "ATTDEF",
    "LEADER", "MLEADER", "WIPEOUT", "VIEWPORT", "SOLID", "3DFACE", "TRACE",
    "IMAGE", "UNDERLAY", "ACAD_PROXY_ENTITY", "ACAD_TABLE",
}


class Settings:
    """导入参数。"""

    def __init__(
        self,
        arc_resolution=64,
        spline_resolution=12,
        scale=1.0,
        import_insert=True,
        use_bulge=True,
        max_block_depth=16,
        import_unreferenced_blocks=True,
        import_text=True,
        max_texts=40000,
    ):
        self.arc_resolution = max(4, int(arc_resolution))
        self.spline_resolution = max(2, int(spline_resolution))
        self.scale = float(scale)
        self.import_insert = bool(import_insert)
        self.use_bulge = bool(use_bulge)
        self.max_block_depth = int(max_block_depth)
        self.import_unreferenced_blocks = bool(import_unreferenced_blocks)
        self.import_text = bool(import_text)
        self.max_texts = max(0, int(max_texts))


class GeoAcc:
    """顶点/边累加器。顶点序号在 add 时平移，避免频繁拼接。"""

    __slots__ = ("_v", "_e", "_l", "nv")

    def __init__(self):
        self._v = []
        self._e = []
        self._l = []
        self.nv = 0

    def add(self, verts: np.ndarray, edges: np.ndarray, layers: np.ndarray):
        if len(verts) == 0 or len(edges) == 0:
            return
        layers = np.asarray(layers)
        if len(layers) != len(edges):
            raise ValueError("layer 数量 %d 与边数 %d 不匹配" % (len(layers), len(edges)))
        self._v.append(verts)
        self._e.append(edges + self.nv)
        self._l.append(layers.astype(np.int32, copy=False))
        self.nv += len(verts)

    def extend(self, acc: "GeoAcc"):
        for v, e, l in zip(acc._v, acc._e, acc._l):
            if len(l):
                self._v.append(v)
                self._e.append(e + self.nv)
                self._l.append(l)
                self.nv += len(v)

    @property
    def empty(self) -> bool:
        return not self._v

    def arrays(self):
        if not self._v:
            return (
                np.empty((0, 3), np.float64),
                np.empty((0, 2), np.int64),
                np.empty(0, np.int32),
            )
        v = self._v[0] if len(self._v) == 1 else np.vstack(self._v)
        e = self._e[0] if len(self._e) == 1 else np.vstack(self._e)
        l = self._l[0] if len(self._l) == 1 else np.concatenate(self._l)
        return v, e, l


# ---------------------------------------------------------------- 几何工具

def chain_edges(counts, closed=None) -> np.ndarray:
    """按每条折线的顶点数，生成相邻顶点之间的边。"""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total < 2 or len(counts) == 0:
        return np.empty((0, 2), dtype=np.int64)
    starts = np.cumsum(counts) - counts
    owner = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    gi = np.arange(total, dtype=np.int64)
    ok = owner[:-1] == owner[1:]
    e = np.stack([gi[:-1][ok], gi[1:][ok]], axis=1)
    if closed is not None and np.any(closed):
        sel = np.flatnonzero(np.asarray(closed, dtype=bool))
        last = starts[sel] + counts[sel] - 1
        e = np.vstack([e, np.stack([last, starts[sel]], axis=1)])
    return e


def arc_points(cx, cy, cz, radius, a0, a1, resolution):
    """生成弧上的点，返回 (verts, n_points_per_arc)。a0/a1 为弧度。"""
    sweep = np.mod(a1 - a0, TWO_PI)
    sweep = np.where(sweep < 1e-9, TWO_PI, sweep)
    counts = np.maximum(2, np.ceil(resolution * sweep / TWO_PI).astype(np.int64))
    npt = counts + 1
    total = int(npt.sum())
    starts = np.cumsum(npt) - npt
    gi = np.arange(total, dtype=np.int64)
    owner = np.repeat(np.arange(len(counts), dtype=np.int64), npt)
    frac = (gi - starts[owner]) / counts[owner]
    ang = a0[owner] + frac * sweep[owner]
    v = np.empty((total, 3), dtype=np.float64)
    v[:, 0] = cx[owner] + radius[owner] * np.cos(ang)
    v[:, 1] = cy[owner] + radius[owner] * np.sin(ang)
    v[:, 2] = cz[owner]
    return v, npt


def _view(seg: Segment, sel):
    starts, ends = seg.starts_ends()
    if len(sel) == 0:
        z = np.empty(0, np.int64)
        return z, z, np.empty(0, np.int32)
    rows, owner = expand_ranges(starts[sel], ends[sel])
    return rows, owner, seg.codes[rows]


def _one(seg, view, code, default=0.0, as_int=False):
    rows, owner, codes = view
    n = int(owner.max()) + 1 if len(owner) else 0
    pos = first_per_owner(owner, codes == code)
    dtype = np.int64 if as_int else np.float64
    out = np.full(n, default, dtype=dtype)
    have = pos >= 0
    if np.any(have):
        src = rows[pos[have]]
        out[have] = gather_ints(seg.vals, src) if as_int else gather_floats(seg.vals, src)
    return out


def _many(seg, view, code, as_int=False):
    rows, owner, codes = view
    m = codes == code
    src = rows[m]
    vals = gather_ints(seg.vals, src) if as_int else gather_floats(seg.vals, src)
    return vals, owner[m]


def edge_layers(vertex_layer: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.empty(0, np.int32)
    return vertex_layer[edges[:, 0]]


# ---------------------------------------------------------------- 转换器

class Converter:
    def __init__(self, settings: Settings):
        self.s = settings
        self.stats = {}
        self.layers = {}
        self.layer_names = []
        self.blocks = {}
        self.warnings = []
        self.block_types = {}
        self.entities_types = {}
        self._zero_layer = 0
        self._stack = ()
        self._silent = False
        self.texts = []          # 最终要画的文字（已换到世界坐标）
        self._text_out = None    # 当前 _emit 的文字落地目标
        self.texts_dropped = 0

    # -- 图层 ------------------------------------------------------
    def layer_id(self, name) -> int:
        key = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        lid = self.layers.get(key)
        if lid is None:
            lid = len(self.layer_names)
            self.layers[key] = lid
            self.layer_names.append(key)
        return lid

    def _count(self, name, n=1):
        if self._silent:  # 块定义里的统计单独记，避免污染 ENTITIES 的统计
            return
        self.stats[name] = self.stats.get(name, 0) + int(n)

    def _layers(self, seg, view) -> np.ndarray:
        rows, owner, codes = view
        n = int(owner.max()) + 1 if len(owner) else 0
        pos = first_per_owner(owner, codes == 8)
        out = np.zeros(n, dtype=np.int32)
        have = pos >= 0
        if np.any(have):
            names = gather_str(seg.vals, rows[pos[have]])
            out[have] = [self.layer_id(nm) for nm in names]
        return out

    # -- 坐标变换 --------------------------------------------------
    def _ocs_axes(self, seg, view):
        """读挤出方向 210/220/230，返回逐实体的 (Ax, Ay, Az)。

        返回 None 表示整批都是默认法向 (0,0,1)，调用方直接跳过变换——
        绝大多数图纸都走这条路，宁可多一次判断也不要白算一遍 200 万个点。
        """
        nx = _one(seg, view, 210, 0.0)
        ny = _one(seg, view, 220, 0.0)
        nz = _one(seg, view, 230, 1.0)
        if ocs.is_identity(nx, ny, nz):
            return None
        return ocs.axes_arrays(nx, ny, nz)

    @staticmethod
    def _sub_axes(axes, idx):
        if axes is None:
            return None
        return tuple(a[idx] for a in axes)

    def _apply_ocs(self, verts, owner, axes):
        return verts if axes is None else ocs.apply(verts, owner, axes)

    # -- LINE ------------------------------------------------------
    def add_lines(self, seg, sel, acc: GeoAcc):
        n = len(sel)
        if n == 0:
            return
        view = _view(seg, sel)
        x1 = _one(seg, view, 10)
        y1 = _one(seg, view, 20)
        z1 = _one(seg, view, 30)
        x2 = _one(seg, view, 11)
        y2 = _one(seg, view, 21)
        z2 = _one(seg, view, 31)
        if not (len(x1) == len(x2) == n):
            self.warnings.append("LINE 组码异常，已跳过")
            return
        v = np.empty((2 * n, 3), np.float64)
        v[0::2, 0], v[0::2, 1], v[0::2, 2] = x1, y1, z1
        v[1::2, 0], v[1::2, 1], v[1::2, 2] = x2, y2, z2
        e = np.arange(2 * n, dtype=np.int64).reshape(-1, 2)
        acc.add(v, e, self._layers(seg, view))
        self._count("LINE", n)

    # -- LWPOLYLINE ------------------------------------------------
    def add_lwpolyline(self, seg, sel, acc: GeoAcc):
        n = len(sel)
        if n == 0:
            return
        view = _view(seg, sel)
        rows, owner, codes = view
        xs, own_x = _many(seg, view, 10)
        ys, own_y = _many(seg, view, 20)
        if len(xs) != len(ys) or len(xs) == 0:
            self.warnings.append("LWPOLYLINE 顶点不成对，已跳过")
            return
        flags = _one(seg, view, 70, 0.0, as_int=True)
        elev = _one(seg, view, 38, 0.0)
        lids = self._layers(seg, view)
        axes = self._ocs_axes(seg, view)
        counts = np.bincount(own_x, minlength=n).astype(np.int64)

        has_bulge = self.s.use_bulge and np.any(codes == 42)
        bul = self._vertex_bulges(seg, view) if has_bulge else None
        if bul is not None and len(bul) != len(xs):
            bul = None

        if np.any(counts < 2):
            keep = counts >= 2
            if not np.all(keep):
                idx = np.flatnonzero(keep)
                starts = np.cumsum(counts) - counts
                rows_v = np.concatenate(
                    [np.arange(starts[i], starts[i] + counts[i]) for i in idx]
                )
                xs, ys = xs[rows_v], ys[rows_v]
                counts = counts[idx]
                flags, elev, lids = flags[idx], elev[idx], lids[idx]
                axes = self._sub_axes(axes, idx)
                if bul is not None:
                    bul = bul[rows_v]
                n = len(idx)

        closed = (flags & 1).astype(bool)
        v = np.stack([xs, ys, np.repeat(elev, counts)], axis=1)

        if bul is not None:
            v, counts = self._bulge_split(v, counts, closed, bul)
            closed = np.zeros(len(counts), bool)  # 已在内部闭合

        if axes is not None:
            v = ocs.apply(v, np.repeat(np.arange(len(counts), dtype=np.int64), counts), axes)

        e = chain_edges(counts, closed)
        vlayer = np.repeat(lids[:n], counts)
        acc.add(v, e, edge_layers(vlayer, e))
        self._count("LWPOLYLINE", n)

    def _vertex_bulges(self, seg, view):
        """把 code 42 对齐到“所属顶点”，返回与顶点等长的数组。

        DXF 里 42 只写在有非零 bulge 的顶点之后，所以不能按出现顺序硬对齐——
        某条多段线只标了 2 个 bulge、却有 5 个顶点时，硬对齐会把圆弧挂错段。
        """
        rows, owner, codes = view
        m10 = codes == 10
        if not np.any(m10):
            return None
        v_owner = owner[m10]
        r42 = np.flatnonzero(codes == 42)
        bul = np.zeros(len(v_owner))
        if len(r42) == 0:
            return bul
        # 每个 42 行前面最近的 10 行，就是它的顶点
        vi = (np.cumsum(m10) - 1)[r42]
        vals = gather_floats(seg.vals, rows[r42])
        elig = np.flatnonzero(vi >= 0)
        idx = vi[elig]
        sameness = v_owner[idx] == owner[r42[elig]]
        bul[idx[sameness]] = vals[elig][sameness]
        return bul

    def _bulge_split(self, v, counts, closed, bul):
        """把带 bulge 的段替换为圆弧折线。仅处理非零 bulge。"""
        out_v = []
        out_counts = np.empty(len(counts), np.int64)
        base = 0
        for i in range(len(counts)):
            k = int(counts[i])
            pts = [(v[base, 0], v[base, 1], v[base, 2])]
            last = k if closed[i] else k - 1
            for j in range(last):
                k2 = (j + 1) % k
                b = float(bul[base + j])
                p0 = (v[base + j, 0], v[base + j, 1])
                p1 = (v[base + k2, 0], v[base + k2, 1])
                if abs(b) > 1e-12:
                    arc = _bulge_arc(p0, p1, b, self.s.arc_resolution)
                    for q in arc[:-1]:
                        pts.append((q[0], q[1], v[base + j, 2]))
                    if arc:
                        pts.append((arc[-1][0], arc[-1][1], v[base + k2, 2]))
                else:
                    pts.append((p1[0], p1[1], v[base + k2, 2]))
            out_counts[i] = len(pts)
            out_v.append(np.asarray(pts, np.float64))
            base += k
        return np.vstack(out_v), out_counts

    # -- ARC / CIRCLE ----------------------------------------------
    def add_arcs(self, seg, sel, acc: GeoAcc, is_circle=False):
        if len(sel) == 0:
            return
        view = _view(seg, sel)
        cx = _one(seg, view, 10)
        cy = _one(seg, view, 20)
        cz = _one(seg, view, 30)
        r = _one(seg, view, 40)
        n = len(cx)
        if is_circle:
            a0 = np.zeros(n)
            a1 = np.full(n, TWO_PI)
        else:
            a0 = _one(seg, view, 50) * DEG
            a1 = _one(seg, view, 51, 360.0) * DEG

        lids = self._layers(seg, view)
        axes = self._ocs_axes(seg, view)
        keep = r > 0
        if not np.all(keep):
            idx = np.flatnonzero(keep)
            cx, cy, cz, r, a0, a1, lids = cx[idx], cy[idx], cz[idx], r[idx], a0[idx], a1[idx], lids[idx]
            axes = self._sub_axes(axes, idx)
            if len(idx) == 0:
                return
        v, npt = arc_points(cx, cy, cz, r, a0, a1, self.s.arc_resolution)
        if axes is not None:
            v = ocs.apply(v, np.repeat(np.arange(len(cx), dtype=np.int64), npt), axes)
        e = chain_edges(npt)
        vlayer = np.repeat(lids, npt)
        acc.add(v, e, edge_layers(vlayer, e))
        self._count("CIRCLE" if is_circle else "ARC", len(cx))

    # -- ELLIPSE ---------------------------------------------------
    def add_ellipses(self, seg, sel, acc: GeoAcc):
        n = len(sel)
        if n == 0:
            return
        view = _view(seg, sel)
        cx = _one(seg, view, 10)
        cy = _one(seg, view, 20)
        cz = _one(seg, view, 30)
        mx = _one(seg, view, 11, 1.0)
        my = _one(seg, view, 21)
        mz = _one(seg, view, 31)
        ratio = _one(seg, view, 40, 1.0)
        t0 = _one(seg, view, 41, 0.0)
        t1 = _one(seg, view, 42, TWO_PI)
        # 注意：ELLIPSE 的中心点和长轴是 **WCS**，不是 OCS（DXF 里只有它和
        # LINE/SPLINE 这样）。法向只用来定短轴方向，不能整体套 OCS 变换，
        # 否则镜像图纸里的椭圆会跑到反方向去。
        nx = _one(seg, view, 210, 0.0)
        ny = _one(seg, view, 220, 0.0)
        nz = _one(seg, view, 230, 1.0)

        sweep = np.mod(t1 - t0, TWO_PI)
        sweep = np.where(sweep < 1e-9, TWO_PI, sweep)
        counts = np.maximum(3, np.ceil(self.s.arc_resolution * sweep / TWO_PI).astype(np.int64))
        npt = counts + 1
        total = int(npt.sum())
        starts = np.cumsum(npt) - npt
        gi = np.arange(total, dtype=np.int64)
        owner = np.repeat(np.arange(n, dtype=np.int64), npt)
        frac = (gi - starts[owner]) / counts[owner]
        t = t0[owner] + frac * sweep[owner]
        ct, st = np.cos(t), np.sin(t)

        maj = np.stack([mx, my, mz], axis=1)
        nor = np.stack([nx, ny, nz], axis=1)
        ln = np.linalg.norm(nor, axis=1)
        ln = np.where(ln < 1e-12, 1.0, ln)
        nor = nor / ln[:, None]
        mi = np.cross(nor, maj) * ratio[:, None]

        v = np.stack([cx, cy, cz], axis=1)[owner] + ct[:, None] * maj[owner] + st[:, None] * mi[owner]
        e = chain_edges(npt)
        vlayer = np.repeat(self._layers(seg, view), npt)
        acc.add(v, e, edge_layers(vlayer, e))
        self._count("ELLIPSE", n)

    # -- SPLINE ----------------------------------------------------
    def add_splines(self, seg, sel, acc: GeoAcc):
        n = len(sel)
        if n == 0:
            return
        starts, ends = seg.starts_ends()
        s_idx, e_idx = starts[sel], ends[sel]
        out_v, lids, out_counts = [], [], []
        for i in range(n):
            a, b = int(s_idx[i]), int(e_idx[i])
            deg, closed, lid = 3, False, 0
            for p in range(a, b):
                c = seg.codes[p]
                if c == 71:
                    deg = max(1, int(float(seg.vals[p])))
                elif c == 70:
                    closed = bool(int(float(seg.vals[p])) & 1)
                elif c == 8:
                    lid = self.layer_id(seg.vals[p].strip())
            ctrl = _collect(seg, a, b, (10, 20, 30))
            if len(ctrl) >= 2:
                pts = _sample_spline(ctrl, _collect(seg, a, b, (40,)), deg,
                                     self.s.spline_resolution)
            else:
                pts = _collect(seg, a, b, (11, 21, 31))
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, np.float64)
            out_v.append(arr)
            out_counts.append(len(arr))
            lids.append(lid)
        if not out_v:
            return
        v = np.vstack(out_v)
        counts = np.asarray(out_counts, np.int64)
        e = chain_edges(counts)
        vlayer = np.repeat(np.asarray(lids, np.int32), counts)
        acc.add(v, e, edge_layers(vlayer, e))
        self._count("SPLINE", len(out_v))

    # -- POLYLINE（老式多段线，几何在后续 VERTEX 子实体里）----------
    def add_polylines(self, seg, sel, acc: GeoAcc):
        n = len(sel)
        if n == 0:
            return
        zero = seg.zero
        pl_rows = zero[sel]
        vsel = seg.select("VERTEX")
        if len(vsel) == 0:
            self._count("POLYLINE", n)
            return

        # VERTEX 归属于它前面最近的 POLYLINE
        slot = np.searchsorted(pl_rows, zero[vsel], side="right") - 1
        ok = (slot >= 0) & (slot < n)
        slot, vsel = slot[ok], vsel[ok]

        view = _view(seg, sel)
        flags = _one(seg, view, 70, 0.0, as_int=True)
        lids = self._layers(seg, view)
        axes = self._ocs_axes(seg, view)

        # polyface mesh(64) / polygon mesh(16) 是面记录，不是线框
        mesh_like = ((flags & 64) | (flags & 16)) != 0
        if np.any(mesh_like):
            self.warnings.append(
                "%d 条 polyface/polygon mesh 多段线未展开" % int(mesh_like.sum())
            )
        okv = ~mesh_like[slot]
        slot, vsel = slot[okv], vsel[okv]
        if len(vsel) == 0:
            self._count("POLYLINE", n)
            return

        vview = _view(seg, vsel)
        vx = _one(seg, vview, 10)
        vy = _one(seg, vview, 20)
        vz = _one(seg, vview, 30)
        vbul = _one(seg, vview, 42, 0.0)
        m = min(len(vx), len(slot))
        vx, vy, vz, vbul, slot = vx[:m], vy[:m], vz[:m], vbul[:m], slot[:m]

        counts = np.bincount(slot, minlength=n).astype(np.int64)
        keep = counts >= 2
        if not np.all(keep):
            starts_v = np.cumsum(counts) - counts
            idx = np.flatnonzero(keep)
            rows_v = (
                np.concatenate([np.arange(starts_v[i], starts_v[i] + counts[i]) for i in idx])
                if len(idx)
                else np.empty(0, np.int64)
            )
            vx, vy, vz, vbul = vx[rows_v], vy[rows_v], vz[rows_v], vbul[rows_v]
            counts, flags, lids = counts[idx], flags[idx], lids[idx]
            axes = self._sub_axes(axes, idx)
        if len(counts) == 0:
            self._count("POLYLINE", n)
            return

        closed = (flags & 1) != 0
        v = np.stack([vx, vy, vz], axis=1)
        if self.s.use_bulge and np.any(vbul != 0.0):
            v, counts = self._bulge_split(v, counts, closed, vbul)
            closed = np.zeros(len(counts), bool)

        if axes is not None:
            v = ocs.apply(v, np.repeat(np.arange(len(counts), dtype=np.int64), counts), axes)

        e = chain_edges(counts, closed)
        vlayer = np.repeat(lids, counts)
        acc.add(v, e, edge_layers(vlayer, e))
        self._count("POLYLINE", int(len(counts)))

    # -- HATCH -----------------------------------------------------
    def add_hatches(self, seg, sel, acc: GeoAcc):
        """填充的边界线。只取边界路径，不取填充图案本身。"""
        n = len(sel)
        if n == 0:
            return
        starts, ends = seg.starts_ends()
        n_ok = 0
        for idx in sel.tolist():
            s, e = int(starts[idx]), int(ends[idx])
            codes = seg.codes[s:e]
            vals = seg.vals[s:e]
            if len(codes) == 0:
                continue
            elevation = 0.0
            layer_name = b"0"
            got_layer = False
            got_elev = False
            nx, ny, nz = 0.0, 0.0, 1.0
            # 头部固定区域：8=图层, 10/20/30=边界整体高程, 210/220/230=所在平面法向。
            # 91 之后是边界数据，不该再往里找，否则会把某条边界边的 z 当成高程。
            for k in range(min(len(codes), 40)):
                c = int(codes[k])
                if c == 8 and not got_layer:
                    layer_name = vals[k].strip()
                    got_layer = True
                elif c == 30 and k and int(codes[k - 1]) == 20 and not got_elev:
                    try:
                        elevation = float(vals[k])
                    except ValueError:
                        pass
                    got_elev = True
                elif c == 210:
                    try:
                        nx = float(vals[k])
                    except ValueError:
                        pass
                elif c == 220:
                    try:
                        ny = float(vals[k])
                    except ValueError:
                        pass
                elif c == 230:
                    try:
                        nz = float(vals[k])
                    except ValueError:
                        pass
                elif c == 91:
                    break
            haxes = None
            if abs(nx) > 1e-12 or abs(ny) > 1e-12 or abs(nz - 1.0) > 1e-12:
                haxes = ocs.axes_from_normal(nx, ny, nz)
            chains = _hatch_boundary(codes, vals, self.s.arc_resolution)
            if not chains:
                continue
            lid = self.layer_id(layer_name)
            for pts, closed, bul in chains:
                pts = _densify_polyline(pts, closed, bul, self.s.arc_resolution)
                if len(pts) < 2:
                    continue
                v = np.empty((len(pts), 3), np.float64)
                v[:, :2] = pts
                v[:, 2] = elevation
                if haxes is not None:
                    v = ocs.apply_single(v, haxes)
                e = chain_edges(np.array([len(pts)]), [closed])
                vlayer = np.full(len(pts), lid, np.int32)
                acc.add(v, e, edge_layers(vlayer, e))
                n_ok += 1
        if n_ok:
            self._count("HATCH", n_ok)
        if n_ok < n:
            self.warnings.append(
                "%d 个填充未取到边界（边界可能由样条或自定义对象构成）" % (n - n_ok)
            )

    # -- INSERT ----------------------------------------------------
    def _insert_records(self, seg, sel):
        """解析 INSERT 列表为 (块名, 4x4 矩阵, 图层id) 记录。"""
        n = len(sel)
        view = _view(seg, sel)
        rows, owner, codes = view
        pos = first_per_owner(owner, codes == 2)
        names = [None] * n
        have = pos >= 0
        if np.any(have):
            got = gather_str(seg.vals, rows[pos[have]])
            for slot, nm in zip(np.flatnonzero(have).tolist(), got):
                names[slot] = nm.decode("utf-8", "replace")

        ix, iy, iz = _one(seg, view, 10), _one(seg, view, 20), _one(seg, view, 30)
        sx, sy, sz = _one(seg, view, 41, 1.0), _one(seg, view, 42, 1.0), _one(seg, view, 43, 1.0)
        rot = _one(seg, view, 50) * DEG
        ncol = np.maximum(1, _one(seg, view, 70, 1.0, as_int=True))
        nrow = np.maximum(1, _one(seg, view, 71, 1.0, as_int=True))
        dcol, drow = _one(seg, view, 44), _one(seg, view, 45)
        lids = self._layers(seg, view)

        sx = np.where(sx == 0, 1.0, sx)
        sy = np.where(sy == 0, 1.0, sy)
        sz = np.where(sz == 0, 1.0, sz)

        # 块引用的插入点与块自身的 XY 轴都在 OCS 里，镜像块（法向 0,0,-1）
        # 必须靠这一步翻回来，否则整个块会跑到图纸的另一侧。
        nx = _one(seg, view, 210, 0.0)
        ny = _one(seg, view, 220, 0.0)
        nz = _one(seg, view, 230, 1.0)
        if ocs.is_identity(nx, ny, nz):
            ocs_m = None
        else:
            ocs_m = [ocs.matrix(nx[i], ny[i], nz[i]) for i in range(n)]

        out = []
        for i in range(n):
            name = names[i]
            if not name:
                continue
            blk = self.blocks.get(name)
            base = blk["base"] if blk else np.zeros(3)
            for rr in range(int(nrow[i])):
                for cc in range(int(ncol[i])):
                    m = _insert_matrix(
                        ix[i], iy[i], iz[i], sx[i], sy[i], sz[i], rot[i],
                        base, cc * dcol[i], rr * drow[i],
                    )
                    if ocs_m is not None:
                        m = ocs_m[i] @ m
                    out.append((name, m, int(lids[i])))
        return out

    def add_inserts(self, seg, sel, acc: GeoAcc, depth=0):
        records = self._insert_records(seg, sel)
        missing = 0
        for name, m, lid in records:
            if name not in self.blocks:
                missing += 1
                continue
            self._emit(name, m, lid, acc, depth)
        if missing:
            self.warnings.append("%d 个块引用找不到块定义（已跳过）" % missing)
        self._count("INSERT", len(sel))

    def _emit(self, name, matrix, layer, acc: GeoAcc, depth):
        if depth > self.s.max_block_depth or name in self._stack:
            return
        blk = self.blocks.get(name)
        if blk is None:
            return
        self._stack = self._stack + (name,)
        try:
            v, e, el = blk["acc"].arrays()
            if len(v):
                vt = v @ matrix[:3, :3].T + matrix[:3, 3]
                el = np.where(el == self._zero_layer, layer, el)
                acc.add(vt, e, el)
            if self._text_out is not None:
                for it, m0, tlay in blk.get("texts", ()):
                    if len(self._text_out) >= self.s.max_texts:
                        self.texts_dropped += 1
                        continue
                    lay = layer if tlay == self._zero_layer else tlay
                    self._text_out.append((it, matrix @ m0, lay))
            for sub_name, sub_m, sub_lid in blk["children"]:
                m2 = matrix @ sub_m
                lay2 = layer if sub_lid == self._zero_layer else sub_lid
                self._emit(sub_name, m2, lay2, acc, depth + 1)
        finally:
            self._stack = self._stack[:-1]

    def _collect_texts(self, seg, sink):
        """收集 TEXT / MTEXT / ATTRIB，连同它们各自的定位矩阵。

        矩阵在这里一次性乘好（OCS × 平移 × 旋转），后面描字形时直接用它，
        块内文字再由 _emit 左乘块引用矩阵——这样块缩放、块旋转、镜像块
        对文字的作用和几何完全一致。
        """
        if not self.s.import_text or sink is None:
            return
        for kind in ("TEXT", "MTEXT", "ATTRIB"):
            sel = seg.select(kind)
            if len(sel) == 0:
                continue
            items = textmod.read_texts(seg, sel, kind, self.layer_id)
            if not items:
                continue
            view = _view(seg, sel)
            nx = _one(seg, view, 210, 0.0)
            ny = _one(seg, view, 220, 0.0)
            nz = _one(seg, view, 230, 1.0)
            ident = ocs.is_identity(nx, ny, nz)
            kept = 0
            for pos, it in items:
                if len(sink) >= self.s.max_texts:
                    self.texts_dropped += 1
                    continue
                normal = None if ident else (nx[pos], ny[pos], nz[pos])
                sink.append((it, _text_matrix(it, normal), it.layer))
                kept += 1
            self._count(kind, kept)

    def _insert_base_matrices(self, seg, sel):
        """每个 INSERT 的基础矩阵（阵列只取第 0 个），长度与 sel 一致。"""
        n = len(sel)
        if n == 0:
            return []
        view = _view(seg, sel)
        names = [None] * n
        pos = first_per_owner(view[1], view[2] == 2)
        have = pos >= 0
        if np.any(have):
            got = gather_str(seg.vals, view[0][pos[have]])
            for slot, nm in zip(np.flatnonzero(have).tolist(), got):
                names[slot] = nm.decode("utf-8", "replace")
        ix, iy, iz = _one(seg, view, 10), _one(seg, view, 20), _one(seg, view, 30)
        sx = np.where(_one(seg, view, 41, 1.0) == 0, 1.0, _one(seg, view, 41, 1.0))
        sy = np.where(_one(seg, view, 42, 1.0) == 0, 1.0, _one(seg, view, 42, 1.0))
        sz = np.where(_one(seg, view, 43, 1.0) == 0, 1.0, _one(seg, view, 43, 1.0))
        rot = _one(seg, view, 50) * DEG
        nx = _one(seg, view, 210, 0.0)
        ny = _one(seg, view, 220, 0.0)
        nz = _one(seg, view, 230, 1.0)
        ident = ocs.is_identity(nx, ny, nz)
        out = []
        for i in range(n):
            blk = self.blocks.get(names[i]) if names[i] else None
            base = blk["base"] if blk else np.zeros(3)
            m = _insert_matrix(ix[i], iy[i], iz[i], sx[i], sy[i], sz[i], rot[i],
                               base, 0.0, 0.0)
            if not ident:
                m = ocs.matrix(nx[i], ny[i], nz[i]) @ m
            out.append(m)
        return out

    def _fixup_top_attribs(self, seg, start):
        """模型空间里跟在 INSERT 后面的 ATTRIB，坐标其实是块坐标系里的。

        这类实体是"块引用的属性值"，AutoCAD 把它挂在 INSERT 后面写，
        它的 10/20 是相对块定义的，必须再乘一次那个 INSERT 的矩阵，
        否则属性文字会散到图纸外面去。
        """
        if self._text_out is None or len(self._text_out) <= start:
            return
        slots = [i for i in range(start, len(self._text_out))
                 if self._text_out[i][0].kind == "ATTRIB"]
        if not slots:
            return
        attrs = seg.select("ATTRIB")
        ins = seg.select("INSERT")
        if len(attrs) != len(slots) or len(ins) == 0:
            return
        mats = self._insert_base_matrices(seg, ins)
        rows = seg.zero[ins] if len(ins) else np.empty(0, np.int64)
        arows = seg.zero[attrs]
        for slot, arow in zip(slots, arows):
            j = int(np.searchsorted(rows, arow, side="right")) - 1
            if j < 0 or j >= len(mats):
                continue
            it, m, lay = self._text_out[slot]
            self._text_out[slot] = (it, mats[j] @ m, lay)

    # -- 块定义 ----------------------------------------------------
    def parse_blocks(self, seg: Segment):
        if seg is None:
            self._zero_layer = self.layer_id("0")
            return
        types = seg.types
        zero = seg.zero
        i, n = 0, len(types)
        while i < n:
            if types[i] != b"BLOCK":
                i += 1
                continue
            j = i + 1
            while j < n and types[j] != b"ENDBLK":
                j += 1
            name, base = _block_header(seg, zero[i])
            if name is not None:
                # 关键：子段上界必须是本块的 ENDBLK 行，否则块内最后一个实体
                # 的结束行会落到整个 BLOCKS 段末尾，把后续所有块的组码吞进来。
                sub = Segment(seg.codes, seg.vals, seg.nrows, end=int(zero[j]))
                sub._zero = zero[i + 1:j]
                sub._types = types[i + 1:j]
                cacc = GeoAcc()
                tlist = []
                self._collect_into(sub, cacc, silent=True, texts=tlist)
                for t, cnt in sub.counts().items():
                    self.block_types[t] = self.block_types.get(t, 0) + cnt
                children = []
                isel = sub.select("INSERT")
                if len(isel):
                    children = self._insert_records(sub, isel)
                self.blocks[name] = {
                    "acc": cacc,
                    "children": children,
                    "texts": tlist,
                    "base": base,
                    "weight": cacc.nv + len(children),
                }
            i = j + 1
        self._zero_layer = self.layer_id("0")

    def _collect_into(self, sub: Segment, acc: GeoAcc, silent=False, texts=None):
        self._silent = silent
        try:
            for name, fn in (
                ("LINE", lambda s: self.add_lines(sub, s, acc)),
                ("LWPOLYLINE", lambda s: self.add_lwpolyline(sub, s, acc)),
                ("ARC", lambda s: self.add_arcs(sub, s, acc, False)),
                ("CIRCLE", lambda s: self.add_arcs(sub, s, acc, True)),
                ("ELLIPSE", lambda s: self.add_ellipses(sub, s, acc)),
                ("SPLINE", lambda s: self.add_splines(sub, s, acc)),
                ("POLYLINE", lambda s: self.add_polylines(sub, s, acc)),
                ("HATCH", lambda s: self.add_hatches(sub, s, acc)),
            ):
                sel = sub.select(name)
                if len(sel):
                    fn(sel)
            self._collect_texts(sub, texts)
        finally:
            self._silent = False

    # -- 顶层 ------------------------------------------------------
    def convert(self, seg: Segment) -> GeoAcc:
        acc = GeoAcc()
        self.texts = []
        self._text_out = self.texts
        if seg is None:
            self.stats["no_entities_section"] = 1
            self._fallback_blocks(acc)
            return acc
        self._collect_into(seg, acc, texts=self.texts)
        self._fixup_top_attribs(seg, 0)
        if self.s.import_insert:
            sel = seg.select("INSERT")
            if len(sel):
                self.add_inserts(seg, sel, acc)
        self._scan_types(seg)
        if acc.empty:
            self._fallback_blocks(acc)
        return acc

    def _fallback_blocks(self, acc: GeoAcc):
        """ENTITIES 里一条线都没有时的兜底。

        两种现实情况：
        1. 几何全在 `*Model_Space` 块里（部分导出器这么写）；
        2. 这是一个"块库"文件：块定义很多，但没有任何 INSERT 引用它们。
           此时把每个块按网格铺开，至少让人看到东西，而不是空白。
        """
        for key in ("*Model_Space", "*Paper_Space", "*MODEL_SPACE", "*PAPER_SPACE"):
            blk = self.blocks.get(key)
            if blk is not None and not blk["acc"].empty:
                self._emit(key, np.eye(4), self._zero_layer, acc, 0)
                self.stats["fallback"] = "model_space"
                self.warnings.append(
                    "ENTITIES 段没有几何，已从块定义 %s 里取出（该文件把模型空间写成了块）"
                    % key
                )
                return

        if not self.s.import_unreferenced_blocks:
            return
        names = sorted(
            n for n, b in self.blocks.items()
            if not b["acc"].empty or b["children"]
        )
        if not names:
            return

        # 先各自展开成独立几何，再按包围盒排成网格
        items = []
        for name in names:
            tmp = GeoAcc()
            self._emit(name, np.eye(4), self._zero_layer, tmp, 0)
            if tmp.empty:
                continue
            v = tmp.arrays()[0]
            items.append((name, tmp, v.min(0), v.max(0)))
        if not items:
            return

        widths = np.array([max(hi[0] - lo[0], 1e-6) for _, _, lo, hi in items])
        heights = np.array([max(hi[1] - lo[1], 1e-6) for _, _, lo, hi in items])
        gap = 0.08 * max(float(widths.max()), float(heights.max()))
        # 目标行宽：让整体尽量接近正方形
        target = math.sqrt(float((widths + gap).sum() * (heights + gap).max())) * 1.15
        target = max(target, float(widths.max()) + gap)

        x = y = 0.0
        row_h = 0.0
        placed = 0
        for (name, tmp, lo, hi), w, h in zip(items, widths, heights):
            if x > 0 and x + w > target:
                x = 0.0
                y += row_h + gap
                row_h = 0.0
            off = np.array([x - lo[0], y - lo[1], 0.0])
            self._emit(name, _translation(off), self._zero_layer, acc, 0)
            x += w + gap
            row_h = max(row_h, h)
            placed += 1

        self.stats["fallback"] = "block_library"
        self.stats["fallback_blocks"] = placed
        self.warnings.append(
            "ENTITIES 段里没有任何线框，文件里的几何都在 %d 个「块定义」中且没有被引用；"
            "已按块库方式全部铺开导入" % placed
        )

    def diagnose(self) -> dict:
        """给"导入不到几何"提供事实依据，而不是让用户猜。"""
        types = {
            k.decode("utf-8", "replace"): v
            for k, v in sorted(self.entities_types.items(), key=lambda kv: -kv[1])
        }
        blk_types = {
            k.decode("utf-8", "replace"): v
            for k, v in sorted(self.block_types.items(), key=lambda kv: -kv[1])
        }
        usable = [
            n for n, b in self.blocks.items()
            if not b["acc"].empty or b["children"]
        ]
        return {
            "entities_types": types,
            "block_count": len(self.blocks),
            "blocks_with_geometry": len(usable),
            "block_types": blk_types,
            "inserts": int(self.stats.get("INSERT", 0)),
        }

    def _scan_types(self, seg: Segment):
        counts = seg.counts()
        self.entities_types = counts
        for t, c in counts.items():
            key = t.decode("utf-8", "replace")
            if key in CONSUMED:
                continue
            if key in SKIPPED_TYPES:
                k = "skipped_" + key
                self.stats[k] = self.stats.get(k, 0) + c
            elif key not in SUPPORTED and key not in (
                "SECTION", "ENDSEC", "BLOCK", "ENDBLK", "EOF"
            ):
                self.warnings.append("未识别实体 %s x%d（已跳过）" % (key, c))


# ---------------------------------------------------------------- 辅助

def _block_header(seg: Segment, start: int):
    name, base = None, np.zeros(3)
    p, end = start + 1, seg.nrows
    while p < end and seg.codes[p] != 0:
        c = seg.codes[p]
        if c == 2 and name is None:
            name = seg.vals[p].strip().decode("utf-8", "replace")
        elif c == 10:
            base[0] = float(seg.vals[p])
        elif c == 20:
            base[1] = float(seg.vals[p])
        elif c == 30:
            base[2] = float(seg.vals[p])
        p += 1
    return name, base


def _collect(seg: Segment, start: int, end: int, codes_wanted):
    if len(codes_wanted) == 3:
        cx, cy, cz = codes_wanted
        xs, ys, zs = [], [], []
        for p in range(start + 1, end):
            c = seg.codes[p]
            if c == cx:
                xs.append(float(seg.vals[p]))
            elif c == cy:
                ys.append(float(seg.vals[p]))
            elif c == cz:
                zs.append(float(seg.vals[p]))
        k = min(len(xs), len(ys))
        if k == 0:
            return []
        if len(zs) < k:
            zs = zs + [0.0] * (k - len(zs))
        return [(xs[i], ys[i], zs[i]) for i in range(k)]
    target = codes_wanted[0]
    return [float(seg.vals[p]) for p in range(start + 1, end) if seg.codes[p] == target]


def _sample_spline(ctrl, knots, degree, resolution):
    """de Boor 求值采样（对采样点向量化）；缺节点向量时退化为控制点折线。"""
    P = np.asarray(ctrl, np.float64)
    n = len(P) - 1
    if n < 1:
        return P
    p = min(max(1, int(degree)), n)
    U = np.asarray(knots, np.float64)
    if len(U) < n + p + 2:
        return P
    lo, hi = U[p], U[n + 1]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return P
    ns = int(min(4096, max(8, resolution * max(1, n))))
    ts = np.linspace(lo, hi, ns)
    k = np.clip(np.searchsorted(U, ts, side="right") - 1, p, n)
    # d[j] = P[k-p+j]，之后逐层做 de Boor 递推
    d = [P[np.clip(k - p + j, 0, n)] for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            ia = j + 1 + k - r
            ib = j + k - p
            den = U[ia] - U[ib]
            safe = np.abs(den) > 1e-15
            alpha = np.where(safe, (ts - U[ib]) / np.where(safe, den, 1.0), 0.0)
            d[j] = (1.0 - alpha)[:, None] * d[j - 1] + alpha[:, None] * d[j]
    return d[p]


def _bulge_arc(p0, p1, bulge, resolution):
    """LWPOLYLINE 的 bulge 段 -> 圆弧中间点（不含起点，含终点）。

    圆心公式与 ezdxf / Lee Mac 的实现一致：
        r_signed = chord * (1 + b^2) / (4b)
        center   = start + polar(angle(start,end) + pi/2 - 2*atan(b), r_signed)
    b>0 逆时针、b<0 顺时针，圆心角 = 4*atan(|b|)。
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    if chord < 1e-12 or abs(bulge) < 1e-12:
        return []

    b = float(bulge)
    r_signed = chord * (1.0 + b * b) / (4.0 * b)
    base_ang = math.atan2(dy, dx) + (math.pi / 2.0 - 2.0 * math.atan(b))
    cx = x0 + r_signed * math.cos(base_ang)
    cy = y0 + r_signed * math.sin(base_ang)

    radius = abs(r_signed)
    a_start = math.atan2(y0 - cy, x0 - cx)
    theta = 4.0 * math.atan(abs(b))
    sign = 1.0 if b > 0 else -1.0

    steps = max(2, int(math.ceil(resolution * theta / TWO_PI)))
    pts = [
        (cx + radius * math.cos(a_start + sign * theta * i / steps),
         cy + radius * math.sin(a_start + sign * theta * i / steps))
        for i in range(1, steps)
    ]
    pts.append((x1, y1))  # 端点用精确值，避免累积误差
    return pts


def _translation(off):
    m = np.eye(4)
    m[:3, 3] = off
    return m


def _arc_pts(cx, cy, r, a0, a1, resolution):
    """圆弧采样点（含两端）。

    DXF 里弧的起止角总是按逆时针记录，所以这里一律「逆时针从 a0 到 a1」。
    顺时针弧在文件里存的是补角（50=360-真实终点，51=360-真实起点），
    由调用方先换算，不要在这里靠符号去猜方向。
    """
    sweep = (a1 - a0) % TWO_PI
    if sweep < 1e-12:
        sweep = TWO_PI
    steps = max(2, int(math.ceil(resolution * sweep / TWO_PI)))
    t = np.linspace(0.0, sweep, steps + 1)
    return np.stack([cx + r * np.cos(a0 + t), cy + r * np.sin(a0 + t)], axis=1)


def _ellipse_pts(cx, cy, mx, my, ratio, a0, a1, resolution):
    """椭圆弧采样点。mx/my 是长轴向量，ratio 是短轴/长轴，角度同样按逆时针。"""
    nx, ny = -my * ratio, mx * ratio
    sweep = (a1 - a0) % TWO_PI
    if sweep < 1e-12:
        sweep = TWO_PI
    steps = max(4, int(math.ceil(resolution * sweep / TWO_PI)))
    t = a0 + np.linspace(0.0, sweep, steps + 1)
    return np.stack(
        [cx + mx * np.cos(t) + nx * np.sin(t), cy + my * np.cos(t) + ny * np.sin(t)],
        axis=1,
    )


def _densify_polyline(pts, closed, bul, resolution):
    """带 bulge 的边界多段线 -> 纯折线。"""
    if bul is None or not np.any(np.abs(np.asarray(bul)) > 1e-12):
        return np.asarray(pts, np.float64)
    pts = np.asarray(pts, np.float64)
    out = [tuple(pts[0])]
    n = len(pts)
    last = n if closed else n - 1
    for j in range(last):
        k = (j + 1) % n
        b = float(bul[j])
        if abs(b) > 1e-12:
            out.extend(_bulge_arc(tuple(pts[j]), tuple(pts[k]), b, resolution))
        else:
            out.append(tuple(pts[k]))
    return np.array(out, np.float64)


class _Cur:
    """HATCH 边界是嵌套的变长记录，只能顺序读；这里做个最小游标。"""

    __slots__ = ("codes", "vals", "i", "n")

    def __init__(self, codes, vals):
        self.codes = codes
        self.vals = vals
        self.i = 0
        self.n = len(codes)

    def eof(self):
        return self.i >= self.n

    def code(self):
        return int(self.codes[self.i]) if self.i < self.n else -1

    def take(self, want):
        if self.i < self.n and int(self.codes[self.i]) == want:
            v = self.vals[self.i]
            self.i += 1
            return v
        return None

    def num(self, want, default=None):
        v = self.take(want)
        if v is None:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def integer(self, want, default=None):
        v = self.num(want)
        return default if v is None else int(v)

    def skip_to(self, want):
        while self.i < self.n:
            if int(self.codes[self.i]) == want:
                return True
            self.i += 1
        return False


def _hatch_boundary(codes, vals, resolution):
    """解析 HATCH 的边界路径，返回 [(pts Nx2, closed, bulges), ...]。

    只取边界（外环 + 内洞），填充图案本身不取——那需要重新求解图案线，
    对"看轮廓"没有价值，而且极慢。
    """
    c = _Cur(codes, vals)
    if not c.skip_to(91):
        return []
    npaths = c.integer(91)
    if npaths is None or npaths <= 0:
        return []

    out = []
    for _ in range(npaths):
        if c.eof() or c.code() != 92:
            return out
        flags = c.integer(92, 0)
        if flags & 2:  # 边界直接由多段线给出
            chain = _hatch_polyline_path(c)
            if chain is None:
                return out
            out.append(chain)
        else:  # 边界由若干条边首尾相接
            nedge = c.integer(93)
            if nedge is None:
                return out
            for _ in range(nedge):
                seg = _hatch_edge(c, resolution)
                if seg is None:
                    return out
                if len(seg) >= 2:
                    out.append((seg, False, None))
        # 每条边界路径后面跟着 97（源边界对象个数）+ 对应个数的 330/360 句柄
        if c.code() == 97:
            nref = c.integer(97) or 0
            c.i += nref
    return out


def _hatch_polyline_path(c):
    has_bulge = c.integer(72, 0) or 0
    closed = c.integer(73, 0) or 0
    n = c.integer(93)
    if n is None or n < 2:
        return None
    pts = np.empty((n, 2), np.float64)
    bul = np.zeros(n, np.float64) if has_bulge else None
    for k in range(n):
        x = c.num(10)
        y = c.num(20)
        if x is None or y is None:
            return None
        pts[k] = (x, y)
        if has_bulge:
            b = c.num(42)
            if b is not None:
                bul[k] = b
    return (pts, bool(closed), bul)


def _hatch_arc_angles(a0, a1, ccw):
    """HATCH 弧边/椭圆边的起止角换算。

    DXF 对**顺时针**的填充边界弧存的是补角：组码 50 = 360 - 真实终点，
    组码 51 = 360 - 真实起点。直接按字面读会画到完全错误的位置
    （实测一张真实图纸：弧整个跑到轮廓上方，包围盒偏出 3.6）。
    换算后统一按「逆时针从起点到终点」采样即可。
    """
    if ccw:
        return a0, a1
    return 360.0 - a1, 360.0 - a0


def _hatch_edge(c, resolution):
    """读一条边界边，返回折线点。"""
    etype = c.integer(72)
    if etype is None:
        return None
    if etype == 1:  # 直线
        x1, y1 = c.num(10), c.num(20)
        x2, y2 = c.num(11), c.num(21)
        if None in (x1, y1, x2, y2):
            return None
        return np.array([[x1, y1], [x2, y2]], np.float64)
    if etype == 2:  # 圆弧
        cx, cy = c.num(10), c.num(20)
        r = c.num(40)
        a0, a1 = c.num(50), c.num(51)
        ccw = c.integer(73, 1)
        if None in (cx, cy, r, a0, a1) or r <= 0:
            return None
        a0, a1 = _hatch_arc_angles(a0, a1, ccw)
        return _arc_pts(cx, cy, r, a0 * DEG, a1 * DEG, resolution)
    if etype == 3:  # 椭圆弧
        cx, cy = c.num(10), c.num(20)
        mx, my = c.num(11), c.num(21)
        ratio = c.num(40)
        a0, a1 = c.num(50), c.num(51)
        ccw = c.integer(73, 1)
        if None in (cx, cy, mx, my, ratio, a0, a1):
            return None
        a0, a1 = _hatch_arc_angles(a0, a1, ccw)
        return _ellipse_pts(cx, cy, mx, my, ratio, a0 * DEG, a1 * DEG, resolution)
    if etype == 4:  # 样条：用控制点折线近似（填充边界够用）
        c.integer(94)  # 阶数
        c.integer(73)
        c.integer(74)
        nknots = c.integer(95, 0) or 0
        for _ in range(nknots):
            c.num(40)
        nctrl = c.integer(96)
        if nctrl is None or nctrl < 2:
            return None
        pts = np.empty((nctrl, 2), np.float64)
        for k in range(nctrl):
            x, y = c.num(10), c.num(20)
            if x is None or y is None:
                return None
            pts[k] = (x, y)
        return pts
    return None  # 未知边类型：这个填充放弃，报数量提示


def _text_matrix(item, normal=None):
    """文字自身的定位矩阵：OCS × 平移到插入点 × 绕插入点旋转。

    字高、宽度因子、倾斜角不放进来——那些是字形层面的缩放，
    描字形的时候才用；放进来会和块缩放缠在一起算不清。
    """
    m = np.eye(4)
    if normal is not None:
        m = ocs.matrix(normal[0], normal[1], normal[2])
    ang = item.rotation * DEG
    c, s = math.cos(ang), math.sin(ang)
    r = np.eye(4)
    r[0, 0], r[0, 1] = c, -s
    r[1, 0], r[1, 1] = s, c
    t = np.eye(4)
    t[0, 3], t[1, 3], t[2, 3] = item.x, item.y, item.z
    return m @ t @ r


def _insert_matrix(ix, iy, iz, sx, sy, sz, rot, base, offx, offy):
    """T(insert) · Rz · S · T(offset - base)。"""
    c, s = math.cos(rot), math.sin(rot)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c * sx, -s * sy
    m[1, 0], m[1, 1] = s * sx, c * sy
    m[2, 2] = sz
    m[0, 3] = ix + c * sx * (offx - base[0]) - s * sy * (offy - base[1])
    m[1, 3] = iy + s * sx * (offx - base[0]) + c * sy * (offy - base[1])
    m[2, 3] = iz - sz * base[2]
    return m
