"""文字实体：TEXT / MTEXT / ATTRIB 的解析与格式码清理。

本模块只产出纯数据，不依赖 bpy —— 字形转折线在 build.py 里做。
这样拆的好处很实在：解析层可以脱离 Blender 单独跑测试对拍。
"""

from __future__ import annotations

import math

DEG = math.pi / 180.0

# DXF 72（水平对齐）→ 对齐方式
# 3=aligned / 4=middle / 5=fit 都按"以插入点为基准"近似
H_ALIGN = {0: "LEFT", 1: "CENTER", 2: "RIGHT", 3: "LEFT", 4: "CENTER", 5: "LEFT"}
# DXF 73（垂直对齐）
V_ALIGN = {0: "BOTTOM_BASELINE", 1: "BOTTOM", 2: "CENTER", 3: "TOP"}
# MTEXT 71（附着点）1..9：左上 → 右下
MT_ALIGN = {
    1: ("LEFT", "TOP"),
    2: ("CENTER", "TOP"),
    3: ("RIGHT", "TOP"),
    4: ("LEFT", "CENTER"),
    5: ("CENTER", "CENTER"),
    6: ("RIGHT", "CENTER"),
    7: ("LEFT", "BOTTOM"),
    8: ("CENTER", "BOTTOM"),
    9: ("RIGHT", "BOTTOM"),
}

SYMBOLS = (
    ("%%d", "\u00b0"),
    ("%%D", "\u00b0"),
    ("%%c", "\u2300"),
    ("%%C", "\u2300"),
    ("%%p", "\u00b1"),
    ("%%P", "\u00b1"),
    ("%%%", "%"),
)


class TextItem:
    """一条文字。坐标是所在坐标系（模型空间 / 块内）的原始值。"""

    __slots__ = (
        "body", "x", "y", "z", "height", "rotation", "width", "oblique",
        "align_x", "align_y", "style", "layer", "kind", "line_spacing",
        "ref_width",
    )

    def __init__(self, body, x, y, z, height, rotation, width, oblique,
                 align_x, align_y, style, layer, kind, line_spacing=1.0,
                 ref_width=0.0):
        self.body = body
        self.x = x
        self.y = y
        self.z = z
        self.height = height
        self.rotation = rotation
        self.width = width
        self.oblique = oblique
        self.align_x = align_x
        self.align_y = align_y
        self.style = style
        self.layer = layer
        self.kind = kind
        self.line_spacing = line_spacing
        self.ref_width = ref_width

    def __repr__(self):
        return "TextItem(%r @ %.2f,%.2f h=%.2f %s)" % (
            self.body[:12], self.x, self.y, self.height, self.kind,
        )


def decode_value(v) -> str:
    """DXF 字符串解码。

    早期版本只按 UTF-8 解，遇到 ANSI_936（GBK）的老图纸，中文全变成替换符。
    这里按 UTF-8 → GBK → Big5 → Latin-1 依次试，都不成就替换。
    """
    if not isinstance(v, bytes):
        return str(v)
    try:
        return v.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in ("gbk", "big5", "cp1252"):
        try:
            return v.decode(enc)
        except UnicodeDecodeError:
            continue
    return v.decode("utf-8", "replace")


def _unicode_escapes(s: str) -> str:
    """把 MTEXT 里的 `\\U+00B0` 这类转义换成真字符。"""
    if "\\U+" not in s:
        return s
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("\\U+", i) and i + 7 <= n:
            hexpart = s[i + 3:i + 7]
            try:
                out.append(chr(int(hexpart, 16)))
                i += 7
                continue
            except ValueError:
                pass
        out.append(s[i])
        i += 1
    return "".join(out)


def clean_mtext(s: str) -> str:
    """剥掉 MTEXT 的内联格式码，留下人能读的文字。

    `\\P` 段落换行 → 换行；`\\S1^2;` 堆叠 → `1/2`；其余 `\\X...;` 控制码整段丢弃。
    宁可丢样式，也不能把 "\\fSimSun|b0|i0;图纸" 这种原文画成图形。
    """
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break
        nxt = s[i + 1]
        if nxt == "P":
            out.append("\n")
            i += 2
        elif nxt in "Nn":
            out.append("\n")
            i += 2
        elif nxt == "~":
            out.append(" ")
            i += 2
        elif nxt in "\\{}":
            out.append(nxt)
            i += 2
        elif nxt == "S":
            j = s.find(";", i + 2)
            if j < 0:
                i += 2
                continue
            out.append(s[i + 2:j].replace("^", "/"))
            i = j + 1
        elif nxt in "AaCcFfHhQqTtWwXx":
            j = s.find(";", i + 2)
            i = (j + 1) if j > 0 else i + 2
        elif nxt in "LlOoKk":
            i += 2
        else:
            i += 2
    return _finish(out)


def _finish(parts) -> str:
    text = "".join(parts)
    for a, b in SYMBOLS:
        if a in text:
            text = text.replace(a, b)
    # %%u / %%o 是下划线、上划线的开关，不是可见字符
    for tok in ("%%u", "%%U", "%%o", "%%O"):
        if tok in text:
            text = text.replace(tok, "")
    return _unicode_escapes(text)


def read_texts(seg, sel, kind: str, layer_id) -> list:
    """从段里取出指定类型的所有文字，返回 TextItem 列表。

    kind: "TEXT" | "MTEXT" | "ATTRIB"
    layer_id: 字节图层名 → 整数 id 的函数
    """
    if len(sel) == 0:
        return []
    starts, ends = seg.starts_ends()
    out = []
    for pos, idx in enumerate(sel):
        a, b = int(starts[idx]), int(ends[idx])
        if b <= a:
            continue
        item = _parse_one(seg, a, b, kind, layer_id)
        if item is not None:
            out.append((pos, item))
    return out


def _parse_one(seg, a, b, kind, layer_id):
    codes = seg.codes
    vals = seg.vals
    body = []
    x = y = z = 0.0
    ax = ay = 0.0
    has_align = False
    height = 2.5
    rot = 0.0
    width = 1.0
    oblique = 0.0
    style = "Standard"
    layer = 0
    ha = va = 0
    attach = 1
    line_spacing = 1.0
    ref_width = 0.0
    dx = dy = None

    for p in range(a + 1, b):
        c = int(codes[p])
        if c == 101:
            # 「Embedded Object」从这里开始，后面是内嵌对象的另一套组码
            # （10/11/40/41 全部重复一遍，含义不同）。不截断就会把字高读成
            # 参考宽度、把插入点当成方向向量，直接导致字歪 + 字号全错。
            break
        if c == 1 or c == 3:
            body.append(decode_value(vals[p]))
        elif c == 10:
            x = float(vals[p])
        elif c == 20:
            y = float(vals[p])
        elif c == 30:
            z = float(vals[p])
        elif c == 11:
            # TEXT/ATTRIB：第二对齐点；MTEXT：X 轴方向向量
            ax = float(vals[p])
            has_align = True
            if kind == "MTEXT":
                dx = ax
        elif c == 21:
            ay = float(vals[p])
            if kind == "MTEXT":
                dy = ay
        elif c == 40:
            height = float(vals[p])
        elif c == 41:
            # 关键区别：MTEXT 的 41 是「参考矩形宽度」，单位是图形单位
            # （常见几十到几千）；TEXT/ATTRIB 的 41 才是「相对 X 缩放因子」，
            # 默认 1.0。把两者混用会让文字被横向拉伸 ref_width 倍。
            if kind == "MTEXT":
                ref_width = float(vals[p])
            else:
                width = float(vals[p])
        elif c == 50:
            rot = float(vals[p])
        elif c == 51:
            oblique = float(vals[p])
        elif c == 7:
            style = decode_value(vals[p]).strip() or "Standard"
        elif c == 8:
            layer = layer_id(vals[p])
        elif c == 72:
            ha = int(float(vals[p]))
        elif c == 73:
            va = int(float(vals[p]))
        elif c == 71 and kind == "MTEXT":
            attach = int(float(vals[p]))
        elif c == 44 and kind == "MTEXT":
            line_spacing = float(vals[p]) or 1.0

    if kind == "MTEXT":
        text = clean_mtext("".join(body))
        # 11/21 是 X 轴方向向量；只有两个分量都在时才能定角度，
        # 缺一个就退回 50 组的旋转角（早先在这里补默认值 1.0，结果文字全歪 45°）
        if dx is not None and dy is not None:
            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                rot = math.degrees(math.atan2(dy, dx))
        align_x, align_y = MT_ALIGN.get(attach, ("LEFT", "TOP"))
    else:
        text = _finish(body)
        # 有对齐点且对齐方式非默认时，DXF 以 11/21 为实际定位点
        if has_align and (ha != 0 or va != 0):
            x, y = ax, ay
        align_x = H_ALIGN.get(ha, "LEFT")
        align_y = V_ALIGN.get(va, "BOTTOM_BASELINE")

    text = text.replace("\r", "")
    if not text.strip():
        return None
    if height <= 0:
        height = 2.5
    if not (0.01 <= width <= 100.0):
        width = 1.0
    return TextItem(text, x, y, z, height, rot, width, oblique,
                    align_x, align_y, style, layer, kind, line_spacing,
                    ref_width)
