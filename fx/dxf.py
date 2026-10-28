"""DXF 词法与段定位（不依赖 bpy，可单独测试）。

设计要点
--------
DXF(ASCII) 是严格的 `组码\\n值\\n` 交替文本。整文件 split 后按奇偶取列，
比逐行 readline / 逐字符状态机快一个数量级（300 万行约 70ms）。
组码用 numpy 批量转 int，数值用 numpy 批量转 float，避开 Python 逐点循环。
"""

from __future__ import annotations

import re

import numpy as np

# 只解析这两个段：实体 + 块定义。HEADER/TABLES/OBJECTS 直接丢弃。
WANTED_SECTIONS = (b"ENTITIES", b"BLOCKS")


class DxfFormatError(ValueError):
    """文件根本不是 ASCII DXF（二进制 DXF / DWG / 编码异常）。"""


# 带前导空白的组码是合法 DXF：AutoCAD 及多数导出器把组码右对齐到 3 列
# （`  0` / ` 10`），只有 ezdxf 这类现代库才写不带空白的组码。
# 快速路径用固定字节串匹配，失败后回退到这个正则。
_SEC_RE = re.compile(
    rb"(?m)^[ \t]*0[ \t]*\n[ \t]*SECTION[ \t]*\n[ \t]*2[ \t]*\n[ \t]*([^\n]+)"
)
_END_SEC_RE = re.compile(rb"(?m)^[ \t]*0[ \t]*\n[ \t]*ENDSEC\b")
_INSUNITS_RE = re.compile(rb"\$INSUNITS[^\n]*\n[ \t]*70[ \t]*\n[ \t]*(-?\d+)")


class Segment:
    """一个 SECTION 内的扁平组码/值数组。"""

    __slots__ = ("codes", "vals", "_zero", "_types", "nrows", "_end")

    def __init__(self, codes: np.ndarray, vals: list, nrows: int, end=None):
        self.codes = codes
        self.vals = vals
        self.nrows = nrows
        # 子段（如单个块定义）的上界。默认 None 表示到 nrows；
        # 块子段必须显式给定 ENDBLK 行，否则块内最后一个实体会吞掉后续所有块的数据。
        self._end = end
        self._zero = None
        self._types = None

    # ---- 实体边界 -------------------------------------------------
    @property
    def zero(self) -> np.ndarray:
        """所有组码为 0 的行号（每个实体/记录的开头）。"""
        if self._zero is None:
            self._zero = np.flatnonzero(self.codes == 0)
        return self._zero

    @property
    def types(self) -> list:
        """zero 行对应的实体类型名（bytes）。"""
        if self._types is None:
            self._types = [self.vals[p].strip() for p in self.zero]
        return self._types

    def starts_ends(self) -> tuple:
        z = self.zero
        if len(z) == 0:
            return z, z
        end = self.nrows if self._end is None else int(self._end)
        return z, np.append(z[1:], end)

    def select(self, type_name) -> np.ndarray:
        """返回某类型实体的序号数组（在 types 中的位置）。"""
        want = type_name.encode() if isinstance(type_name, str) else type_name
        types = self.types
        return np.fromiter(
            (i for i, t in enumerate(types) if t == want), dtype=np.int64, count=-1
        )

    def counts(self) -> dict:
        out = {}
        for t in self.types:
            out[t] = out.get(t, 0) + 1
        return out


def normalize_newlines(raw: bytes) -> bytes:
    if b"\r" in raw:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return raw


def _sniff(raw: bytes) -> bytes:
    """把各种编码归一成 UTF-8 字节的 ASCII DXF 文本。"""
    if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return raw[3:]
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16").encode("utf-8")
        except UnicodeDecodeError:
            pass
    # 无 BOM 的 UTF-16：偶数位大量 NUL
    head = raw[:4096]
    if head.count(b"\x00") > len(head) // 4:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "SECTION" in text.upper():
                return text.encode("utf-8")
    return raw


def find_sections(raw: bytes) -> dict:
    """定位 SECTION 字节范围: name -> (start, end)。同名段只取第一个。"""
    out = {}
    marker = b"0\nSECTION\n2\n"
    pos = 0
    n = len(raw)
    while True:
        pos = raw.find(marker, pos)
        if pos < 0:
            break
        head = pos + len(marker)
        nl = raw.find(b"\n", head)
        if nl < 0:
            break
        name = raw[head:nl].strip().upper()
        end = raw.find(b"0\nENDSEC", nl)
        if end < 0:
            end = n
        if name not in out:
            # 从段名那一行的下一行开始，保证以组码行开头
            out[name] = (nl + 1, end)
        pos = end + 1
    if out:
        return out
    return _find_sections_tolerant(raw)


def _find_sections_tolerant(raw: bytes) -> dict:
    """组码带前导空白时（`  0` / ` 10`）的慢路径。只在快路径完全失败时跑。"""
    out = {}
    for m in _SEC_RE.finditer(raw):
        name = m.group(1).strip().upper()
        if name in out:
            continue
        tail = _END_SEC_RE.search(raw, m.end())
        end = tail.start() if tail else len(raw)
        out[name] = (m.end() + 1, end)
    return out


def _safe_int_list(chunks) -> np.ndarray:
    out = np.empty(len(chunks), dtype=np.int32)
    for i, c in enumerate(chunks):
        try:
            out[i] = int(c)
        except (ValueError, TypeError):
            out[i] = -1
    return out


def split_pairs(raw: bytes) -> tuple:
    """把 `组码\\n值` 交替文本拆成 (int32 codes, list[bytes] vals)。"""
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) % 2:  # 尾行落单，丢弃容错
        lines.pop()
    code_chunks = lines[0::2]
    vals = lines[1::2]
    try:
        codes = np.array(code_chunks, dtype="S8").astype(np.int32)
    except (ValueError, TypeError):
        codes = _safe_int_list(code_chunks)
    return codes, vals


def parse_file(path: str) -> dict:
    """解析 DXF，返回 {"ENTITIES": Segment, "BLOCKS": Segment, "meta": ...}。

    只对需要的段做 tokenize，其余字节直接跳过（大文件里 OBJECTS/TABLES
    往往占三成以上）。返回值中的 meta 记录文件大小等诊断信息。
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    total = len(raw)
    if raw.startswith(b"AutoCAD Binary DXF"):
        raise DxfFormatError(
            "这是「二进制 DXF」，不是文本格式；请在 CAD 中另存为 ASCII DXF 再导入"
        )
    if raw[:6].startswith(b"AC10") and b"SECTION" not in raw[:4096]:
        raise DxfFormatError(
            "文件头是 DWG（AC10xx），只是扩展名被改成了 .dxf；请用 CAD 导出成 DXF"
        )

    raw = normalize_newlines(_sniff(raw))
    spans = find_sections(raw)
    if not spans:
        head = raw[:80].decode("utf-8", "replace").replace("\n", " ")
        raise DxfFormatError(
            "找不到任何 SECTION，文件可能已损坏或不是 ASCII DXF（开头：%s）" % head
        )

    result = {
        "meta": {
            "bytes": total,
            "sections": [k.decode("ascii", "replace") for k in spans.keys()],
            "padded_codes": bool(re.match(rb"[ \t]+0[ \t]*\n", raw[:64])),
        }
    }

    for name in WANTED_SECTIONS:
        span = spans.get(name)
        if span is None:
            result[name.decode()] = None
            continue
        start, end = span
        codes, vals = split_pairs(raw[start:end])
        result[name.decode()] = Segment(codes, vals, len(codes))

    # 快速抓 $INSUNITS（HEADER 很小区间，单独扫）
    result["meta"]["insunits"] = _read_insunits(raw, spans.get(b"HEADER"))
    return result


def _read_insunits(raw: bytes, span) -> int | None:
    if span is None:
        return None
    s, e = span
    m = _INSUNITS_RE.search(raw, s, e)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ---- 实体属性提取工具 ---------------------------------------------

def expand_ranges(starts: np.ndarray, ends: np.ndarray) -> tuple:
    """把 [start, end) 区间展开成扁平行号 + 所属实体序号。

    例: starts=[0,4], ends=[4,9] -> idx=[0..8], owner=[0,0,0,0,1,1,1,1,1]
    """
    lens = (ends - starts).astype(np.int64)
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    offsets = np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(starts, lens) + (np.arange(total) - offsets)
    owner = np.repeat(np.arange(len(starts), dtype=np.int64), lens)
    return idx, owner


def gather_floats(vals: list, rows: np.ndarray, dtype="S32") -> np.ndarray:
    """取指定行的值并转 float64。空输入返回空数组。"""
    if len(rows) == 0:
        return np.empty(0, np.float64)
    picked = [vals[i] for i in rows]
    return np.array(picked, dtype=dtype).astype(np.float64)


def gather_ints(vals: list, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.empty(0, np.int64)
    picked = [vals[i] for i in rows]
    return np.array(picked, dtype="S24").astype(np.int64)


def gather_str(vals: list, rows: np.ndarray) -> list:
    return [vals[i].strip() for i in rows]


def first_per_owner(owner: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """对每个实体，取 mask 命中的第一条行的值位置。

    返回长度为实体数的数组，未命中的位置为 -1。
    """
    n_owner = int(owner.max()) + 1 if len(owner) else 0
    out = np.full(n_owner, -1, dtype=np.int64)
    hit = np.flatnonzero(mask)
    if len(hit) == 0:
        return out
    own = owner[hit]
    uniq, first_idx = np.unique(own, return_index=True)  # 每个 owner 首次出现
    out[uniq] = hit[first_idx]
    return out
