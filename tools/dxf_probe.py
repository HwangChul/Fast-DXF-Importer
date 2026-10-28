# -*- coding: utf-8 -*-
"""DXF 体检工具 —— 零依赖，只用 Python 标准库。

用途：在不装 Blender 的机器上，快速看清一份 DXF 里到底有什么，
      以及哪些类型是 Fast DXF 不会导进 Blender 的。

用法：
    python dxf_probe.py 图纸.dxf
    python dxf_probe.py 图纸.dxf > 体检.txt     （输出重定向到文件）

支持 ASCII DXF（UTF-8 / GBK / UTF-16 都行）。二进制 DXF 会直接说明。
"""

import collections
import io
import os
import sys

# Fast DXF 目前会导进 Blender 的实体类型
IMPORTED = {
    "LINE", "LWPOLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "INSERT", "POLYLINE", "HATCH",
}
# 只画边框、不画填充图案
PARTIAL = {"HATCH"}
# 完全跳过
SKIPPED = {
    "TEXT", "MTEXT", "ATTDEF", "ATTRIB", "DIMENSION", "LEADER", "MLEADER",
    "POINT", "SOLID", "3DFACE", "TRACE", "IMAGE", "WIPEOUT", "VIEWPORT",
    "ACAD_TABLE", "ACAD_PROXY_ENTITY", "UNDERLAY", "OLE2FRAME", "SHAPE",
}
TEXTY = {"TEXT", "MTEXT", "ATTDEF", "ATTRIB"}


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    size = len(raw)
    if raw[:22].startswith(b"AutoCAD Binary DXF"):
        raise SystemExit("这是二进制 DXF，本工具只读 ASCII DXF。请在 CAD 里另存为 ASCII DXF 再试。")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), size, "UTF-16(带BOM)"
    if b"\x00" in raw[:4096]:
        head = raw[:4096]
        if head[1::2].count(0) > len(head) // 8:
            return raw.decode("utf-16-le", "replace"), size, "UTF-16(无BOM)"
        raise SystemExit("看起来是二进制文件（DWG 改名？），无法按 ASCII DXF 读取。")
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc), size, enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), size, "latin-1"


def load_pairs(path):
    text, size, enc = read_text(path)
    lines = text.splitlines()
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        code = lines[i].strip()
        if not code:
            continue
        try:
            pairs.append((int(code), lines[i + 1]))
        except ValueError:
            continue
    return pairs, size, enc


def sections(pairs):
    out = collections.OrderedDict()
    name = None
    buf = []
    for code, val in pairs:
        if code == 0 and val.strip() == "SECTION":
            name = "(未命名)"
            buf = []
            continue
        if code == 2 and name == "(未命名)":
            name = val.strip()
            continue
        if code == 0 and val.strip() == "ENDSEC":
            if name:
                out[name] = buf
            name = None
            continue
        if name:
            buf.append((code, val))
    return out


def entities(seg):
    """把段切成 [(类型, [(code,val), ...]), ...]"""
    out = []
    cur = None
    for code, val in seg:
        if code == 0:
            cur = (val.strip(), [])
            out.append(cur)
        elif cur is not None:
            cur[1].append((code, val))
    return out


def first(ent, code, default=None):
    for c, v in ent:
        if c == code:
            return v
    return default


def header_var(pairs, var):
    for i, (code, val) in enumerate(pairs):
        if val.strip() == var and code == 9:
            for c, v in pairs[i + 1:i + 4]:
                if c in (70, 1, 2, 3, 40):
                    return v.strip()
    return None


def main(path):
    pairs, size, enc = load_pairs(path)
    secs = sections(pairs)
    acadver = header_var(pairs, "$ACADVER") or "?"
    insunits = header_var(pairs, "$INSUNITS") or "?"
    unit_names = {
        "0": "未指定", "1": "英寸", "2": "英尺", "4": "毫米", "5": "厘米",
        "6": "米", "7": "千米",
    }

    print("=" * 66)
    print("文件: %s" % os.path.basename(path))
    print("大小: %.2f MB   编码: %s   DXF 版本: %s   $INSUNITS: %s (%s)"
          % (size / 1048576.0, enc, acadver, insunits,
             unit_names.get(insunits, "未知")))
    print("段: %s" % ", ".join(secs.keys()))
    print("=" * 66)

    ents = entities(secs.get("ENTITIES", []))
    msp = collections.Counter(t for t, _ in ents)

    # BLOCKS：按块名分组
    blk_ents = collections.Counter()
    blk_names = []
    blocks = {}
    cur_name = None
    for t, body in entities(secs.get("BLOCKS", [])):
        if t == "BLOCK":
            cur_name = first(body, 2) or "?"
            blk_names.append(cur_name.strip())
            blocks[cur_name.strip()] = collections.Counter()
        elif t == "ENDBLK":
            cur_name = None
        elif t not in ("SEQEND",):
            blk_ents[t] += 1
            if cur_name:
                blocks[cur_name.strip()][t] += 1

    def show(counter, title):
        if not counter:
            print("%s: 空" % title)
            return
        total = sum(counter.values())
        print("%s: %d 个" % (title, total))
        for t, n in counter.most_common():
            if t in SKIPPED:
                mark = "  ← 不会导入"
            elif t in PARTIAL:
                mark = "  ← 只导边界，不导填充图案"
            elif t in IMPORTED:
                mark = ""
            elif t in ("SEQEND", "VERTEX"):
                mark = ""
            else:
                mark = "  ← 未知类型"
            print("   %-22s %7d%s" % (t, n, mark))

    show(msp, "模型空间 ENTITIES")
    print()
    show(blk_ents, "块定义内实体合计（%d 个块）" % len(blk_names))

    # ---- 文字体检 ----
    print()
    print("-" * 66)
    txt_items = [(t, b) for t, b in ents if t in TEXTY]
    print("模型空间文字/标注: %d 个" % len(txt_items))
    if txt_items:
        xs, ys = [], []
        for t, body in txt_items:
            x, y = first(body, 10), first(body, 20)
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                pass
        if xs:
            print("   位置范围: X %.1f ~ %.1f   Y %.1f ~ %.1f"
                  % (min(xs), max(xs), min(ys), max(ys)))
        sample = collections.Counter(t for t, _ in txt_items)
        print("   类型分布: %s" % dict(sample))
        print("   内容样本（前 15 条）:")
        shown = 0
        for t, body in txt_items:
            if shown >= 15:
                break
            s = first(body, 1) or first(body, 3) or ""
            s = s.replace("\\P", " / ").strip()
            if not s:
                continue
            print("     %-7s %s" % (t, s[:60]))
            shown += 1

    # ---- 块内文字（图框标题栏常在这里）----
    blk_text = {n: c for n, c in blocks.items()
                if sum(c.get(k, 0) for k in TEXTY)}
    if blk_text:
        print()
        print("含文字的块定义（标题栏/图例通常在这里）: %d 个" % len(blk_text))
        for n, c in sorted(blk_text.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
            marks = ", ".join("%s×%d" % (k, v) for k, v in c.items() if k in TEXTY)
            print("   %-28s %s" % (n, marks))

    # ---- 字体表 ----
    styles = []
    for t, body in entities(secs.get("TABLES", [])):
        if t == "STYLE":
            styles.append((first(body, 2), first(body, 3), first(body, 4)))
    if styles:
        print()
        print("文字样式表 TABLES/STYLE（%d 个）:" % len(styles))
        for name, font, bigfont in styles[:12]:
            print("   %-16s 字体=%-18s 大字体=%s" % (name, font, bigfont))

    # ---- 结论 ----
    print()
    print("=" * 66)
    lost = {t: n for t, n in msp.items() if t in SKIPPED}
    lost_blk = {t: n for t, n in blk_ents.items() if t in SKIPPED}
    if lost or lost_blk:
        print("结论：有内容不会被导入 Blender")
        if lost:
            print("   模型空间丢失: %s" % ", ".join("%s×%d" % kv for kv in lost.items()))
        if lost_blk:
            print("   块内丢失:     %s" % ", ".join("%s×%d" % kv for kv in lost_blk.items()))
        n_text = sum(lost.get(k, 0) for k in TEXTY)
        if n_text:
            print("   → 其中 %d 条文字在 CAD 里可见、Blender 里完全没有" % n_text)
    else:
        print("结论：没有发现会被丢弃的实体类型")
    print("=" * 66)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main(sys.argv[1])
