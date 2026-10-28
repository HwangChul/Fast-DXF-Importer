"""生成公开测试用例与期望值。

开发期脚本，**需要 ezdxf**（测试本身不需要）。用法：

    python tests/make_cases.py

产物写进 tests/cases/：
    xxx.dxf    合成图纸
    xxx.json   期望值（由 ezdxf 独立算出，含包围盒、文字位置、实体计数）

用例覆盖的都是实际踩过的坑，不是凑数的样例。
"""
import json
import os
import sys

import ezdxf
from ezdxf import bbox

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")


# ---------------------------------------------------------------- 工具

def text_anchor(e):
    """TEXT 的有效定位点：halign/valign 非 0 时 10 组码只是占位，真点位在 11。"""
    kind = e.dxftype()
    if kind == "TEXT":
        if e.dxf.halign or e.dxf.valign:
            return e.dxf.align_point
        return e.dxf.insert
    return e.dxf.insert


def collect_expect(doc, name, desc):
    msp = doc.modelspace()
    e = bbox.extents(msp, fast=True)
    texts = []
    for en in msp:
        if en.dxftype() in ("TEXT", "MTEXT"):
            p = text_anchor(en)
            body = en.dxf.text if en.dxftype() == "TEXT" else en.text
            texts.append({"body": body, "pos": [p.x, p.y, p.z]})
        elif en.dxftype() == "INSERT":
            # 块内文字也算（验证块内文字跟着矩阵走）
            for sub in en.virtual_entities():
                if sub.dxftype() in ("TEXT", "MTEXT"):
                    p = text_anchor(sub)
                    body = sub.dxf.text if sub.dxftype() == "TEXT" else sub.text
                    texts.append({"body": body, "pos": [p.x, p.y, p.z]})

    return {
        "name": name,
        "desc": desc,
        "bbox_min": [round(e.extmin.x, 4), round(e.extmin.y, 4), round(e.extmin.z, 4)],
        "bbox_max": [round(e.extmax.x, 4), round(e.extmax.y, 4), round(e.extmax.z, 4)],
        "texts": texts,
    }


def save(doc, name, desc, **extra):
    path = os.path.join(CASES, name + ".dxf")
    doc.saveas(path)
    data = collect_expect(doc, name, desc)
    data.update(extra)
    with open(os.path.join(CASES, name + ".json"), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print("  %-22s 包围盒 %s -> %s  文字 %d 条%s" % (
        name, data["bbox_min"][:2], data["bbox_max"][:2], len(data["texts"]),
        "  (跳过包围盒比对)" if extra.get("skip_bbox") else ""))


def new_doc():
    return ezdxf.new("R2013", setup=True)


# ---------------------------------------------------------------- 用例

def case_leading_space():
    """组码行带前导空格（AutoCAD 的标准写法）。原实现会整个读空。"""
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 6))
    msp.add_circle((5, 3), radius=2)
    path = os.path.join(CASES, "leading_space.dxf")
    doc.saveas(path)
    data = collect_expect(doc, "leading_space", "组码前导空格")

    # 逐行处理：只给"组码行"加两个前导空格，值行不动
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fp:
        lines = fp.read().splitlines()
    out = []
    for i, line in enumerate(lines):
        if i % 2 == 0:  # 偶数行是组码
            out.append("  " + line.strip())
        else:
            out.append(line)
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as fp:
        fp.write("\n".join(out) + "\n")

    with open(os.path.join(CASES, "leading_space.json"), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print("  %-22s 包围盒 %s -> %s  文字 %d 条（已加前导空格）" % (
        data["name"], data["bbox_min"][:2], data["bbox_max"][:2], len(data["texts"])))


def case_block_name_special():
    """块名含 $ 和 #（真实工程图里很常见，Blender 官方导入器正死在这）。"""
    doc = new_doc()
    name = "J26-9#楼底图$0$$lvtry$00000179"
    blk = doc.blocks.new(name=name)
    blk.add_line((0, 0), (4, 0))
    blk.add_line((4, 0), (4, 3))
    blk.add_text("底图", dxfattribs={"height": 0.8, "insert": (0.5, 1.0)})

    msp = doc.modelspace()
    msp.add_blockref(name, (0, 0))
    msp.add_blockref(name, (10, 0), dxfattribs={"rotation": 90})
    save(doc, "block_name_special", "块名含 $ 与 #")


def case_mirror_ocs():
    """法向 (0,0,-1) 的实体：OCS 会把 (x,y) 映射到 (-x,y)。"""
    doc = new_doc()
    msp = doc.modelspace()
    ext = {"extrusion": (0, 0, -1)}

    msp.add_line((2, 0), (2, 5))                       # LINE 是 WCS，不该被翻转
    msp.add_circle((2, 3), radius=1, dxfattribs=ext)   # 应落到 (-2, 3)
    msp.add_lwpolyline([(0, 0), (3, 0), (3, 2)], dxfattribs=ext)
    msp.add_arc((1, 1), radius=1.5, start_angle=0, end_angle=90, dxfattribs=ext)

    blk = doc.blocks.new(name="MIRBLK")
    blk.add_line((0, 0), (2, 0))
    msp.add_blockref("MIRBLK", (5, 5), dxfattribs={"xscale": -1})
    msp.add_blockref("MIRBLK", (-5, 5), dxfattribs=ext)

    save(doc, "mirror_ocs", "镜像 OCS：circle/lwpolyline/arc/insert")


def case_text_width():
    """TEXT 的 41 是宽度因子；MTEXT 的 41 是参考宽度。混用会横向拉伸。"""
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_text("W06", dxfattribs={"height": 2.5, "width": 0.6, "insert": (0, 0)})
    msp.add_text("W18", dxfattribs={"height": 2.5, "width": 1.8, "insert": (0, 5)})
    for ref in (0.0, 40.0, 1000.0):
        msp.add_mtext("参考宽度%.0f" % ref, dxfattribs={
            "char_height": 2.5, "width": ref, "insert": (0, 10 + ref * 0.01),
        })
    # MTEXT 的参考宽度会让 ezdxf 按折行算包围盒，而本插件刻意不折行，
    # 所以这个用例只比文字位置，不比包围盒。
    save(doc, "text_width", "TEXT 宽度因子 vs MTEXT 参考宽度", skip_bbox=True)


def case_bulge():
    """直线与 bulge 圆弧混排：顶点过滤后 bulge 数组必须同步截断。"""
    doc = new_doc()
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0, 0, 0, 0.5), (4, 0, 0, 0, 0), (4, 3, 0, 0, -0.5), (0, 3, 0, 0, 0)],
        format="xyseb", close=True,
    )
    msp.add_lwpolyline([(6, 0), (9, 0), (9, 2), (6, 2)], close=True)
    save(doc, "bulge", "LWPOLYLINE 正负 bulge 与直线混排")


def case_nested_block():
    """嵌套块：内层缩放 + 旋转，外层阵列两次。"""
    doc = new_doc()
    inner = doc.blocks.new(name="INNER")
    inner.add_line((0, 0), (2, 0))
    inner.add_circle((1, 0), radius=0.5)

    outer = doc.blocks.new(name="OUTER")
    outer.add_blockref("INNER", (0, 0), dxfattribs={"xscale": 2, "yscale": 2, "rotation": 45})
    outer.add_line((0, 0), (0, 1))

    msp = doc.modelspace()
    msp.add_blockref("OUTER", (0, 0))
    msp.add_blockref("OUTER", (10, 0))
    save(doc, "nested_block", "嵌套块 + 缩放旋转 + 多次引用")


BUILDERS = [
    case_leading_space,
    case_block_name_special,
    case_mirror_ocs,
    case_text_width,
    case_bulge,
    case_nested_block,
]


def main():
    os.makedirs(CASES, exist_ok=True)
    print("生成用例 ->", CASES)
    for fn in BUILDERS:
        fn()
    names = sorted(f[:-4] for f in os.listdir(CASES) if f.endswith(".dxf"))
    print("\n共 %d 个用例：%s" % (len(names), ", ".join(names)))


if __name__ == "__main__":
    sys.exit(main())
