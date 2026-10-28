"""编排：解析 -> 几何 -> 建对象。"""

from __future__ import annotations

import time

from . import build, dxf
from .geom import Converter, Settings


def run(
    path,
    global_scale=0.001,
    auto_unit=True,
    arc_resolution=64,
    spline_resolution=12,
    import_insert=True,
    use_bulge=True,
    by_layer=False,
    center=False,
    base_name=None,
    import_unreferenced_blocks=True,
    import_text=True,
    font_path="",
    text_resolution=6,
    solid_faces=False,
    line_width=0.0,
):
    """完整导入流程，返回统计字典（含各阶段耗时）。"""
    t0 = time.perf_counter()
    parsed = dxf.parse_file(path)
    t1 = time.perf_counter()

    meta = parsed["meta"]
    scale = global_scale
    if auto_unit:
        unit_scale = build.INSUNITS_TO_METER.get(meta.get("insunits"))
        if unit_scale:
            scale = unit_scale

    st = Settings(
        arc_resolution=arc_resolution,
        spline_resolution=spline_resolution,
        scale=scale,
        import_insert=import_insert,
        use_bulge=use_bulge,
        import_unreferenced_blocks=import_unreferenced_blocks,
        import_text=import_text,
    )
    conv = Converter(st)
    conv.parse_blocks(parsed["BLOCKS"])
    acc = conv.convert(parsed["ENTITIES"])
    t2 = time.perf_counter()

    import os

    name = base_name or os.path.splitext(os.path.basename(path))[0] or "DXF"
    objects, bstats = build.build_objects(
        acc,
        conv.layer_names,
        base_name=name,
        scale=scale,
        by_layer=by_layer,
        center=center,
        text_records=conv.texts,
        font_path=font_path or None,
        text_resolution=text_resolution,
        solid_faces=solid_faces,
        line_width=line_width,
    )
    t3 = time.perf_counter()

    return {
        "path": path,
        "file_mb": meta["bytes"] / 1048576.0,
        "scale": scale,
        "units": meta.get("insunits"),
        "layers": conv.layer_names,
        "stats": conv.stats,
        "texts": len(conv.texts),
        "texts_dropped": conv.texts_dropped,
        "warnings": conv.warnings[:20],
        "warning_count": len(conv.warnings),
        "objects": objects,
        "build": bstats,
        "diagnose": conv.diagnose(),
        "meta": {
            "padded_codes": meta.get("padded_codes", False),
            "sections": meta.get("sections", []),
            "entities_section": meta.get("sections") is not None
            and "ENTITIES" in meta.get("sections", []),
        },
        "time_parse": t1 - t0,
        "time_geom": t2 - t1,
        "time_build": t3 - t2,
        "time_total": t3 - t0,
    }


def explain(rep) -> str:
    """把"为什么没有几何"翻译成一句人话，供算子在界面上显示。"""
    d = rep["diagnose"]
    total = sum(d["entities_types"].values())
    top = sorted(d["entities_types"].items(), key=lambda kv: -kv[1])[:4]
    top_txt = "、".join("%s×%d" % (k, v) for k, v in top) or "无"
    if not d["entities_types"]:
        base = "ENTITIES 段是空的"
    else:
        base = "ENTITIES 段有 %d 个实体（%s）" % (total, top_txt)
    blk = "BLOCKS 里 %d 个块定义，其中 %d 个有几何" % (
        d["block_count"], d["blocks_with_geometry"]
    )
    if d["blocks_with_geometry"] and not d["inserts"]:
        hint = "但没有任何块引用（INSERT）；可勾选「无引用块按块库导入」"
    elif d["blocks_with_geometry"]:
        hint = "块引用 %d 个，但展开后没有线框" % d["inserts"]
    else:
        hint = "这些类型本插件不转成线框（文字/标注/代理对象），只支持线框类实体"
    return "%s；%s。%s" % (base, blk, hint)
