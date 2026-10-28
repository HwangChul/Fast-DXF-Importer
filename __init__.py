bl_info = {
    "name": "Fast DXF Importer",
    "author": "小八",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > Fast DXF",
    "description": "高性能 DXF 导入：numpy 向量化解析 + 单 mesh 合并，面向数十 MB 图纸",
    "category": "Import-Export",
}

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper


class IMPORT_OT_fast_dxf(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.fast_dxf"
    bl_label = "Fast DXF"
    bl_description = "高性能 DXF 导入（线框几何合并为单个 mesh）"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".dxf"
    filter_glob: StringProperty(default="*.dxf;*.DXF", options={"HIDDEN"})

    global_scale: FloatProperty(
        name="缩放",
        description="仅在关闭「自动单位」时生效。默认 0.001 表示按毫米导入",
        default=0.001,
        min=1e-9,
        soft_max=1000.0,
    )
    auto_unit: BoolProperty(
        name="自动单位",
        description="读取 DXF 的 $INSUNITS 并换算到米（毫米图自动 ×0.001）",
        default=True,
    )
    arc_resolution: IntProperty(
        name="圆弧细分",
        description="整圆分成多少段。越小越快，越大越圆滑",
        default=64,
        min=4,
        max=512,
    )
    spline_resolution: IntProperty(
        name="样条细分",
        description="每条样条曲线的采样密度",
        default=12,
        min=2,
        max=128,
    )
    import_insert: BoolProperty(
        name="展开块引用",
        description="展开 INSERT（块引用），阵列也一并展开",
        default=True,
    )
    import_unreferenced_blocks: BoolProperty(
        name="无引用块按块库导入",
        description="ENTITIES 里没有任何线框时，把 BLOCKS 中的块定义按网格铺开导入"
        "（块库文件、或把模型空间写成块的文件）",
        default=True,
    )
    use_bulge: BoolProperty(
        name="多段线圆弧",
        description="把 LWPOLYLINE 的 bulge 段还原成圆弧",
        default=True,
    )
    by_layer: BoolProperty(
        name="按图层拆分对象",
        description="每个图层生成一个对象。关闭时全部合并成一个（最快）",
        default=False,
    )
    center: BoolProperty(
        name="居中到原点",
        description="把几何平移到原点附近。大坐标图纸可避免 float32 精度损失",
        default=False,
    )
    import_text: BoolProperty(
        name="导入文字",
        description="把 TEXT / MTEXT / ATTRIB 描成字形轮廓折线，并入同一个网格。"
        "关闭可省时间（几千条文字会明显变慢）",
        default=True,
    )
    text_resolution: IntProperty(
        name="字形精度",
        description="字形轮廓的细分密度。越小顶点越少、越快，越大越圆滑",
        default=6,
        min=2,
        max=24,
    )
    font_path: StringProperty(
        name="字体",
        description="留空则自动找一个中英文都有的开源字体（思源黑体 / 文泉驿 / 微软雅黑…）",
        subtype="FILE_PATH",
        default="",
    )
    solid_faces: BoolProperty(
        name="线框转为实体面",
        description="把每条边扩成一条极细的四边形带，使导出 OBJ 后是真正的面。"
        "Isaac Sim / Omniverse 这类引擎会丢弃只有线段（l）的 OBJ，需要打开这个",
        default=False,
    )
    line_width: FloatProperty(
        name="线宽",
        description="转面时的带宽，图纸单位。0 表示按图纸实际跨度自动估算",
        default=0.0,
        min=0.0,
        soft_max=100.0,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="单位")
        box.prop(self, "auto_unit")
        row = box.row()
        row.enabled = not self.auto_unit
        row.prop(self, "global_scale")
        box.prop(self, "center")

        box = layout.box()
        box.label(text="几何")
        box.prop(self, "arc_resolution")
        box.prop(self, "spline_resolution")
        box.prop(self, "use_bulge")
        box.prop(self, "import_insert")
        box.prop(self, "import_unreferenced_blocks")

        box = layout.box()
        box.label(text="文字")
        box.prop(self, "import_text")
        col = box.column()
        col.enabled = self.import_text
        col.prop(self, "text_resolution")
        col.prop(self, "font_path")

        box = layout.box()
        box.label(text="导出兼容")
        box.prop(self, "solid_faces")
        row = box.row()
        row.enabled = self.solid_faces
        row.prop(self, "line_width")

        box = layout.box()
        box.label(text="组织方式")
        box.prop(self, "by_layer")

    def execute(self, context):
        from .fx import importer

        try:
            rep = importer.run(
                self.filepath,
                global_scale=self.global_scale,
                auto_unit=self.auto_unit,
                arc_resolution=self.arc_resolution,
                spline_resolution=self.spline_resolution,
                import_insert=self.import_insert,
                use_bulge=self.use_bulge,
                by_layer=self.by_layer,
                center=self.center,
                import_unreferenced_blocks=self.import_unreferenced_blocks,
                import_text=self.import_text,
                font_path=self.font_path,
                text_resolution=self.text_resolution,
                solid_faces=self.solid_faces,
                line_width=self.line_width,
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "DXF 导入失败: %s" % exc)
            return {"CANCELLED"}

        b = rep["build"]
        if b["verts"] == 0:
            why = importer.explain(rep)
            for w in rep["warnings"]:
                print("[FastDXF]", w)
            print("[FastDXF] 没有可导入的几何：", why)
            print("[FastDXF] 实体分类：", rep["diagnose"]["entities_types"])
            print("[FastDXF] 块内分类：", rep["diagnose"]["block_types"])
            self.report({"WARNING"}, "没有可导入的线框几何 — %s" % why)
            return {"CANCELLED"}

        for w in rep["warnings"]:
            print("[FastDXF]", w)

        msg = "%.2f MB -> %d 顶点 / %d 边, 耗时 %.2fs" % (
            rep["file_mb"], b["verts"], b["edges"], rep["time_total"]
        )
        print("[FastDXF]", msg)
        if rep.get("texts"):
            drawn = b.get("texts", 0)
            print(
                "[FastDXF] 文字 %d 条，已描成 %d 条字形轮廓（并入同一网格）"
                % (rep["texts"], drawn)
            )
            if not drawn:
                self.report(
                    {"WARNING"},
                    "读到 %d 条文字但没能描出字形：没找到可用中文字体，"
                    "请在「字体」里指定一个 .ttf/.otf" % rep["texts"],
                )
        if rep.get("texts_dropped"):
            print("[FastDXF] 文字超过上限，丢弃 %d 条" % rep["texts_dropped"])
        if b.get("faces"):
            print(
                "[FastDXF] 线框已转为实体面：%d 个四边形，带宽 %g"
                % (b["faces"], b.get("line_width", 0.0))
            )
        print(
            "[FastDXF] parse %.3fs | geom %.3fs | build %.3fs"
            % (rep["time_parse"], rep["time_geom"], rep["time_build"])
        )
        if rep["stats"]:
            print("[FastDXF] 实体:", rep["stats"])
        if rep["warning_count"]:
            print("[FastDXF] 警告 %d 条" % rep["warning_count"])

        self.report({"INFO"}, msg)
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_fast_dxf.bl_idname, text="Fast DXF (.dxf)")


CLASSES = (IMPORT_OT_fast_dxf,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
