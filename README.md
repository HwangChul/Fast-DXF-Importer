# Fast DXF Importer

[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange.svg)](https://www.blender.org)

面向数十 MB 图纸的 Blender DXF 导入器：numpy 向量化解析 + 单 mesh 合并。

## 特点

| | |
|---|---|
| **块引用** | 阵列、嵌套、缩放、旋转，以及带挤出方向（210/220/230，OCS）的镜像块 |
| **文字** | `TEXT` / `MTEXT` / `ATTRIB` 描成字形轮廓折线并入同一网格，中英文混排正常 |
| **可导出给仿真** | 可选把线框转成实体面，导出 OBJ 后是真正的面，供 Isaac Sim / Omniverse 使用 |
| **脏文件容错** | 组码前导空格、UTF-16、GBK / Big5 编码、二进制 DXF、DWG 改扩展名，都能读出内容或给出明确提示 |
| **体检脚本** | `tools/dxf_probe.py` 只依赖标准库，可在不开 Blender 的情况下看一份图纸里到底有什么 |

## 安装

1. 下载 `fast_dxf-0.2.0.zip`（GitHub Release 或本地 `dist/`）
2. Blender 4.2+ → `编辑 > 偏好设置 > 扩展` → 右上角下拉 → `从磁盘安装`
3. 选择 zip，启用扩展

菜单位置：`文件 > 导入 > Fast DXF (.dxf)`。

## 用法

1. `文件 > 导入 > Fast DXF (.dxf)`，选图纸。
2. 单位保持「自动单位」即可；尺寸明显不对时关掉它，按 `缩放 = 目标米数 ÷ 图纸单位数` 手填。
3. 要交给 Isaac Sim / Omniverse，勾上「线框转为实体面」。
4. 实体着色模式（Solid）下线框默认看不见，按 `Shift+Z` 切线框模式。

导入结束后控制台会打印顶点数、边数、耗时、图层统计和最终尺寸。

## 参数

**单位**

- `自动单位`：读 DXF 的 `$INSUNITS` 换算到米（毫米图自动 ×0.001）。
  这个字段由出图软件填写，尺寸明显错误时手动填缩放。
- `缩放`：`缩放 = 你想让模型多大（米） ÷ 图纸单位数`。例如某张模型图宽 31115 单位，按毫米填 `0.001`。
- `居中到原点`：大坐标图纸（如测量坐标 4,000,000 级）建议打开，避免 float32 精度损失。

**几何**

- `圆弧细分`（默认 64）：整圆分多少段。越小越快。
- `样条细分`（默认 12）：每条样条的采样点密度。
- `多段线圆弧`：把 `LWPOLYLINE` 的 bulge 段还原成圆弧。
- `展开块引用`：展开 INSERT，阵列一并展开。
- `无引用块按块库导入`：`ENTITIES` 里没有线框时，把 `BLOCKS` 里的块定义按网格铺开。

**文字**

- `导入文字`（默认开）：关掉可省时间（几千条文字会明显变慢）。
- `字形精度`（默认 6）：字形轮廓的细分密度。
- `字体`：留空时自动找一个中英文都有的字体（思源黑体 / 文泉驿 / 微软雅黑 / 苹方）。

**导出兼容**

- `线框转为实体面`：见「已知限制」。
- `线宽`：转面时的带宽，图纸单位；`0` 表示按图纸实际跨度自动估算。

## 支持的实体

`LINE`、`LWPOLYLINE`（含 bulge 圆弧）、老式 `POLYLINE` / `VERTEX`（2D/3D）、
`ARC`、`CIRCLE`、`ELLIPSE`、`SPLINE`、`INSERT`（含阵列、嵌套、镜像）、
`HATCH`（只导**边界**，不导填充图案）、`TEXT`、`MTEXT`、`ATTRIB`。

## 已知限制

**导出到 Isaac Sim 什么都看不到**

插件导入的是纯线框（0 个面，只有边）。Blender 导出 OBJ 时只能把边写成 `l` 线段，
而 Isaac Sim / Omniverse 的 OBJ 导入器只构建多边形面，会直接丢掉线段——
症状是 Blender 里看得见，导进仿真什么都没有。

打开 `线框转为实体面` 后：

- 每条边扩成一条很细的四边形带，OBJ 里变成真正的 `f` 面；
- 线宽填 `0` 时按图纸实际跨度自动估算，用 1%~99% 分位数排除远处的游离图块；
- 代价：顶点数 ×4、面数 = 线段数。2 百万条边大约产生 8 百万顶点。

**文字是折线，不可编辑**

文字描成字形轮廓后和几何进同一个网格，不是可编辑的 Text 对象。图上文字要改，回 CAD 改。

**不支持的类型**（跳过并在控制台统计）

- `SOLID` / `3DFACE` / `REGION` / `BODY`：实心填充和三维面域（`REGION` 的几何只在 ACIS 二进制里）
- `ATTDEF`：块定义里的属性**模板**（真正的值在 `ATTRIB` 里，已经导入了）
- `LEADER` / `MLEADER` / `DIMENSION`：引线标注、尺寸标注的匿名块不展开
- `POINT`、`IMAGE` / `UNDERLAY` / `WIPEOUT`、`ACAD_PROXY_ENTITY` / `ACAD_TABLE`
- polyface mesh / polygon mesh 多段线（标志位 64 / 16）

**其他取舍**

- 顶点存进 mesh 时是 float32。坐标绝对值超过 ~10 万单位时开「居中到原点」。
- 不写材质、不建集合结构、不保留线型/线宽。
- `MTEXT` 不按参考宽度自动折行。

## 故障排查

提示「没有可导入的线框几何」时，会把文件里的实际内容报出来，对照下表定位：

| 现象 | 真实原因 | 处理 |
|---|---|---|
| 提示"找不到 SECTION" | 组码带前导空格（`  0` / ` 10`），AutoCAD 的标准写法 | 自动回退到容错扫描（23 MB 多花 0.19 s） |
| 提示"二进制 DXF" | 存成了 Binary DXF | 让 CAD 另存 ASCII DXF |
| 提示"文件头是 DWG" | DWG 改了扩展名 | 导出真正的 DXF |
| 提示"ENTITIES 段是空的" | 块库文件，或模型空间被写成 `*Model_Space` 块 | 「无引用块按块库导入」会铺开全部块 |
| 只有 `ACAD_PROXY_ENTITY` | 来自 Civil3D / Architecture 的代理对象，几何是不透明二进制 | 在原软件里 `EXPLODE` 或另存低版本 |

看不出来源时，用体检脚本列一遍文件内容：

```bash
python tools/dxf_probe.py 图纸.dxf
```

## 开发

```bash
python -m unittest discover -s tests -v
```

只依赖 **numpy**，不需要 Blender、不需要 ezdxf。测的是 `dxf.py` / `geom.py` / `ocs.py` / `text.py`
四个不依赖 `bpy` 的模块（`build.py` 要 bpy，不在测试范围内）。

用例是合成图纸，期望值由 ezdxf 独立算出并冻结在 `tests/cases/*.json`，所以跑测试时不用装 ezdxf。
要重新生成（改了实现、想加用例时）：

```bash
pip install ezdxf
python tests/make_cases.py
```

```
fast-dxf/
├── __init__.py          # 算子注册、参数面板、菜单、后置报告
├── blender_manifest.toml
├── fx/
│   ├── dxf.py           # 词法、段定位、实体分组（不依赖 bpy）
│   ├── ocs.py           # OCS（挤出方向）到 WCS 的变换
│   ├── geom.py          # 实体 -> 顶点/边数组 + 文字收集（不依赖 bpy）
│   ├── text.py          # TEXT/MTEXT/ATTRIB 解析与格式码清理（不依赖 bpy）
│   ├── build.py         # 数组 -> mesh、字形描边、线框转面（依赖 bpy）
│   └── importer.py      # 编排与计时
├── tools/
│   └── dxf_probe.py     # 零依赖体检脚本
└── tests/
    ├── make_cases.py    # 生成用例与期望值（开发期用，需要 ezdxf）
    ├── test_import.py   # 测试本体（只需要 numpy）
    └── cases/           # 合成图纸 + 期望值 JSON
```

## 许可证

GNU General Public License v3.0 or later，全文见 [LICENSE](LICENSE)。

Copyright (C) 2026 HwangChul
