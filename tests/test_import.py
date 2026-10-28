"""公开测试：解析 + 几何 + 文字，不需要 bpy，也不需要 ezdxf。

    python -m unittest discover -s tests -v

只被测不依赖 Blender 的模块（fx.dxf / fx.geom / fx.ocs / fx.text）。
build.py 依赖 bpy，不在此列。

用例全部是用 ezdxf 生成的合成图纸，覆盖实际踩过的坑，期望值由 ezdxf 独立算出
并存在同名 .json 里——所以跑测试时不再需要 ezdxf。
"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fx import dxf as fdxf  # noqa: E402
from fx import geom  # noqa: E402

CASES = os.path.join(ROOT, "tests", "cases")

# 圆/弧按 64 段折线近似，弦高带来系统误差，按跨度给 0.5% 的余量
BBOX_RTOL = 0.005
TEXT_TOL_FRAC = 0.01


def case_names():
    if not os.path.isdir(CASES):
        return []
    return sorted(f[:-4] for f in os.listdir(CASES) if f.endswith(".dxf"))


def load_json(name):
    with open(os.path.join(CASES, name + ".json"), encoding="utf-8") as fp:
        return json.load(fp)


def run_case(name):
    """跑一遍转换，返回 (顶点数组, 边数组, Converter)。"""
    parsed = fdxf.parse_file(os.path.join(CASES, name + ".dxf"))
    conv = geom.Converter(geom.Settings())
    conv.parse_blocks(parsed["BLOCKS"])
    acc = conv.convert(parsed["ENTITIES"])
    v, e, _layers = acc.arrays()
    return v, e, conv


def text_points(conv):
    """文字落点（世界坐标）与内容。"""
    out = []
    for item, matrix, _layer in conv.texts:
        out.append((item.body, matrix[:3, 3].astype(float)))
    return out


class TestCases(unittest.TestCase):
    """逐用例核对包围盒与文字。"""

    def test_cases_exist(self):
        names = case_names()
        self.assertTrue(names, "tests/cases 为空，先跑 tests/make_cases.py")
        self.assertGreaterEqual(len(names), 6)

    def test_bbox(self):
        for name in case_names():
            with self.subTest(case=name):
                exp = load_json(name)
                if exp.get("skip_bbox"):
                    continue
                v, _e, _c = run_case(name)
                self.assertGreater(len(v), 0, "没有顶点")
                lo, hi = v.min(axis=0)[:2], v.max(axis=0)[:2]
                elo = np.array(exp["bbox_min"][:2])
                ehi = np.array(exp["bbox_max"][:2])
                span = max(float(np.max(ehi - elo)), 1e-6)
                tol = max(span * BBOX_RTOL, 1e-3)
                self.assertLessEqual(
                    float(np.max(np.abs(lo - elo))), tol,
                    "%s 包围盒下界 %s 期望 %s（容差 %.4g）" % (name, np.round(lo, 3), elo, tol))
                self.assertLessEqual(
                    float(np.max(np.abs(hi - ehi))), tol,
                    "%s 包围盒上界 %s 期望 %s（容差 %.4g）" % (name, np.round(hi, 3), ehi, tol))

    def test_texts(self):
        for name in case_names():
            exp = load_json(name)
            expected = exp.get("texts") or []
            if not expected:
                continue
            with self.subTest(case=name):
                _v, _e, conv = run_case(name)
                got = text_points(conv)
                self.assertEqual(
                    len(got), len(expected),
                    "文字条数 %d != 期望 %d" % (len(got), len(expected)))

                span = max(
                    abs(exp["bbox_max"][0] - exp["bbox_min"][0]),
                    abs(exp["bbox_max"][1] - exp["bbox_min"][1]), 1.0)
                tol = span * TEXT_TOL_FRAC

                pool = list(got)
                for want in expected:
                    target = np.array(want["pos"], dtype=float)
                    dist = [float(np.linalg.norm(p[:2] - target[:2])) for _b, p in pool]
                    k = int(np.argmin(dist))
                    self.assertLess(
                        dist[k], tol,
                        "%s 文字 %r 位置 %s 附近无匹配（最近 %.3f > %.3f）" % (
                            name, want["body"][:12], np.round(target[:2], 2), dist[k], tol))
                    body, _p = pool.pop(k)
                    self.assertEqual(body.strip(), want["body"].strip(),
                                     "%s 最近点内容不符" % name)


class TestSpecific(unittest.TestCase):
    """针对单个历史 bug 的定向断言，失败时比通用断言更好定位。"""

    def test_leading_space_parsed(self):
        """组码带前导空格也要读到内容（否则顶点为 0）。"""
        v, e, _c = run_case("leading_space")
        self.assertGreater(len(v), 0)
        self.assertGreater(len(e), 0)

    def test_special_block_name(self):
        """块名含 $ 与 # 时块内容不能丢（官方导入器在此报 KeyError）。"""
        v, _e, _c = run_case("block_name_special")
        self.assertGreater(len(v), 4)
        # 两次引用：(0,0) 原样占 x∈[0,4]；(10,0) 旋转 90° 占 x=10、y∈[0,4]
        self.assertGreaterEqual(float(v[:, 0].max()), 9.5, "第二次引用丢失")
        self.assertGreaterEqual(float(v[:, 1].max()), 3.5, "旋转后的块高度不对")

    def test_mirror_flips_x(self):
        """法向 (0,0,-1) 的圆必须落在 -x 侧，而不是 +x 侧。"""
        v, _e, _c = run_case("mirror_ocs")
        self.assertLess(float(v[:, 0].min()), -2.5)
        # 原始 LINE (x=2) 不受 OCS 影响，仍应在 +2
        on_x2 = np.abs(v[:, 0] - 2.0) < 0.05
        self.assertTrue(bool(on_x2.any()), "WCS 的 LINE 不该被镜像")

    def test_text_width_factor(self):
        """TEXT 宽度因子不能把字拉成 0.6/1.8 倍宽度以外的形状。"""
        v, _e, conv = run_case("text_width")
        got = {body: p for body, p in text_points(conv)}
        self.assertIn("W06", got)
        self.assertIn("W18", got)
        # 宽度因子 1.8 的比 0.6 的占位更长：用同一行内的 x 跨度粗验
        self.assertLess(got["W06"][0], got["W18"][0] + 1e-6)

    def test_bulge_kept(self):
        """bulge 段不能被静默丢弃，包围盒在 -y 侧要有凸起。"""
        v, _e, _c = run_case("bulge")
        self.assertLess(float(v[:, 1].min()), -0.5)

    def test_nested_block_depth(self):
        """嵌套块的缩放与旋转要逐层叠加。"""
        v, _e, _c = run_case("nested_block")
        self.assertGreaterEqual(float(np.ptp(v[:, 0])), 12.0)


class TestProbe(unittest.TestCase):
    """tools/dxf_probe.py 是零依赖脚本，至少不能崩。"""

    def test_probe_runs(self):
        import subprocess

        probe = os.path.join(ROOT, "tools", "dxf_probe.py")
        sample = os.path.join(CASES, "mirror_ocs.dxf")
        res = subprocess.run(
            [sys.executable, probe, sample],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stderr[-800:])
        self.assertIn("CIRCLE", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
