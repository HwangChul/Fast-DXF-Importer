"""OCS（对象坐标系）到 WCS 的变换。

DXF 里有一类实体，坐标不是世界坐标，而是画在实体自己的「对象坐标系」里：
点 (x, y, z) 的含义是 x·Ax + y·Ay + z·Az，其中 Az 就是组码 210/220/230 给出的
挤出方向（法向），Ax/Ay 由 AutoCAD 的「任意轴算法」推出来。

**为什么必须处理**：当法向是 (0,0,-1) 时，这套基是左手翻转的，同一个 (x, y)
落到世界坐标就是镜像的。图纸里用这种方式放置的块会整个翻到反方向——表现就是
"这个块应该在右边，却跑到左边去了"。只做平移+旋转+缩放是补不回来的。

受影响实体：INSERT、LWPOLYLINE、POLYLINE、ARC、CIRCLE、ELLIPSE、TEXT、MTEXT、
ATTRIB、HATCH。LINE 和 SPLINE 的坐标本来就是 WCS，不需要。
"""

from __future__ import annotations

import numpy as np

# AutoCAD 任意轴算法的阈值：法向在 XY 平面上太短时，换成 Y 轴做参考
_AXIS_EPS = 1.0 / 64.0


def axes_from_normal(nx, ny, nz):
    """单个法向 -> (Ax, Ay, Az)，三个单位向量。"""
    n = np.array([nx, ny, nz], dtype=np.float64)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    n /= ln
    if abs(n[0]) < _AXIS_EPS and abs(n[1]) < _AXIS_EPS:
        w = np.array([0.0, 1.0, 0.0])
    else:
        w = np.array([0.0, 0.0, 1.0])
    ax = np.cross(w, n)
    ax /= np.linalg.norm(ax)
    ay = np.cross(n, ax)
    return ax, ay, n


def axes_arrays(nx, ny, nz):
    """向量化版本：逐实体法向 -> 逐实体 (Ax, Ay, Az)，形状都是 (n, 3)。"""
    n = np.stack([np.asarray(nx, np.float64),
                  np.asarray(ny, np.float64),
                  np.asarray(nz, np.float64)], axis=1)
    ln = np.linalg.norm(n, axis=1)
    bad = ln < 1e-12
    if np.any(bad):
        n[bad] = (0.0, 0.0, 1.0)
        ln[bad] = 1.0
    n = n / ln[:, None]

    small = (np.abs(n[:, 0]) < _AXIS_EPS) & (np.abs(n[:, 1]) < _AXIS_EPS)
    w = np.where(small[:, None], np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    ax = np.cross(w, n)
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ay = np.cross(n, ax)
    return ax, ay, n


def is_identity(nx, ny, nz) -> bool:
    """法向全为 (0,0,1)——绝大多数图纸都是这样，走这条快路径不做任何计算。"""
    return (
        np.all(np.abs(np.asarray(nx)) < 1e-12)
        and np.all(np.abs(np.asarray(ny)) < 1e-12)
        and np.all(np.abs(np.asarray(nz) - 1.0) < 1e-12)
    )


def matrix(nx, ny, nz):
    """单个法向 -> 4x4 的 OCS->WCS 矩阵（纯旋转/反射，无平移）。"""
    ax, ay, az = axes_from_normal(nx, ny, nz)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = ax, ay, az
    return m


def apply(verts: np.ndarray, owner: np.ndarray, axes) -> np.ndarray:
    """把 OCS 坐标换成 WCS。

    verts: (N, 3)，每行是 OCS 里的 (x, y, z)；
    owner: (N,)，指明该行属于第几个实体，用来取对应的 Ax/Ay/Az。
    """
    ax, ay, az = axes
    if len(verts) == 0:
        return verts
    x = verts[:, 0:1]
    y = verts[:, 1:2]
    z = verts[:, 2:3]
    return x * ax[owner] + y * ay[owner] + z * az[owner]


def apply_single(verts: np.ndarray, axes) -> np.ndarray:
    """单实体版本：vertss (N,3)，axes 为三个 3 维向量。"""
    ax, ay, az = axes
    if len(verts) == 0:
        return verts
    return verts[:, 0:1] * ax + verts[:, 1:2] * ay + verts[:, 2:3] * az
