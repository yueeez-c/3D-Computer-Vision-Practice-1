"""Regenerate every data asset used by Exercise 1.

Usage:  python code/make_data.py  [--data-dir DIR]

Needs only numpy and the standard library. No network access, no downloads.

Note: the demosaicing task (Task B) does NOT use this script -- it ships the original ten-image
dataset under data/demosaic/ and is run from code/p2-demosaic/evalDemosaicing.py.

Outputs (in data/):
  scene_wireframe.npz  unit cube + ground-plane grid, as vertices and edge index pairs
  two_view.npz         non-planar point cloud, two calibrated cameras with a real baseline,
                       and the exact F / E / relative pose to score estimates against
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, os.pardir, "data")

# Shared camera for Tasks C/D and the two-view demo: 640x480 image, 6.4x4.8 mm sensor, 5 mm lens.
# This yields exactly K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]], the matrix used in the written
# Part 3, Problem 4, so the coding self-check can reuse those hand-computed values.
IMG_WH = (640, 480)
SENSOR_MM = (6.4, 4.8)
FOCAL_MM = 5.0


# ---------------------------------------------------------------------------
# Camera helpers (kept here so make_data.py does not depend on student code)
# ---------------------------------------------------------------------------
def make_K(f_mm, sensor_mm, img_wh):
    W, H = img_wh
    sw, sh = sensor_mm
    return np.array(
        [[f_mm * W / sw, 0.0, W / 2.0], [0.0, f_mm * H / sh, H / 2.0], [0.0, 0.0, 1.0]]
    )


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera (R, t) in the OpenCV convention: x right, y down, z forward."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z = target - eye
    z /= np.linalg.norm(z)
    if abs(np.dot(z, up)) > 1.0 - 1e-8:
        raise ValueError("viewing direction is parallel to `up`; look_at is undefined")
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    return R, t


def skew(t):
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def project(K, R, t, X):
    Xc = X @ R.T + t
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]


# ---------------------------------------------------------------------------
# Wireframe scene for Task C
# ---------------------------------------------------------------------------
def make_wireframe(data_dir):
    # Unit cube, axis-aligned, sitting on the ground plane y = 0 (y is up in the world).
    c = np.array(
        [[x, y, z] for y in (0.0, 1.0) for z in (-0.5, 0.5) for x in (-0.5, 0.5)]
    )
    # Edges grouped by direction, so the render can colour them and the vanishing point of each
    # direction can be picked out by eye.
    edges_x = [(0, 1), (2, 3), (4, 5), (6, 7)]
    edges_z = [(0, 2), (1, 3), (4, 6), (5, 7)]
    edges_y = [(0, 4), (1, 5), (2, 6), (3, 7)]

    lines = []
    ticks = np.linspace(-3.0, 3.0, 13)
    for v in ticks:
        lines.append([[v, 0.0, -3.0], [v, 0.0, 3.0]])
        lines.append([[-3.0, 0.0, v], [3.0, 0.0, v]])
    grid = np.array(lines)  # (L, 2, 3)

    out = os.path.join(data_dir, "scene_wireframe.npz")
    np.savez_compressed(
        out,
        cube_vertices=c,
        cube_edges_x=np.array(edges_x),
        cube_edges_y=np.array(edges_y),
        cube_edges_z=np.array(edges_z),
        grid_segments=grid,
        dir_x=np.array([1.0, 0.0, 0.0]),
        dir_z=np.array([0.0, 0.0, 1.0]),
        ground_normal=np.array([0.0, 1.0, 0.0]),
    )
    print(f"  wrote {out}  ({len(c)} cube vertices, {len(grid)} grid segments)")


# ---------------------------------------------------------------------------
# Two-view scene for the multiview demo
# ---------------------------------------------------------------------------
def make_two_view(data_dir, n_points=120, seed=7):
    rng = np.random.default_rng(seed)

    X = np.column_stack(
        [
            rng.uniform(-1.6, 1.6, n_points),
            rng.uniform(-1.2, 1.2, n_points),
            rng.uniform(-1.6, 1.6, n_points),
        ]
    )
    # A planar cloud makes F unrecoverable (see Part 5), so assert we are well clear of that.
    assert np.linalg.svd(X - X.mean(0), compute_uv=False)[2] > 0.5, "point cloud is near-planar"

    K = make_K(FOCAL_MM, SENSOR_MM, IMG_WH)
    R1, t1 = look_at([-1.5, 0.5, 5.0], [0.0, 0.0, 0.0])
    R2, t2 = look_at([1.5, -0.3, 5.2], [0.0, 0.0, 0.0])

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    E = skew(t_rel) @ R_rel
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    F /= np.linalg.norm(F)

    x1, z1 = project(K, R1, t1, X)
    x2, z2 = project(K, R2, t2, X)

    W, H = IMG_WH
    visible = (
        (z1 > 0) & (z2 > 0)
        & (x1[:, 0] >= 0) & (x1[:, 0] < W) & (x1[:, 1] >= 0) & (x1[:, 1] < H)
        & (x2[:, 0] >= 0) & (x2[:, 0] < W) & (x2[:, 1] >= 0) & (x2[:, 1] < H)
    )
    X, x1, x2 = X[visible], x1[visible], x2[visible]
    assert len(X) >= 40, f"only {len(X)} visible points; loosen the scene bounds"

    x1h = np.c_[x1, np.ones(len(x1))]
    x2h = np.c_[x2, np.ones(len(x2))]
    resid = np.abs(np.sum(x2h * (x1h @ F.T), axis=1)).max()
    assert resid < 1e-9, f"ground-truth F is wrong: residual {resid:g}"

    out = os.path.join(data_dir, "two_view.npz")
    np.savez_compressed(
        out, X=X, x1=x1, x2=x2, K1=K, K2=K, R1=R1, t1=t1, R2=R2, t2=t2,
        R_rel=R_rel, t_rel=t_rel, E=E, F=F,
    )
    print(f"  wrote {out}  ({len(X)} correspondences, baseline {np.linalg.norm(t_rel):.3f})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"writing to {data_dir}")

    make_wireframe(data_dir)
    make_two_view(data_dir)

    print("done.")


if __name__ == "__main__":
    main()
