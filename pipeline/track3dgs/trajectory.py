"""Pose math. Convention: OpenCV/COLMAP camera (x-right, y-down, z-forward).
TUM lines are camera-to-world; COLMAP images.txt is world-to-camera."""
import math
from pathlib import Path

import numpy as np


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_quat(R):
    qw = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if qw > 1e-8:
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    else:  # qw ~ 0: fall back to largest diagonal element branch
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        q = np.zeros(4)
        q[i + 1] = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) / 2
        q[0] = (R[k, j] - R[j, k]) / (4 * q[i + 1])
        q[j + 1] = (R[j, i] + R[i, j]) / (4 * q[i + 1])
        q[k + 1] = (R[k, i] + R[i, k]) / (4 * q[i + 1])
        qw, qx, qy, qz = q
    return qw, qx, qy, qz


def load_tum(path):
    poses = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in line.split())
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = [tx, ty, tz]
        poses.append((t, T))
    return poses


def apply_scale(T_list, s):
    out = []
    for T in T_list:
        T2 = T.copy()
        T2[:3, 3] *= s
        out.append(T2)
    return out


def yaw_view_pose(T_wc, yaw_deg):
    a = math.radians(yaw_deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)],
                   [0, 1, 0],
                   [-math.sin(a), 0, math.cos(a)]])
    T = T_wc.copy()
    T[:3, :3] = T_wc[:3, :3] @ Ry
    return T


def world_to_camera(T_wc):
    R_cw = T_wc[:3, :3].T
    t = -R_cw @ T_wc[:3, 3]
    qw, qx, qy, qz = R_to_quat(R_cw)
    return qw, qx, qy, qz, t[0], t[1], t[2]


def arc_length(positions):
    d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(positions, step):
    s = arc_length(positions)
    total = s[-1]
    s_new = np.arange(0.0, total + step, step)
    s_new = s_new[s_new <= total]
    pts = np.stack([np.interp(s_new, s, positions[:, i]) for i in range(3)], axis=1)
    return pts, s_new
