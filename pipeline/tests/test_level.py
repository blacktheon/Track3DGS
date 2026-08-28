import numpy as np
from track3dgs.level import make_mount_key, orientation_from_key, _cam_axes
from track3dgs.trajectory import yaw_view_pose


def _frames(R_frame):
    """Rig moving along R_frame@x with camera up at R_frame@(-y)."""
    out = []
    for i in range(6):
        T = np.eye(4)
        T[:3, :3] = R_frame
        T[:3, 3] = R_frame @ np.array([2.0 * i, 0, 0])
        out.append({"name": f"f{i}.jpg", "t": float(i), "s": 2.0 * i,
                    "T_wc": [float(v) for v in T.reshape(-1)]})
    return out


def _rot(axis, deg):
    a = np.radians(deg)
    x, y, z = np.asarray(axis) / np.linalg.norm(axis)
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    return np.array([
        [x*x*C + c, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])


def test_mount_key_roundtrip_levels_any_frame():
    # "truth": mount tilted 40 deg about z relative to gravity, then the whole
    # old reconstruction sat in an arbitrary frame A
    mount_tilt = _rot([0, 0, 1], 40.0)
    A = _rot([1, 2, 3], 71.0)
    frames_old = _frames(A @ mount_tilt)
    # the user's manual leveling of the old frame is exactly A^-1
    key = make_mount_key(frames_old, A.T)

    # a NEW section lands in a different arbitrary frame B
    B = _rot([-2, 1, 5], -113.0)
    frames_new = _frames(B @ mount_tilt)
    R, t0 = orientation_from_key(frames_new, key)

    # applying R must reproduce the same leveled camera axes as the key
    up_new, travel_new = _cam_axes(frames_new)
    assert np.allclose(R @ up_new, key["cam_up"], atol=1e-9)
    assert np.allclose(R @ travel_new, key["travel"], atol=1e-9)
    assert np.allclose(t0, frames_new[0]["T_wc"] and
                       np.array(frames_new[0]["T_wc"]).reshape(4, 4)[:3, 3])
