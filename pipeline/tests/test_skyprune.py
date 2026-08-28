import numpy as np
from track3dgs.skyprune import equirect_uv, sky_fractions


def test_equirect_uv_directions():
    # camera at origin, identity pose (x-right, y-down, z-forward)
    dirs = np.array([
        [0.0, -1.0, 0.0],    # straight up   -> v ~ 0
        [0.0, 1.0, 0.0],     # straight down -> v ~ H
        [0.0, 0.0, 1.0],     # forward       -> centre
        [1.0, 0.0, 0.0],     # right         -> u ~ 3/4 W
    ])
    u, v = equirect_uv(dirs, W=360, H=180)
    assert v[0] < 2 and v[1] > 178
    assert abs(u[2] - 180) < 1 and abs(v[2] - 90) < 1
    assert abs(u[3] - 270) < 1


def test_sky_colored_rules():
    from track3dgs.skyprune import sky_colored
    rgb = np.array([
        [0.35, 0.55, 0.85],   # sky blue -> flagged
        [0.95, 0.96, 0.97],   # blown white -> flagged
        [0.20, 0.55, 0.25],   # leaf green -> kept
        [0.55, 0.50, 0.45],   # gravel grey -> kept
    ])
    m = sky_colored(rgb)
    assert list(m) == [True, True, False, False]


def test_sky_fractions_flags_sky_splat():
    H, W = 90, 180
    mask = np.full((H, W), 255, np.uint8)
    mask[: H // 3, :] = 0                       # sky above +30deg latitude
    masks = {"frame_000000": mask, "frame_000001": mask}
    T0 = np.eye(4)
    T1 = np.eye(4); T1[:3, 3] = [0, 0, 5.0]
    frames = [{"name": "frame_000000.jpg",
               "T_wc": [float(x) for x in T0.reshape(-1)]},
              {"name": "frame_000001.jpg",
               "T_wc": [float(x) for x in T1.reshape(-1)]}]
    centers = np.array([
        [0.0, -50.0, 2.0],    # high above both cameras -> sky from both
        [0.0, -1.0, 40.0],    # far ahead, slightly up -> low latitude -> kept
        [0.0, 2.0, 2.0],      # below horizon -> kept
    ])
    frac = sky_fractions(centers, frames, masks, max_range=100.0)
    assert frac[0] > 0.9
    assert frac[1] < 0.1
    assert frac[2] < 0.1
