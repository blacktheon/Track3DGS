import numpy as np
import cv2
from track3dgs.skymask import sky_keep_mask, combine_keep_masks


def test_sky_keep_mask_marks_sky_black_and_dilates():
    class_map = np.zeros((64, 128), dtype=np.int64)   # ADE20K ids; 4 = tree
    class_map[:] = 4
    class_map[:20, :] = 2                             # sky band on top
    keep = sky_keep_mask(class_map, dilate_px=8)
    assert keep.dtype == np.uint8 and set(np.unique(keep)) <= {0, 255}
    assert keep[10, 64] == 0                          # sky -> excluded
    assert keep[22, 64] == 0                          # dilation ate the boundary
    assert keep[40, 64] == 255                        # tree kept


def test_color_sky_mask_blue_white_above_cutoff_only():
    from track3dgs.skymask import color_sky_mask
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :] = (40, 120, 60)                  # BGR greenish forest everywhere
    img[10:20, 20:40] = (230, 160, 60)         # sky blue, above cutoff -> sky
    img[10:20, 60:80] = (245, 245, 245)        # white cloud, above -> sky
    img[70:80, 20:40] = (230, 160, 60)         # blue but below cutoff -> kept
    img[70:80, 60:80] = (200, 200, 200)        # bright grey road below -> kept
    img[30:40, 120:140] = (60, 200, 90)        # sunlit green leaf above -> kept
    img[10:20, 150:170] = (190, 195, 200)      # warm grey rock above -> kept
    img[30:40, 20:40] = (180, 200, 160)        # pale green-cyan haze -> kept
    m = color_sky_mask(img, lat_cutoff_frac=0.52)
    assert m[15, 30] == 1 and m[15, 70] == 1
    assert m[75, 30] == 0 and m[75, 70] == 0
    assert m[35, 130] == 0
    assert m[15, 160] == 0                     # warm grey spared (tuned rule)
    assert m[35, 30] == 0                      # green-ish haze spared
    assert m[50, 100] == 0                     # forest background untouched


def test_fill_enclosed_sky_fills_clouds_keeps_trees():
    from track3dgs.skymask import fill_enclosed_sky
    prior = np.zeros((100, 100), bool); prior[:60, :] = True   # sky zone: top 60
    sky = prior.copy()
    sky[20:30, 40:60] = False              # cloud hole enclosed by sky
    sky[40:60, 10:25] = False              # "tree crown": touches zone bottom
    out = fill_enclosed_sky(sky, prior)
    assert out[25, 50]                     # cloud filled
    assert not out[50, 15]                 # tree crown preserved
    assert not out[80, 50]                 # below zone untouched


def test_bright_sky_mask_flags_blown_whites_above_horizon_only():
    from track3dgs.skymask import bright_sky_mask
    img = np.full((100, 200, 3), 60, np.uint8)        # dark scene
    img[10:20, 50:90] = 250                            # blown white, upper part
    img[80:90, 50:90] = 250                            # bright road, lower part
    m = bright_sky_mask(img, row_frac=0.45)
    assert m[15, 70] == 1                              # upper blown white = sky
    assert m[85, 70] == 0                              # below horizon: untouched
    assert m[50, 100] == 0                             # dark pixels untouched


def test_combine_keep_masks_is_logical_and():
    a = np.full((8, 8), 255, np.uint8); a[0, 0] = 0
    b = np.full((8, 8), 255, np.uint8); b[7, 7] = 0
    c = combine_keep_masks(a, b)
    assert c[0, 0] == 0 and c[7, 7] == 0 and c[4, 4] == 255


def test_combine_resizes_smaller_mask():
    a = np.full((64, 128), 255, np.uint8)
    b = np.full((32, 64), 255, np.uint8); b[:16, :] = 0   # top half masked
    c = combine_keep_masks(a, b)
    assert c.shape == (64, 128)
    assert c[10, 64] == 0 and c[50, 64] == 255
