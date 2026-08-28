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
