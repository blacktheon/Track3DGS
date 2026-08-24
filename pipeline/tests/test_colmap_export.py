import numpy as np
from track3dgs.colmap_export import write_colmap_model, load_images_txt


def test_write_and_load_model(tmp_path):
    intr = {"width": 1600, "height": 1600, "fx": 671.0, "fy": 671.0,
            "cx": 800.0, "cy": 800.0}
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    images = [{"name": "a.jpg", "T_wc": T}]
    pts = np.array([[0.0, 0.0, 5.0]])
    write_colmap_model(tmp_path, intr, images, pts)

    cams = (tmp_path / "cameras.txt").read_text()
    assert "PINHOLE 1600 1600" in cams
    loaded = load_images_txt(tmp_path / "images.txt")
    assert loaded[0]["name"] == "a.jpg"
    assert np.allclose(loaded[0]["t"], [-1.0, -2.0, -3.0])   # world-to-camera
    p3d = (tmp_path / "points3D.txt").read_text()
    assert "5.0" in p3d


def test_load_images_txt_skips_observation_lines(tmp_path):
    # COLMAP-style file: header comments + image line + non-empty observations line
    (tmp_path / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1.0 0.0 0.0 0.0 -1.0 -2.0 -3.0 1 a.jpg\n"
        "100.5 200.5 55 300.1 400.2 -1\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 b.jpg\n"
        "\n")
    loaded = load_images_txt(tmp_path / "images.txt")
    assert [im["name"] for im in loaded] == ["a.jpg", "b.jpg"]


def test_points_none(tmp_path):
    intr = {"width": 10, "height": 10, "fx": 5, "fy": 5, "cx": 5, "cy": 5}
    write_colmap_model(tmp_path, intr, [], None)
    assert (tmp_path / "points3D.txt").exists()
