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


def test_load_points3d_full_filters(tmp_path):
    from track3dgs.colmap_export import load_points3d_full
    # PID X Y Z R G B ERROR TRACK(image_id, point2d_idx)...
    (tmp_path / "points3D.txt").write_text(
        "# header\n"
        "1 0 0 5 10 10 10 0.5 1 0 2 0 3 0\n"      # 3 obs, low error -> keep
        "2 0 0 6 10 10 10 0.5 1 1 2 1\n"          # 2 obs: dropped by min_track
        "3 0 0 7 10 10 10 9.9 1 2 2 2 3 2\n"      # high error: dropped
        "4 0 0 8 10 10 10 0.5\n")                 # no track info -> keep (unknown)
    pts = load_points3d_full(tmp_path / "points3D.txt", min_track=3, max_error=2.0)
    assert [round(p[2]) for p in pts] == [5, 8]


def test_points_none(tmp_path):
    intr = {"width": 10, "height": 10, "fx": 5, "fy": 5, "cx": 5, "cy": 5}
    write_colmap_model(tmp_path, intr, [], None)
    assert (tmp_path / "points3D.txt").exists()
