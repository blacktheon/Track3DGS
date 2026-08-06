import json
from pathlib import Path
from track3dgs.io_utils import Project, read_json, write_json, read_jsonl, write_jsonl, ensure_dir

def test_project_paths(tmp_path):
    p = Project(tmp_path / "section01")
    assert p.frames_dir == tmp_path / "section01" / "frames"
    assert p.views_meta == tmp_path / "section01" / "views" / "views_meta.json"
    assert p.manifest_json == tmp_path / "section01" / "tiles" / "manifest.json"
    assert p.colmap_dir == tmp_path / "section01" / "track" / "colmap"

def test_json_roundtrip(tmp_path):
    f = tmp_path / "x.json"
    write_json(f, {"a": 1})
    assert read_json(f) == {"a": 1}

def test_jsonl_roundtrip(tmp_path):
    f = tmp_path / "x.jsonl"
    write_jsonl(f, [{"i": 0}, {"i": 1}])
    assert read_jsonl(f) == [{"i": 0}, {"i": 1}]

def test_ensure_dir(tmp_path):
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()
