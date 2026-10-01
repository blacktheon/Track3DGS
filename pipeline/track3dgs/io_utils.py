import json
from pathlib import Path


class Project:
    """On-disk layout for one track section. All stages read/write through this."""

    def __init__(self, root):
        self.root = Path(root)

    @property
    def section_video(self): return self.root / "section.mp4"
    @property
    def proxy_video(self): return self.root / "proxy_vslam.mp4"
    @property
    def frames_dir(self): return self.root / "frames"
    @property
    def frames_meta(self): return self.root / "frames_meta.jsonl"
    @property
    def mask_path(self): return self.root / "mask_equirect.png"
    @property
    def mask_reference(self): return self.root / "mask_reference.jpg"
    @property
    def views_dir(self): return self.root / "views"
    @property
    def views_masks_dir(self): return self.root / "views_masks"
    @property
    def views_meta(self): return self.root / "views" / "views_meta.json"
    @property
    def track_dir(self): return self.root / "track"
    @property
    def poses_json(self): return self.root / "track" / "poses.json"
    @property
    def colmap_dir(self): return self.root / "track" / "colmap"
    @property
    def qc_trajectory(self): return self.root / "track" / "qc_trajectory.png"
    @property
    def cells_dir(self): return self.root / "cells"
    @property
    def cells_json(self): return self.root / "cells" / "cells.json"
    @property
    def train_dir(self): return self.root / "train"
    @property
    def export_dir(self): return self.root / "train" / "export"
    @property
    def tiles_dir(self): return self.root / "tiles"
    @property
    def manifest_json(self): return self.root / "tiles" / "manifest.json"
    @property
    def demo_dir(self): return self.root / "demo"


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_jsonl(path, records):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
