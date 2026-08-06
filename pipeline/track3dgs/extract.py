"""Stage 1: trim section, extract sharp frames, build proxy + mask template."""
import argparse
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .io_utils import Project, ensure_dir, write_jsonl


def sharpness(img_bgr):
    h, w = img_bgr.shape[:2]
    if w > 960:
        img_bgr = cv2.resize(img_bgr, (960, int(h * 960 / w)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharp(scores, group):
    keep = []
    for g0 in range(0, len(scores), group):
        chunk = scores[g0:g0 + group]
        keep.append(g0 + int(np.argmax(chunk)))
    return keep


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                   check=True)


def _ffprobe_field(video, field):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={field}",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True)
    return out.stdout.strip()


def probe_fps(video):
    return float(Fraction(_ffprobe_field(video, "r_frame_rate")))


def probe_duration(video):
    return float(_ffprobe_field(video, "duration"))


def run_extract(video, out, start=0.0, end=None, group=3, proxy_width=1920):
    p = Project(out)
    ensure_dir(p.root)
    duration = probe_duration(video)
    if end is None:
        end = duration
    if start <= 0.0 and end >= duration - 0.01:
        # already trimmed (e.g. by LosslessCut at a keyframe): lossless copy
        _ffmpeg("-i", str(video), "-c", "copy", "-an", str(p.section_video))
    else:
        # re-encode (not stream-copy) so t=0 lands exactly on --start
        _ffmpeg("-ss", str(start), "-to", str(end), "-i", str(video),
                "-c:v", "libx264", "-crf", "18", "-an", str(p.section_video))
    _ffmpeg("-i", str(p.section_video),
            "-vf", f"scale={proxy_width}:{proxy_width // 2}",
            "-c:v", "libx264", "-crf", "23", "-an", str(p.proxy_video))

    fps = probe_fps(p.section_video)
    with tempfile.TemporaryDirectory() as td:
        _ffmpeg("-i", str(p.section_video), "-qscale:v", "2", "-start_number", "0",
                str(Path(td) / "f_%06d.jpg"))
        all_frames = sorted(Path(td).glob("f_*.jpg"))
        scores = [sharpness(cv2.imread(str(f))) for f in all_frames]
        keep = select_sharp(scores, group)

        ensure_dir(p.frames_dir)
        records = []
        for idx in keep:
            name = f"frame_{idx:06d}.jpg"
            shutil.copy2(all_frames[idx], p.frames_dir / name)
            records.append({"name": name, "src_index": idx,
                            "t": idx / fps, "sharpness": scores[idx]})
        write_jsonl(p.frames_meta, records)

        first = cv2.imread(str(all_frames[0]))
        cv2.imwrite(str(p.mask_reference), first)
        if not p.mask_path.exists():  # never clobber a hand-painted mask
            cv2.imwrite(str(p.mask_path),
                        np.full(first.shape[:2], 255, dtype=np.uint8))
    print(f"kept {len(keep)}/{len(all_frames)} frames at {fps:.3f} fps")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--proxy-width", type=int, default=1920)
    a = ap.parse_args()
    run_extract(a.video, a.out, a.start, a.end, a.group, a.proxy_width)


if __name__ == "__main__":
    main()
