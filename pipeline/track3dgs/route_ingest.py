"""Bounded-memory video decoding; original presentation timestamps stay authoritative."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .extract import sharpness, select_sharp
from .io_utils import Project, ensure_dir, read_jsonl, write_jsonl
from .route_config import atomic_json, file_hash, stage_fingerprint


def select_records(pts, scores, source_hash, group=3, keyframe_hz=2, holdout_stride=10):
    if len(pts) != len(scores) or not pts or not np.all(np.isfinite(pts)):
        raise ValueError('Decoded frames and presentation timestamps must agree')
    if np.any(np.diff(pts) <= 0):
        raise ValueError('Presentation timestamps must increase strictly')
    winners = select_sharp(scores, group)
    if keyframe_hz is not None:
        kept, previous = [], None
        for i in winners:
            bucket = int((pts[i] - pts[0]) * keyframe_hz + 1e-7)
            if bucket != previous:
                kept.append(i)
                previous = bucket
        if kept[-1] != winners[-1]:
            kept.append(winners[-1])
        winners = kept
    return [{'name': f'frame_{i:06d}.jpg', 'src_index': i, 't': pts[i] - pts[0],
             'source_pts_seconds': pts[i], 'sharpness': scores[i],
             'frame_id': f'{source_hash}:{i}', 'source_sha256': source_hash,
             'split': 'held_out' if (j+1) % holdout_stride == 0 else 'train'}
            for j, i in enumerate(winners)]


def preserve_mask(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError('Existing workspace mask differs; use a new revision to preserve manual edits')
        return
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[1] != 2 * mask.shape[0]:
        raise ValueError('Expected a 2:1 equirectangular vehicle mask')
    shutil.copy2(source, destination)


def probe_timestamps(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
                          '-of', 'json', str(video)], capture_output=True, text=True, check=True)
    return [float(f['best_effort_timestamp_time']) for f in json.loads(out.stdout)['frames']]


def score_video(video, log_path):
    # One 960x480 BGR frame at a time; no full-video intermediate image directory.
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '8', '-i', str(video),
           '-map', '0:v:0', '-vf', 'scale=960:480', '-fps_mode', 'passthrough',
           '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    scores = []
    with open(log_path, 'wb') as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log) as proc:
        while True:
            raw = proc.stdout.read(960 * 480 * 3)
            if not raw:
                break
            if len(raw) != 960 * 480 * 3:
                raise RuntimeError('Truncated decoded frame')
            scores.append(sharpness(np.frombuffer(raw, np.uint8).reshape(480, 960, 3)))
            if len(scores) % 600 == 0:
                print(f'sharpness: {len(scores)} decoded frames', flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f'FFmpeg decode failed; see {log_path}')
    return scores


def ingest_route(config):
    p = Project(config['workspace'])
    ensure_dir(p.root)
    state = ensure_dir(p.root/'state')
    preserve_mask(config['vehicle_mask'], p.mask_path)
    print('Hashing source and checking workspace identity', flush=True)
    inputs = {k: file_hash(config[k]) if config[k] else None
              for k in ('source_video', 'vehicle_mask', 'sky_prior', 'mount_calibration')}
    fingerprint = stage_fingerprint('ingest', inputs, config['capture'], {'route_ingest': 1})
    identity = state/'ingest_identity.json'
    if identity.exists() and json.loads(identity.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Workspace inputs changed; select a new revision instead of mixing caches')
    atomic_json(identity, {'fingerprint': fingerprint, 'inputs': inputs})
    atomic_json(p.root/'route_config.resolved.json', config | {'input_hashes': inputs})
    done = state/'ingest_done.json'
    if done.exists() and p.frames_meta.exists():
        recs = read_jsonl(p.frames_meta)
        if recs and all((p.frames_dir/r['name']).is_file() for r in recs):
            print(f'Ingest reused: {len(recs)} route keyframes', flush=True)
            return p.root
    pts_path, scores_path = state/'timestamps.json', state/'sharpness.json'
    if not pts_path.exists():
        atomic_json(pts_path, probe_timestamps(config['source_video']))
    if not scores_path.exists():
        atomic_json(scores_path, score_video(config['source_video'], state/'decode.log'))
    pts, scores = json.loads(pts_path.read_text()), json.loads(scores_path.read_text())
    cap = config['capture']
    records = select_records(pts, scores, inputs['source_video'], cap['sharpness_group'],
                             cap['keyframe_hz'], cap['holdout_stride'])
    ensure_dir(p.frames_dir)
    select_path = state/'select_frames.filter'
    select_path.write_text("select='" + '+'.join(f'eq(n,{r["src_index"]})' for r in records) + "'",
                           encoding='ascii')
    print(f'Extracting {len(records)} route keyframes from {len(pts)} decoded frames', flush=True)
    with open(state/'extract.log', 'wb') as log:
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-threads', '8',
                        '-i', config['source_video'], '-filter_script:v', str(select_path),
                        '-fps_mode', 'passthrough', '-q:v', '2', '-start_number', '0',
                        str(p.frames_dir/'pending_%06d.jpg')], stderr=log, check=True)
    for j, rec in enumerate(records):
        pending = p.frames_dir/f'pending_{j:06d}.jpg'
        if not pending.exists():
            raise RuntimeError('Frame extraction did not produce every requested source index')
        pending.replace(p.frames_dir/rec['name'])
    write_jsonl(p.frames_meta, records)
    atomic_json(done, {'fingerprint': fingerprint, 'source_frames': len(pts),
                       'selected_frames': len(records), 'pts_interval': [pts[0], pts[-1]]})
    print(f'Ingest complete: {len(records)} keyframes', flush=True)
    return p.root
