import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from track3dgs.route_config import load_route_config, stage_fingerprint
from track3dgs.route_ingest import select_records, preserve_mask


def test_best_of_three_preserves_vfr_pts_and_source_identity():
    pts = [8.0, 8.03, 8.09, 8.2, 8.21, 8.4, 8.65]
    records = select_records(pts, [1, 9, 2, 2, 3, 8, 4], "abc", 3, None, 2)
    assert [r['src_index'] for r in records] == [1, 5, 6]
    assert [r['source_pts_seconds'] for r in records] == [8.03, 8.4, 8.65]
    assert [r['split'] for r in records] == ['train', 'held_out', 'train']
    assert records[0]['frame_id'] == 'abc:1'
    assert records[0]['t'] == pytest.approx(.03)
    other = select_records(pts, [1, 9, 2, 2, 3, 8, 4], 'def', 3, None, 2)
    assert other[0]['frame_id'] != records[0]['frame_id']


def test_keyframe_thinning_uses_time_and_keeps_last_candidate():
    records = select_records([0, .1, .6, 1.0, 1.1], [1]*5, 'a', 1, 2, 10)
    assert [r['src_index'] for r in records] == [0, 2, 3, 4]


def test_mask_is_never_silently_overwritten(tmp_path):
    src, dst = tmp_path/'src.png', tmp_path/'dst.png'
    cv2.imwrite(str(src), np.full((4, 8), 255, np.uint8))
    cv2.imwrite(str(dst), np.zeros((4, 8), np.uint8))
    with pytest.raises(ValueError, match='mask'):
        preserve_mask(src, dst)
    assert cv2.imread(str(dst), 0).sum() == 0


def test_config_resolves_relative_paths_and_rejects_invalid_lengths(tmp_path):
    (tmp_path/'video.mp4').write_bytes(b'fixture')
    cv2.imwrite(str(tmp_path/'mask.png'), np.full((4, 8), 255, np.uint8))
    cfg = {'schema_version': 1, 'route_id': 'test', 'revision': 'r1',
           'source_video': 'video.mp4', 'workspace': 'out', 'vehicle_mask': 'mask.png',
           'sky_prior': None, 'mount_calibration': None,
           'scale': {'method': 'nominal_speed', 'speed_kmh': 10},
           'regions': {'core_length': 100, 'context_length': 20,
                       'start_margin': 20, 'end_margin': 20}}
    path = tmp_path/'route.json'
    path.write_text(json.dumps(cfg))
    result = load_route_config(path)
    assert Path(result['source_video']) == tmp_path/'video.mp4'
    assert result['scale']['status'] == 'approximate'
    cfg['regions']['core_length'] = float('nan')
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_route_config(path)


def test_fingerprint_changes_with_inputs_settings_and_tool_versions():
    a = stage_fingerprint('views', {'source': 'a'}, {'size': 1600}, {'code': 1})
    assert a == stage_fingerprint('views', {'source': 'a'}, {'size': 1600}, {'code': 1})
    assert a != stage_fingerprint('views', {'source': 'b'}, {'size': 1600}, {'code': 1})
    assert a != stage_fingerprint('views', {'source': 'a'}, {'size': 800}, {'code': 1})


def test_long_frame_selection_does_not_exceed_ffmpeg_expression_depth():
    from track3dgs.route_ingest import build_select_filter
    result = subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
        'testsrc=size=16x8:rate=30:duration=1','-vf',build_select_filter(list(range(0,1200,2))),
        '-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True)
    assert result.returncode == 0, result.stderr.decode()[-500:]
    assert len(result.stdout) == 15*16*8*3
