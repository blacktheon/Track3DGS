import copy

import numpy as np
import pytest

from track3dgs.route_quality import evaluate_route_quality, withhold_region_plan
from track3dgs.route_regions import plan_regions
from track3dgs.route_config import atomic_json
from track3dgs.io_utils import read_json


def route_at_speeds(speeds):
    distances = np.r_[0., np.cumsum(speeds)]
    return {'coverage': {'passed': True}, 'length': float(distances[-1]),
            'samples': [{'timestamp_seconds': float(i), 's': float(s)}
                        for i, s in enumerate(distances)]}


def test_steady_capture_rejects_large_local_scale_drift_without_warping_route():
    route = route_at_speeds([30.] * 15 + [1.] * 195)
    before = copy.deepcopy(route)
    quality = evaluate_route_quality(route, {'capture_motion': 'broadly_similar_speed'})
    assert not quality['passed']
    assert quality['motion']['window_max_to_median'] == pytest.approx(30)
    assert 'motion' in quality['failures']
    assert route == before


def test_broadly_steady_motion_allows_brief_stops_and_small_speed_changes():
    route = route_at_speeds([2.8] * 80 + [0.] * 4 + [2.] * 60 + [3.2] * 66)
    quality = evaluate_route_quality(route, {'capture_motion': 'broadly_similar_speed'})
    assert quality['passed']
    assert quality['motion']['window_p95_to_median'] < 2


def test_unknown_capture_motion_does_not_assume_constant_speed():
    route = route_at_speeds([30.] * 15 + [1.] * 195)
    quality = evaluate_route_quality(route, {})
    assert quality['passed']
    assert quality['motion']['decision'] == 'diagnostic_only'


def test_manual_rejection_withholds_old_region_plan_and_preserves_evidence(tmp_path):
    route = route_at_speeds([2.] * 100)
    quality = evaluate_route_quality(route, {}, {'status': 'rejected_for_training', 'reason': 'Bad geometry'})
    assert not quality['passed']
    assert 'visual_review' in quality['failures']
    atomic_json(tmp_path/'regions.json', {'regions': [{'region_id': 'old'}]})
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports'/'training_regions.csv').write_text('old cuts\n')
    withhold_region_plan(tmp_path, quality)
    assert read_json(tmp_path/'regions.json')['regions'] == []
    assert not read_json(tmp_path/'regions.json')['training_ready']
    assert read_json(tmp_path/'reports'/'superseded_regions.json')['regions'][0]['region_id'] == 'old'
    assert (tmp_path/'reports'/'superseded_training_regions.csv').read_text() == 'old cuts\n'
    withhold_region_plan(tmp_path, quality)
    assert read_json(tmp_path/'reports'/'superseded_regions.json')['regions'][0]['region_id'] == 'old'
    route['quality'] = quality
    with pytest.raises(ValueError, match='quality'):
        plan_regions(route, [], {})
