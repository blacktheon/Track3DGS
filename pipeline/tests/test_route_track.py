import numpy as np
import pytest

from track3dgs.route_track import build_route_metadata, check_route_coverage, route_transform


def fixture_route():
    frames, poses = [], []
    # Curve, slope, then reverse; direction and height must survive the one transform.
    for i, xyz in enumerate([[0,0,0], [3,-4,0], [3,-4,5], [3,-4,0]]):
        frames.append({'name': f'frame_{i:06d}.jpg', 'frame_id': f'abc:{i}',
                       'source_sha256': 'abc', 'source_pts_seconds': i+8.,
                       't': float(i), 'split': 'held_out' if i == 2 else 'train'})
        T = np.eye(4)
        T[:3, 3] = xyz
        poses.append({'name': f'frame_{i:06d}_y+000.jpg', 'T_wc': T})
    return frames, poses


def test_one_similarity_preserves_curve_slope_and_reverse():
    frames, poses = fixture_route()
    R = np.diag([1., -1, -1])
    M = np.eye(4)
    M[:3, :3] = R
    route, cameras = build_route_metadata(frames, poses,
        {'method': 'manual', 'metres_per_sfm_unit': 2, 'status': 'approximate'}, M)
    samples = route['samples']
    assert [r['s'] for r in samples] == [0, 10, 20, 30]
    matrices = [np.array(r['rig_to_package']).reshape(4,4) for r in samples]
    assert matrices[1][:3,3] == pytest.approx([6,8,0])
    assert matrices[2][:3,3] == pytest.approx([6,8,-10])
    assert matrices[3][:3,3] == pytest.approx([6,8,0])
    assert np.linalg.det(matrices[1][:3,:3]) == pytest.approx(1)
    assert cameras[2]['split'] == 'held_out'
    assert cameras[2]['source_pts_seconds'] == 10


def test_nominal_scale_is_approximate_and_not_applied_to_rotation():
    frames, poses = fixture_route()
    route, _ = build_route_metadata(frames, poses,
        {'method': 'nominal_speed', 'speed_kmh': 36}, np.eye(4))
    assert route['length'] == pytest.approx(30)
    assert route['scale']['status'] == 'approximate'
    assert route['scale']['metres_per_sfm_unit'] == pytest.approx(2)
    assert np.array(route['samples'][1]['rig_to_package']).reshape(4,4)[:3,:3] == pytest.approx(np.eye(3))


def test_internal_registration_gap_blocks_training_plan():
    frames = [{'frame_id': str(i), 't': float(i)} for i in range(10)]
    result = check_route_coverage(frames, {'0','1','8','9'}, [{'id': '0'}], max_gap_seconds=2)
    assert not result['passed']
    assert result['gaps'][0]['time_interval'] == [1, 8]
    assert result['registered_fraction'] == .4


def test_disconnected_components_are_not_silently_hidden():
    frames = [{'frame_id': str(i), 't': float(i)} for i in range(3)]
    result = check_route_coverage(frames, {'0','1','2'}, [{'id':'0'}, {'id':'1'}])
    assert not result['passed']
    assert len(result['components']) == 2


def test_stationary_nominal_speed_cannot_fabricate_route():
    frames, poses = fixture_route()
    for pose in poses:
        pose['T_wc'][:3,3] = 0
    with pytest.raises(ValueError, match='stationary'):
        build_route_metadata(frames, poses, {'method':'nominal_speed', 'speed_kmh':10}, np.eye(4))


def test_route_orientation_is_single_rigid_transform_at_first_camera():
    frames, poses = fixture_route()
    transform = route_transform(poses)
    assert np.linalg.det(transform[:3,:3]) == pytest.approx(1)
    assert transform @ np.array([0,0,0,1]) == pytest.approx([0,0,0,1])
