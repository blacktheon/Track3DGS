"""Protect camera/mask conventions and immutable row selection during extraction."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement


def api():
    assert importlib.util.find_spec('track3dgs.sky_volume') is not None, 'Reusable sky-volume module missing'
    from track3dgs import sky_volume
    return sky_volume


def test_upper_sky_excludes_vehicle_and_isolated_dark_pixels():
    mask = np.zeros((10, 10), bool)
    mask[:3, :4] = True
    mask[5, 6] = True
    mask[8:, :] = True
    result = api().top_connected_sky(mask)
    assert result.sum() == 12
    assert not result[5, 6] and not result[9, 1]


def test_projection_respects_pose_front_clipping_and_vehicle_zone():
    sky = np.zeros((100, 100), bool); sky[:40] = True
    K = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]])
    camera_points = np.array([[0, -.5, 1], [0, .8, 1], [0, .5, -1], [5, -.5, 1.]])
    c2w = np.array([[0, 0, 1, 8], [0, 1, 0, 3], [-1, 0, 0, -4], [0, 0, 0, 1.]])
    world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    assert api().center_hits(world, np.linalg.inv(c2w), K, sky).tolist() == [True, False, False, False]


def test_projected_vertical_spike_is_detected_without_deleting_horizontal_road():
    sky = np.zeros((100, 100), bool); sky[:40] = True
    uv = np.array([[50, 55], [50, 55], [50, 55], [50, 55], [50, 90.]])
    conics = np.array([[1, 0, .0025], [.0025, 0, 1], [1, 0, 1], [1, 0, .0025], [1, 0, 1.]])
    hit, _ = api().footprint_hits(uv, conics, np.array([60, 60, 3, 0, 3]), np.ones(5), sky)
    assert hit.tolist() == [True, False, False, False, False]


def test_consensus_does_not_double_count_yaws_and_protects_foreground():
    center = np.array([0, 0, 0, 3, 1, 0], np.uint64)
    sky = np.array([1 | 1, 3, 3, 0, 2, 0], np.uint64)
    mass = np.array([100., 100., 1., 0., 1., 0.])
    foreground = np.array([0., 0., 100., 100., 5., 0.])
    assert api().decide(center, sky, mass, foreground).tolist() == [False, True, False, True, True, False]


def cameras():
    return [dict(frame_id=f'f{i}', name=f'f{i}_y{yaw}.jpg', s=float(i * 10),
                 split='train' if i % 2 == 0 else 'held_out', yaw_degrees=yaw)
            for i in range(6) for yaw in (0, 90)]


def test_view_selection_groups_yaws_by_position_and_separates_validation():
    groups = {'edit': {'split': 'train', 'stations': [0, 20], 'yaws': [0, 90]},
              'validation': {'split': 'held_out', 'stations': [10, 30], 'yaws': [0, 90]}}
    selected = api().select_cameras(cameras(), groups)
    assert [c['position_index'] for c in selected['edit']] == [0, 0, 1, 1]
    assert {c['frame_id'] for c in selected['validation']} == {'f1', 'f3'}
    groups['validation'] = {'names': ['f0_y90.jpg']}
    with pytest.raises(ValueError, match='validation'):
        api().select_cameras(cameras(), groups)


def test_sparse_requested_positions_fail_instead_of_inventing_extra_votes():
    with pytest.raises(ValueError, match='distinct'):
        api().select_cameras(cameras(), {'edit': {'split': 'train', 'stations': [0, 1], 'yaws': [0]}})


def test_filtered_export_keeps_all_attributes_and_mixed_source_provenance(tmp_path):
    from track3dgs.route_assembly import PROVENANCE_DTYPE
    source = tmp_path / 'source.ply'
    data = np.zeros(4, dtype=[('x', '<f4'), ('f_rest_44', '<f4'), ('scale_0', '<f4')])
    data['x'] = [0, 1, 2, 3]; data['f_rest_44'] = [.1, -.2, .3, -.4]
    PlyData([PlyElement.describe(data, 'vertex')], byte_order='<').write(str(source))
    rows = np.array([(0, 10), (1, 20), (0, 30), (1, 40)], dtype=PROVENANCE_DTYPE)
    rows.tofile(source.with_suffix('.provenance.bin'))
    before = source.read_bytes()
    out = tmp_path / 'result.ply'
    api().write_filtered(source, np.array([False, True, False, True]), out)
    assert PlyData.read(str(out))['vertex'].data.tobytes() == data[[0, 2]].tobytes()
    assert np.fromfile(out.with_suffix('.provenance.bin'), PROVENANCE_DTYPE).tobytes() == rows[[0, 2]].tobytes()
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        api().write_filtered(source, np.zeros(4, bool), out)


def test_filtered_export_requires_explicit_identity_for_untracked_raw_rows(tmp_path):
    source = tmp_path / 'raw.ply'
    data = np.zeros(2, dtype=[('x', '<f4')])
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(source))
    with pytest.raises(ValueError, match='source.index|source_index'):
        api().write_filtered(source, np.zeros(2, bool), tmp_path / 'out.ply')


def test_raw_cleanup_transfers_to_core_by_provenance_not_row_position(tmp_path):
    from track3dgs.route_assembly import PROVENANCE_DTYPE
    from track3dgs.route_config import atomic_json, file_hash
    from track3dgs.route_preview_models import write_preview_ply
    module = api()
    assert hasattr(module, 'derive_core'), 'Core provenance transfer missing'
    raw = tmp_path/'raw.ply'; core = tmp_path/'core.ply'; output = tmp_path/'cleanup'
    (output/'models').mkdir(parents=True)
    data = np.zeros(4, dtype=[('x','<f4')]); data['x']=[1,2,3,4]
    ids = np.array([(0,0),(0,1),(0,2),(0,3)], dtype=PROVENANCE_DTYPE)
    write_preview_ply(data,raw); ids.tofile(raw.with_suffix('.provenance.bin'))
    write_preview_ply(data[[3,1,2]],core); ids[[3,1,2]].tofile(core.with_suffix('.provenance.bin'))
    module.write_filtered(raw,np.array([False,True,False,False]),output/'models/sky.ply')
    atomic_json(output/'processing.json',dict(source='raw.ply',source_sha256=file_hash(raw),
        file='models/sky.ply',sha256=file_hash(output/'models/sky.ply'),
        provenance_sha256=file_hash(output/'models/sky.provenance.bin')))
    module.derive_core(tmp_path,output,core)
    assert PlyData.read(output/'models/sky_core.ply')['vertex'].data['x'].tolist()==[4,3]
    assert np.fromfile(output/'models/sky_core.provenance.bin',PROVENANCE_DTYPE)['source_row'].tolist()==[3,2]


def test_core_transfer_rejects_changed_retained_provenance_before_writing(tmp_path):
    from track3dgs.route_assembly import PROVENANCE_DTYPE
    from track3dgs.route_config import atomic_json, file_hash
    from track3dgs.route_preview_models import write_preview_ply
    module=api();output=tmp_path/'cleanup';(output/'models').mkdir(parents=True)
    raw=tmp_path/'raw.ply';core=tmp_path/'core.ply';data=np.zeros(3,dtype=[('x','<f4')]);data['x']=[1,2,3]
    ids=np.array([(0,0),(0,1),(0,2)],dtype=PROVENANCE_DTYPE)
    for p in (raw,core):write_preview_ply(data,p);ids.tofile(p.with_suffix('.provenance.bin'))
    module.write_filtered(raw,np.array([False,True,False]),output/'models/sky.ply')
    atomic_json(output/'processing.json',dict(source='raw.ply',source_sha256=file_hash(raw),
        file='models/sky.ply',sha256=file_hash(output/'models/sky.ply'),
        provenance_sha256=file_hash(output/'models/sky.provenance.bin')))
    ids[[0,1]].tofile(output/'models/sky.provenance.bin')
    with pytest.raises(ValueError,match='provenance|Provenance'):
        module.derive_core(tmp_path,output,core)
    assert not (output/'models/sky_core.ply').exists()
