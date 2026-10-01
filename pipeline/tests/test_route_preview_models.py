import numpy as np
import pytest
from plyfile import PlyData
from track3dgs.route_preview_models import recenter_vertices


def preview_fixture(tmp_path):
    from track3dgs.route_config import atomic_json, file_hash
    from track3dgs.route_preview_models import write_preview_ply
    root, run, unity = tmp_path/'route', tmp_path/'run', tmp_path/'viewer'
    folder = run/'models/cell_000'; folder.mkdir(parents=True)
    region = dict(region_id='cell_000', cell_index=0, core_s=[0, 100])
    a = np.eye(4); a[:3, 3] = [10, 20, -30]
    b = a.copy(); b[0, 3] = 110
    atomic_json(root/'route.json', {'route_id': 'different-route', 'samples': [
        {'s': 0, 'rig_to_package': a.ravel().tolist()}, {'s': 100, 'rig_to_package': b.ravel().tolist()}]})
    atomic_json(root/'reports/route_preview.json', {'schemaVersion': 1, 'routeSha256': file_hash(root/'route.json'), 'markers': []})
    atomic_json(folder/'model.json', {'local_to_package': np.eye(4).ravel().tolist()})
    v = np.zeros(2, dtype=[(n, '<f4') for n in ('x','y','z','f_rest_44')]); v['x'] = [60, 61]; v['z'] = -30
    for filename in ['splat.ply','clean.ply','core.ply']:
        write_preview_ply(v, folder/filename)
    atomic_json(folder/'processing.json', {'raw_count':2, 'clean_count':2, 'core_count':2})
    return root, run, unity, region


def test_publisher_uses_generic_route_folder_and_copies_route_markers(tmp_path):
    from track3dgs.route_preview_models import publish_preview
    from track3dgs.io_utils import read_json
    root, run, unity, region = preview_fixture(tmp_path)
    publish_preview(root, run, unity, region)
    path = unity/'Assets/Track3DGSData/catalog.json'
    assert path.exists(), 'Publication still requires the old QuestSBTC layout'
    catalog = read_json(path)
    assert catalog['routeId'] == 'different-route'
    assert catalog['entries'][0]['x'] == 60 and catalog['entries'][0]['z'] == 30
    assert (unity/'Assets/Track3DGSData/route_preview.json').read_bytes() == (root/'reports/route_preview.json').read_bytes()
    source = PlyData.read(run/'models/cell_000/core.ply')['vertex'].data
    cached = PlyData.read(unity/catalog['entries'][0]['corePath'])['vertex'].data
    assert cached['f_rest_44'].tobytes() == source['f_rest_44'].tobytes()


def test_publisher_rejects_route_mismatch_before_overwriting_data(tmp_path):
    from track3dgs.route_preview_models import publish_preview
    from track3dgs.route_config import atomic_json
    root, run, unity, region = preview_fixture(tmp_path)
    publish_preview(root, run, unity, region)
    before = {p: p.read_bytes() for p in unity.rglob('*') if p.is_file()}
    atomic_json(root/'route.json', {'route_id':'another', 'samples':[]})
    with pytest.raises(ValueError, match='route|Route'):
        publish_preview(root, run, unity, region)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_review_camera_reflects_rub_once_and_converts_opencv_down():
    from track3dgs import route_preview_models as module
    assert hasattr(module, 'review_view'), 'Reusable recorded-camera adapter missing'
    pose = np.eye(4); pose[:3,3] = [1,2,3]
    view = module.review_view({'camera_to_package':pose.ravel().tolist(), 's':7, 'yaw_degrees':45}, 0, 'edit', 100)
    assert view['position'] == {'x':1.,'y':2.,'z':-3.}
    assert view['forward'] == {'x':0.,'y':0.,'z':-1.}
    assert view['up'] == {'x':0.,'y':-1.,'z':0.}


def test_translation_cache_preserves_native_sh_and_restores_global_placement():
    v=np.zeros(3,dtype=[(n,'<f4') for n in ('x','y','z','f_rest_44','rot_0','scale_0')])
    v['x']=[400.11,420.43,450.51];v['y']=[-10,0,10];v['z']=[-400,-440,-480]
    v['f_rest_44']=[.123,-.237,.956];v['rot_0']=.707;v['scale_0']=-3
    original=v.copy();origin=np.array([420.,0.,-440.])
    cache=recenter_vertices(v,origin)
    for n in ('f_rest_44','rot_0','scale_0'): assert cache[n].tobytes()==v[n].tobytes()
    np.testing.assert_allclose(np.column_stack([cache[n] for n in ('x','y','z')])+origin,
                               np.column_stack([v[n] for n in ('x','y','z')]),atol=1e-5)
    assert original.tobytes()==v.tobytes()
    assert max(abs(cache['x'].astype(np.float16).astype(np.float32)-cache['x'])) < .02


def test_incomplete_preview_is_not_published_to_a_live_unity_project(tmp_path, monkeypatch):
    from track3dgs.route_preview_models import write_preview_ply
    destination = tmp_path / 'core.ply'
    destination.write_bytes(b'existing complete asset')
    vertices = np.zeros(1, dtype=[('x', '<f4')])
    def interrupted_write(self, path):
        assert str(path) != str(destination)
        assert destination.read_bytes() == b'existing complete asset'
        with open(path, 'wb') as stream:
            stream.write(b'incomplete header')
        raise OSError('Interrupted cache publication')
    with monkeypatch.context() as patch:
        patch.setattr(PlyData, 'write', interrupted_write)
        with pytest.raises(OSError, match='Interrupted'):
            write_preview_ply(vertices, destination)
    assert destination.read_bytes() == b'existing complete asset'
    assert list(tmp_path.iterdir()) == [destination]
    write_preview_ply(vertices, destination)
    assert len(PlyData.read(destination)['vertex'].data) == 1
