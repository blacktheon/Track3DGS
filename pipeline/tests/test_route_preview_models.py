import numpy as np
import pytest
from plyfile import PlyData
from track3dgs.route_preview_models import recenter_vertices


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
