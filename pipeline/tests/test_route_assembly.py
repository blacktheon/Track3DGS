import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from track3dgs.route_assembly import project_to_route, core_mask, write_subset, PROVENANCE_DTYPE


def test_continuous_projection_retains_curve_and_slope():
    route = np.array([[0.,0,0],[10,0,0],[10,10,0]])
    s, d, nearest, ambiguous = project_to_route(np.array([[4,2,0],[12,6,0]]),route,[0,10,20])
    np.testing.assert_allclose(s,[4,16])
    np.testing.assert_allclose(d,[2,2])
    np.testing.assert_allclose(nearest,[[4,0,0],[10,6,0]])
    assert not ambiguous.any()


def test_half_open_ownership_and_context_not_exported_twice():
    s = np.array([-1,0,9.999,10,15,20,21])
    a = core_mask(s,{'core_s':[0,10],'core_end_inclusive':False})
    b = core_mask(s,{'core_s':[10,20],'core_end_inclusive':True})
    assert not (a&b).any()
    assert np.array_equal(a|b,[False,True,True,True,True,True,False])


def test_hairpin_ambiguity_is_flagged_and_stops_have_no_division_by_zero():
    xyz=np.array([[0.,0,0],[0,0,0],[100,0,0],[100,0,1],[0,0,1]])
    s,d,p,ambiguous=project_to_route(np.array([[10.,0,.5],[10,0,-5]]),xyz,[0,0,100,101,201])
    assert ambiguous[0]
    assert np.isfinite(s).all() and np.isfinite(d).all()


def test_subset_preserves_every_attribute_and_64bit_source_identity(tmp_path):
    fields=['x','y','z','rot_0','opacity','scale_0','f_dc_0','f_rest_44','auxiliary']
    v=np.zeros(4,dtype=[(f,'<f4') for f in fields])
    for i,f in enumerate(fields): v[f]=np.arange(4)+i*.1
    source=tmp_path/'raw.ply';PlyData([PlyElement.describe(v,'vertex')]).write(source)
    before=source.read_bytes()
    rows=np.array([2**33,2**33+1,2**33+2,2**33+3],dtype=np.uint64)
    write_subset(v,np.array([False,True,False,True]),rows,7,tmp_path/'core.ply')
    actual=PlyData.read(tmp_path/'core.ply')['vertex'].data
    assert actual.tobytes()==v[[1,3]].tobytes()
    assert source.read_bytes()==before
    ids=np.fromfile(tmp_path/'core.provenance.bin',dtype=PROVENANCE_DTYPE)
    assert ids.itemsize==12 and ids['source_index'].tolist()==[7,7]
    assert ids['source_row'].tolist()==[2**33+1,2**33+3]
