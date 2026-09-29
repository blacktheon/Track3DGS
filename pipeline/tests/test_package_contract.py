import json
import shutil
import numpy as np
import pytest
from plyfile import PlyData,PlyElement

from track3dgs.package_contract import file_ref, validate_package, validate_similarity
from track3dgs.route_assembly import PROVENANCE_DTYPE


def fixture(root):
    root.mkdir(parents=True,exist_ok=True)
    fields=['x','y','z','opacity',*[f'rot_{i}' for i in range(4)],*[f'scale_{i}' for i in range(3)],*[f'f_dc_{i}' for i in range(3)]]
    v=np.zeros(2,dtype=[(f,'<f4') for f in fields]);v['rot_0']=1
    PlyData([PlyElement.describe(v,'vertex')],byte_order='<').write(root/'model.ply')
    ids=np.empty(2,dtype=PROVENANCE_DTYPE);ids['source_index']=0;ids['source_row']=[0,1];ids.tofile(root/'model.bin')
    for name in ['regions','ownership','quality']:(root/(name+'.json')).write_text('{}')
    manifest={'format':'gs-asset-package','schema_version':1,'kind':'reconstruction',
        'package_id':'f7ad5443-717b-4621-a920-e5d95dd27ab5','revision':'test','created_utc':'2026-09-29T00:00:00Z',
        'producer':{'name':'fixture','version':'1','git_commit':None},'parents':[],
        'required_capabilities':['ply.sh0','transform.similarity','provenance.u32-u64'],
        'coordinate_system':{'axes':'RUB','handedness':'right','matrix_layout':'row-major','vectors':'column',
            'unit':'meter','scale_status':'approximate','scale_basis':'fixture','level_status':'unknown'},
        'sources':[{'source_id':'source-0','kind':'original','count':2,'sha256':file_ref(root,root/'model.ply')['sha256'],'producer':{}}],
        'assets':[{'asset_id':'model','file':file_ref(root,root/'model.ply'),'codec':'ply-3dgs-f32-le','count':2,
            'sh_degree':0,'radiance_encoding':'srgb','local_to_package':np.eye(4).ravel().tolist(),
            'provenance':file_ref(root,root/'model.bin'),'render_bounds':{'min':[-4]*3,'max':[4]*3,'support_sigma':4.0}}],
        'reconstruction':{k:file_ref(root,root/(f+'.json')) for k,f in [('regions','regions'),('ownership','ownership'),('quality_report','quality')]},
        'validation':{'state':'draft','reports':[],'limitations':['Fixture without a route']}}
    (root/'manifest.json').write_text(json.dumps(manifest));return manifest


def test_generic_route_free_package_relocates_and_validates(tmp_path):
    fixture(tmp_path/'original');shutil.copytree(tmp_path/'original',tmp_path/'elsewhere')
    result=validate_package(tmp_path/'elsewhere')
    assert result['asset_count']==1 and result['splat_count']==2


@pytest.mark.parametrize('mutation',['hash','traversal','count','quaternion','duplicate_identity','reflection','case_alias'])
def test_rejects_corrupt_package(tmp_path,mutation):
    m=fixture(tmp_path);asset=m['assets'][0]
    if mutation=='hash': (tmp_path/'model.bin').write_bytes(b'x'*24)
    elif mutation=='traversal': asset['file']['path']='../model.ply'
    elif mutation=='count': asset['count']=3
    elif mutation=='quaternion':
        ply=PlyData.read(tmp_path/'model.ply',mmap=False);v=ply['vertex'].data.copy();v['rot_0']=0
        PlyData([PlyElement.describe(v,'vertex')],byte_order='<').write(tmp_path/'model.ply')
        asset['file']=file_ref(tmp_path,tmp_path/'model.ply')
    elif mutation=='duplicate_identity':
        ids=np.zeros(2,dtype=PROVENANCE_DTYPE);ids.tofile(tmp_path/'model.bin');asset['provenance']=file_ref(tmp_path,tmp_path/'model.bin')
    elif mutation=='reflection': asset['local_to_package'][0]=-1
    else:
        extra=dict(asset);extra['asset_id']='second';extra['file']=dict(asset['file'],path='MODEL.ply');m['assets'].append(extra)
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):validate_package(tmp_path)


def test_similarity_rejects_shear_and_nonuniform_scale():
    T=np.eye(4);T[0,1]=.2
    with pytest.raises(ValueError):validate_similarity(T.ravel())
    T=np.eye(4);T[0,0]=2
    with pytest.raises(ValueError):validate_similarity(T.ravel())
