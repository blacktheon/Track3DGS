import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from track3dgs.package_export import export_reconstruction
from track3dgs.package_contract import validate_package
from track3dgs.route_assembly import PROVENANCE_DTYPE
from track3dgs.route_config import atomic_json, file_hash


def training_fixture(tmp_path):
    root=tmp_path/'workspace';root.mkdir();run=root/'training'/'t001';out=run/'models'/'cell_002';out.mkdir(parents=True)
    fields=['x','y','z','opacity',*[f'rot_{i}' for i in range(4)],*[f'scale_{i}' for i in range(3)],*[f'f_dc_{i}' for i in range(3)]]
    vertices=np.zeros(2,dtype=[(k,'<f4') for k in fields]);vertices['rot_0']=1
    PlyData([PlyElement.describe(vertices,'vertex')],byte_order='<').write(out/'core.ply')
    ids=np.empty(2,dtype=PROVENANCE_DTYPE);ids['source_index']=2;ids['source_row']=[0,3];ids.tofile(out/'core.provenance.bin')
    raw=np.zeros(5,dtype=vertices.dtype);raw['rot_0']=1
    PlyData([PlyElement.describe(raw,'vertex')],byte_order='<').write(out/'splat.ply')
    atomic_json(out/'model.json',{'region_id':'cell_002','raw_file':str(out/'splat.ply'),'raw_sha256':file_hash(out/'splat.ply'),'count':5,'sh_degree':0,
        'local_to_package':np.eye(4).ravel().tolist(),'fingerprint':'b'*64})
    atomic_json(out/'processing.json',{'core_count':2,'core_sha256':file_hash(out/'core.ply'),'raw_count':5,'clean_count':4,
        'core_provenance_sha256':file_hash(out/'core.provenance.bin'),'core_provenance_bytes':24})
    atomic_json(root/'route.json',{'route_id':'fixture','scale':{'status':'approximate','basis':'assumed speed'},'samples':[]})
    atomic_json(root/'regions.json',{'playable_s':[0,30],'capture_margins':[0,0],'regions':[{'region_id':'cell_002','cell_index':2,'core_s':[20,30],'context_s':[15,35]}]})
    atomic_json(root/'views'/'views_meta.json',{'width':1600,'height':1600,'fx':600,'fy':600,'cx':800,'cy':800})
    (root/'cameras.jsonl').write_text(json.dumps({'split':'train','camera_id':'camera','camera_to_package':np.eye(4).ravel().tolist()})+'\n')
    return root,run


def test_draft_export_is_relocatable_and_remaps_registry_without_losing_rows(tmp_path):
    root,run=training_fixture(tmp_path);output=tmp_path/'published'
    manifest=export_reconstruction(root,run,output,'test',git_commit=None)
    shutil.copytree(output,tmp_path/'relocated');result=validate_package(tmp_path/'relocated')
    assert result['splat_count']==2 and result['state']=='draft'
    m=json.loads(manifest.read_text());asset=m['assets'][0]
    ids=np.fromfile(output/asset['provenance']['path'],dtype=PROVENANCE_DTYPE)
    assert ids['source_index'].tolist()==[0,0] and ids['source_row'].tolist()==[0,3]
    assert len((output/asset['file']['path']).read_bytes())>0
    assert not str(root) in manifest.read_text()
    with pytest.raises(FileExistsError): export_reconstruction(root,run,output,'test')


def test_changed_input_hash_prevents_publication(tmp_path):
    root,run=training_fixture(tmp_path)
    with (run/'models'/'cell_002'/'core.ply').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError):export_reconstruction(root,run,tmp_path/'published','test')
    assert not (tmp_path/'published').exists()


@pytest.mark.parametrize('mutation',['extra_byte','in_range_row','wrong_attributes'])
def test_provenance_corruption_cannot_publish(tmp_path,mutation):
    root,run=training_fixture(tmp_path);out=run/'models'/'cell_002'
    path=out/'core.provenance.bin'
    if mutation=='extra_byte':
        with path.open('ab') as f:f.write(b'x')
    elif mutation=='in_range_row':
        ids=np.fromfile(path,dtype=PROVENANCE_DTYPE);ids['source_row'][1]=2;ids.tofile(path)
    else:
        v=PlyData.read(out/'core.ply',mmap=False)['vertex'].data;v['x'][1]=50
        PlyData([PlyElement.describe(v,'vertex')],byte_order='<').write(out/'core.ply')
        report=json.loads((out/'processing.json').read_text());report['core_sha256']=file_hash(out/'core.ply')
        atomic_json(out/'processing.json',report)
    with pytest.raises(ValueError):export_reconstruction(root,run,tmp_path/'published','test')
