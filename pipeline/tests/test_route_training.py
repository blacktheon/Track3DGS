from pathlib import Path
import json
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from track3dgs.route_training import (build_region_train_cmd, prepare_region_dataset,
    export_local_to_package, describe_native_export, completed_model)
from track3dgs.route_gpu import BoundedImageCache
from track3dgs.io_utils import write_json, write_jsonl
from track3dgs.colmap_export import write_colmap_model, load_images_txt


def test_resume_runs_only_remaining_iterations_and_handles_complete_checkpoint(tmp_path):
    from track3dgs.route_training import configure_resume
    directory=tmp_path/'checkpoints';directory.mkdir()
    (directory/'step-000028000.ckpt').write_bytes(b'checkpoint')
    command=build_region_train_cmd(tmp_path,tmp_path/'run',0,30000,Path('python.exe'))
    resumed,completed=configure_resume(command,directory,30000)
    assert completed==28001
    assert resumed[resumed.index('--max-num-iterations')+1]=='1999'
    assert resumed[resumed.index('--load-step')+1]=='28000'
    (directory/'step-000029999.ckpt').write_bytes(b'complete')
    assert configure_resume(command,directory,30000)==(None,30000)
    (directory/'step-000030000.ckpt').write_bytes(b'overtrained')
    with pytest.raises(ValueError,match='past'):
        configure_resume(command,directory,30000)


def test_region_commands_keep_distinct_paths_and_fixed_frame(tmp_path):
    cmds = [build_region_train_cmd(tmp_path, tmp_path/'run', i, 30000, Path('python.exe')) for i in [0,1,2]]
    assert len({c[c.index('--data')+1] for c in cmds}) == 3
    for cmd in cmds:
        text = ' '.join(map(str,cmd))
        assert '--orientation-method none' in text and '--auto-scale-poses False' in text
        assert '--assume-colmap-world-coordinate-convention False' in text
        assert '--pipeline.model.camera-optimizer.mode off' in text
        assert '--downscale-factor 1' in text


def test_dataset_preserves_global_poses_and_whole_panorama_holdouts(tmp_path):
    root=tmp_path/'route'; run=tmp_path/'run'
    (root/'views').mkdir(parents=True); (root/'views_masks').mkdir()
    cameras=[]
    for frame,split in [(1,'train'),(2,'held_out')]:
        for yaw in [0,90]:
            T=np.eye(4);T[:3,3]=[100+frame,10,20]
            name=f'frame_{frame:06d}_y{yaw:+04d}.jpg'
            (root/'views'/name).write_bytes(b'image');(root/'views_masks'/Path(name).with_suffix('.png')).write_bytes(b'mask')
            cameras.append({'camera_id':name,'name':name,'frame_id':str(frame),'split':split,'camera_to_package':T.ravel().tolist()})
    write_jsonl(root/'cameras.jsonl',cameras)
    meta=dict(width=1600,height=1600,fx=670,fy=670,cx=800,cy=800)
    write_json(root/'views'/'views_meta.json',meta)
    write_colmap_model(root/'cells'/'cell_000'/'train',meta,[],np.array([[102,10,22,50,100,200]]))
    region={'region_id':'cell_000','cell_index':0,'train_camera_ids':[c['camera_id'] for c in cameras if c['split']=='train'],
            'held_out_camera_ids':[c['camera_id'] for c in cameras if c['split']=='held_out']}
    folder=prepare_region_dataset(root,run,region)
    train=set((folder/'train_list.txt').read_text().splitlines()); val=set((folder/'val_list.txt').read_text().splitlines())
    assert train.isdisjoint(val) and len(train)==len(val)==2
    assert set((folder/'test_list.txt').read_text().splitlines())==val
    ims=load_images_txt(folder/'colmap'/'images.txt')
    assert len(ims)==4
    for im in ims:
        assert im['t']==pytest.approx([-int(im['name'][6:12])-100,-10,-20])
    region['train_camera_ids'].append(region['held_out_camera_ids'][0])
    with pytest.raises(ValueError,match='split|overlap'):
        prepare_region_dataset(root,run,region)


def test_native_export_retains_all_sh3_bytes_and_reports_inverse_transform(tmp_path):
    fields=[(n,'f4') for n in ['x','y','z','opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)],*[f'f_dc_{i}' for i in range(3)],*[f'f_rest_{i}' for i in range(45)]]]
    v=np.zeros(2,dtype=fields);v['x']=[1,2];v['rot_0']=1;v['f_rest_44']=[.2,.4]
    path=tmp_path/'raw.ply';PlyData([PlyElement.describe(v,'vertex')]).write(path)
    before=path.read_bytes()
    A=np.array([[0.,-1,0,2],[1,0,0,3],[0,0,1,4],[0,0,0,1]])
    parser={'transform':A[:3].tolist(),'scale':2}
    L=export_local_to_package(parser);x=np.array([8.,5.,7.,1.])
    expected=A@x;expected[:3]*=2
    assert L@expected==pytest.approx(x)
    model=describe_native_export(path,parser,'cell_000','fingerprint')
    assert model['count']==2 and model['sh_degree']==3 and path.read_bytes()==before
    marker=tmp_path/'model.json';write_json(marker,model)
    assert completed_model(marker,'fingerprint') is not None
    assert completed_model(marker,'different') is None
    path.write_bytes(before+b'tampered')
    assert completed_model(marker,'fingerprint') is None


def test_bounded_image_cache_evicts_without_changing_data():
    calls=[]
    def load(i):
        calls.append(i);return {'image':np.full((2,3,3),i,dtype=np.uint8)}
    cache=BoundedImageCache(100,load,2)
    assert len(cache)==100 and cache[4]['image'][0,0,0]==4
    assert cache[4]['image'][0,0,0]==4 and calls==[4]
    cache[5];cache[6];cache[4]
    assert calls==[4,5,6,4] and cache.resident_count==2
    with pytest.raises(IndexError):cache[100]
