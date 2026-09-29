"""Sequential Splatfacto training in the reviewed route frame, with native SH3."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from plyfile import PlyData

from .colmap_export import load_points3d_txt, write_colmap_model
from .io_utils import Project, read_json, read_jsonl
from .route_config import atomic_json, file_hash, stage_fingerprint
from .train import build_train_cmd, check_alignment


def build_region_train_cmd(workspace, run_root, cell_index, iters, python):
    cmd = build_train_cmd(Project(workspace), cell_index, iters)
    cmd[:1] = [str(python), '-u', '-m', 'track3dgs.route_gpu', 'train']
    cmd[cmd.index('--data')+1] = str(Path(run_root)/'datasets'/f'cell_{cell_index:03d}')
    cmd[cmd.index('--output-dir')+1] = str(Path(run_root)/'checkpoints')
    at = cmd.index('colmap')
    cmd[at:at] = ['--vis','tensorboard','--steps-per-save','2000',
        '--steps-per-eval-image','2000','--steps-per-eval-all-images','1000000',
        '--logging.steps-per-log','100','--pipeline.datamanager.cache-images','cpu',
        '--pipeline.model.camera-optimizer.mode','off']
    cmd += ['--assume-colmap-world-coordinate-convention','False','--downscale-factor','1']
    return [str(c) for c in cmd]


def prepare_region_dataset(workspace, run_root, region):
    root = Path(workspace)
    cameras = {c['camera_id']:c for c in read_jsonl(root/'cameras.jsonl')}
    train, held = region['train_camera_ids'], region['held_out_camera_ids']
    if set(train)&set(held): raise ValueError('Training and held-out lists overlap')
    selected = [cameras[i] for i in [*train,*held]]
    frame_splits = {}
    for c in selected:
        expected = 'train' if c['camera_id'] in train else 'held_out'
        if c['split'] != expected: raise ValueError('Source split changed')
        previous = frame_splits.setdefault(c['frame_id'], expected)
        if previous != expected: raise ValueError('One panorama crossed photographic split')
        if not (root/'views'/c['name']).is_file() or not (root/'views_masks'/Path(c['name']).with_suffix('.png')).is_file():
            raise ValueError('Missing training image or mask: '+c['name'])
    out = Path(run_root)/'datasets'/region['region_id']
    meta = read_json(root/'views'/'views_meta.json')
    points = load_points3d_txt(root/'cells'/region['region_id']/'train'/'points3D.txt')
    write_colmap_model(out/'colmap',meta,
        [{'name':c['name'],'T_wc':np.array(c['camera_to_package']).reshape(4,4)} for c in selected], points)
    for split, ids in [('train',train),('val',held),('test',held)]:
        (out/(split+'_list.txt')).write_text(''.join(cameras[i]['name']+'\n' for i in ids),encoding='utf-8')
    atomic_json(out/'region.json',region)
    return out


def export_local_to_package(parser):
    transform = np.eye(4)
    transform[:3] = np.asarray(parser['transform']).reshape(3,4)
    scale = float(parser['scale'])
    if not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid dataparser scale')
    R = transform[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or np.linalg.det(R)<0:
        raise ValueError('Dataparser transform must be rigid')
    transform[:3] *= scale
    return np.linalg.inv(transform)


def describe_native_export(path, parser, region_id, fingerprint):
    path = Path(path).resolve()
    v = PlyData.read(str(path))['vertex'].data
    names = v.dtype.names
    required = ['x','y','z','opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)],*[f'f_dc_{i}' for i in range(3)],*[f'f_rest_{i}' for i in range(45)]]
    if not len(v) or not all(k in names for k in required): raise ValueError('Expected nonempty native SH3 Gaussian PLY')
    if any(not np.all(np.isfinite(v[k])) for k in required): raise ValueError('Nonfinite Gaussian attributes')
    return {'schema_version':1,'region_id':region_id,'status':'trained_draft','fingerprint':fingerprint,
        'raw_file':str(path),'raw_sha256':file_hash(path),'count':len(v),'sh_degree':3,
        'local_to_package':export_local_to_package(parser).ravel().tolist(),
        'coordinate_frame':'right-handed RUB package frame; SH evaluated in PLY-local frame',
        'sh_layout':'Inria channel-major real SH3','color_space':'sRGB coefficients; classic rasterization',
        'training_ready':False,'visual_review':'pending'}


def completed_model(path, fingerprint):
    path = Path(path)
    if not path.exists(): return None
    model = read_json(path)
    raw = Path(model['raw_file'])
    if model.get('fingerprint')!=fingerprint or not raw.is_file() or file_hash(raw)!=model['raw_sha256']:
        return None
    return model


@contextmanager
def gpu_lease(path):
    import msvcrt
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as lock:
        if not path.stat().st_size: lock.write(b'0');lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error: raise RuntimeError('Another route GPU job holds the training lease') from error
        try: yield
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def run_gpu(cmd, log):
    launcher = Path(__file__).resolve().parents[1]/'cuda_env.bat'
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    with Path(log).open('a',encoding='utf-8') as stream:
        stream.write('\nCOMMAND '+json.dumps(cmd)+'\n');stream.flush()
        subprocess.run([str(launcher),*cmd],cwd=launcher.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def run_region_training(workspace, region, settings):
    root = Path(workspace);run = Path(settings['run_root'])
    rid = region['region_id'];out = run/'models'/rid;out.mkdir(parents=True,exist_ok=True)
    reports = run/'reports';reports.mkdir(exist_ok=True)
    fingerprint = stage_fingerprint('regional-training',
        {'route':file_hash(root/'route.json'),'cameras':file_hash(root/'cameras.jsonl'),
         'views':file_hash(root/'state'/'views_identity.json')},
        {'region':region,'iterations':settings['iterations'],'recipe':'splatfacto-native-rub-v1','image_cache':32},
        {'nerfstudio':'1.1.5','gsplat':'1.4.0','torch':'2.7.1+cu128'})
    marker = out/'model.json'
    existing = completed_model(marker,fingerprint)
    if existing:
        print(f'REUSED {rid}: {existing["count"]:,} splats',flush=True);return existing
    identity = out/'identity.json'
    if identity.exists() and read_json(identity)['fingerprint']!=fingerprint:
        raise ValueError(f'{rid} training inputs changed; choose a new training revision')
    atomic_json(identity,{'fingerprint':fingerprint,'region':rid})
    dataset = prepare_region_dataset(root,run,region)
    python = Path(settings['training_python'])
    cfg = run/'checkpoints'/rid/'splatfacto'/'run'/'config.yml'
    train_done = out/'training_done.json'
    start = time.monotonic()
    if not train_done.exists():
        command = build_region_train_cmd(root,run,region['cell_index'],settings['iterations'],python)
        checkpoints = cfg.parent/'nerfstudio_models'
        if checkpoints.exists() and list(checkpoints.glob('step-*.ckpt')):
            command[command.index('colmap'):command.index('colmap')] = ['--load-dir',str(checkpoints)]
        atomic_json(out/'progress.json',{'status':'training','pid':os.getpid(),'fingerprint':fingerprint})
        print(f'TRAINING {rid}; log={reports/(rid+"_train.log")}',flush=True)
        run_gpu(command,reports/(rid+'_train.log'))
        atomic_json(train_done,{'fingerprint':fingerprint,'elapsed_seconds':time.monotonic()-start})
    parser = read_json(cfg.parent/'dataparser_transforms.json')
    if not np.allclose(export_local_to_package(parser),np.eye(4),atol=1e-6):
        raise ValueError('Expected explicit native route frame; dataparser transform drifted')
    raw = out/'splat.ply'
    export_done = out/'export_done.json'
    if not (raw.exists() and export_done.exists() and read_json(export_done)['sha256']==file_hash(raw)):
        run_gpu([str(python),'-u','-m','track3dgs.route_gpu','export','gaussian-splat','--load-config',str(cfg),'--output-dir',str(out)], reports/(rid+'_export.log'))
        atomic_json(export_done,{'sha256':file_hash(raw)})
    model = describe_native_export(raw,parser,rid,fingerprint)
    alignment = check_alignment(raw,dataset/'colmap')
    model.update({'alignment':alignment,'training_seconds':read_json(train_done)['elapsed_seconds'],
        'iterations':settings['iterations'],'source_keyframe_hz':2,'dataset':str(dataset),'config_yml':str(cfg)})
    if not alignment['ok']:
        atomic_json(out/'alignment_failed.json',model)
        raise ValueError(f'{rid}: native export failed route alignment check')
    atomic_json(marker,model)
    atomic_json(out/'progress.json',{'status':'exported','count':model['count'],'alignment':alignment})
    print(f'EXPORTED {rid}: {model["count"]:,} splats; alignment={alignment["offset_m"]:.4f}',flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True);parser.add_argument('--run-root',required=True)
    parser.add_argument('--training-python',required=True);parser.add_argument('--regions',default='0,1,2,3,4,5')
    parser.add_argument('--iterations',type=int,default=30000)
    args=parser.parse_args();root=Path(args.workspace)
    route=read_json(root/'route.json')
    if not route.get('quality',{}).get('passed'): raise ValueError('Route quality gate failed')
    plan=read_json(root/'regions.json')
    settings={'run_root':args.run_root,'training_python':args.training_python,'iterations':args.iterations}
    with gpu_lease(root.parent.parent/'.gpu-training.lock'):
        for index in map(int,args.regions.split(',')):
            run_region_training(root,plan['regions'][index],settings)


if __name__ == '__main__': main()
