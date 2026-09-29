"""Route-wide SfM ingestion. All regions inherit one scale and one rigid orientation."""
import json
import subprocess
from pathlib import Path

import numpy as np

from .colmap_export import load_images_txt, load_points3d_full, write_colmap_model
from .io_utils import Project, ensure_dir, read_json, read_jsonl, write_jsonl
from .route_config import atomic_json, file_hash, stage_fingerprint
from .track import (rig_poses_from_views, drop_rig_outliers, build_feature_cmd,
                    build_matcher_cmd, build_mapper_cmd, build_converter_cmd,
                    prepare_colmap_masks)
from .trajectory import arc_length, quat_to_R


def read_view_poses(path):
    views = []
    for im in load_images_txt(path):
        qw,qx,qy,qz = im['q']
        R = quat_to_R(qx,qy,qz,qw)
        T = np.eye(4)
        T[:3,:3], T[:3,3] = R.T, -R.T @ np.asarray(im['t'])
        views.append({'name':im['name'], 'T_wc':T})
    return views


def route_transform(views):
    rigs = rig_poses_from_views(views)
    ordered = [rigs[k] for k in sorted(rigs)]
    if len(ordered) < 2:
        raise ValueError('Need at least two rig poses')
    up = -np.mean([T[:3,1] for T in ordered], axis=0)
    up /= np.linalg.norm(up)
    origin = ordered[0][:3,3]
    forward = None
    for T in ordered[min(6,len(ordered)-1):]:
        delta = T[:3,3]-origin
        delta -= up * np.dot(delta,up)
        if np.linalg.norm(delta) > 1e-7:
            forward = delta/np.linalg.norm(delta)
            break
    if forward is None:
        raise ValueError('A stationary or vertical-only trajectory cannot establish a route heading')
    right = np.cross(forward, up)
    M = np.eye(4)
    M[:3,:3] = np.stack([right, up, -forward])
    M[:3,3] = -M[:3,:3] @ origin
    return M


def check_route_coverage(selected_frames, registered_ids, components, max_gap_seconds=2):
    frames = sorted(selected_frames, key=lambda f:f['t'])
    available = [f for f in frames if f['frame_id'] in registered_ids]
    gaps = []
    if available:
        pairs = [(frames[0],available[0]), *zip(available,available[1:]), (available[-1],frames[-1])]
        for a,b in pairs:
            if b['t']-a['t'] > max_gap_seconds:
                gaps.append({'time_interval':[a['t'],b['t']], 'duration_seconds':b['t']-a['t']})
    fraction = len(available)/len(frames) if frames else 0
    return {'passed':bool(len(available)>=2 and fraction>=.95 and not gaps and len(components)==1),
            'selected_frames':len(frames), 'registered_frames':len(available),
            'registered_fraction':fraction, 'gaps':gaps, 'components':components,
            'max_gap_seconds':max_gap_seconds,
            'missing_frame_ids':[f['frame_id'] for f in frames if f['frame_id'] not in registered_ids]}


def build_route_metadata(frames, camera_poses, scale, route_transform):
    ids = [f['frame_id'] for f in frames]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source frame identity')
    rigs = rig_poses_from_views(camera_poses)
    frames = sorted([f for f in frames if Path(f['name']).stem in rigs], key=lambda f:f['t'])
    if len(frames)<2 or np.any(np.diff([f['t'] for f in frames])<=0):
        raise ValueError('Route needs increasing timestamps and at least two registered frames')
    positions = np.array([rigs[Path(f['name']).stem][:3,3] for f in frames])
    length = float(arc_length(positions)[-1])
    if length < 1e-8:
        raise ValueError('Cannot scale a stationary reconstruction')
    scale = dict(scale)
    if scale['method']=='nominal_speed':
        factor = scale['speed_kmh']/3.6*(frames[-1]['t']-frames[0]['t'])/length
        scale['status'] = 'approximate'
    elif scale['method']=='manual':
        factor = scale['metres_per_sfm_unit']
    elif scale['method']=='unscaled':
        factor = 1.
        scale['status'] = 'unscaled'
    else:
        raise ValueError('Unknown scale method')
    if not np.isfinite(factor) or factor<=0:
        raise ValueError('Invalid global scale')
    M = np.asarray(route_transform, dtype=float)
    if M.shape!=(4,4) or not np.all(np.isfinite(M)) or not np.allclose(M[3],[0,0,0,1]) or not np.allclose(M[:3,:3].T @ M[:3,:3],np.eye(3)) or np.linalg.det(M[:3,:3])<0:
        raise ValueError('Route orientation must be one proper rigid transform')
    scale['metres_per_sfm_unit'] = factor
    scale['basis'] = 'assumed average capture speed; not measured' if scale['method']=='nominal_speed' else scale['method']
    def transform(T):
        out = M @ T
        out[:3,3] *= factor
        return out
    rig_matrices = [transform(rigs[Path(f['name']).stem]) for f in frames]
    distances = arc_length(np.array([T[:3,3] for T in rig_matrices]))
    samples = [{'frame_id':f['frame_id'], 'name':f['name'], 'timestamp_seconds':f['t'],
                'source_pts_seconds':f['source_pts_seconds'], 's':float(s), 'split':f['split'],
                'rig_to_package':T.reshape(-1).tolist()} for f,T,s in zip(frames,rig_matrices,distances)]
    by_stem = {Path(f['name']).stem:f for f in samples}
    cameras = []
    for v in camera_poses:
        stem,yaw = Path(v['name']).stem.rsplit('_y',1)
        if stem not in by_stem:
            continue
        f = by_stem[stem]
        cameras.append({'camera_id':f['frame_id']+f'/y{int(yaw):+04d}', 'frame_id':f['frame_id'],
                        'name':v['name'], 'yaw_degrees':int(yaw), 's':f['s'], 'split':f['split'],
                        'timestamp_seconds':f['timestamp_seconds'], 'source_pts_seconds':f['source_pts_seconds'],
                        'camera_to_package':transform(v['T_wc']).reshape(-1).tolist()})
    sim = M.copy()
    sim[:3,:] *= factor
    return {'schema_version':1, 'kind':'route-reconstruction', 'source_sha256':frames[0]['source_sha256'],
            'coordinates':{'handedness':'right', 'axes':'RUB', 'camera_axes':'OpenCV RDF',
                           'matrix_storage':'row-major; column vectors', 'units':'sfm_units' if scale['method']=='unscaled' else 'nominal_metres',
                           'orientation':'mean camera up and initial travel; gravity/absolute grade unmeasured'},
            'scale':scale, 'sfm_to_package':sim.reshape(-1).tolist(),
            'length':float(distances[-1]), 'samples':samples}, cameras


def run_logged(cmd, log_path):
    print('Running '+str(cmd[1])+'; log: '+str(log_path), flush=True)
    with open(log_path,'wb') as log:
        subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT, check=True)


def reconstruct_route(config):
    p = Project(config['workspace'])
    work = ensure_dir(p.track_dir/'colmap_work')
    reports = ensure_dir(p.root/'reports')
    state = ensure_dir(p.root/'state')
    exe = config['toolchain']['colmap']
    meta = read_json(p.views_meta)
    fingerprint = stage_fingerprint('sfm', {'frames':file_hash(p.frames_meta), 'views':file_hash(p.views_meta),
                                            'mask':file_hash(p.mask_path)}, config['reconstruction'],
                                    {'route_track':1,'colmap':file_hash(exe)})
    identity = state/'sfm_identity.json'
    if identity.exists() and read_json(identity)['fingerprint']!=fingerprint:
        raise ValueError('SfM inputs changed; use a new workspace revision')
    atomic_json(identity, {'fingerprint':fingerprint})
    db = work/'database.db'
    masks = prepare_colmap_masks(p, work/'masks')
    for name,cmd in [('features',build_feature_cmd(exe,db,p.views_dir,masks,meta)),
                     ('matching',build_matcher_cmd(exe,db,config['reconstruction']['overlap']))]:
        done = state/(name+'_done.json')
        if not (done.exists() and db.exists()):
            run_logged(cmd,reports/(name+'.log'))
            atomic_json(done, {'fingerprint':fingerprint})
    mapper_done = state/'mapper_done.json'
    if mapper_done.exists():
        sparse = Path(read_json(mapper_done)['sparse'])
    else:
        attempt = len(list(work.glob('sparse_attempt_*')))+1
        sparse = ensure_dir(work/f'sparse_attempt_{attempt:03d}')
        cmd = build_mapper_cmd(exe,db,p.views_dir,sparse,config['reconstruction']['mapper'])
        if config['reconstruction']['mapper']=='glomap':
            # Intrinsics came from a deterministic panorama projection and must stay fixed.
            cmd += ['--GlobalMapper.ba_refine_focal_length','0', '--GlobalMapper.ba_refine_extra_params','0']
        run_logged(cmd,reports/f'mapper_{attempt:03d}.log')
        atomic_json(mapper_done, {'fingerprint':fingerprint,'sparse':str(sparse)})
    components, models = [], []
    for folder in sorted(sparse.iterdir()):
        if not (folder/'images.bin').exists():
            continue
        txt = ensure_dir(work/('model_txt_'+folder.name))
        run_logged(build_converter_cmd(exe,folder,txt), reports/('convert_'+folder.name+'.log'))
        views = read_view_poses(txt/'images.txt')
        rigs = rig_poses_from_views(views)
        components.append({'id':folder.name, 'registered_views':len(views), 'rig_frames':len(rigs)})
        models.append((len(rigs),txt,views))
    if not models:
        raise RuntimeError('Mapper produced no reconstruction; inspect mapper log')
    _,txt,views = max(models,key=lambda row:row[0])
    frames = read_jsonl(p.frames_meta)
    # Reuse existing optical-centre validation before applying the one accepted transform.
    preliminary,_ = build_route_metadata(frames,views,config['scale'],np.eye(4))
    views,bad = drop_rig_outliers(views, max_dev=.5/preliminary['scale']['metres_per_sfm_unit'])
    M = route_transform(views)
    route,cameras = build_route_metadata(frames,views,config['scale'],M)
    coverage = check_route_coverage(frames,{r['frame_id'] for r in route['samples']},components,
                                    config['reconstruction']['max_gap_seconds'])
    coverage['dropped_inconsistent_views'] = len(bad)
    coverage['registered_views'] = len(views)
    coverage['expected_views'] = len(frames)*len(config['capture']['yaws'])
    route.update({'route_id':config['route_id'],'revision':config['revision'],'coverage':coverage,
                  'status':'ready_for_visual_review' if coverage['passed'] else 'coverage_failed',
                  'training_ready':False, 'training_note':'Step 1 keyframe solve; denser pose registration and visual acceptance precede training'})
    sim = np.array(route['sfm_to_package']).reshape(4,4)
    points = load_points3d_full(txt/'points3D.txt')
    if points is not None:
        points[:,:3] = points[:,:3] @ sim[:3,:3].T + sim[:3,3]
        from scipy.spatial import cKDTree
        xyz = np.array([r['rig_to_package'] for r in route['samples']]).reshape(-1,4,4)[:,:3,3]
        distances,_ = cKDTree(xyz).query(points[:,:3])
        points = points[distances<=60] if config['scale']['method']!='unscaled' else points
    write_colmap_model(p.colmap_dir,meta,[{'name':c['name'],'T_wc':np.array(c['camera_to_package']).reshape(4,4)} for c in cameras],points)
    atomic_json(p.root/'route.json',route)
    write_jsonl(p.root/'cameras.jsonl',cameras)
    atomic_json(reports/'coverage.json',coverage)
    print(f"Route: {len(route['samples'])} camera positions, {route['length']:.1f} nominal metres; coverage={coverage['passed']}",flush=True)
    return p.root/'route.json'
