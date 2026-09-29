"""Calibrated zero-baseline panorama rig for the existing pinhole projections.

COLMAP 4.1 rig workflow: https://colmap.github.io/rigs.html
Keep the existing projection, mask and SIFT algorithms; constrain the virtual
cameras to one real exposure instead of optimizing eight unrelated centres.
"""
import argparse
import math
import shutil
import sqlite3
from pathlib import Path

from .io_utils import ensure_dir, read_json
from .route_config import atomic_json, load_route_config


def rig_image_name(name):
    frame,yaw = Path(name).stem.rsplit('_y',1)
    return f'yaw_{int(yaw):+04d}/{frame}.jpg'


def rig_configuration(yaws, intr):
    if 0 not in yaws or len(yaws)!=len(set(yaws)):
        raise ValueError('Panorama rig requires one reference yaw 0 and unique sensors')
    cameras = []
    for yaw in yaws:
        cam = {'image_prefix':f'yaw_{yaw:+04d}/','camera_model_name':'PINHOLE',
               'camera_params':[intr[k] for k in ('fx','fy','cx','cy')]}
        if yaw==0:
            cam['ref_sensor'] = True
        else:
            angle = math.radians(-yaw)/2
            cam.update({'cam_from_rig_rotation':[math.cos(angle),0,math.sin(angle),0],
                        'cam_from_rig_translation':[0,0,0]})
        cameras.append(cam)
    # COLMAP's Rig requires its reference sensor to be added before other sensors.
    cameras.sort(key=lambda c:not c.get('ref_sensor',False))
    return [{'cameras':cameras}]


def rename_feature_database(database, yaws):
    """Operate only on a private database copy. Keypoint/image IDs stay intact."""
    aliases = {}
    with sqlite3.connect(database) as c:
        cameras = c.execute('SELECT * FROM cameras ORDER BY camera_id').fetchall()
        if len(cameras)!=1:
            raise ValueError('Expected the shared-intrinsic flat observation cache')
        base = cameras[0]
        images = c.execute('SELECT image_id,name FROM images').fetchall()
        if any('/' in name or '\\' in name for _,name in images):
            raise ValueError('Source database already uses a different image layout')
        ids = {}
        for index,yaw in enumerate(yaws):
            if index==0:
                ids[yaw] = base[0]
            else:
                cur = c.execute('INSERT INTO cameras(model,width,height,params,prior_focal_length) VALUES(?,?,?,?,?)',base[1:])
                ids[yaw] = cur.lastrowid
        for image_id,name in images:
            yaw = int(Path(name).stem.rsplit('_y',1)[1])
            new = rig_image_name(name)
            c.execute('UPDATE images SET name=?,camera_id=? WHERE image_id=?',(new,ids[yaw],image_id))
            aliases[new] = name
        # These geometries assumed independent camera centres. Reverify after
        # rig_configurator establishes the correct simultaneous frame membership.
        c.execute('DELETE FROM matches')
        c.execute('DELETE FROM two_view_geometries')
    return aliases


def prepare_rig_layout(project, work, config, fingerprint):
    from .route_track import build_feature_cmd, run_logged
    p = project
    state = ensure_dir(p.root/'state')
    reports = ensure_dir(p.root/'reports')
    layout = ensure_dir(work/'rig_images')
    masks = ensure_dir(work/'rig_masks')
    intr = read_json(p.views_meta)
    aliases = {}
    for source in sorted(p.views_dir.glob('*.jpg')):
        name = rig_image_name(source.name)
        aliases[name] = source.name
        target = layout/name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            # Independent copies protect earlier revisions from future regeneration.
            shutil.copy2(source,target)
        target_mask = masks/(name+'.png')
        target_mask.parent.mkdir(exist_ok=True)
        if not target_mask.exists():
            shutil.copy2(p.views_masks_dir/source.with_suffix('.png').name,target_mask)
    atomic_json(work/'rig_image_aliases.json',aliases)
    db = work/'database.db'
    ready = state/'rig_ready.json'
    if ready.exists():
        if read_json(ready)['fingerprint']!=fingerprint:
            raise ValueError('Rig settings changed; select a new revision')
        return layout,aliases
    cache = config['reconstruction'].get('reuse_feature_database')
    if cache:
        cache = Path(cache).resolve()
        if cache==db.resolve():
            raise ValueError('Feature source must not be the destination database')
        source_config = read_json(cache.parents[2]/'route_config.resolved.json')
        current_config = read_json(p.root/'route_config.resolved.json')
        if source_config['input_hashes']!=current_config['input_hashes'] or source_config['capture']!=config['capture']:
            raise ValueError('Feature cache inputs do not match this route revision')
        if not db.exists():
            with sqlite3.connect(cache.as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(db) as dst:
                src.backup(dst)
            renamed = rename_feature_database(db,config['capture']['yaws'])
        else:
            # A configurator failure can occur after the transactional layout
            # conversion. Validate that complete layout before retrying its step.
            with sqlite3.connect(db) as c:
                rows = c.execute('SELECT name,camera_id FROM images').fetchall()
                groups = {}
                for name,cid in rows:
                    groups.setdefault(name.split('/')[0],set()).add(cid)
            if (set(name for name,_ in rows)!=set(aliases)
                    or any(len(ids)!=1 for ids in groups.values())
                    or len(set(cid for _,cid in rows))!=len(config['capture']['yaws'])):
                raise ValueError('Partial rig database layout; select a new revision')
            renamed = {name:aliases[name] for name,_ in rows}
        if renamed!=aliases:
            raise ValueError('Feature database image membership differs from selected observations')
    else:
        cmd = build_feature_cmd(config['toolchain']['colmap'],db,layout,masks,intr)
        index = cmd.index('--ImageReader.single_camera')
        cmd[index:index+2] = ['--ImageReader.single_camera_per_folder','1']
        run_logged(cmd,reports/'features.log')
    rig_path = work/'rig_config.json'
    atomic_json(rig_path,rig_configuration(config['capture']['yaws'],intr))
    run_logged([config['toolchain']['colmap'],'rig_configurator','--database_path',db,
                '--rig_config_path',rig_path],reports/'rig_configuration.log')
    with sqlite3.connect(db) as c:
        counts = {'rigs':c.execute('SELECT COUNT(*) FROM rigs').fetchone()[0],
                  'frames':c.execute('SELECT COUNT(*) FROM frames').fetchone()[0],
                  'cameras':c.execute('SELECT COUNT(*) FROM cameras').fetchone()[0],
                  'images':c.execute('SELECT COUNT(*) FROM images').fetchone()[0]}
    if counts['rigs']!=1 or counts['frames']*len(config['capture']['yaws'])!=counts['images']:
        raise ValueError(f'Rig grouping failed: {counts}')
    atomic_json(ready,{'fingerprint':fingerprint,**counts})
    atomic_json(state/'features_done.json',{'fingerprint':fingerprint})
    return layout,aliases


def fork_observations(source_config_path, target_config_path, revision):
    """Create a new immutable reconstruction experiment without decoding again."""
    config = load_route_config(source_config_path)
    source = Path(config['workspace'])
    target = source.parent/revision
    if target.exists():
        raise ValueError('Target revision already exists')
    if not (source/'state'/'views_done.json').exists():
        raise ValueError('Source observations are incomplete')
    target.mkdir(parents=True)
    for directory in ('frames','sky_masks','views','views_masks'):
        shutil.copytree(source/directory,target/directory)
    for filename in ('frames_meta.jsonl','mask_equirect.png'):
        shutil.copy2(source/filename,target/filename)
    state = ensure_dir(target/'state')
    for filename in ('ingest_identity.json','ingest_done.json','timestamps.json','sharpness.json',
                     'sky_done.json','views_identity.json','views_done.json'):
        shutil.copy2(source/'state'/filename,state/filename)
    config['workspace'],config['revision'] = str(target),revision
    config['reconstruction'].update({'panorama_rig':True,'overlap':6,
        'reuse_feature_database':str(source/'track'/'colmap_work'/'database.db')})
    atomic_json(target_config_path,config)
    return config


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-config',required=True)
    ap.add_argument('--target-config',required=True)
    ap.add_argument('--revision',required=True)
    args = ap.parse_args()
    fork_observations(args.source_config,args.target_config,args.revision)
