"""Opt-in continuous-route Step 1. Does not start Gaussian training."""
import argparse
import json
from pathlib import Path

from .io_utils import Project, read_json, read_jsonl
from .route_config import load_route_config, atomic_json, stage_fingerprint, file_hash
from .route_ingest import ingest_route
from .route_track import reconstruct_route, run_logged
from .route_regions import plan_regions, write_region_subsets


def prepare_views(config):
    p = Project(config['workspace'])
    reports, state = p.root/'reports', p.root/'state'
    reports.mkdir(exist_ok=True)
    fingerprint = stage_fingerprint('views', {'frames':file_hash(p.frames_meta), 'mask':file_hash(p.mask_path),
        'sky_prior':file_hash(config['sky_prior']) if config['sky_prior'] else None}, config['capture'], {'views':1,'sky_model':'union'})
    identity = state/'views_identity.json'
    if identity.exists() and read_json(identity)['fingerprint']!=fingerprint:
        raise ValueError('View inputs changed; use a new revision')
    atomic_json(identity, {'fingerprint':fingerprint})
    frames = read_jsonl(p.frames_meta)
    sky_dir = p.root/'sky_masks'
    sky_done = state/'sky_done.json'
    if not (sky_done.exists() and all((sky_dir/Path(r['name']).with_suffix('.png')).exists() for r in frames)):
        if config['sky_prior'] is None:
            raise ValueError('This union-mask route profile requires an explicit sky prior')
        python = config['toolchain']['training_python']
        run_logged([python,'-u','-m','track3dgs.skymask','--project',p.root,
                    '--model','union','--prior',config['sky_prior']],reports/'sky_masks.log')
        atomic_json(sky_done, {'fingerprint':fingerprint})
    expected = len(frames)*len(config['capture']['yaws'])
    done = state/'views_done.json'
    if not (done.exists() and p.views_meta.exists() and len(list(p.views_dir.glob('*.jpg')))==expected
            and len(list(p.views_masks_dir.glob('*.png')))==expected):
        from .views import run_views
        print(f'Projecting {len(frames)} panoramas into {expected} pinhole views',flush=True)
        run_views(p.root,config['capture']['yaws'],config['capture']['fov_degrees'],config['capture']['view_size'])
        atomic_json(done, {'fingerprint':fingerprint,'view_count':expected})


def run_route(config_path, through='regions', resume=True, dry_run=False):
    config = load_route_config(config_path)
    if through not in ('ingest','views','reconstruct','regions'):
        raise ValueError('Step 1 only: through must be ingest, views, reconstruct or regions')
    if dry_run:
        return {'config':config,'through':through,'training_started':False}
    if not resume and Path(config['workspace']).exists():
        raise ValueError('Choose a new revision for a fresh run; destructive reset is unsupported')
    workspace = ingest_route(config)
    if through=='ingest':
        return {'workspace':str(workspace),'through':through}
    prepare_views(config)
    if through=='views':
        return {'workspace':str(workspace),'through':through}
    reconstruct_route(config)
    route = read_json(workspace/'route.json')
    cameras = read_jsonl(workspace/'cameras.jsonl')
    plan = None
    if through=='regions' and route['coverage']['passed']:
        plan = plan_regions(route,cameras,config['regions'])
        write_region_subsets(workspace,plan)
    from .route_report import write_reports
    write_reports(workspace,route,plan)
    if not route['coverage']['passed']:
        raise RuntimeError('Coverage failed. Diagnostic visualizations written; training plan withheld.')
    return {'workspace':str(workspace),'through':through,'status':route['status'],
            'regions':len(plan['regions']) if plan else 0,'training_started':False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--through',choices=['ingest','views','reconstruct','regions'],default='regions')
    ap.add_argument('--dry-run',action='store_true')
    args = ap.parse_args()
    print(json.dumps(run_route(args.config,args.through,dry_run=args.dry_run),indent=2),flush=True)


if __name__=='__main__':
    main()
