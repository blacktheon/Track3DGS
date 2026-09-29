"""Publish immutable draft reconstruction packages from processed route models."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from plyfile import PlyData

from .io_utils import read_json, read_jsonl
from .package_contract import file_ref, gaussian_bounds, validate_package
from .route_assembly import PROVENANCE_DTYPE
from .route_config import atomic_json, file_hash


def export_reconstruction(workspace,run_root,output,revision,git_commit=None):
    root=Path(workspace);run=Path(run_root);output=Path(output)
    if output.exists(): raise FileExistsError('Immutable package output already exists')
    plan=read_json(root/'regions.json');route=read_json(root/'route.json')
    selected=[r for r in plan['regions'] if (run/'models'/r['region_id']/'processing.json').exists()]
    if not selected: raise ValueError('No processed models to export')
    staging=output.with_name(output.name+'.staging-'+uuid.uuid4().hex[:8]);staging.mkdir(parents=True)
    family=uuid.uuid5(uuid.NAMESPACE_URL,'track3dgs-route:'+file_hash(root/'route.json'))
    sources=[];assets=[];regions=[];quality=[]
    for source_index,region in enumerate(selected):
        rid=region['region_id'];folder=run/'models'/rid
        model=read_json(folder/'model.json');processing=read_json(folder/'processing.json')
        src=folder/'core.ply'
        if file_hash(src)!=processing['core_sha256']: raise ValueError('Processed model changed: '+rid)
        ply=staging/'regions'/(rid+'.ply');ply.parent.mkdir(exist_ok=True);shutil.copy2(src,ply)
        ids=np.fromfile(folder/'core.provenance.bin',dtype=PROVENANCE_DTYPE)
        if len(ids)!=processing['core_count'] or np.any(ids['source_index']!=region['cell_index']):
            raise ValueError('Unexpected source identity in '+rid)
        ids['source_index']=source_index
        provenance=ply.with_suffix('.provenance.bin');ids.tofile(provenance)
        source_id=str(uuid.uuid5(family,rid+':'+model['raw_sha256']))
        sources.append({'source_id':source_id,'kind':'original','sha256':model['raw_sha256'],'count':model['count'],
            'producer':{'name':'Track3DGS Splatfacto','region_id':rid,'training_fingerprint':model['fingerprint']}})
        vertices=PlyData.read(str(ply))['vertex'].data
        assets.append({'asset_id':rid,'file':file_ref(staging,ply),'codec':'ply-3dgs-f32-le',
            'count':processing['core_count'],'sh_degree':model['sh_degree'],'radiance_encoding':'srgb',
            'local_to_package':model['local_to_package'],'provenance':file_ref(staging,provenance),
            'render_bounds':gaussian_bounds(vertices,model['local_to_package'])})
        # Release plyfile's mapped Windows handle before atomically publishing
        # the staging directory (an open mapping prevents directory renames).
        del vertices
        regions.append({**region,'asset_ids':[rid]});quality.append(processing)
    metadata=staging/'metadata';metadata.mkdir()
    atomic_json(metadata/'regions.json',{'schema_version':1,'regions':regions})
    atomic_json(metadata/'ownership.json',{'schema_version':1,'partition_version':'continuous-polyline-v1',
        'rule':'Nearest continuous polyline segment; half-open core station, final endpoint inclusive',
        'owners':[{'region_id':r['region_id'],'core_s':r['core_s']} for r in selected],
        'spatial_overrides':[],'gaussian_footprints_clipped':False,
        'note':'Centre ownership prevents duplicate region domains; independent model geometry still needs seam review.'})
    route.update(route_id=route.get('route_id',root.parent.name),playable_s=plan['playable_s'],capture_margins=plan['capture_margins'])
    atomic_json(metadata/'route.json',route)
    intrinsics=read_json(root/'views'/'views_meta.json')
    import json
    with (metadata/'cameras.jsonl').open('w',encoding='utf-8') as stream:
        for camera in read_jsonl(root/'cameras.jsonl'):
            camera=dict(camera,intrinsics={k:intrinsics[k] for k in ('width','height','fx','fy','cx','cy')})
            if camera['split']=='train': camera['split']='development'
            stream.write(json.dumps(camera,allow_nan=False)+'\n')
    reports=staging/'reports';reports.mkdir()
    atomic_json(reports/'reconstruction.json',{'schema_version':1,'status':'draft','regions':quality,
        'route_source_sha256':file_hash(root/'route.json'),'observations':'2 Hz globally registered panorama keyframes',
        'training_images_included':False})
    seams=[]
    for r in selected:
        index=r['cell_index']
        if index==0: continue
        name=f'boundary_{index:03d}';qc=run/'reports'/'qc'/name/'report.json'
        seams.append({'boundary_id':name,'station':r['core_s'][0],
            'evidence_available':qc.exists(),'status':'pending_human_visual_review',
            'metrics':read_json(qc)['metrics'] if qc.exists() else None})
    atomic_json(reports/'seams.json',{'schema_version':1,'status':'draft','boundaries':seams})
    evidence=[]
    if (run/'reports'/'qc').exists():
        shutil.copytree(run/'reports'/'qc',reports/'qc')
        evidence=[file_ref(staging,p) for p in sorted((reports/'qc').rglob('*')) if p.is_file()]
    def ref(path): return file_ref(staging,staging/path)
    manifest={'format':'gs-asset-package','schema_version':1,'kind':'reconstruction',
        'package_id':str(family),'revision':revision,'created_utc':datetime.now(timezone.utc).isoformat(),
        'producer':{'name':'Track3DGS','version':'route-step2-v1','git_commit':git_commit},'parents':[],
        'required_capabilities':sorted({'transform.similarity','provenance.u32-u64',*[f'ply.sh{a["sh_degree"]}' for a in assets]}),
        'coordinate_system':{'axes':'RUB','handedness':'right','matrix_layout':'row-major','vectors':'column',
            'unit':'meter','scale_status':route['scale']['status'],'scale_basis':route['scale']['basis'],'level_status':'approximate'},
        'sources':sources,'assets':assets,
        'reconstruction':{'regions':ref('metadata/regions.json'),'ownership':ref('metadata/ownership.json'),
            'quality_report':ref('reports/reconstruction.json'),'route':ref('metadata/route.json'),
            'cameras':ref('metadata/cameras.jsonl'),'seam_report':ref('reports/seams.json')},
        'validation':{'state':'draft','reports':evidence,'limitations':[
            'Human review of every assembled boundary is still required.',
            'Route scale and absolute grade are approximate; original panorama photography is not included.',
            'First pass uses 2 Hz keyframes; sparse or blurred regions may require denser registration and retraining.',
            'No Quest performance claim or Stage1 reduction/Stage2 LOD is included.',
            f'{len(selected)} of {len(plan["regions"])} planned regions are included.']}}
    atomic_json(staging/'manifest.json',manifest)
    validated=validate_package(staging)
    atomic_json(staging/'validation-result.json',validated)
    staging.rename(output)
    return output/'manifest.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--output',required=True);p.add_argument('--revision',required=True)
    a=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    print(export_reconstruction(a.workspace,a.run_root,a.output,a.revision,commit))


if __name__=='__main__': main()
