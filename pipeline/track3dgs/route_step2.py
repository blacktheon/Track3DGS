"""Train, clean, check and publish route chunks serially on one GPU."""
import argparse
from .route_regions import region_indices
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import read_json
from .route_config import atomic_json, file_hash
from .route_training import gpu_lease, run_gpu, run_region_training
from .route_assembly import cleanup_and_partition, processed_files_match
from .route_preview_models import publish_preview


def process_regions(regions,train,process,qc,seam,publish,state_path):
    state={'status':'running','completed':[],'active_region':None,'stage':None}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(state_path,state)
    try:
        for region in regions:
            state['active_region']=region['region_id']
            for stage,action in [('train',train),('clean_and_own',process),('photographic_qc',qc)]:
                state['stage']=stage;save();action(region)
            if region['cell_index']>0:
                state['stage']='seam_qc';save();seam(region['cell_index'])
            state['stage']='unity_cache';save();publish(region)
            state['completed'].append(region['region_id']);save()
        state.update(status='processed_draft',stage=None,active_region=None);save()
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));save()
        raise
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--training-python',required=True)
    p.add_argument('--unity-project',help='Optional Unity viewer path; omit for Python-only processing')
    p.add_argument('--preview-folder',default='Assets/Track3DGSData')
    p.add_argument('--regions',default=None)
    p.add_argument('--iterations',type=int,default=30000);a=p.parse_args()
    root=Path(a.workspace);run=Path(a.run_root);plan=read_json(root/'regions.json')
    review=read_json(run/'route_review.json')
    if review['route_sha256']!=file_hash(root/'route.json'): raise ValueError('Reviewed route hash changed')
    if not read_json(root/'route.json').get('quality',{}).get('passed'): raise ValueError('Route quality gate failed')
    settings={'run_root':str(run),'training_python':a.training_python,'iterations':a.iterations}
    indices=region_indices(plan['regions'],a.regions)
    if indices!=sorted(set(indices)): raise ValueError('Regions must be unique and increasing')
    def process(region):
        folder=run/'models'/region['region_id'];marker=folder/'processing.json'
        if marker.exists():
            report=read_json(marker);model=read_json(folder/'model.json')
            if (report['raw_sha256']==model['raw_sha256'] and
                all(processed_files_match(folder,report,kind) for kind in ['clean','core'])):
                return
        cleanup_and_partition(root,run,region)
    def qc(flag,index,key):
        command=[a.training_python,'-u','-m','track3dgs.route_qc','--workspace',str(root),'--run-root',str(run),flag,str(index)]
        run_gpu(command,run/'reports'/(key+'_qc.log'))
    def seam(index):
        if not(run/'models'/f'cell_{index-1:03d}'/'core.ply').exists():
            raise ValueError('Previous region is required for adjacent seam validation')
        qc('--boundary',index,f'boundary_{index:03d}')
    with gpu_lease(root.parent.parent/'.gpu-training.lock'):
        process_regions([plan['regions'][i] for i in indices],
            lambda r:run_region_training(root,r,settings),process,
            lambda r:qc('--region',r['cell_index'],r['region_id']),seam,
            lambda r:publish_preview(root,run,a.unity_project,r,a.preview_folder) if a.unity_project else None,
            run/'step2_state.json')


if __name__=='__main__': main()
