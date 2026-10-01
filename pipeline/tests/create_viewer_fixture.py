"""Generate tiny, synthetic data for the standalone Unity integration check.

Run from pipeline: python tests/create_viewer_fixture.py --unity-project ../unity/Track3DGSViewer
Only use an empty viewer data folder. No video, learned model or QuestSBTC needed.
"""
import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track3dgs.route_config import atomic_json, file_hash
from track3dgs.route_preview_models import publish_preview, publish_sky, write_preview_ply


def create_fixture(unity):
    unity = Path(unity).resolve()
    if (unity/'Assets/Track3DGSData').exists():
        raise FileExistsError('Use a fresh viewer copy; refusing to replace existing preview data')
    with tempfile.TemporaryDirectory(prefix='track3dgs-fixture-') as temporary:
        root = Path(temporary); run = root/'training'; pose = np.diag([1., -1., -1., 1.])
        samples, markers = [], []
        for i in range(3):
            p = pose.copy(); p[0, 3] = i*10
            samples.append({'s':i*10, 'rig_to_package':p.ravel().tolist()})
            markers.append(dict(frameId=f'fixture{i}', time=i, distance=i*10, x=i*10, y=0, z=0,
                fx=1, fy=0, fz=0, ux=0, uy=1, uz=0, region=min(i,1)))
        atomic_json(root/'route.json', {'route_id':'synthetic-smoke', 'samples':samples})
        region_preview = [dict(index=i,name=f'cell_{i:03d}',coreStart=i*10,coreEnd=(i+1)*10,
            contextStart=0,contextEnd=20,startTime=0,endTime=2) for i in range(2)]
        atomic_json(root/'reports/route_preview.json', dict(schemaVersion=1,routeId='synthetic-smoke',
            status='synthetic',scaleNote='Synthetic fixture, not a reconstructed capture',qualityNote='Test data',
            routeSha256=file_hash(root/'route.json'),length=20,maxGapSeconds=2,markers=markers,regions=region_preview))
        fields=['x','y','z','opacity','rot_0','rot_1','rot_2','rot_3','scale_0','scale_1','scale_2']
        fields += [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]
        for index in range(2):
            folder=run/f'models/cell_{index:03d}';folder.mkdir(parents=True)
            vertices=np.zeros(4,dtype=[(n,'<f4') for n in fields])
            vertices['x']=[-.3,.3,-.2,.2];vertices['z']=-np.arange(2,6)-index*.25
            vertices['rot_0']=1;vertices['f_dc_1']=.8
            for name in ('scale_0','scale_1','scale_2'): vertices[name]=np.log(.35)
            for name in ('splat','clean','core'): write_preview_ply(vertices,folder/(name+'.ply'))
            atomic_json(folder/'model.json',{'local_to_package':np.eye(4).ravel().tolist()})
            atomic_json(folder/'processing.json',{'raw_count':4,'clean_count':4,'core_count':4})
            publish_preview(root,run,unity,dict(region_id=f'cell_{index:03d}',cell_index=index,core_s=[index*10,(index+1)*10]))
            cleanup=root/f'cleanup{index}';(cleanup/'models').mkdir(parents=True)
            write_preview_ply(vertices[:3],cleanup/'models/sky.ply')
            camera={'camera_to_package':pose.ravel().tolist(),'s':index*10+5,'yaw_degrees':0}
            atomic_json(cleanup/'selection.json',{'edit':[camera], 'intrinsics':{'height':1600,'fy':671.279704941824}})
            atomic_json(cleanup/'processing.json',{'route_sha256':file_hash(root/'route.json'),'region_id':f'cell_{index:03d}',
                'source_sha256':file_hash(folder/'core.ply'),'file':'models/sky.ply','sha256':file_hash(cleanup/'models/sky.ply'),
                'selection_sha256':file_hash(cleanup/'selection.json')})
            publish_sky(root,cleanup,unity,index)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unity-project',required=True)
    create_fixture(parser.parse_args().unity_project)
