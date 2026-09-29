"""Disposable, translation-centred Unity cache; portable masters stay unchanged."""
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import read_json
from .route_config import atomic_json, file_hash


def recenter_vertices(vertices,origin):
    result=vertices.copy()
    for axis,name in enumerate(('x','y','z')): result[name]-=origin[axis]
    return result


def publish_preview(workspace,run_root,unity_project,region):
    root=Path(workspace);run=Path(run_root);unity=Path(unity_project)
    source=run/'models'/region['region_id']
    processing=read_json(source/'processing.json');model=read_json(source/'model.json')
    if not np.allclose(np.asarray(model['local_to_package']).reshape(4,4),np.eye(4)):
        raise ValueError('Preview adapter expects native package-frame masters')
    route=read_json(root/'route.json');samples=route['samples'];stations=[s['s'] for s in samples]
    xyz=np.asarray([s['rig_to_package'] for s in samples]).reshape(-1,4,4)[:,:3,3]
    middle=sum(region['core_s'])/2
    origin=np.array([np.interp(middle,stations,xyz[:,i]) for i in range(3)])
    base=unity/'Assets'/'TrackTrainingPreview'/'Track02'
    output=base/'Models'/region['region_id'];output.mkdir(parents=True,exist_ok=True)
    entry={'index':region['cell_index'],'id':region['region_id'],'coreStart':region['core_s'][0],
        'coreEnd':region['core_s'][1],'x':float(origin[0]),'y':float(origin[1]),'z':float(-origin[2]),
        'rawCount':processing['raw_count'],'cleanCount':processing['clean_count'],'coreCount':processing['core_count'],
        'status':'draft; photographic seam review required','sourceModel':str(source/'model.json')}
    cache_records=[]
    for variant,filename in [('raw','splat.ply'),('clean','clean.ply'),('core','core.ply')]:
        src=source/filename;digest=file_hash(src)
        dst=output/(variant+'.ply');identity=output/(variant+'.cache.json')
        cached=read_json(identity) if identity.exists() else {}
        if not(dst.exists() and cached.get('source_sha256')==digest and cached.get('origin')==origin.tolist() and cached.get('sha256')==file_hash(dst)):
            vertices=PlyData.read(str(src))['vertex'].data
            local=recenter_vertices(vertices,origin)
            PlyData([PlyElement.describe(local,'vertex')],byte_order='<').write(str(dst))
            positions=np.column_stack([local[n] for n in ('x','y','z')])
            errors=np.linalg.norm(positions.astype(np.float16).astype(np.float32)-positions,axis=1)
            cached={'adapter':'native-rub-translation-cache-v1','source_sha256':digest,'sha256':file_hash(dst),
                'origin':origin.tolist(),'translation_only':True,'SH_unchanged':True,
                'spark_position_error_nominal_m_p95':float(np.percentile(errors,95)),
                'spark_position_error_nominal_m_max':float(errors.max())}
            atomic_json(identity,cached)
        entry[variant+'Path']=dst.relative_to(unity).as_posix();cache_records.append(cached)
    catalog_path=base/'catalog.json'
    catalog=read_json(catalog_path) if catalog_path.exists() else {'schemaVersion':1,'routeId':'track02','entries':[]}
    catalog['entries']=[e for e in catalog['entries'] if e['index']!=entry['index']]+[entry]
    catalog['entries'].sort(key=lambda e:e['index'])
    catalog['routeSha256']=file_hash(root/'route.json')
    catalog['note']='Draft training preview. Approximate scale. At most two renderer slots; originals and provenance remain in Track3DGS.'
    atomic_json(catalog_path,catalog)
    atomic_json(source/'unity_cache.json',{'catalog':str(catalog_path),'records':cache_records})
    print('UNITY_CACHE '+region['region_id']+' '+str(catalog_path),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--unity-project',required=True)
    p.add_argument('--regions',default='0,1,2,3,4,5');a=p.parse_args()
    regions=read_json(Path(a.workspace)/'regions.json')['regions']
    for index in map(int,a.regions.split(',')): publish_preview(a.workspace,a.run_root,a.unity_project,regions[index])


if __name__=='__main__': main()
