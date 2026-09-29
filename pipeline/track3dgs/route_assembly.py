"""Non-destructive route cleanup and continuous-polyline core ownership."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import read_json
from .route_config import atomic_json, file_hash
from .skyprune import sky_fractions, sky_colored, splat_rgb, needle_mask


PROVENANCE_DTYPE = np.dtype([('source_index','<u4'),('source_row','<u8')],align=False)


def processed_files_match(folder,report,kind):
    folder=Path(folder);ply=folder/(kind+'.ply');provenance=folder/(kind+'.provenance.bin')
    return (ply.is_file() and provenance.is_file() and
        report.get(kind+'_provenance_bytes')==12*report[kind+'_count']==provenance.stat().st_size and
        report.get(kind+'_sha256')==file_hash(ply) and report.get(kind+'_provenance_sha256')==file_hash(provenance))


def project_to_route(points, route_xyz, route_s, batch_size=2048):
    """Exact nearest segment, with explicit close nonadjacent-branch ambiguity.

    Nearest samples would create stair-step cuts. Zero-length stop segments are
    skipped. Longitudinally distant branches within 20 nominal metres and two
    metres of the best distance must be reviewed instead of silently assigned.
    """
    xyz=np.asarray(route_xyz,dtype=float); stations=np.asarray(route_s,dtype=float)
    points=np.asarray(points,dtype=float)
    delta=np.diff(xyz,axis=0);length2=np.sum(delta*delta,axis=1)
    valid=length2>1e-12
    if not valid.any(): raise ValueError('Route has no nonzero segments')
    a=xyz[:-1][valid];v=delta[valid];length2=length2[valid]
    start=stations[:-1][valid];ds=np.diff(stations)[valid]
    if np.any(ds<=0): raise ValueError('Nonzero route segments require increasing station')
    out_s=np.empty(len(points));out_d=np.empty(len(points));out_p=np.empty_like(points)
    ambiguous=np.zeros(len(points),dtype=bool)
    for offset in range(0,len(points),batch_size):
        p=points[offset:offset+batch_size]
        dot=p@v.T-np.sum(a*v,axis=1)
        u=np.clip(dot/length2,0,1)
        d2=np.maximum(np.sum(p*p,axis=1)[:,None]+np.sum(a*a,axis=1)-2*p@a.T-2*u*dot+u*u*length2,0)
        idx=np.argmin(d2,axis=1);row=np.arange(len(p));frac=u[row,idx]
        best=np.sqrt(d2[row,idx]);station=start[idx]+frac*ds[idx]
        projected=start[None,:]+u*ds[None,:]
        other=(np.abs(projected-station[:,None])>60)&(d2<20**2)&(d2<=(best[:,None]+2)**2)
        sl=slice(offset,offset+len(p))
        out_s[sl]=station;out_d[sl]=best;out_p[sl]=a[idx]+frac[:,None]*v[idx]
        ambiguous[sl]=other.any(axis=1)
    return out_s,out_d,out_p,ambiguous


def core_mask(stations,region):
    lo,hi=region['core_s'];s=np.asarray(stations)
    return (s>=lo)&((s<=hi) if region['core_end_inclusive'] else (s<hi))


def write_subset(vertices,keep,source_rows,source_index,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    chosen=np.asarray(vertices[keep])
    PlyData([PlyElement.describe(chosen,'vertex')],text=False,byte_order='<').write(str(path))
    ids=np.empty(len(chosen),dtype=PROVENANCE_DTYPE)
    ids['source_index']=source_index;ids['source_row']=np.asarray(source_rows,dtype=np.uint64)[keep]
    ids.tofile(path.with_suffix('.provenance.bin'))
    return len(chosen)


def cleanup_and_partition(workspace,run_root,region):
    root=Path(workspace);out=Path(run_root)/'models'/region['region_id']
    model=read_json(out/'model.json');raw=Path(model['raw_file'])
    if file_hash(raw)!=model['raw_sha256']: raise ValueError('Raw model hash changed')
    v=PlyData.read(str(raw))['vertex'].data
    xyz=np.column_stack([v[k] for k in ('x','y','z')]).astype(float)
    T=np.asarray(model['local_to_package']).reshape(4,4)
    xyz=xyz@T[:3,:3].T+T[:3,3]
    route=read_json(root/'route.json');samples=route['samples']
    route_xyz=np.asarray([s['rig_to_package'] for s in samples]).reshape(-1,4,4)[:,:3,3]
    stations,distance,nearest,ambiguous=project_to_route(xyz,route_xyz,[s['s'] for s in samples])
    eligible=core_mask(stations,region)
    if (ambiguous&eligible).any():
        atomic_json(out/'ownership_ambiguity.json',{'ambiguous_rows':int((ambiguous&eligible).sum()),
            'status':'manual spatial ownership review required'})
        raise ValueError('Nearby nonadjacent route branches need explicit ownership review')
    frames=[];masks={}
    for sample in samples:
        if sample['split']!='train' or not region['context_s'][0]<=sample['s']<=region['context_s'][1]: continue
        stem=Path(sample['name']).stem
        mask=cv2.imread(str(root/'sky_masks'/(stem+'.png')),cv2.IMREAD_GRAYSCALE)
        if mask is None: raise ValueError('Missing sky mask '+stem)
        masks[stem]=cv2.resize(mask,(960,480),interpolation=cv2.INTER_NEAREST)
        frames.append({'name':sample['name'],'T_wc':sample['rig_to_package']})
    frac=sky_fractions(xyz,frames,masks,max_range=50)
    sky=frac>=.6
    glitter=(frac>=.12)&sky_colored(splat_rgb(v))&(xyz[:,1]-nearest[:,1]>2)&~sky
    uniform_scale=np.linalg.norm(T[:3,0])
    log_scales=np.column_stack([v[f'scale_{i}'] for i in range(3)])+np.log(uniform_scale)
    needles=needle_mask(log_scales)&~sky&~glitter
    keep=~(sky|glitter|needles)
    rows=np.arange(len(v),dtype=np.uint64)
    nclean=write_subset(v,keep,rows,region['cell_index'],out/'clean.ply')
    ncore=write_subset(v,keep&eligible,rows,region['cell_index'],out/'core.ply')
    result={'schema_version':1,'region_id':region['region_id'],'raw_count':len(v),
        'raw_sha256':model['raw_sha256'],'clean_count':nclean,'core_count':ncore,
        'sky_removed':int(sky.sum()),'glitter_removed':int(glitter.sum()),'needle_removed':int(needles.sum()),
        'context_removed_after_cleanup':int((keep&~eligible).sum()),'ambiguous_core_rows':0,
        'core_s':region['core_s'],'core_end_inclusive':region['core_end_inclusive'],
        'ownership':'closest continuous polyline segment, half-open cores, final endpoint inclusive',
        'local_to_package':model['local_to_package'],
        'cleanup':{'sky_fraction':.6,'sky_range_nominal_m':50,'mask_width':960,'train_frames_only':True,
            'color_fraction':.12,'color_height_above_local_route':2,'needle_max_len':.5,'needle_ratio':8,'hard_max':5},
        'clean_sha256':file_hash(out/'clean.ply'),'core_sha256':file_hash(out/'core.ply'),
        'clean_provenance_sha256':file_hash(out/'clean.provenance.bin'),'clean_provenance_bytes':12*nclean,
        'core_provenance_sha256':file_hash(out/'core.provenance.bin'),'core_provenance_bytes':12*ncore,
        'seam_status':'pending render review; unique centre ownership does not establish a gap-free seam',
        'status':'draft'}
    atomic_json(out/'processing.json',result)
    print(f'PROCESSED {region["region_id"]}: {len(v):,} raw -> {nclean:,} clean -> {ncore:,} core',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--regions',default='0,1,2,3,4,5')
    a=p.parse_args();plan=read_json(Path(a.workspace)/'regions.json')
    for index in map(int,a.regions.split(',')): cleanup_and_partition(a.workspace,a.run_root,plan['regions'][index])


if __name__=='__main__': main()
