"""Portable reconstruction-package validation, independent of Unity or a route."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import uuid

import numpy as np
from plyfile import PlyData

from .route_config import file_hash
from .route_assembly import PROVENANCE_DTYPE


CAPABILITIES={'ply.sh0','ply.sh1','ply.sh2','ply.sh3','transform.similarity','provenance.u32-u64'}


def load_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    def invalid(value): raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,parse_constant=invalid)


def file_ref(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    return {'path':path.relative_to(root).as_posix(),'sha256':file_hash(path),'bytes':path.stat().st_size}


def validate_similarity(values):
    T=np.asarray(values,dtype=float)
    if T.size!=16 or not np.isfinite(T).all(): raise ValueError('Invalid transform')
    T=T.reshape(4,4);A=T[:3,:3];scale2=float(np.sum(A*A)/3)
    if (not np.allclose(T[3],[0,0,0,1],atol=1e-8) or scale2<=0 or np.linalg.det(A)<=0 or
        not np.allclose(A.T@A,np.eye(3)*scale2,atol=1e-7*scale2,rtol=1e-5)):
        raise ValueError('Transform must be a positive uniform similarity')
    return T


def validate_ply(path,count,sh_degree):
    if type(count)!=int or not 0<=count<=2147483647: raise ValueError('Unsupported PLY count (limit 2147483647)')
    if sh_degree not in (0,1,2,3): raise ValueError('Unsupported SH degree')
    try: ply=PlyData.read(str(path))
    except Exception as error: raise ValueError('Invalid PLY') from error
    if ply.text or ply.byte_order!='<' or len(ply.elements)!=1 or ply.elements[0].name!='vertex':
        raise ValueError('Expected one binary little-endian vertex element')
    v=ply['vertex'].data;names=v.dtype.names
    if len(v)!=count or any(v.dtype[n].kind!='f' or v.dtype[n].itemsize!=4 for n in names):
        raise ValueError('PLY count or scalar float32 fields invalid')
    required=['x','y','z','opacity',*[f'rot_{i}' for i in range(4)],*[f'scale_{i}' for i in range(3)],*[f'f_dc_{i}' for i in range(3)]]
    rest=[f'f_rest_{i}' for i in range(3*((sh_degree+1)**2-1))]
    if not set(required+rest)<=set(names) or set(n for n in names if n.startswith('f_rest_'))!=set(rest):
        raise ValueError('PLY fields disagree with declared SH degree')
    if any(not np.isfinite(v[n]).all() for n in names): raise ValueError('Nonfinite Gaussian attributes')
    if np.any(sum(v[f'rot_{i}'].astype(float)**2 for i in range(4))<=1e-16): raise ValueError('Zero quaternion')
    with np.errstate(over='ignore',under='ignore'):
        for i in range(3):
            scale=np.exp(v[f'scale_{i}'].astype(float))
            if np.any(~np.isfinite(scale)|(scale<=0)): raise ValueError('Invalid decoded scale')
    with open(path,'rb') as f:
        while f.tell()<65536:
            line=f.readline()
            if line.strip()==b'end_header': break
            if not line: raise ValueError('Missing PLY header terminator')
        else: raise ValueError('PLY header too long')
        if Path(path).stat().st_size!=f.tell()+v.dtype.itemsize*count: raise ValueError('PLY has truncated or trailing rows')
    return v


def gaussian_bounds(vertices,transform,sigma=4.):
    T=validate_similarity(transform);v=vertices
    if not len(v): return {'min':[0.,0.,0.],'max':[0.,0.,0.],'support_sigma':sigma}
    lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
    for offset in range(0,len(v),65536):
        batch=v[offset:offset+65536]
        xyz=np.column_stack([batch[n] for n in ('x','y','z')]).astype(float)@T[:3,:3].T+T[:3,3]
        q=np.column_stack([batch[f'rot_{i}'] for i in range(4)]).astype(float)
        q/=np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
        A=T[:3,:3][None]@R
        scales=np.exp(np.column_stack([batch[f'scale_{i}'] for i in range(3)]).astype(float))
        extent=sigma*np.sqrt(np.sum((A*scales[:,None,:])**2,axis=2))
        lo=np.minimum(lo,np.min(xyz-extent,axis=0));hi=np.maximum(hi,np.max(xyz+extent,axis=0))
    return {'min':lo.tolist(),'max':hi.tolist(),'support_sigma':sigma}


def validate_package(root):
    root=Path(root).resolve();m=load_json(root/'manifest.json')
    required={'format','schema_version','kind','package_id','revision','created_utc','producer','parents',
              'required_capabilities','coordinate_system','sources','assets','validation','reconstruction'}
    if not required<=m.keys(): raise ValueError('Missing manifest fields')
    if m['format']!='gs-asset-package' or m['schema_version']!=1 or m['kind']!='reconstruction':
        raise ValueError('This validator supports reconstruction packages v1')
    uuid.UUID(m['package_id'])
    if not m['revision'] or set(m['required_capabilities'])-CAPABILITIES: raise ValueError('Unknown required capability or empty revision')
    c=m['coordinate_system']
    if any(c.get(k)!=v for k,v in {'axes':'RUB','handedness':'right','matrix_layout':'row-major','vectors':'column'}.items()):
        raise ValueError('Unsupported coordinate declaration')
    if c.get('scale_status') not in ('calibrated','approximate','unscaled') or c.get('unit')!=('unit' if c.get('scale_status')=='unscaled' else 'meter'):
        raise ValueError('Scale declaration invalid')
    if c.get('level_status') not in ('calibrated','approximate','unknown') or not c.get('scale_basis'): raise ValueError('Missing calibration basis')
    if m['validation']['state'] not in ('draft','accepted'): raise ValueError('Unknown validation state')
    seen_paths={}
    def check(ref):
        name=ref['path'];parts=PurePosixPath(name).parts
        if not name or '\\' in name or ':' in name or name.startswith('/') or '..' in parts:
            raise ValueError('Unsafe package path')
        key=name.casefold()
        if key in seen_paths and seen_paths[key]!=ref: raise ValueError('Case-insensitive path collision')
        seen_paths[key]=ref
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Missing file or path escaping package')
        if not re.fullmatch('[0-9a-f]{64}',ref['sha256']) or path.stat().st_size!=ref['bytes'] or file_hash(path)!=ref['sha256']:
            raise ValueError('File hash or length mismatch: '+name)
        return path
    sources=m['sources'];source_ids=[s['source_id'] for s in sources]
    if len(set(source_ids))!=len(source_ids): raise ValueError('Source ID collision')
    for source in sources:
        if source['kind'] not in ('original','derived') or type(source['count'])!=int or source['count']<0 or not re.fullmatch('[0-9a-f]{64}',source['sha256']):
            raise ValueError('Invalid source registry')
    assets=set();total=0;identities={}
    for asset in m['assets']:
        if asset['asset_id'] in assets: raise ValueError('Asset ID collision')
        assets.add(asset['asset_id'])
        if asset['codec']!='ply-3dgs-f32-le' or asset['radiance_encoding'] not in ('srgb','linear'): raise ValueError('Unsupported PLY appearance')
        v=validate_ply(check(asset['file']),asset['count'],asset['sh_degree']);T=validate_similarity(asset['local_to_package'])
        for cap in (f'ply.sh{asset["sh_degree"]}','transform.similarity','provenance.u32-u64'):
            if cap not in m['required_capabilities']: raise ValueError('Required asset capability missing')
        path=check(asset['provenance'])
        if path.stat().st_size!=12*len(v): raise ValueError('Provenance length mismatch')
        ids=np.fromfile(path,dtype=PROVENANCE_DTYPE)
        if len(ids) and (ids['source_index'].max()>=len(sources)): raise ValueError('Unknown provenance source')
        for index in np.unique(ids['source_index']):
            rows=ids['source_row'][ids['source_index']==index]
            if len(rows) and rows.max()>=sources[index]['count']: raise ValueError('Source row out of range')
            if len(np.unique(rows))!=len(rows): raise ValueError('Duplicate row identity')
            old=identities.get(int(index),np.array([],dtype=np.uint64))
            if len(np.intersect1d(old,rows,assume_unique=True)): raise ValueError('Duplicate identity across assets')
            identities[int(index)]=np.concatenate([old,rows])
        bounds=asset['render_bounds'];lo=np.asarray(bounds['min']);hi=np.asarray(bounds['max'])
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi) or bounds['support_sigma']!=4.:
            raise ValueError('Invalid render bounds')
        expected=gaussian_bounds(v,T,4.)
        if np.any(lo>np.array(expected['min'])+1e-4) or np.any(hi<np.array(expected['max'])-1e-4): raise ValueError('Bounds omit Gaussian support')
        total+=len(v)
    for key in ('regions','ownership','quality_report'):
        if key not in m['reconstruction']: raise ValueError('Missing reconstruction metadata')
    for ref in m['reconstruction'].values(): check(ref)
    for ref in m['validation']['reports']: check(ref)
    return {'valid':True,'asset_count':len(assets),'splat_count':total,'state':m['validation']['state']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('package');a=parser.parse_args()
    print(json.dumps(validate_package(a.package),indent=2))


if __name__=='__main__': main()
