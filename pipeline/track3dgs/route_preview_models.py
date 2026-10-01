"""Disposable, translation-centred Unity cache; portable masters stay unchanged."""
import argparse
from .route_regions import region_indices
from pathlib import Path
import shutil
import uuid

import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import read_json
from .route_config import atomic_json, file_hash
from .route_training import region_training_fingerprint


def recenter_vertices(vertices,origin):
    result=vertices.copy()
    for axis,name in enumerate(('x','y','z')): result[name]-=origin[axis]
    return result


def write_preview_ply(vertices, destination):
    destination = Path(destination)
    # Unity ignores dot-prefixed files. Publish only once every row is written.
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        PlyData([PlyElement.describe(vertices, 'vertex')], byte_order='<').write(str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


DEFAULT_PREVIEW_FOLDER = 'Assets/Track3DGSData'


def preview_catalog(workspace, unity_project, preview_folder=DEFAULT_PREVIEW_FOLDER):
    root, unity = Path(workspace), Path(unity_project).resolve()
    base = (unity / preview_folder).resolve()
    if not base.is_relative_to(unity / 'Assets') or base == unity / 'Assets':
        raise ValueError('Preview folder must be inside Unity Assets')
    path = base / 'catalog.json'
    route_hash = file_hash(root / 'route.json')
    catalog = read_json(path) if path.exists() else {'schemaVersion': 1, 'entries': []}
    if catalog.get('routeSha256', route_hash) != route_hash:
        raise ValueError('Catalog belongs to a different route revision; choose a fresh viewer/data folder')
    route_preview = root / 'reports/route_preview.json'
    if read_json(route_preview).get('routeSha256') != route_hash:
        raise ValueError('Route marker export is stale; regenerate Step 1 reports')
    catalog.update(routeSha256=route_hash, routeId=read_json(root / 'route.json')['route_id'])
    return base, catalog


def publish_preview(workspace,run_root,unity_project,region,preview_folder=DEFAULT_PREVIEW_FOLDER):
    root=Path(workspace);run=Path(run_root);unity=Path(unity_project)
    base,catalog=preview_catalog(root,unity,preview_folder)
    unity=unity.resolve()
    if read_json(run/'route_review.json')['route_sha256'] != catalog['routeSha256']:
        raise ValueError('Training run was reviewed for a different route revision')
    source=run/'models'/region['region_id']
    processing=read_json(source/'processing.json');model=read_json(source/'model.json')
    if model.get('fingerprint') != region_training_fingerprint(root,region,model['iterations']):
        raise ValueError('Training model fingerprint does not match this route, cameras, views and region')
    if not np.allclose(np.asarray(model['local_to_package']).reshape(4,4),np.eye(4)):
        raise ValueError('Preview adapter expects native package-frame masters')
    route=read_json(root/'route.json');samples=route['samples'];stations=[s['s'] for s in samples]
    xyz=np.asarray([s['rig_to_package'] for s in samples]).reshape(-1,4,4)[:,:3,3]
    middle=sum(region['core_s'])/2
    origin=np.array([np.interp(middle,stations,xyz[:,i]) for i in range(3)])
    output=base/'Models'/region['region_id'];output.mkdir(parents=True,exist_ok=True)
    entry={'index':region['cell_index'],'id':region['region_id'],'coreStart':region['core_s'][0],
        'coreEnd':region['core_s'][1],'x':float(origin[0]),'y':float(origin[1]),'z':float(-origin[2]),
        'rawCount':processing['raw_count'],'cleanCount':processing['clean_count'],'coreCount':processing['core_count'],
        'status':'draft; photographic seam review required','sourceModel':str((source/'model.json').resolve()),
        'coreSourceSha256':file_hash(source/'core.ply')}
    cache_records=[]
    for variant,filename in [('raw','splat.ply'),('clean','clean.ply'),('core','core.ply')]:
        src=source/filename;digest=file_hash(src)
        dst=output/(variant+'.ply');identity=output/(variant+'.cache.json')
        cached=read_json(identity) if identity.exists() else {}
        if not(dst.exists() and cached.get('source_sha256')==digest and cached.get('origin')==origin.tolist() and cached.get('sha256')==file_hash(dst)):
            vertices=PlyData.read(str(src))['vertex'].data
            local=recenter_vertices(vertices,origin)
            write_preview_ply(local,dst)
            positions=np.column_stack([local[n] for n in ('x','y','z')])
            errors=np.linalg.norm(positions.astype(np.float16).astype(np.float32)-positions,axis=1)
            cached={'adapter':'native-rub-translation-cache-v1','source_sha256':digest,'sha256':file_hash(dst),
                'origin':origin.tolist(),'translation_only':True,'SH_unchanged':True,
                'spark_position_error_nominal_m_p95':float(np.percentile(errors,95)),
                'spark_position_error_nominal_m_max':float(errors.max())}
            atomic_json(identity,cached)
        entry[variant+'Path']=dst.relative_to(unity).as_posix();cache_records.append(cached)
    catalog_path=base/'catalog.json'
    old=next((e for e in catalog['entries'] if e['index']==entry['index']),{})
    if (old.get('coreSourceSha256') == entry['coreSourceSha256'] and
            all(old.get(k) == entry[k] for k in ('x','y','z'))):
        entry.update({k:v for k,v in old.items() if k.startswith('sky')})
    else:
        catalog['reviewViews']=[v for v in catalog.get('reviewViews',[]) if v['cellIndex']!=entry['index']]
    catalog['entries']=[e for e in catalog['entries'] if e['index']!=entry['index']]+[entry]
    catalog['entries'].sort(key=lambda e:e['index'])
    catalog['routeSha256']=file_hash(root/'route.json')
    catalog['note']='Draft training preview. Approximate scale. At most two renderer slots; originals and provenance remain in Track3DGS.'
    shutil.copyfile(root/'reports/route_preview.json',base/'route_preview.json')
    atomic_json(catalog_path,catalog)
    atomic_json(source/'unity_cache.json',{'catalog':str(catalog_path),'records':cache_records})
    print('UNITY_CACHE '+region['region_id']+' '+str(catalog_path),flush=True)


def review_view(camera, index, role, field_of_view):
    transform = np.asarray(camera['camera_to_package']).reshape(4, 4)
    reflection = np.array([1., 1., -1.])
    def vector(values):
        return {k: float(v) for k, v in zip(('x', 'y', 'z'), values)}
    return {'label': f'Model {index+1} / {role} / {camera["s"]:.1f} m / yaw {camera["yaw_degrees"]}',
        'cellIndex': index, 'station': camera['s'], 'fieldOfView': field_of_view,
        'position': vector(transform[:3, 3] * reflection), 'forward': vector(transform[:3, 2] * reflection),
        'up': vector(-transform[:3, 1] * reflection)}


def publish_sky(workspace, cleanup, unity_project, index, preview_folder=DEFAULT_PREVIEW_FOLDER):
    root, cleanup, unity = Path(workspace), Path(cleanup), Path(unity_project).resolve()
    base, catalog = preview_catalog(root, unity, preview_folder)
    report = read_json(cleanup / 'processing.json')
    selection = read_json(cleanup / 'selection.json')
    entry = next(e for e in catalog['entries'] if e['index'] == index)
    if report['route_sha256'] != catalog['routeSha256'] or report['region_id'] != entry['id']:
        raise ValueError('Cleanup belongs to a different route or region')
    if report.get('base_core_sha256', report['source_sha256']) != entry['coreSourceSha256']:
        raise ValueError('Sky comparison must be derived from the catalog core (or use the core subset command)')
    if file_hash(cleanup / 'selection.json') != report['selection_sha256']:
        raise ValueError('Cleanup camera selection changed')
    src = cleanup / report.get('core_file', report['file'])
    expected = report.get('core_sha256', report['sha256'])
    if file_hash(src) != expected:
        raise ValueError('Cleanup result changed')
    origin = np.array([entry['x'], entry['y'], -entry['z']])
    vertices = PlyData.read(str(src))['vertex'].data
    dst = base / 'Models' / entry['id'] / 'sky.ply'
    write_preview_ply(recenter_vertices(vertices, origin), dst)
    entry.update(skyPath=dst.relative_to(unity).as_posix(), skyCount=len(vertices),
        skyReport=str((cleanup / 'index.html').resolve()), skySourceSha256=expected)
    intrinsics = selection['intrinsics']
    fov = float(np.degrees(2 * np.arctan(intrinsics['height'] / (2 * intrinsics['fy']))))
    views = [review_view(c, index, role, fov) for role in ('edit', 'validation', 'check') for c in selection.get(role, [])]
    catalog['reviewViews'] = [v for v in catalog.get('reviewViews', []) if v['cellIndex'] != index] + views
    atomic_json(base / 'catalog.json', catalog)
    return entry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--unity-project',required=True)
    p.add_argument('--regions',default=None)
    p.add_argument('--preview-folder',default=DEFAULT_PREVIEW_FOLDER)
    p.add_argument('--sky-cleanup',help='Prepared cleanup result; use exactly one region')
    a=p.parse_args()
    regions=read_json(Path(a.workspace)/'regions.json')['regions']
    indices=region_indices(regions,a.regions)
    if a.sky_cleanup and len(indices)!=1: p.error('--sky-cleanup requires exactly one region')
    for index in indices:
        publish_preview(a.workspace,a.run_root,a.unity_project,regions[index],a.preview_folder)
        if a.sky_cleanup: publish_sky(a.workspace,a.sky_cleanup,a.unity_project,index,a.preview_folder)


if __name__=='__main__': main()
