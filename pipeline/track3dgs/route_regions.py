"""Training core/context planning in the unchanged global route coordinate frame."""
import csv
from pathlib import Path

import numpy as np

from .io_utils import read_json, read_jsonl, ensure_dir
from .colmap_export import load_points3d_txt, write_colmap_model
from .route_config import atomic_json


def plan_regions(route, cameras, settings):
    if not route['coverage']['passed']:
        raise ValueError('Route coverage failed; repair registration before planning training regions')
    if not route.get('quality', {}).get('passed', True):
        raise ValueError('Route quality failed; training region plan withheld')
    total = route['length']
    start,end = settings['start_margin'], total-settings['end_margin']
    core,pad = settings['core_length'], settings['context_length']
    if not np.all(np.isfinite([total,start,end,core,pad])) or core<=0 or pad<0 or start<0 or end>total or end<=start:
        raise ValueError('Route too short for margins, or invalid region settings')
    samples = route['samples']
    svals = np.array([r['s'] for r in samples])
    times = np.array([r['timestamp_seconds'] for r in samples])
    pts = np.array([r['source_pts_seconds'] for r in samples])
    def time_at(values, distance, include_stop_end=False):
        first = int(np.searchsorted(svals,distance,side='left'))
        if first<len(svals) and svals[first]==distance:
            index = int(np.searchsorted(svals,distance,side='right'))-1 if include_stop_end else first
            return float(values[index])
        if first==0: return float(values[0])
        if first==len(svals): return float(values[-1])
        # Travel begins after the lower station's stop and ends upon first arrival
        # at the upper station. Interpolating unique arrays spreads stops over travel.
        lower = first-1
        alpha = (distance-svals[lower])/(svals[first]-svals[lower])
        return float(values[lower]+alpha*(values[first]-values[lower]))
    def span(values, interval, end_inclusive=True):
        return [time_at(values,interval[0]),time_at(values,interval[1],end_inclusive)]
    regions = []
    left = start
    while left < end-1e-8:
        right = min(left+core,end)
        context = [max(0,left-pad),min(total,right+pad)]
        idx = len(regions)
        selected = [c for c in cameras if context[0]<=c['s']<=context[1]]
        train = [c['camera_id'] for c in selected if c['split']=='train']
        held = [c['camera_id'] for c in selected if c['split']=='held_out']
        boundary_counts = [sum(abs(r['s']-boundary)<=10 for r in samples) for boundary in (left,right)]
        regions.append({'region_id':f'cell_{idx:03d}','cell_index':idx,'core_s':[left,right],
                        'core_end_inclusive':right==end, 'context_s':context,
                        'core_time_seconds':span(times,[left,right],right==end),
                        'context_time_seconds':span(times,context),
                        'context_source_pts_seconds':span(pts,context),
                        'train_camera_ids':train,'held_out_camera_ids':held,
                        'boundary_ids':[f'boundary_{idx:03d}',f'boundary_{idx+1:03d}'],
                        'boundary_camera_frame_counts_10m':boundary_counts,
                        'coverage_state':'proposed' if train and held and min(boundary_counts)>=3 else 'needs_review'})
        left = right
    xyz = np.array([r['rig_to_package'] for r in samples]).reshape(-1,4,4)[:,:3,3]
    from scipy.spatial import cKDTree
    # Diagnostic only: ownership needs segment-aware seam validation, never nearest sample alone.
    ambiguous = [(int(i),int(j)) for i,j in cKDTree(xyz).query_pairs(20)
                 if abs(svals[i]-svals[j])>60]
    return {'schema_version':1,'kind':'training-region-plan','playable_s':[start,end],
            'capture_margins':[start,total-end],'regions':regions,
            'ownership_rule':'half-open cores, final endpoint inclusive; training context overlaps',
            'nonadjacent_nearby_samples':ambiguous,
            'seam_status':'not validated; Gaussian training has not run',
            'training_ready':False}


def write_region_subsets(workspace, plan):
    workspace = Path(workspace)
    cameras = read_jsonl(workspace/'cameras.jsonl')
    by_id = {c['camera_id']:c for c in cameras}
    meta = read_json(workspace/'views'/'views_meta.json')
    points = load_points3d_txt(workspace/'track'/'colmap'/'points3D.txt')
    for region in plan['regions']:
        out = ensure_dir(workspace/'cells'/region['region_id'])
        for split,key in [('train','train_camera_ids'),('held_out','held_out_camera_ids')]:
            selected = [by_id[cid] for cid in region[key]]
            views = [{'name':c['name'],'T_wc':np.array(c['camera_to_package']).reshape(4,4)} for c in selected]
            subset = points
            if points is not None and views:
                xyz = np.array([v['T_wc'][:3,3] for v in views])
                lo,hi = xyz.min(axis=0)-20,xyz.max(axis=0)+20
                subset = points[np.all((points[:,:3]>=lo)&(points[:,:3]<=hi),axis=1)]
            write_colmap_model(out/split,meta,views,subset)
        atomic_json(out/'region.json',region)
    atomic_json(workspace/'regions.json',plan)
    with open(workspace/'reports'/'training_regions.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['region','core_start_nominal_m','core_end_nominal_m','context_start_seconds','context_end_seconds','train_views','held_out_views'])
        for r in plan['regions']:
            writer.writerow([r['region_id'],*r['core_s'],*r['context_time_seconds'],len(r['train_camera_ids']),len(r['held_out_camera_ids'])])
