"""Portable route visualization and explicitly converted Unity marker data."""
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .route_config import atomic_json, file_hash

COLORS = ['#38bdf8','#2dd4bf','#a3e635','#fbbf24','#fb923c','#f472b6','#c084fc','#818cf8']


def unity_preview(route, plan):
    markers = []
    regions = plan['regions'] if plan else []
    for sample in route['samples']:
        T = np.array(sample['rig_to_package']).reshape(4,4)
        xyz = T[:3,3]*[1,1,-1]
        forward = T[:3,2]*[1,1,-1]
        up = -T[:3,1]*[1,1,-1]
        region = next((r['cell_index'] for r in regions
            if r['core_s'][0]<=sample['s'] and (sample['s']<r['core_s'][1] or r['core_end_inclusive'] and sample['s']<=r['core_s'][1])), -1)
        markers.append({'frameId':sample['frame_id'],'time':sample['timestamp_seconds'],'distance':sample['s'],
                        'x':float(xyz[0]),'y':float(xyz[1]),'z':float(xyz[2]),
                        'fx':float(forward[0]),'fy':float(forward[1]),'fz':float(forward[2]),
                        'ux':float(up[0]),'uy':float(up[1]),'uz':float(up[2]),'region':region,
                        'heldOut':sample.get('split')=='held_out'})
    return {'schemaVersion':1,'routeId':route.get('route_id','route'),
            'status':route.get('status','diagnostic'), 'length':route['length'],
            'scaleNote':f"{route['scale']['status']} scale / {route['scale']['method']}; absolute grade unmeasured",
            'coordinateNote':'Unity LH +X right +Y up +Z forward. RUB Z reflection applied once by exporter.',
            'maxGapSeconds':route['coverage'].get('max_gap_seconds',2),'markers':markers,
            'regions':[{'index':r['cell_index'],'name':r['region_id'],'coreStart':r['core_s'][0],
                        'coreEnd':r['core_s'][1],'contextStart':r['context_s'][0],
                        'contextEnd':r['context_s'][1],'startTime':r['context_time_seconds'][0],
                        'endTime':r['context_time_seconds'][1]} for r in regions]}


def write_reports(workspace, route, plan):
    workspace = Path(workspace)
    out = workspace/'reports'
    out.mkdir(exist_ok=True)
    preview = unity_preview(route,plan)
    atomic_json(out/'route_preview.json',preview | {'routeSha256':file_hash(workspace/'route.json')})
    samples = route['samples']
    xyz = np.array([r['rig_to_package'] for r in samples]).reshape(-1,4,4)[:,:3,3]*[1,1,-1]
    times = np.array([r['timestamp_seconds'] for r in samples])
    s = np.array([r['s'] for r in samples])
    fig = plt.figure(figsize=(15,10),facecolor='#0b1220')
    grid = fig.add_gridspec(2,2,height_ratios=[3,1],width_ratios=[2,1],hspace=.35,wspace=.3)
    ax = fig.add_subplot(grid[0,0]); elev = fig.add_subplot(grid[1,0]); legend = fig.add_subplot(grid[:,1])
    for a in (ax,elev):
        a.set_facecolor('#111c30'); a.tick_params(colors='#cbd5e1'); a.grid(color='#334155',alpha=.4)
        for spine in a.spines.values(): spine.set_color('#334155')
        a.xaxis.label.set_color('#cbd5e1'); a.yaxis.label.set_color('#cbd5e1')
    colors = [COLORS[m['region']%len(COLORS)] if m['region']>=0 else '#64748b' for m in preview['markers']]
    segments = np.stack([xyz[:-1][:,[0,2]],xyz[1:][:,[0,2]]],axis=1)
    valid = np.diff(times)<=preview['maxGapSeconds']
    ax.add_collection(LineCollection(segments[valid],colors=np.array(colors)[1:][valid],linewidths=3))
    ax.scatter(xyz[:,0],xyz[:,2],s=7,c=colors,zorder=3)
    ax.autoscale(); ax.set_aspect('equal',adjustable='datalim'); ax.margins(.15)
    ax.scatter(xyz[[0,-1],0],xyz[[0,-1],2],s=100,c=['#fff','#fb7185'],edgecolors='#0b1220',zorder=4)
    for i,label in [(0,'START'),(-1,'END')]:
        ax.annotate(f'{label}  {times[i]:.1f}s',(xyz[i,0],xyz[i,2]),xytext=(8,8),textcoords='offset points',color='white',fontsize=10)
    ax.set_xlabel('X / nominal metres'); ax.set_ylabel('Z / nominal metres')
    ax.set_title('Reconstructed camera path · top view',loc='left',color='white',pad=15)
    for i in range(len(samples)-1):
        if valid[i]: elev.plot(s[i:i+2],xyz[i:i+2,1],color=colors[i],lw=1.6)
    elev.set_xlabel('Travelled distance / nominal metres'); elev.set_ylabel('Camera height change')
    elev.set_title('Relative elevation · approximate camera-up alignment',loc='left',color='white',fontsize=10)
    legend.axis('off')
    legend.text(0,1,'TRACK02 / ROUTE STUDY',color='#38bdf8',fontsize=11,weight='bold',transform=legend.transAxes)
    legend.text(0,.945,'One shared coordinate frame',color='white',fontsize=18,weight='bold',transform=legend.transAxes)
    coverage = route['coverage']
    text = (f"{len(samples):,} camera positions\n{coverage.get('registered_views',0):,} registered pinhole views\n"
            f"{route['length']:.1f} nominal metres\n{coverage['registered_fraction']:.1%} keyframe coverage\n"
            f"{len(coverage['components'])} reconstruction component(s)")
    legend.text(0,.88,text,color='#cbd5e1',fontsize=12,linespacing=1.8,va='top',transform=legend.transAxes)
    y=.64
    legend.text(0,y,'PROPOSED TRAINING REGIONS',color='white',fontsize=11,weight='bold',transform=legend.transAxes)
    for r in (plan['regions'] if plan else []):
        y-=.065
        a,b = r['core_s']; t0,t1 = r['context_time_seconds']
        legend.text(0,y,f"{r['region_id']}   {a:.0f}–{b:.0f} m\nVideo context  {t0:.1f}–{t1:.1f} s",color=COLORS[r['cell_index']%len(COLORS)],fontsize=10,linespacing=1.4,transform=legend.transAxes)
    note = 'Scale assumes 10 km/h average speed.\nNo GPS, metric reference or gravity measurement.\nGrey route ends are capture margins.\nContext overlaps; future output cores do not.\nCamera trajectory is not a road surface.\nGaussian training has not started.'
    legend.text(0,.015,note,color='#94a3b8',fontsize=9,linespacing=1.6,va='bottom',transform=legend.transAxes)
    fig.suptitle('Continuous capture → shared route → training regions',color='white',fontsize=21,x=.08,ha='left',y=.975)
    fig.savefig(out/'route_overview.png',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)
    rows = ''.join(f"<tr><td>{r['region_id']}</td><td>{r['core_s'][0]:.1f}–{r['core_s'][1]:.1f}</td><td>{r['context_time_seconds'][0]:.2f}–{r['context_time_seconds'][1]:.2f}</td><td>{len(r['train_camera_ids'])}</td><td>{len(r['held_out_camera_ids'])}</td></tr>" for r in (plan['regions'] if plan else []))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Track02 route review</title>'
        '<style>body{background:#0b1220;color:#dbeafe;font:16px system-ui;margin:40px auto;max-width:1400px}img{width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #334155}a{color:#38bdf8}</style>'
        f'<h1>Track02 route review</h1><p>{html.escape(preview["scaleNote"])}. Status: {html.escape(preview["status"])}.</p>'
        '<img src="route_overview.png" alt="Reconstructed route and proposed training regions">'
        '<h2>Proposed video intervals</h2><p>Use overlapping context for training. Retain only validated core ownership after training. These are source video seconds, not speed-based equal-duration cuts.</p>'
        '<table><tr><th>Region</th><th>Core / nominal m</th><th>Video context / s</th><th>Training views</th><th>Held-out views</th></tr>'+rows+'</table>'
        '<p>Not yet approved for dense training. Inspect reconstruction shape, coverage, grades and transitions first. Future dense registration must keep this global frame and held-out frame membership.</p>'
        '<p><a href="coverage.json">Coverage evidence</a> · <a href="training_regions.csv">Region CSV</a> · <a href="route_preview.json">Unity marker data</a></p>',encoding='utf-8')
