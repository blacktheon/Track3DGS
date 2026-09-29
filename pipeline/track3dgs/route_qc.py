"""Bounded GPU photographic and seam evidence for native-frame route models."""
import argparse
import html
from pathlib import Path

import cv2
import numpy as np

from .io_utils import read_json, read_jsonl
from .route_config import atomic_json


def choose_views(cameras, stations):
    held=[c for c in cameras if c['split']=='held_out']
    if not held: raise ValueError('No held-out photographs for QC')
    frames={min(held,key=lambda c:abs(c['s']-s))['frame_id'] for s in stations}
    return sorted([c for c in held if c['frame_id'] in frames],key=lambda c:(c['s'],c['yaw_degrees']))


def camera_matrices(camera,intrinsics,width):
    ratio=width/intrinsics['width'];height=round(intrinsics['height']*ratio)
    K=np.array([[intrinsics['fx']*ratio,0,intrinsics['cx']*ratio],
        [0,intrinsics['fy']*ratio,intrinsics['cy']*ratio],[0,0,1]],dtype=np.float32)
    return np.linalg.inv(np.asarray(camera['camera_to_package']).reshape(4,4)).astype(np.float32),K,width,height


def masked_metrics(pred,gt,mask,alpha):
    if not mask.any(): raise ValueError('QC view has no foreground mask')
    mse=float(np.mean((pred[mask]-gt[mask])**2))
    return {'psnr_db':float(-10*np.log10(max(mse,1e-12))),
        'foreground_low_alpha_fraction':float(np.mean(alpha[mask]<.5))}


class RenderModel:
    def __init__(self,paths):
        import torch
        from .evalpsnr import load_splats
        if not 1<=len(paths)<=2: raise ValueError('QC supports at most two resident models')
        arrays=[load_splats(str(path)) for path in paths]
        joined=[np.concatenate([a[i] for a in arrays],axis=0) for i in range(6)]
        self.means,self.scales,self.quats,self.opacity=[torch.tensor(a,device='cuda') for a in joined[:4]]
        self.quats=self.quats/self.quats.norm(dim=-1,keepdim=True)
        self.colors=torch.tensor(np.concatenate(joined[4:],axis=1),device='cuda')

    def render(self,camera,intrinsics,width=800):
        import torch
        from gsplat import rasterization
        vm,K,w,h=camera_matrices(camera,intrinsics,width)
        with torch.no_grad():
            rgb,alpha,_=rasterization(means=self.means,scales=self.scales,quats=self.quats,
                opacities=self.opacity,colors=self.colors,sh_degree=3,
                viewmats=torch.tensor(vm[None],device='cuda'),Ks=torch.tensor(K[None],device='cuda'),
                width=w,height=h,rasterize_mode='classic',backgrounds=torch.zeros((1,3),device='cuda'))
        return rgb[0,:,:,:3].clamp(0,1).cpu().numpy(),alpha[0,:,:,0].cpu().numpy()

    def close(self):
        import torch
        self.means=self.scales=self.quats=self.opacity=self.colors=None
        torch.cuda.empty_cache()


def evaluate(workspace,run_root,key,variants,views,width=800):
    """Each variant is one model or a globally sorted adjacent pair, never more."""
    root=Path(workspace);folder=Path(run_root)/'reports'/'qc'/key;folder.mkdir(parents=True,exist_ok=True)
    intrinsics=read_json(root/'views'/'views_meta.json')
    results=[]
    for label,paths in variants.items():
        model=RenderModel(paths)
        try:
            for index,camera in enumerate(views):
                pred,alpha=model.render(camera,intrinsics,width)
                gt=cv2.cvtColor(cv2.imread(str(root/'views'/camera['name'])),cv2.COLOR_BGR2RGB)
                gt=cv2.resize(gt,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_AREA)/255.
                mask=cv2.imread(str(root/'views_masks'/Path(camera['name']).with_suffix('.png')),cv2.IMREAD_GRAYSCALE)
                mask=cv2.resize(mask,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)>127
                metrics=masked_metrics(pred,gt,mask,alpha)
                # Ground truth masked to the same black sky as the reference renderer.
                gt[~mask]=0
                rgb=np.concatenate([gt,pred,np.repeat(alpha[:,:,None],3,axis=2)],axis=1)
                image=f'{label}_{index:03d}.jpg'
                cv2.imwrite(str(folder/image),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
                results.append({'variant':label,'camera_id':camera['camera_id'],'frame_id':camera['frame_id'],
                    'station':camera['s'],'timestamp_seconds':camera['timestamp_seconds'],'yaw':camera['yaw_degrees'],
                    'image':image,**metrics})
        finally: model.close()
    means={label:{'mean_psnr_db':float(np.mean([r['psnr_db'] for r in results if r['variant']==label])),
                  'mean_low_alpha_fraction':float(np.mean([r['foreground_low_alpha_fraction'] for r in results if r['variant']==label]))}
           for label in variants}
    report={'schema_version':1,'key':key,'status':'draft_visual_review_required','width':width,
        'view_count':len(views),'metrics':means,'views':results,
        'limitations':['Masked photographic PSNR includes residual exposure and reconstruction error.',
            'Alpha is a diagnostic; vegetation translucency is not automatically a hole.',
            'Black sky is intentionally removed. No synthetic view is counted as held-out photography.',
            'Unique ownership alone does not establish seam quality.']}
    atomic_json(folder/'report.json',report)
    rows=[]
    for r in results:
        title=f"{r['variant']} · {r['station']:.1f} nominal m · yaw {r['yaw']} · PSNR {r['psnr_db']:.2f} dB · low alpha {r['foreground_low_alpha_fraction']:.1%}"
        rows.append(f'<figure><figcaption>{html.escape(title)}</figcaption><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></figure>')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(key)+
        '</title><style>body{background:#101923;color:#dce8f3;font:16px system-ui;margin:32px}img{width:100%}figure{margin:24px 0}figcaption{padding:8px}</style>'+
        '<h1>'+html.escape(key)+'</h1><p>Held-out photograph | rendered model | alpha. Draft evidence; inspect the road and both sides at every seam.</p>'+''.join(rows),encoding='utf-8')
    print('QC '+key+' '+str(means),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--region',type=int);p.add_argument('--boundary',type=int)
    p.add_argument('--width',type=int,default=800);a=p.parse_args()
    root=Path(a.workspace);run=Path(a.run_root);plan=read_json(root/'regions.json')
    cameras=read_jsonl(root/'cameras.jsonl')
    if a.region is not None:
        region=plan['regions'][a.region];out=run/'models'/region['region_id'];lo,hi=region['core_s']
        selected=[c for c in cameras if c['camera_id'] in region['held_out_camera_ids']]
        views=choose_views(selected,[lo+min(15,(hi-lo)/4),(lo+hi)/2,hi-min(15,(hi-lo)/4)])
        evaluate(root,run,region['region_id'],{'raw':[out/'splat.ply'],'clean':[out/'clean.ply']},views,a.width)
    elif a.boundary is not None:
        index=a.boundary
        if not 1<=index<len(plan['regions']): raise ValueError('Boundary must have two adjacent regions')
        left,right=plan['regions'][index-1:index+1];station=right['core_s'][0]
        selected=[c for c in cameras if c['camera_id'] in set(left['held_out_camera_ids'])&set(right['held_out_camera_ids'])]
        views=choose_views(selected,[station-5,station,station+5])
        l=run/'models'/left['region_id'];r=run/'models'/right['region_id']
        evaluate(root,run,f'boundary_{index:03d}',{'left_reference':[l/'clean.ply'],'right_reference':[r/'clean.ply'],
            'assembled':[l/'core.ply',r/'core.ply']},views,a.width)
    else: p.error('Choose --region or --boundary')


if __name__=='__main__': main()
