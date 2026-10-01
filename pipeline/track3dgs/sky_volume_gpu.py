"""Native renderer contribution refinement and photographic cleanup review."""
import html
from pathlib import Path

import cv2
import numpy as np

from .io_utils import read_json
from .route_config import atomic_json, file_hash
from .route_qc import RenderModel, camera_matrices, masked_metrics
from .sky_volume import load_selection, masks, center_hits, footprint_hits, bit_count, decide, write_filtered


def carve(workspace, source, output, source_index=None):
    import torch
    from gsplat import rasterization
    from gsplat.cuda._wrapper import fully_fused_projection
    root, source, output = Path(workspace).resolve(), Path(source).resolve(), Path(output).resolve()
    selection = load_selection(root, output)
    if (output / 'processing.json').exists() or (output / 'models/sky.ply').exists():
        raise FileExistsError('Cleanup result exists; prepare a new revision')
    params = selection['parameters']; width = params['projection_width']
    single = params['mode'] == 'single'
    digest = file_hash(source)
    model = RenderModel([source]); n = len(model.means)
    center_bits = np.zeros(n, np.uint64)
    prepared = []
    try:
        xyz = model.means.cpu().numpy()
        remove = np.zeros(n, bool)
        for camera in selection['edit']:
            vm, K, w, h = camera_matrices(camera, selection['intrinsics'], width)
            sky, foreground = masks(root, output, camera, width, params['erosion_px'])
            center_mask = masks(root, output, camera, width)[0] if single else sky
            hit = center_hits(xyz, vm, K, center_mask)
            center_bits[hit] |= np.uint64(1 << camera['position_index'])
            prepared.append((camera, vm, K, sky, foreground, w, h))
            if single:
                # Preserve the original single-view experiment's sampled ellipse seed.
                with torch.no_grad():
                    projected = fully_fused_projection(model.means, None, model.quats, model.scales,
                        torch.tensor(vm[None], device='cuda'), torch.tensor(K[None], device='cuda'), w, h, packed=False)
                radii, means2d, _, conics = [a[0].cpu().numpy() for a in projected[:4]]
                footprint, _ = footprint_hits(means2d, conics, radii, model.opacity.cpu().numpy(), center_mask, params['erosion_px'])
                remove |= hit | footprint
                del projected
        if not single:
            remove = bit_count(center_bits) >= params['minimum_votes']
        seeded = int(remove.sum()); rounds = []
        print('SEED', n, seeded, flush=True)
        for iteration in range(params['max_rounds']):
            sky_bits = np.zeros(n, np.uint64)
            sky_mass, foreground_mass = np.zeros(n, np.float64), np.zeros(n, np.float64)
            opacity = model.opacity * torch.tensor(~remove, device='cuda')
            view_stats = []
            for camera, vm, K, sky, foreground, w, h in prepared:
                colors = torch.ones((n, 2), device='cuda', requires_grad=True)
                image, alpha, _ = rasterization(means=model.means, quats=model.quats,
                    scales=model.scales, opacities=opacity, colors=colors,
                    viewmats=torch.tensor(vm[None], device='cuda'), Ks=torch.tensor(K[None], device='cuda'),
                    width=w, height=h, rasterize_mode='classic', backgrounds=torch.zeros((1, 2), device='cuda'))
                sm, fm = torch.tensor(sky, device='cuda'), torch.tensor(foreground, device='cuda')
                loss = image[0, :, :, 0][sm].sum() + image[0, :, :, 1][fm].sum()
                loss.backward()
                contribution = colors.grad.detach().cpu().numpy()
                if not np.isfinite(contribution).all() or contribution.min() < -1e-5:
                    raise ValueError('Invalid renderer contribution gradient')
                hit = contribution[:, 0] > params['alpha_pixels']
                sky_bits[hit] |= np.uint64(1 << camera['position_index'])
                sky_mass += contribution[:, 0]; foreground_mass += contribution[:, 1]
                view_stats.append({'name': camera['name'], 'sky_alpha_before':
                    float(alpha[0, :, :, 0][sm].mean().item()) if sky.any() else None})
                del colors, image, alpha, sm, fm, loss, contribution
            chosen = (sky_mass > params['alpha_pixels']) if single else decide(
                center_bits, sky_bits, sky_mass, foreground_mass, params['minimum_votes'], params['sky_fraction'])
            additional = chosen & ~remove
            rounds.append({'iteration': iteration, 'additional_removed': int(additional.sum()), 'views': view_stats})
            remove |= additional
            np.savez_compressed(output / 'classification.npz', removed=remove, center_bits=center_bits,
                sky_bits=sky_bits, sky_mass=sky_mass, foreground_mass=foreground_mass)
            atomic_json(output / 'progress.json', {'iteration': iteration, 'removed': int(remove.sum()), 'rounds': rounds})
            print('ROUND', iteration, 'additional', int(additional.sum()), 'total', int(remove.sum()), flush=True)
            if not additional.any():
                break
    finally:
        model.close()
    if file_hash(source) != digest:
        raise ValueError('Source changed during cleanup')
    out = output / 'models/sky.ply'
    count = write_filtered(source, remove, out, source_index)
    report = {'schema_version': 1, 'status': 'experimental_pending_visual_review',
        'region_id': selection['region_id'], 'route_sha256': selection['route_sha256'],
        'selection_sha256': file_hash(output / 'selection.json'),
        'source': str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
        'source_sha256': digest, 'source_index': source_index, 'source_count': n, 'count': count,
        'removed_count': int(remove.sum()), 'removed_percent': float(remove.mean() * 100),
        'file': 'models/sky.ply', 'sha256': file_hash(out),
        'provenance_sha256': file_hash(out.with_suffix('.provenance.bin')),
        'coordinate_frame': 'native package RUB; exact retained attributes including SH3',
        'parameters': params, 'center_seed_removed': seeded,
        'converged': rounds[-1]['additional_removed'] == 0, 'rounds': rounds}
    atomic_json(output / 'processing.json', report)
    print('RESULT', count, 'removed', int(remove.sum()), flush=True)
    return report


def evaluate(workspace, output, width=800):
    root, output = Path(workspace), Path(output)
    selection = load_selection(root, output)
    report = read_json(output / 'processing.json')
    source, cleaned = root / report['source'], output / report['file']
    if (file_hash(source) != report['source_sha256'] or file_hash(cleaned) != report['sha256'] or
            file_hash(output / 'selection.json') != report['selection_sha256']):
        raise ValueError('Cleanup source/result/selection changed')
    views = [c for role in ('edit', 'validation', 'check') for c in selection.get(role, [])]
    folder = output / 'renders'; folder.mkdir(exist_ok=True)
    results = []
    for label, path in [('before', source), ('after', cleaned)]:
        model = RenderModel([path])
        try:
            for index, camera in enumerate(views):
                rgb, alpha = model.render(camera, selection['intrinsics'], width)
                gt = cv2.imread(str(root / 'views' / camera['name']))
                if gt is None:
                    raise FileNotFoundError(camera['name'])
                gt = cv2.cvtColor(cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) / 255.
                sky, foreground = masks(root, output, camera, width)
                values = masked_metrics(rgb, gt, foreground, alpha)
                values['sky_mean_alpha'] = float(alpha[sky].mean()) if sky.any() else None
                results.append({'variant': label, 'index': index, 'camera': camera, **values})
                cv2.imwrite(str(folder / f'{label}_{index:02d}.jpg'), cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255), cv2.COLOR_RGB2BGR))
                if label == 'before':
                    gt[~foreground] = 0
                    cv2.imwrite(str(folder / f'photo_{index:02d}.jpg'), cv2.cvtColor(np.uint8(gt * 255), cv2.COLOR_RGB2BGR))
        finally:
            model.close()
    atomic_json(output / 'metrics.json', {'views': results, 'validation_frames_disjoint': True})
    cards = []
    for index, camera in enumerate(views):
        caption = html.escape(f"{camera['role']} · {camera['s']:.1f} nominal m · yaw {camera['yaw_degrees']} · {camera['name']}")
        imgs = ''.join(f'<img loading="lazy" src="renders/{kind}_{index:02d}.jpg" alt="{kind}">' for kind in ('photo', 'before', 'after'))
        cards.append(f'<figure><figcaption>{caption}</figcaption><div>{imgs}</div></figure>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sky-volume review</title>'
        '<style>body{background:#101923;color:#dce8f3;font:16px system-ui;margin:30px}figure{margin:24px 0}figure div{display:flex}img{width:33.33%}figcaption{padding:8px}</style>'
        f'<h1>Sky-volume review · {html.escape(report["region_id"])}</h1><p>{report["source_count"]:,} → {report["count"]:,} splats. '
        f'{report["removed_percent"]:.2f}% removed.</p><p>Photograph | before | after. Edit views informed removal; only validation views are independent. '
        'Check canopy thinning, road coverage and neighboring views. Experimental; no automatic quality acceptance.</p>' + ''.join(cards), encoding='utf-8')
    return results
