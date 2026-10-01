"""Synthetic lateral seam diagnostics; never count these views as photographs."""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from track3dgs.io_utils import read_json, read_jsonl
from track3dgs.route_config import atomic_json, file_hash
from track3dgs.route_qc import RenderModel
from track3dgs.route_training import gpu_lease

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--offset', type=float, default=1.5)
    parser.add_argument('--width', type=int, default=600)
    parser.add_argument('--forward-yaw', type=float, default=180, help='Depends on camera mounting')
    parser.add_argument('--backward-yaw', type=float, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.offset) or args.offset <= 0 or args.width < 16:
        parser.error('Positive finite offset and width >=16 required')
    ROOT, RUN = Path(args.workspace), Path(args.run_root)
    regions = read_json(ROOT / 'regions.json')['regions']
    cameras = read_jsonl(ROOT / 'cameras.jsonl')
    intrinsics = read_json(ROOT / 'views' / 'views_meta.json')
    output = RUN / 'reports' / 'qc' / 'lateral_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with gpu_lease(ROOT.parent.parent / '.gpu-training.lock'):
        for boundary in range(1, len(regions)):
            left, right = regions[boundary - 1:boundary + 1]
            station = right['core_s'][0]
            common = set(left['held_out_camera_ids']) & set(right['held_out_camera_ids'])
            views = []
            for heading, yaw, sign in [('forward', args.forward_yaw, -1), ('backward', args.backward_yaw, 1)]:
                eligible = [c for c in cameras if c['camera_id'] in common and
                            c['yaw_degrees'] == yaw and (c['s'] - station) * sign >= 0]
                if not eligible:
                    raise ValueError(f'No suitable source camera for boundary {boundary}, {heading}')
                source = min(eligible, key=lambda c: abs(c['s'] - (station + sign * 5)))
                for offset in (-args.offset, args.offset):
                    transform = np.asarray(source['camera_to_package']).reshape(4, 4).copy()
                    transform[:3, 3] += transform[:3, 0] * offset
                    views.append({'camera_to_package': transform.ravel().tolist(),
                                  'source_camera_id': source['camera_id'],
                                  'source_station': source['s'], 'heading': heading,
                                  'offset_along_camera_right_nominal_m': offset})
            l = RUN / 'models' / left['region_id']
            r = RUN / 'models' / right['region_id']
            variants = {'left_context': [l / 'clean.ply'],
                        'assembled': [l / 'core.ply', r / 'core.ply'],
                        'right_context': [r / 'clean.ply']}
            renders = {}
            for label, paths in variants.items():
                model = RenderModel(paths)
                try:
                    renders[label] = [model.render(view, intrinsics, width=args.width) for view in views]
                finally:
                    model.close()
            for index, view in enumerate(views):
                ordered = [renders[key][index] for key in variants]
                top = np.concatenate([pair[0] for pair in ordered], axis=1)
                bottom = np.concatenate([np.repeat(pair[1][:, :, None], 3, axis=2) for pair in ordered], axis=1)
                collage = np.uint8(np.clip(np.concatenate([top, bottom]), 0, 1) * 255)
                name = f'boundary_{boundary:03d}_{index:02d}.jpg'
                cv2.imwrite(str(output / name), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 94])
                reference_opaque = (ordered[0][1] > .9) & (ordered[2][1] > .9)
                candidates = reference_opaque & (ordered[1][1] < .5)
                records.append({'boundary': boundary, 'boundary_station': station,
                                'image': name, 'camera': view,
                                'both_parents_opaque_pixels': int(reference_opaque.sum()),
                                'new_low_alpha_candidate_pixels': int(candidates.sum()),
                                'core_sha256': [file_hash(l / 'core.ply'), file_hash(r / 'core.ply')]})
            print(f'LATERAL boundary_{boundary:03d}: four synthetic poses rendered', flush=True)
    atomic_json(output / 'report.json', {
        'schema_version': 1, 'status': 'diagnostic_visual_review_required',
        'route_sha256': file_hash(ROOT / 'route.json'), 'views': records,
        'limitations': ['Synthetic translations at the recorded camera height, not held-out photography.',
                        'At most two models rendered together. Left context | assembled cores | right context; alpha below.',
                        'Low-alpha candidates are prompts for inspection, not automatic seam failures or acceptance.',
                        'Does not establish quality for every drivable position, height or heading.']})
    figures = []
    for record in records:
        v = record['camera']
        caption = (f"Boundary {record['boundary_station']} m Â· {v['heading']} Â· "
                   f"lateral offset {v['offset_along_camera_right_nominal_m']:+.1f} m Â· "
                   f"new low-alpha candidate pixels {record['new_low_alpha_candidate_pixels']}")
        figures.append(f'<figure><figcaption>{html.escape(caption)}</figcaption><img loading="lazy" src="{record["image"]}"></figure>')
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Synthetic lateral seam checks</title>'
        '<style>body{background:#101923;color:#dce8f3;font:16px system-ui;margin:32px}img{width:100%}figure{margin:24px 0}</style>'
        '<h1>Synthetic lateral seam checks</h1><p>Left context | assembled cores | right context. Alpha below. '
        'These are model-to-model diagnostics, without photographic ground truth.</p>' + ''.join(figures), encoding='utf-8')


if __name__ == '__main__':
    main()
