"""Calibrated sky-volume cleanup: CPU geometry, selection and CLI. GPU imports are lazy."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

from .io_utils import read_json, read_jsonl
from .route_config import atomic_json, file_hash
from .route_assembly import PROVENANCE_DTYPE
from .route_preview_models import write_preview_ply


def center_hits(xyz, viewmat, K, sky):
    p = xyz @ viewmat[:3, :3].T + viewmat[:3, 3]
    valid = np.isfinite(p).all(axis=1) & (p[:, 2] > .01)
    uv = np.full((len(p), 2), -1., dtype=float)
    uv[valid] = p[valid, :2] / p[valid, 2, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    valid &= (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < sky.shape[1]) & (uv[:, 1] < sky.shape[0])
    result = np.zeros(len(p), dtype=bool)
    pix = np.floor(uv[valid]).astype(int)
    result[valid] = sky[pix[:, 1], pix[:, 0]]
    return result


def footprint_hits(means2d, conics, radii, opacity, sky, erosion_px=8):
    """Sample significant projected ellipse support; no centre-only shortcut.

    Uses the actual renderer's projected covariance (including its 0.3 pixel
    blur), 16 radial rings and 48 angles out to 3 sigma, with alpha >= 1/255.
    This finite, sampled support is an approximation, not exact Gaussian CSG.
    An 8-pixel sky erosion protects the immediate foliage/sky boundary.
    """
    safe_sky = cv2.erode(sky.astype(np.uint8), np.ones((2 * erosion_px + 1,) * 2, np.uint8)) > 0
    valid = (radii > 0) & np.isfinite(means2d).all(axis=1) & np.isfinite(conics).all(axis=1)
    ids = np.flatnonzero(valid)
    c = conics[ids].astype(float)
    det = c[:, 0] * c[:, 2] - c[:, 1] ** 2
    good = det > 0
    ids = ids[good]; c = c[good]; det = det[good]
    cov = np.empty((len(ids), 2, 2))
    cov[:, 0, 0] = c[:, 2] / det
    cov[:, 1, 1] = c[:, 0] / det
    cov[:, 0, 1] = cov[:, 1, 0] = -c[:, 1] / det
    eig, axes = np.linalg.eigh(cov)
    axes *= np.sqrt(np.maximum(eig, 0))[:, None, :]
    centers = means2d[ids]
    hit_any = np.zeros(len(ids), bool)
    weighted_sky = np.zeros(len(ids), dtype=float)
    weighted_in_image = np.zeros(len(ids), dtype=float)
    h, w = sky.shape
    # Uniform-area rings for overlap estimates, includes principal axes.
    for j in range(16):
        radius = 3 * np.sqrt((j + .5) / 16)
        weight = math.exp(-.5 * radius * radius)
        significant = opacity[ids] * weight >= 1 / 255
        for angle in np.arange(48) * (2 * np.pi / 48):
            direction = radius * np.array([np.cos(angle), np.sin(angle)])
            uv = centers + axes @ direction
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            indices = np.flatnonzero(inside & significant)
            pix = np.floor(uv[indices]).astype(int)
            hit = safe_sky[pix[:, 1], pix[:, 0]]
            hit_any[indices] |= hit
            weighted_sky[indices] += weight * hit
            weighted_in_image[indices] += weight
    fraction = np.divide(weighted_sky, weighted_in_image, out=np.zeros_like(weighted_sky), where=weighted_in_image > 0)
    all_hit = np.zeros(len(radii), bool); all_fraction = np.zeros(len(radii))
    all_hit[ids] = hit_any; all_fraction[ids] = fraction
    return all_hit, all_fraction


def top_connected_sky(sky):
    """Input is a sky-only boolean mask, never the combined sky/vehicle mask."""
    _, labels = cv2.connectedComponents(np.asarray(sky, np.uint8), connectivity=8)
    ids = np.unique(labels[0]); ids = ids[ids > 0]
    return np.isin(labels, ids)


def bit_count(bits):
    values = np.ascontiguousarray(bits, dtype=np.uint64)
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    return table[values.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def decide(center_bits, sky_bits, sky_mass, foreground_mass, minimum_votes=2, sky_fraction=.05):
    total = sky_mass + foreground_mass
    fraction = np.divide(sky_mass, total, out=np.zeros_like(sky_mass), where=total > 0)
    return ((bit_count(center_bits) >= minimum_votes) |
            ((bit_count(sky_bits | center_bits) >= minimum_votes) & (fraction >= sky_fraction)))


def select_cameras(cameras, groups):
    selected = {}
    for role, group in groups.items():
        if role not in ('edit', 'validation', 'check'):
            raise ValueError('Unknown camera role: ' + role)
        if 'names' in group:
            by_name = {c['name']: c for c in cameras}
            views = [by_name[n] for n in group['names']]
        else:
            available = [c for c in cameras if c['split'] == group['split']]
            if not available:
                raise ValueError('No cameras for ' + group['split'])
            frames = [min(available, key=lambda c: abs(c['s'] - s))['frame_id'] for s in group['stations']]
            if len(set(frames)) != len(frames):
                raise ValueError('Requested stations do not resolve to distinct positions')
            by_pose = {(c['frame_id'], c['yaw_degrees']): c for c in available}
            views = [by_pose[(frame, yaw)] for frame in frames for yaw in group['yaws']]
        positions = {}
        selected[role] = [dict(c, role=role, position_index=positions.setdefault(c['frame_id'], len(positions))) for c in views]
        if len(positions) > 64:
            raise ValueError('At most 64 distinct positions per cleanup pass')
        if len({c['name'] for c in views}) != len(views):
            raise ValueError('Duplicate camera would bias contribution mass')
    if not selected.get('edit'):
        raise ValueError('At least one edit camera required')
    edited = {c['frame_id'] for c in selected['edit']}
    if edited & {c['frame_id'] for c in selected.get('validation', [])}:
        raise ValueError('Independent validation frames must be disjoint from edit frames')
    if any(c['split'] != 'held_out' for c in selected.get('validation', [])):
        raise ValueError('Independent validation must use held_out source frames')
    return selected


def write_filtered(source, remove, destination, source_index=None):
    source, destination = Path(source), Path(destination)
    provenance = destination.with_suffix('.provenance.bin')
    if destination.exists() or provenance.exists():
        raise FileExistsError('Choose a new output; never overwrite a model or its provenance')
    original_hash = file_hash(source)
    vertices = PlyData.read(str(source))['vertex'].data
    remove = np.asarray(remove)
    if remove.dtype != bool or remove.shape != (len(vertices),):
        raise ValueError('Removal mask must contain one boolean per source row')
    source_provenance = source.with_suffix('.provenance.bin')
    if source_provenance.exists():
        if source_provenance.stat().st_size != len(vertices) * PROVENANCE_DTYPE.itemsize:
            raise ValueError('Source provenance length mismatch')
        ids = np.fromfile(source_provenance, PROVENANCE_DTYPE)
    else:
        if source_index is None or not 0 <= source_index < 2**32:
            raise ValueError('Raw PLY needs an explicit source_index')
        ids = np.zeros(len(vertices), PROVENANCE_DTYPE)
        ids['source_index'] = source_index
        ids['source_row'] = np.arange(len(vertices), dtype=np.uint64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_preview_ply(vertices[~remove], destination)
    ids[~remove].tofile(provenance)
    if PlyData.read(str(destination))['vertex'].data.tobytes() != vertices[~remove].tobytes():
        raise ValueError('Export changed retained Gaussian attributes')
    if file_hash(source) != original_hash:
        raise ValueError('Source changed while exporting')
    return int((~remove).sum())


def prepare(workspace, config_path, output):
    from .views import _e2p
    root, output = Path(workspace), Path(output)
    config = read_json(config_path)
    groups = select_cameras(read_jsonl(root / 'cameras.jsonl'), config['views'])
    params = {'mode': 'consensus', 'minimum_votes': 2, 'sky_fraction': .05, 'alpha_pixels': .05,
              'erosion_px': 8, 'max_rounds': 8, 'projection_width': 1600, **config.get('parameters', {})}
    mode = params['mode']
    positions = len({c['position_index'] for c in groups['edit']})
    if mode not in ('consensus', 'single') or (mode == 'single' and len(groups['edit']) != 1):
        raise ValueError('Single mode requires exactly one edit view')
    if mode == 'consensus' and not 1 <= params['minimum_votes'] <= positions:
        raise ValueError('Minimum votes exceed distinct edit positions')
    if (not 0 <= params['sky_fraction'] <= 1 or params['alpha_pixels'] < 0 or
            params['max_rounds'] < 1 or params['erosion_px'] < 0 or params['projection_width'] < 16):
        raise ValueError('Invalid cleanup parameters')
    intrinsics = read_json(root / 'views/views_meta.json')
    if intrinsics['width'] != intrinsics['height'] or not np.isclose(intrinsics['fx'], intrinsics['fy']):
        raise ValueError('Mask projection currently requires square symmetric pinhole crops')
    fov = math.degrees(2 * math.atan(intrinsics['width'] / (2 * intrinsics['fx'])))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'masks').mkdir()
    hashes, source_hashes = {}, {}
    for camera in [c for views in groups.values() for c in views]:
        name = Path(camera['name']).with_suffix('.png').name
        src = root / 'sky_masks' / (camera['name'].split('_y')[0] + '.png')
        sky = cv2.imread(str(src), 0)
        if sky is None:
            raise FileNotFoundError(src)
        crop = _e2p(np.repeat(sky[:, :, None], 3, axis=2), camera['yaw_degrees'], fov, intrinsics['width'])[:, :, 0] <= 127
        top = top_connected_sky(crop)
        target = output / 'masks' / name
        if not cv2.imwrite(str(target), top.astype(np.uint8) * 255):
            raise IOError('Could not write sky mask: ' + str(target))
        hashes[camera['name']] = file_hash(target)
        source_hashes[src.name] = file_hash(src)
    selection = {'schema_version': 1, 'region_id': config['region_id'], **groups,
                 'parameters': params, 'intrinsics': intrinsics, 'mask_hashes': hashes,
                 'source_sky_hashes': source_hashes, 'source_cameras_sha256': file_hash(root / 'cameras.jsonl'),
                 'route_sha256': file_hash(root / 'route.json'),
                 'note': 'Sky-only top-connected masks. Vehicle pixels are never a deletion target.'}
    atomic_json(output / 'selection.json', selection)
    return selection


def load_selection(workspace, output):
    root, output = Path(workspace), Path(output)
    selection = read_json(output / 'selection.json')
    for file, key in [('route.json', 'route_sha256'), ('cameras.jsonl', 'source_cameras_sha256')]:
        if file_hash(root / file) != selection[key]:
            raise ValueError('Source route/cameras changed; prepare a new cleanup revision')
    for name, digest in selection['mask_hashes'].items():
        if file_hash(output / 'masks' / Path(name).with_suffix('.png').name) != digest:
            raise ValueError('Prepared mask changed: ' + name)
    return selection


def derive_core(workspace, output, core):
    """Transfer a raw cleanup to its core using exact original row identities."""
    root, output, core = Path(workspace), Path(output), Path(core)
    report = read_json(output / 'processing.json')
    source, result = root / report['source'], output / report['file']
    if file_hash(source) != report['source_sha256'] or file_hash(result) != report['sha256']:
        raise ValueError('Cleanup source or result changed')
    raw = PlyData.read(str(source))['vertex'].data
    vertices = PlyData.read(str(core))['vertex'].data
    provenance = source.with_suffix('.provenance.bin')
    if provenance.exists():
        ids = np.fromfile(provenance, PROVENANCE_DTYPE)
    else:
        ids = np.zeros(len(raw), PROVENANCE_DTYPE)
        ids['source_index'] = report['source_index']
        ids['source_row'] = np.arange(len(raw), dtype=np.uint64)
    core_ids = np.fromfile(core.with_suffix('.provenance.bin'), PROVENANCE_DTYPE)
    retained = np.fromfile(result.with_suffix('.provenance.bin'), PROVENANCE_DTYPE)
    if len(ids) != len(raw) or len(core_ids) != len(vertices) or not np.isin(core_ids, ids).all():
        raise ValueError('Core provenance is not a subset of the cleaned source')
    order = np.argsort(ids)
    rows = order[np.searchsorted(ids[order], core_ids)]
    if vertices.dtype != raw.dtype or vertices.tobytes() != raw[rows].tobytes():
        raise ValueError('Core attributes/frame differ from the original raw rows')
    destination = output / 'models/sky_core.ply'
    count = write_filtered(core, ~np.isin(core_ids, retained), destination)
    report.update(core_file='models/sky_core.ply', core_count=count, core_sha256=file_hash(destination),
                  base_core_sha256=file_hash(core), base_core_count=len(vertices))
    atomic_json(output / 'processing.json', report)
    return count


def masks(workspace, output, camera, width, erosion_px=0):
    sky = cv2.imread(str(Path(output) / 'masks' / Path(camera['name']).with_suffix('.png').name), 0)
    fg = cv2.imread(str(Path(workspace) / 'views_masks' / Path(camera['name']).with_suffix('.png')), 0)
    if sky is None or fg is None:
        raise FileNotFoundError('Sky or foreground mask missing: ' + camera['name'])
    if erosion_px:
        sky = cv2.erode(sky, np.ones((2 * erosion_px + 1,) * 2, np.uint8))
    size = (width, round(sky.shape[0] * width / sky.shape[1]))
    return tuple(cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 127 for m in (sky, fg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'carve', 'evaluate', 'subset-core'):
        p = sub.add_parser(command)
        p.add_argument('--workspace', required=True)
        p.add_argument('--output', required=True, help='New cleanup revision directory')
        if command == 'prepare':
            p.add_argument('--config', required=True)
        elif command == 'carve':
            p.add_argument('--source', required=True, help='Native route-frame PLY; never a recentered Unity cache')
            p.add_argument('--source-index', type=int, help='Required for raw PLY without source-row provenance')
        elif command == 'subset-core':
            p.add_argument('--core', required=True, help='Existing core with source-row provenance')
        else:
            p.add_argument('--width', type=int, default=800)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.workspace, args.config, args.output)
    elif args.command == 'subset-core':
        derive_core(args.workspace, args.output, args.core)
    else:
        from .route_training import gpu_lease
        from .sky_volume_gpu import carve, evaluate
        with gpu_lease(Path(args.workspace).parent.parent / '.gpu-training.lock'):
            if args.command == 'carve':
                carve(args.workspace, args.source, args.output, args.source_index)
            else:
                evaluate(args.workspace, args.output, args.width)


if __name__ == '__main__':
    main()
