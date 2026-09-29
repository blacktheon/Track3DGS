"""Configuration and immutable workspace identity for the opt-in route workflow."""
import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_fingerprint(stage, inputs, settings, tool_versions):
    data = json.dumps([stage, inputs, settings, tool_versions], sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def pin_route_calibration(config):
    """Never revise accepted coordinates in-place, including pre-pin workspaces."""
    root = Path(config['workspace'])
    settings = {k:config[k] for k in ('route_id','revision','scale')}
    identity = root/'state'/'calibration_identity.json'
    previous_config = root/'route_config.resolved.json'
    if previous_config.exists():
        previous = json.loads(previous_config.read_text(encoding='utf-8'))
        if {k:previous[k] for k in settings} != settings:
            raise ValueError('Route calibration changed; select a new revision to preserve existing coordinates')
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8')) != settings:
        raise ValueError('Route calibration changed; select a new revision to preserve existing coordinates')
    atomic_json(identity,settings)


def load_route_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    if cfg['schema_version'] != 1:
        raise ValueError('Unsupported route configuration version')
    for key in ('route_id', 'revision'):
        if not cfg.get(key) or any(c in cfg[key] for c in '/\\:'):
            raise ValueError(f'Invalid {key}')
    for key in ('source_video', 'workspace', 'vehicle_mask', 'sky_prior', 'mount_calibration'):
        if key not in cfg:
            raise ValueError(f'{key} must be explicit (null allowed for optional priors)')
        if cfg[key] is not None:
            cfg[key] = str((path.parent / cfg[key]).resolve())
            if key != 'workspace' and not Path(cfg[key]).is_file():
                raise ValueError(f'Missing {key}: {cfg[key]}')
    if not cfg['vehicle_mask'] or not cfg['source_video'] or not cfg['workspace']:
        raise ValueError('source_video, workspace and vehicle_mask are required')
    cfg['capture'] = {'sharpness_group': 3, 'keyframe_hz': 2.0, 'view_size': 1600,
                      'fov_degrees': 100, 'yaws': [-135,-90,-45,0,45,90,135,180],
                      'holdout_stride': 10, **cfg.get('capture', {})}
    cfg['reconstruction'] = {'mapper': 'glomap', 'overlap': 48,
                             'max_gap_seconds': 2, **cfg.get('reconstruction', {})}
    cfg['toolchain'] = {'colmap': r'C:\Work\tools\colmap\bin\colmap.exe',
                       **cfg.get('toolchain', {})}
    for key in ('sharpness_group', 'view_size', 'holdout_stride'):
        value = cfg['capture'][key]
        if not isinstance(value, int) or value < (2 if key == 'holdout_stride' else 1):
            raise ValueError(f'Invalid capture.{key}')
    for key in ('core_length', 'context_length', 'start_margin', 'end_margin'):
        value = cfg['regions'][key]
        if not math.isfinite(value) or value < 0 or (key == 'core_length' and value == 0):
            raise ValueError(f'Invalid regions.{key}')
    for value in (cfg['capture']['keyframe_hz'], cfg['capture']['fov_degrees'],
                  cfg['reconstruction']['max_gap_seconds']):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Capture/coverage settings must be positive and finite')
    scale = cfg['scale']
    if scale['method'] == 'nominal_speed':
        if not math.isfinite(scale['speed_kmh']) or scale['speed_kmh'] <= 0:
            raise ValueError('Nominal speed must be positive')
        scale['status'] = 'approximate'
    elif scale['method'] == 'manual':
        if not math.isfinite(scale['metres_per_sfm_unit']) or scale['metres_per_sfm_unit'] <= 0:
            raise ValueError('Manual scale must be positive')
        scale.setdefault('status', 'approximate')
    elif scale['method'] == 'unscaled':
        scale['status'] = 'unscaled'
    else:
        raise ValueError('Unknown scale method')
    return cfg
