"""Quality gates for derived plans. Motion priors never modify reconstructed poses."""
from pathlib import Path
import shutil

import numpy as np

from .route_config import atomic_json


def evaluate_route_quality(route, settings, visual_review=None):
    profile = settings.get('capture_motion', 'unknown')
    if profile not in ('unknown', 'broadly_similar_speed'):
        raise ValueError('Unknown capture motion profile')
    window = float(settings.get('speed_window_seconds', 5))
    p95_limit = float(settings.get('max_p95_to_median', 3))
    max_limit = float(settings.get('max_window_to_median', 5))
    if not np.all(np.isfinite([window, p95_limit, max_limit])) or window <= 0 or min(p95_limit, max_limit) <= 1:
        raise ValueError('Invalid motion quality thresholds')
    times = np.array([r['timestamp_seconds'] for r in route['samples']])
    distances = np.array([r['s'] for r in route['samples']])
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(np.diff(distances) < 0):
        raise ValueError('Motion evidence requires increasing timestamps and nondecreasing distance')
    window = min(window, times[-1] - times[0])
    starts = np.unique(np.r_[np.arange(times[0], times[-1] - window, 1.), times[-1] - window])
    speeds = (np.interp(starts + window, times, distances) - np.interp(starts, times, distances)) / window
    median, p95, maximum = np.quantile(speeds, [.5, .95, 1])
    # Epsilon keeps diagnostic JSON finite for predominantly stationary captures.
    denominator = max(float(median), 1e-12)
    p95_ratio, max_ratio = float(p95 / denominator), float(maximum / denominator)
    rejected = profile == 'broadly_similar_speed' and (p95_ratio > p95_limit or max_ratio > max_limit)
    motion = {'capture_motion': profile, 'source': settings.get('source', 'not supplied'),
              'decision': 'rejected' if rejected else ('plausible' if profile != 'unknown' else 'diagnostic_only'),
              'speed_window_seconds': float(window), 'window_speed_median': float(median),
              'window_speed_p95': float(p95), 'window_speed_max': float(maximum),
              'window_p95_to_median': p95_ratio, 'window_max_to_median': max_ratio,
              'thresholds': {'max_p95_to_median': p95_limit, 'max_window_to_median': max_limit},
              'window_centres_seconds': (starts + window/2).tolist(), 'window_speeds': speeds.tolist(),
              'note': 'Provisional plausibility check, not proof of geometric accuracy. No pose warping or speed equalization.'}
    review = visual_review or {'status': 'pending'}
    failures = []
    if not route['coverage']['passed']: failures.append('coverage')
    if rejected: failures.append('motion')
    if review.get('status') in ('rejected', 'rejected_for_training'): failures.append('visual_review')
    return {'passed': not failures, 'failures': failures, 'motion': motion, 'visual_review': review,
            'training_ready': False, 'note': 'Passing these checks permits a provisional plan; dense registration and visual acceptance are still required.'}


def withhold_region_plan(workspace, quality):
    """Invalidate stale cuts while keeping prior generated evidence available."""
    root = Path(workspace)
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    for source, destination in [(root/'regions.json', reports/'superseded_regions.json'),
                                (reports/'training_regions.csv', reports/'superseded_training_regions.csv')]:
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    atomic_json(root/'regions.json', {'schema_version': 1, 'kind': 'training-region-plan',
                'status': 'withheld', 'regions': [], 'training_ready': False, 'failures': quality['failures']})
    (reports/'training_regions.csv').write_text('status,reason\nwithheld,' + '+'.join(quality['failures']) + '\n', encoding='utf-8')
