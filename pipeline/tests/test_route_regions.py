import pytest

from track3dgs.route_regions import plan_regions


def route_fixture():
    return {'length': 285, 'coverage': {'passed': True}, 'samples': [
        {'s': i*5., 'timestamp_seconds': i*2., 'source_pts_seconds': i*2.+8,
         'frame_id': str(i), 'rig_to_package': [1,0,0,i*5., 0,1,0,0, 0,0,1,0, 0,0,0,1]}
        for i in range(58)]}


def test_cores_cover_playable_range_once_context_overlaps_and_holdouts_stay_grouped():
    route = route_fixture()
    cameras = [{'camera_id': f'{i}/{yaw}', 'frame_id': str(i), 's': i*5.,
                'split': 'held_out' if i % 10 == 9 else 'train'}
               for i in range(58) for yaw in range(8)]
    result = plan_regions(route, cameras, {'core_length':100, 'context_length':20,
                                          'start_margin':20, 'end_margin':20})
    regions = result['regions']
    assert [r['core_s'] for r in regions] == [[20,120], [120,220], [220,265]]
    assert [r['context_s'] for r in regions] == [[0,140], [100,240], [200,285]]
    assert regions[0]['core_time_seconds'] == pytest.approx([8,48])
    assert regions[0]['context_source_pts_seconds'] == pytest.approx([8,64])
    assert not regions[0]['core_end_inclusive']
    assert regions[-1]['core_end_inclusive']
    training = set().union(*(set(r['train_camera_ids']) for r in regions))
    holdout = set().union(*(set(r['held_out_camera_ids']) for r in regions))
    assert not training.intersection(holdout)
    assert all(f'29/{yaw}' in holdout for yaw in range(8))


def test_short_route_and_failed_coverage_do_not_produce_training_regions():
    route = route_fixture()
    with pytest.raises(ValueError):
        plan_regions(route, [], {'core_length':100,'context_length':20,'start_margin':200,'end_margin':200})
    route['coverage']['passed'] = False
    with pytest.raises(ValueError, match='coverage'):
        plan_regions(route, [], {'core_length':100,'context_length':20,'start_margin':20,'end_margin':20})


def test_video_intervals_interpolate_travel_without_spreading_stopped_time():
    route = route_fixture()
    route['length'] = 10
    route['samples'] = [dict(route['samples'][i],s=s,timestamp_seconds=t,source_pts_seconds=t+8)
                        for i,(s,t) in enumerate(zip([0,5,5,5,10],[0,1,20,30,31]))]
    plan = plan_regions(route,[],{'core_length':2.5,'context_length':0,'start_margin':0,'end_margin':0})
    r = plan['regions']
    assert r[0]['context_time_seconds'] == [0,.5]
    assert r[1]['context_time_seconds'] == [.5,30]  # Inclusive context includes the complete stop.
    assert r[1]['core_time_seconds'] == [.5,1]  # Half-open core gives the stop to the next region.
    assert r[2]['core_time_seconds'] == [1,30.5]
    assert r[3]['context_time_seconds'] == [30.5,31]
    assert r[3]['context_source_pts_seconds'] == [38.5,39]
