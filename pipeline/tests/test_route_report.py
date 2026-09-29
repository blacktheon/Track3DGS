import numpy as np
import pytest

from track3dgs.route_report import unity_preview


def test_unity_reflects_package_z_once_and_converts_camera_down_to_up():
    # OpenCV camera axes in RUB at an identity-facing camera: down=-Y, forward=-Z.
    route = {'route_id':'test','length':3,'scale':{'status':'approximate','method':'manual'},
             'coverage':{'max_gap_seconds':2},'samples':[
                 {'frame_id':'a:1','timestamp_seconds':1,'s':0,
                  'rig_to_package':[1,0,0,1, 0,-1,0,2, 0,0,-1,-3, 0,0,0,1]}]}
    m = unity_preview(route,None)['markers'][0]
    assert [m['x'],m['y'],m['z']] == [1,2,3]
    assert [m['fx'],m['fy'],m['fz']] == [0,0,1]
    assert [m['ux'],m['uy'],m['uz']] == [0,1,0]


def test_internal_core_boundary_belongs_only_to_next_region():
    route = {'length':20,'scale':{'status':'approximate','method':'manual'},
             'coverage':{},'samples':[{'frame_id':'a:1','timestamp_seconds':1,'s':10,
                                     'rig_to_package':np.eye(4).reshape(-1).tolist()}]}
    def region(index,start,end,last):
        return {'cell_index':index,'region_id':f'cell_{index:03d}','core_s':[start,end],
                'core_end_inclusive':last,'context_s':[0,20],'context_time_seconds':[0,10]}
    data = unity_preview(route,{'regions':[region(0,0,10,False),region(1,10,20,True)]})
    assert data['markers'][0]['region'] == 1
