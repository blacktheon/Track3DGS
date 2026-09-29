import numpy as np

from track3dgs.route_qc import choose_views, masked_metrics, camera_matrices


def test_qc_selects_held_out_panorama_groups_instead_of_training_neighbours():
    cameras=[dict(camera_id=f'{t}/{yaw}',frame_id=str(t),s=t,split=split,yaw_degrees=yaw)
             for t,split in [(0,'train'),(2,'held_out'),(9,'held_out'),(14,'train')]
             for yaw in [-90,0,90,180]]
    selected=choose_views(cameras,[1,8,9])
    assert {c['frame_id'] for c in selected}=={'2','9'}
    assert len(selected)==8 and all(c['split']=='held_out' for c in selected)


def test_projection_uses_real_camera_pose_and_scaled_intrinsics():
    T=np.eye(4);T[:3,3]=[10,2,-3]
    vm,K,w,h=camera_matrices({'camera_to_package':T.ravel().tolist()},
        dict(fx=400,fy=500,cx=800,cy=600,width=1600,height=1200),800)
    np.testing.assert_allclose(vm@T,np.eye(4))
    np.testing.assert_allclose(K,[[200,0,400],[0,250,300],[0,0,1]])
    assert (w,h)==(800,600)


def test_masked_metrics_exclude_sky_and_report_holes():
    gt=np.zeros((2,2,3));pred=np.zeros_like(gt);pred[0]=1
    mask=np.array([[False,False],[True,True]])
    alpha=np.array([[0,0],[.1,.9]])
    result=masked_metrics(pred,gt,mask,alpha)
    assert result['psnr_db']>=100 and result['foreground_low_alpha_fraction']==.5
