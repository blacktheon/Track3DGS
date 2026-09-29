import sqlite3
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from track3dgs.route_rig import rig_configuration, rig_image_name, rename_feature_database
from track3dgs.trajectory import quat_to_R
from track3dgs.views import _e2p


def test_calibrated_rig_rotation_agrees_with_actual_panorama_projection():
    intr = dict(fx=50,fy=50,cx=50,cy=50,width=101,height=101)
    config = rig_configuration([-90,0,90,180],intr)[0]['cameras']
    lon,lat = np.meshgrid(np.linspace(-np.pi,np.pi,480),np.linspace(np.pi/2,-np.pi/2,240))
    directions = np.stack([np.sin(lon)*np.cos(lat),-np.sin(lat),np.cos(lon)*np.cos(lat)],-1).astype(np.float32)
    for yaw in [-90,0,90,180]:
        cam = next(c for c in config if c['image_prefix']==f'yaw_{yaw:+04d}/')
        if yaw==0:
            assert cam['ref_sensor']
            continue
        w,x,y,z = cam['cam_from_rig_rotation']
        rig_ray = quat_to_R(x,y,z,w).T @ np.array([0.,0.,1.])
        actual = _e2p(directions,yaw,100,101)[50,50]
        assert rig_ray == pytest.approx(actual,abs=.008)
        assert cam['cam_from_rig_translation'] == [0,0,0]


def test_database_rig_layout_keeps_feature_ids_but_clears_unconstrained_matches(tmp_path):
    db = tmp_path/'database.db'
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER); CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER); CREATE TABLE descriptors(image_id INTEGER,data BLOB); CREATE TABLE matches(pair_id INTEGER,data BLOB); CREATE TABLE two_view_geometries(pair_id INTEGER,data BLOB);')
        c.execute('INSERT INTO cameras VALUES(1,1,1600,1600,?,1)',(np.array([671,671,800,800],np.float64).tobytes(),))
        c.executemany('INSERT INTO images VALUES(?,?,1)',[(11,'frame_000000_y+000.jpg'),(12,'frame_000000_y+090.jpg'),(13,'frame_000015_y+090.jpg')])
        c.execute('INSERT INTO descriptors VALUES(12,?)',(b'unchanged',))
        c.execute('INSERT INTO matches VALUES(1,?)',(b'old',))
        c.execute('INSERT INTO two_view_geometries VALUES(1,?)',(b'old',))
    aliases = rename_feature_database(db,[0,90])
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT data FROM descriptors WHERE image_id=12').fetchone()[0] == b'unchanged'
        assert c.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 0
        assert c.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0] == 0
        records = c.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id').fetchall()
        assert records == [(11,'yaw_+000/frame_000000.jpg',1),(12,'yaw_+090/frame_000000.jpg',2),(13,'yaw_+090/frame_000015.jpg',2)]
    assert aliases['yaw_+090/frame_000015.jpg']=='frame_000015_y+090.jpg'
    assert rig_image_name('frame_000015_y-090.jpg')=='yaw_-090/frame_000015.jpg'


@pytest.mark.skipif(not Path(r'C:\Work\tools\colmap\bin\colmap.exe').exists(),reason='COLMAP integration requires installed executable')
def test_installed_colmap_groups_each_panorama_into_one_calibrated_frame(tmp_path):
    exe = r'C:\Work\tools\colmap\bin\colmap.exe'
    db = tmp_path/'database.db'
    subprocess.run([exe,'database_creator','--database_path',str(db)],capture_output=True,check=True)
    with sqlite3.connect(db) as c:
        for cid,yaw in enumerate([-90,0,90],1):
            c.execute('INSERT INTO cameras VALUES(?,1,101,101,?,1)',(cid,np.array([50,50,50,50],np.float64).tobytes()))
            for frame in range(2):
                c.execute('INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)',(cid*10+frame,f'yaw_{yaw:+04d}/frame_{frame:06d}.jpg',cid))
    cfg = tmp_path/'rig.json'
    cfg.write_text(json.dumps(rig_configuration([-90,0,90],dict(fx=50,fy=50,cx=50,cy=50))))
    result = subprocess.run([exe,'rig_configurator','--database_path',str(db),'--rig_config_path',str(cfg)],capture_output=True)
    assert result.returncode==0,result.stderr.decode()[-500:]
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT COUNT(*) FROM rigs').fetchone()[0]==1
        assert c.execute('SELECT COUNT(*) FROM frames').fetchone()[0]==2
        assert c.execute('SELECT COUNT(DISTINCT sensor_id) FROM frame_data WHERE frame_id=(SELECT MIN(frame_id) FROM frames)').fetchone()[0]==3
