import json
from pathlib import Path
import subprocess
import sys

from track3dgs.route_config import load_route_config


def test_toolchain_paths_are_config_relative_and_colmap_can_come_from_path(tmp_path, monkeypatch):
    from track3dgs import route_config
    (tmp_path/'video.mp4').write_bytes(b'fixture')
    (tmp_path/'mask.png').write_bytes(b'fixture')
    config = dict(schema_version=1,route_id='test',revision='r1',source_video='video.mp4',
        vehicle_mask='mask.png',sky_prior=None,mount_calibration=None,workspace='out',
        scale={'method':'unscaled'},regions=dict(core_length=100,context_length=20,start_margin=20,end_margin=20),
        toolchain={'colmap':'../tools/colmap.exe','training_python':'../gpu/python.exe'},
        reconstruction={'reuse_feature_database':'../older/database.db'})
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    result=load_route_config(path)
    assert Path(result['toolchain']['colmap'])==(tmp_path/'../tools/colmap.exe').resolve()
    assert Path(result['toolchain']['training_python'])==(tmp_path/'../gpu/python.exe').resolve()
    assert Path(result['reconstruction']['reuse_feature_database'])==(tmp_path/'../older/database.db').resolve()
    config['toolchain'].pop('colmap');path.write_text(json.dumps(config))
    assert hasattr(route_config,'shutil'), 'Tool discovery is still hard-coded to the original PC'
    monkeypatch.setattr(route_config.shutil,'which',lambda name: '/portable-tools/colmap.exe')
    assert load_route_config(path)['toolchain']['colmap']=='/portable-tools/colmap.exe'


def test_step2_without_unity_reaches_route_validation_instead_of_argparse_rejection(tmp_path):
    # No GPU work is launched: a missing route plan is the intentional boundary.
    result=subprocess.run([sys.executable,'-m','track3dgs.route_step2','--workspace',str(tmp_path),
        '--run-root',str(tmp_path/'run'),'--training-python',sys.executable],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'required: --unity-project' not in result.stderr
    assert 'regions.json' in result.stderr


def test_review_json_written_by_windows_powershell_is_readable(tmp_path):
    from track3dgs.io_utils import read_json
    path=tmp_path/'route_review.json'
    path.write_text('{"accepted_for":"Step 2"}',encoding='utf-8-sig')
    assert read_json(path)['accepted_for']=='Step 2'


def test_default_region_selection_handles_any_route_length():
    import pytest
    from track3dgs import route_regions
    assert hasattr(route_regions, 'region_indices'), 'Default regions are still hard-coded to six'
    regions=[{},{}]
    assert route_regions.region_indices(regions,None)==[0,1]
    assert route_regions.region_indices(regions,'1')==[1]
    with pytest.raises(ValueError): route_regions.region_indices(regions,'-1')
    with pytest.raises(ValueError): route_regions.region_indices(regions,'1,0')
