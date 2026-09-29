from pathlib import Path
import pytest
from track3dgs.route_step2 import process_regions


def test_coordinator_is_sequential_and_checks_pair_only_after_both_exist(tmp_path):
    events=[]
    regions=[{'cell_index':i,'region_id':f'cell_{i:03d}'} for i in range(3)]
    def train(region): events.append(('train',region['cell_index']))
    def process(region): events.append(('clean',region['cell_index']))
    def qc(region): events.append(('qc',region['cell_index']))
    def seam(index): events.append(('seam',index))
    def publish(region): events.append(('publish',region['cell_index']))
    process_regions(regions,train,process,qc,seam,publish,tmp_path/'state.json')
    assert events==[('train',0),('clean',0),('qc',0),('publish',0),
        ('train',1),('clean',1),('qc',1),('seam',1),('publish',1),
        ('train',2),('clean',2),('qc',2),('seam',2),('publish',2)]


def test_failure_keeps_earlier_outputs_and_records_partial_state(tmp_path):
    events=[]
    def train(region):
        events.append(region['cell_index'])
        if region['cell_index']==1: raise RuntimeError('test GPU failure')
    noop=lambda _:None
    with pytest.raises(RuntimeError,match='test GPU failure'):
        process_regions([{'cell_index':i,'region_id':str(i)} for i in range(3)],train,noop,noop,noop,noop,tmp_path/'state.json')
    import json
    state=json.loads((tmp_path/'state.json').read_text())
    assert events==[0,1] and state['completed']==['0'] and state['status']=='failed'
    assert state['active_region']=='1' and state['error']=='test GPU failure'
