"""H1 finite complete-tile interventions. No data-dependent mask generation."""
import contextlib
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys
import types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
H0DIR=HERE.parent/'implementation_h0'
TILES=tuple((y,x) for y in range(3) for x in range(4))


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_h0():
    for name,digest in json.loads((HERE/'source_pins.json').read_text())['accepted_h0_sources'].items():
        require(sha(H0DIR/name)==digest,'Accepted H0 source differs: '+name)
    sys.path.insert(0,str(H0DIR))
    h0=importlib.import_module('h0_core')
    require(Path(h0.__file__).resolve()==H0DIR/'h0_core.py','Wrong H0 core import')
    runner=h0.load_file(H0DIR/'run_h0.py','_h1_accepted_h0_entry_helpers')
    return h0,runner,h0.load_frozen()


def mask_name(tiles):
    return 'k%d_' % len(tiles)+'_'.join('r%dc%d' % tile for tile in tiles)


def actions():
    masks=[(tile,) for tile in TILES]+list(itertools.combinations(TILES,2))
    require(len(masks)==78 and len(set(masks))==78,'Exactly12 singleton and66 pair masks')
    return [{'action':mask_name(mask),'tiles':[list(t) for t in mask],'k':len(mask)} for mask in masks]


def plan(rows,segment=None):
    require(len(rows)==8 and len({r['target_id'] for r in rows})==8,'Fixed H8 required')
    result=[]
    for index,row in enumerate(rows):
        for action in actions():
            item=dict(action,target_id=row['target_id'],sequence=row['sequence'],segment=index//2+1,mode='actual')
            result.append(item)
            if index==0 and action['tiles']==[[0,0],[0,1]]:result.append(dict(item,mode='dual_reference'))
    require(len(result)==625,'Exactly625 additional full forwards')
    result=[dict(item,global_forward_id=index+1) for index,item in enumerate(result)]
    if segment is not None:
        require(segment in (1,2,3,4),'Segment must be1..4')
        result=[item for item in result if item['segment']==segment]
        require(len(result)==(157 if segment==1 else 156),'Fixed two-frame segment budget differs')
    counts={}
    for item in result:
        counts[item['segment']]=counts.get(item['segment'],0)+1
        item['segment_forward_id']=counts[item['segment']]
    return result


def validate_tiles(tiles,shape):
    require(isinstance(tiles,tuple) and len(tiles) in (1,2) and tuple(sorted(tiles))==tiles and len(set(tiles))==len(tiles),
            'Canonical unique singleton/pair tuple required')
    complete={(r,c) for r in range(shape[2]//9) for c in range(shape[3]//9)}
    require(set(tiles)<=set(TILES) and set(tiles)<=complete,'Only the twelve real complete target tiles are legal')


def forward(h0,runtime,model,functional,targets,shapes,chunk,tiles,merge,torch,mode='actual',deadline=lambda:None,capture_sink=None):
    validate_tiles(tiles,shapes[h0.TARGET]);require(mode in ('actual','dual_reference'),'Invalid execution mode')
    probe=runtime[2]
    @contextlib.contextmanager
    def wrap(model,targets,shapes,unused,mode,merge,torch):
        require(unused==(),'Mask only from fixed enumeration')
        wrappers={}
        with contextlib.ExitStack() as stack:
            for name in targets:
                if h0.selected_layer(name):
                    mask=h0.mask_for(shapes[name],name,'CC',torch)
                    if name==h0.TARGET:
                        for y,x in tiles:mask[y,x]=True
                    wrappers[name]=stack.enter_context(merge.replace_mlp(model,name,mask,3,mode,expected_shape=shapes[name]))
            require(len(wrappers)==8 and h0.TARGET in wrappers,'Exactly stage02 eight MLPs')
            yield wrappers
    namespace=dict(probe.forward_probe.__globals__);namespace.update(wrap=wrap,TARGET=h0.TARGET)
    fn=types.FunctionType(probe.forward_probe.__code__,namespace,'_h1_frozen_forward',probe.forward_probe.__defaults__,probe.forward_probe.__closure__)
    flows,execution,branches,captured=fn(model,functional,targets,shapes,chunk,(),merge,torch,mode=mode,capture=True,capture_u=capture_sink is not None,deadline=deadline)
    geometry=h0.tile_geometry(shapes[h0.TARGET]);increments={tuple(r['tile']):r['fine_extra_fc_rows'] for r in geometry}
    expected=h0.expected_rows(shapes,'CC',torch,mode=='dual_reference')+(0 if mode=='dual_reference' else sum(increments[t] for t in tiles))
    require(execution['summed_fc1_rows']==execution['summed_fc2_rows']==expected,'Independent H1 producer row formula differs')
    inputs={n:r['input_sha256'] for n,r in execution['layers'].items()}
    require(set(branches)==set(inputs) and len(branches)==8,'Complete eight branch/input capture required')
    execution.update(action=mask_name(tiles),tiles=[list(t) for t in tiles],k=len(tiles),mode=mode,
                     head_records=[{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(torch.isfinite(v).all()),'sha256':merge.tensor_sha(v)} for v in flows],
                     branch_records={n:{'input_sha256':inputs[n],'output_sha256':merge.tensor_sha(v),'shape':list(v.shape)} for n,v in branches.items()},
                     target_geometry=geometry,tail_rule='All8 real partial target tiles stay coarse; every valid output GT pixel remains evaluated.')
    require(all(model.get_submodule(n) is m for n,m in targets.items()),'Original modules not restored')
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),'Unremoved hooks')
    if capture_sink is not None:
        require(not capture_sink and set(captured)=={'u'},'Single explicit predecision input capture required')
        require(tuple(captured['u'].shape)==tuple(shapes[h0.TARGET]) and merge.tensor_sha(captured['u'])==inputs[h0.TARGET],
                'Captured predecision u differs from actual target input')
        capture_sink['u']=captured['u']
    return flows,execution,branches,inputs


def ordinary_descriptors(u,native,probe,merge,torch):
    """Frozen ordinary scoring, separately metered; no full-model invocation."""
    calls=[]
    def count(module,args):
        calls.append({'shape':list(args[0].shape),'elements':args[0].numel()})
    handle=native.sn1.register_forward_pre_hook(count)
    try:
        with torch.no_grad():scores,cost=probe.feature_scores(u,native,merge,torch)
    finally:handle.remove()
    complete=tuple((r,c) for r in range(u.shape[2]//9) for c in range(u.shape[3]//9))
    require([tuple(row['tile']) for row in scores]==list(complete),'Frozen score order/complete tiles differ')
    require(len(calls)==len(complete)==cost['diagnostic_only_extra_psn1_calls'],'Actual extra PSN1 calls differ')
    require(all(call['shape']==[10,1,9,9,u.shape[-1]] for call in calls),'Extra PSN1 shape differs')
    require(sum(call['elements'] for call in calls)==cost['diagnostic_only_extra_psn1_elements'],'Actual extra PSN1 elements differ')
    require(all(set(row)=={'tile',*probe.SCORE_NAMES} for row in scores),'Seven ordinary score fields required')
    return scores,cost,{'observed_extra_psn1_calls':len(calls),'observed_extra_psn1_elements':sum(call['elements'] for call in calls),
                       'call_shapes':[call['shape'] for call in calls],'full_model_forwards':0,'temporary_hook_removed':True,
                       'scope':'Additional descriptor work only; reductions, reads and PSN nominal work are not cycle measurements.'}


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_descriptor_artifacts(result_path,value,target):
    import math
    import numpy as np
    names=('amplitude_abs_mean','feature_rms','pooling_residual_mse','spatial_neighbor_abs_difference',
           'temporal_neighbor_abs_difference','psn1_activity','psn1_spatial_neighbor_hamming')
    artifacts=[]
    for frame in value['frames']:
        record=frame['predecision_u'];filename=record['file'];require(Path(filename).name==filename,'Local u basename required')
        path=Path(result_path).parent/filename;require(sha(path)==record['file_sha256'],'Predecision u archive SHA differs')
        with np.load(path,allow_pickle=False) as values:
            require(values.files==['u'],'Exactly one u tensor per archive')
            u=values['u'];require(u.shape==(10,1,30,40,384) and u.dtype==np.float32 and bool(np.isfinite(u).all()),'Native predecision u shape/dtype/finite differs')
            digest=hashlib.sha256(u.tobytes()).hexdigest()
        require(record['shape_TBHWC']==[10,1,30,40,384] and record['dtype']=='float32' and record['payload_sha256']==digest and
                digest==frame['H0_CC_reference']['execution']['branch_records'][target]['input_sha256'],'Saved u differs from actual H0 CC prefix')
        require(record['source_action']==mask_name(((0,0),)) and record['full_forward_reused'] is True,'u must come from first existing singleton')
        scores=frame['ordinary_scores'];require([tuple(row['tile']) for row in scores]==list(TILES),'Twelve complete ordinary-score rows required')
        require(frame['ordinary_scores_sha256']==json_sha(scores),'Ordinary score payload SHA differs')
        require(all(set(row)=={'tile',*names} and all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) for k in names) for row in scores),
                'Finite seven-field ordinary scores required')
        cost=frame['ordinary_score_cost'];audit=frame['ordinary_score_actual_audit']
        require(cost['diagnostic_only_extra_psn1_calls']==12 and cost['diagnostic_only_extra_psn1_elements']==3732480 and
                cost['diagnostic_only_extra_psn1_temporal_MACs_nominal']==37324800 and cost['extra_fc_calls']==0 and
                cost['score_reductions_and_feature_reads_counted_in_cycles'] is False,'Descriptor diagnostic cost differs')
        require(audit['observed_extra_psn1_calls']==12 and audit['observed_extra_psn1_elements']==3732480 and
                audit['call_shapes']==[[10,1,9,9,384]]*12 and audit['full_model_forwards']==0 and audit['temporary_hook_removed'] is True,
                'Actual descriptor PSN1 service audit differs')
        artifacts.append({'target_id':frame['target_id'],'path':str(path),'sha256':sha(path),'u_payload_sha256':digest,'ordinary_scores_sha256':frame['ordinary_scores_sha256']})
    return artifacts


def drive(segment_plan,coarse_by_id,infer,save_actual,pair_checker,prefix_checker,expected_heads=4):
    require(segment_plan and len({p['segment'] for p in segment_plan})==1,'One fixed segment at a time')
    segment=segment_plan[0]['segment'];require(len(segment_plan)==(157 if segment==1 else 156),'Incomplete segment plan')
    records=[];pair=None;previous=None
    for item in segment_plan:
        output=infer(item);require(len(output[0])==expected_heads,'Complete native heads required')
        output[1]['same_H0_CC_prefix_audit']=prefix_checker(output[1],coarse_by_id[item['target_id']])
        if item['mode']=='actual':save_actual(item,output);previous=(item,output)
        else:
            require(segment==1 and previous is not None and previous[0]['target_id']==item['target_id'] and previous[0]['action']==item['action'],
                    'Only immediately matched first pair has an extra dual')
            pair=pair_checker(previous[1],output);previous=None
        records.append(dict(item,execution=output[1]))
    require((pair is not None)==(segment==1),'Exactly first-segment pair dual audit required')
    return records,pair


def validate_h0_receipt(value,prepared,h0):
    require(value.get('schema')=='flow_h0_action_screen_v1' and value.get('status')=='H0_69_FORWARD_COMPLETE','Completed actual H0 required')
    require(value.get('full_forward_count')==69 and value.get('completed_targets')==8 and value.get('inference_executed') is True,
            'Complete H0 actual count/coverage required')
    for key in ('all_recovered_states_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'):
        require(value.get(key) is True,'H0 closure missing: '+key)
    require(value.get('training_performed') is False and value.get('new_BN_calibration_performed') is False,'Unexpected H0 training/calibration')
    stable=('inputs_sha256','cpu_checks_sha256','source_sha256','common','H8_ids','condition','W_arm','scientific_scope','region_sha256',
            'all_eight_audit_receipt','independent_execution_audit','completed_analysis_admission_audit','independent_execution_admission_audit')
    require(all(value['binding'][k]==prepared['binding'][k] for k in stable),'H0 state/runtime/admission/source binding differs')
    require(value['data_evidence']==prepared['data_evidence'],'H0 actual input/timestamp source differs')
    require(set(value['recovered_state_loads'])=={'W','N','F'},'H0 recovered loads incomplete')
    for role,descriptor in prepared['descriptors'].items():
        record=value['recovered_state_loads'][role]
        require(record['state_sha256']==descriptor['final_state_sha256'] and record['strict_full_state_load'] is True and record['common_BN_exact'] is True,
                'H0 used different recovered state or BN')
    expected=h0.forward_plan(prepared['rows'])
    require(len(value['forward_records'])==len(expected)==69,'H0 execution ledger incomplete')
    totals={'native':558000,'matched_N':558000,'FF':558000,'matched_F':558000,'CC':153600,'P':164200,'F12':162240,'T8':155560}
    for row,item in zip(value['forward_records'],expected):
        require(all(row[k]==v for k,v in item.items()),'H0 forward order/identity/arm differs')
        ex=row['execution'];total=609600 if item['mode']=='dual_reference' else totals[item['anchor']]
        require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total,'H0 actual producer row audit differs')
        require(len(ex['branch_records'])==8 and len(ex['head_records'])==4 and all(h['shape']==[1,2,480,640] and h['dtype']=='torch.float32' and h['finite'] is True for h in ex['head_records']),
                'H0 complete heads/branches missing')
    require(set(value['first_frame_dual_audits'])==set(h0.DUALS),'Five H0 dual audits required')
    require(all(v.get('head_count')==4 and all(v.get(k) is True for k in ('all_flow_heads_exact','all8_branch_inputs_exact','all8_branches_exact'))
                for v in value['first_frame_dual_audits'].values()),'H0 full-head/branch dual audit failed')
    require([f['target_id'] for f in value['frames']]==[r['target_id'] for r in prepared['rows']],'H0 member order differs')
    for frame,row in zip(value['frames'],prepared['rows']):
        require(frame['sequence']==row['sequence'] and set(frame['arms'])==set(h0.ANCHORS),'H0 all anchor/sequence coverage differs')
        for anchor,arm in frame['arms'].items():
            ledger=[r['execution'] for r in value['forward_records'] if r['target_id']==row['target_id'] and r['anchor']==anchor and r['mode']=='actual']
            require(ledger==[arm['execution']],'H0 prediction/execution binding differs')
        require(frame['arms']['CC']['state']=='W','CC must be same recovered W')
    return {'completed_H0_69_bound':True,'all8_H8_input_state_CC_sources_bound':True,'no_H0_forwards_repeated':True}


def validate_prediction_artifacts(result_path,value,names,torch=None):
    """Hash every retained final prediction; no quality ranking or recomputation."""
    import numpy as np
    artifacts=[]
    for frame in value['frames']:
        filename=frame['prediction_file'];require(Path(filename).name==filename,'Local prediction basename required')
        path=Path(result_path).parent/filename;require(sha(path)==frame['prediction_file_sha256'],'Prediction archive bytes differ')
        with np.load(path,allow_pickle=False) as arrays:
            require(set(arrays.files)==set(names),'Prediction archive action set differs')
            for name in names:
                pred=arrays[name]
                require(pred.shape==(2,480,640) and pred.dtype==np.float32 and bool(np.isfinite(pred).all()),'Full-resolution finite FP32 prediction required')
                require(hashlib.sha256(pred.tobytes()).hexdigest()==frame['arms'][name]['prediction_sha256'],'Prediction payload SHA differs')
        artifacts.append({'target_id':frame['target_id'],'path':str(path),'sha256':sha(path),'predictions':len(names)})
    return artifacts
