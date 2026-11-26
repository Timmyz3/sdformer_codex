#!/usr/bin/env python3
"""One fixed two-frame H1 segment. Default preflight; same accepted H0 W only."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode=True
import h1_core as core
HERE=core.HERE
OWN=('h1_core.py','run_h1.py','finalize_h1.py','test_h1_cpu.py','source_pins.json')
require,sha=core.require,core.sha


def prepare(a,h0,h0runner,runtime):
    require(0<a.wall_minutes<=30 and not a.output_dir.exists(),'Fresh output and bounded30minute segment required')
    require(sha(a.h0_result)==a.h0_result_sha256,'Actual H0 result SHA differs')
    # Accepted new H0 preflight has no forwards or calibration. Reuse its legal
    # sources/data/BN/admission checks, never a historical collector prepare.
    parent_args=argparse.Namespace(inputs=a.h0_inputs,inputs_sha256=a.h0_inputs_sha256,cpu_checks=a.h0_cpu_checks,
                                   output_dir=a.output_dir,wall_minutes=a.wall_minutes)
    p=h0runner.prepare(parent_args,runtime)
    previous=json.loads(a.h0_result.read_text())
    inherited_audit=core.validate_h0_receipt(previous,p,h0)
    artifacts=core.validate_prediction_artifacts(a.h0_result,previous,h0.ANCHORS)
    cpu_sha=runtime[3].checked_cpu_receipt(a.cpu_checks,[HERE/n for n in OWN])
    h1_plan=core.plan(p['rows'],a.segment);rows=p['rows'][2*(a.segment-1):2*a.segment]
    frames={f['target_id']:f for f in previous['frames']}
    binding={'schema':'flow_h1_binding_v1','source_sha256':{n:sha(HERE/n) for n in OWN},'cpu_checks_sha256':cpu_sha,
             'h0_result_sha256':a.h0_result_sha256,'h0_inputs_sha256':a.h0_inputs_sha256,'h0_cpu_checks_sha256':sha(a.h0_cpu_checks),
             'h0_binding':previous['binding'],'inherited_H0_audit':inherited_audit,'H8_ids':[r['target_id'] for r in p['rows']],
             'H8_sequences':[r['sequence'] for r in p['rows']],'W_descriptor':p['descriptors']['W'],
             'W_weights_sha256':p['inputs']['states']['W']['weights']['sha256'],'BN_buffers_sha256':p['inputs']['bn_buffers']['sha256'],
             'h0_prediction_artifacts':artifacts,'actions':core.actions(),'all_segments_full_forward_budget':625,
             'actual_new_masks':624,'extra_duals':1,'segment':a.segment,'segment_ids':[r['target_id'] for r in rows],
             'additional_descriptor_work':{'extra_PSN1_calls_all8':96,'extra_PSN1_elements_all8':29859840,
                 'PSN1_temporal_MACs_nominal_all8':298598400,'extra_full_model_forwards':0,'GT_used':False,'runtime_cost_claim':False},
             'segment_forward_budget':len(h1_plan),'forward_plan':h1_plan,'wall_minutes':a.wall_minutes,
             'GPU_lock':str(runtime[3].GPU_LOCK),'GT_used_to_generate_masks':False,'repeated_H0_or_CC_forwards':0,
             'new_BN_calibration_forwards':0,'training_forwards':0,'flow_specific_increment_claim':False,'hardware_benefit_claim':False}
    return {'parent':p,'rows':rows,'H8_rows':p['rows'],'h0_frames':frames,'binding':binding,'h0_result':previous}


def execute(a,p,result,h0,runtime):
    import numpy as np
    import torch
    r13,collector,probe,parent,r9,r8=runtime
    require(not torch.cuda.is_initialized(),'CUDA before shared GPU lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261016)
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'H1 fixed segment budget exhausted')
    inherited=p['parent'];b=inherited['b'];context=inherited['context'];base=inherited['base'];scope=inherited['scope'];loader=inherited['loader']
    native,trainer,metrics,dev32,hashes,config,saved,side,regions,old_plan=context
    model,functional=loader.load_model(b,context,base,torch,device)
    bn=r13.apply_fresh_bn(model,inherited['files']['bn_buffers'],inherited['files']['bn_receipt'],inherited['manifest'],h0.MEMBER_SHA,scope,base,torch)
    buffers={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n.endswith(h0.BUFFERS)}
    require(len(buffers)==72 and bn['BN_count']==78 and bn['MLP_BN_count']==24,'Same legal BN required')
    asset=inherited['inputs']['states']['W']['weights'];require(sha(asset['path'])==asset['sha256'],'W changed after preflight')
    payload=torch.load(asset['path'],map_location='cpu',weights_only=True)
    loaded=h0.install_recovered(model,payload,inherited['descriptors']['W'],buffers,torch);del payload
    require(loaded==p['h0_result']['recovered_state_loads']['W'],'Actual H1 W differs from actual H0 W')
    targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
    require(tuple(shapes[h0.TARGET])==(10,1,30,40,384),'Production action geometry differs')
    require({tuple(r['tile']) for r in h0.tile_geometry(shapes[h0.TARGET]) if r['complete']}==set(core.TILES),'Twelve complete candidates differ')
    before=r13.state_digest(model);bn_before=base.bn_signature(model,torch)
    require(bn_before==p['h0_result']['BN_scope']['BN_signature_before_inference'],'Actual H0/H1 BN state differs')
    cache={};frames={};predictions={};rows={r['target_id']:r for r in p['rows']}
    for row in p['rows']:
        deadline();evidence={};expected={'events' if k=='input' else k:hashes[rel]['sha256'] for k,rel in native.target_files(row).items()}
        arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        require(evidence==p['h0_frames'][row['target_id']]['input_evidence'],'Actual arrays differ from H0 input hashes')
        cache[row['target_id']]=arrays
        frames[row['target_id']]={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'arms':{},
                                  'H0_CC_reference':{'result_sha256':a.h0_result_sha256,'prediction_sha256':p['h0_frames'][row['target_id']]['arms']['CC']['prediction_sha256'],
                                                     'execution':p['h0_frames'][row['target_id']]['arms']['CC']['execution']}}
        predictions[row['target_id']]={}
    meters={act['action']:metrics.MetricAccumulator() for act in core.actions()}
    result.update(status='H1_SEGMENT_RUNNING',full_forward_count=0,training_performed=False,new_BN_calibration_performed=False,
                  recovered_state_load=loaded,BN_scope=bn,runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                  'cudnn':torch.backends.cudnn.version(),'device_name':torch.cuda.get_device_name(device),'TF32':False,'AMP':False,'precision':'float32'})
    def infer(item):
        deadline();chunk=base.prepare_input(cache[item['target_id']]['events'],device,torch)
        capture={} if item['mode']=='actual' and item['tiles']==[[0,0]] else None
        output=core.forward(h0,runtime,model,functional,targets,shapes,chunk,tuple(tuple(t) for t in item['tiles']),merge,torch,item['mode'],deadline,capture_sink=capture)
        del chunk
        require(len(output[0])==4 and all(tuple(f.shape)==(1,2,480,640) and f.dtype==torch.float32 for f in output[0]),'Four full-resolution FP32 heads required')
        if capture is not None:
            frame=frames[item['target_id']];require('predecision_u' not in frame,'One u capture per member')
            u=capture['u'];u_sha=merge.tensor_sha(u)
            require(u_sha==frame['H0_CC_reference']['execution']['branch_records'][h0.TARGET]['input_sha256'],'Captured u must equal saved H0 CC predecision input')
            scores,cost,audit=core.ordinary_descriptors(u,targets[h0.TARGET],probe,merge,torch)
            u_path=a.output_dir/('U_'+item['target_id']+'.npz');np.savez_compressed(u_path,u=u.detach().cpu().numpy())
            frame.update(predecision_u={'file':u_path.name,'file_sha256':sha(u_path),'payload_sha256':u_sha,'shape_TBHWC':list(u.shape),
                                        'dtype':'float32','source_action':item['action'],'full_forward_reused':True},
                         ordinary_scores=scores,ordinary_scores_sha256=core.json_sha(scores),ordinary_score_cost=cost,ordinary_score_actual_audit=audit)
            del u,capture
        result['full_forward_count']+=1
        require(result['full_forward_count']==item['segment_forward_id'],'Fixed segment forward order differs')
        result['last_completed_forward']=dict(item)
        parent.write(a.output_dir/'results.json',result)
        return output
    def save_actual(item,output):
        row=rows[item['target_id']];arrays=cache[item['target_id']]
        masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions);require(len(masks)==16,'All16 valid-GT regions required')
        prediction=output[0][-1][0].numpy().copy();predictions[row['target_id']][item['action']]=prediction
        frames[row['target_id']]['arms'][item['action']]={'prediction_sha256':merge.tensor_sha(output[0][-1][0]),'execution':output[1],
                                                       'metrics':meters[item['action']].update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks)}
    coarse={identity:frame['arms']['CC']['execution'] for identity,frame in p['h0_frames'].items()}
    records,pair=core.drive(p['binding']['forward_plan'],coarse,infer,save_actual,lambda x,y:h0.check_pair(x,y,torch),h0.check_prefix)
    require(r13.state_digest(model)==before and base.bn_signature(model,torch)==bn_before,'W/BN changed during H1')
    require(all(model.get_submodule(n) is m for n,m in targets.items()),'Original modules not restored')
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),'Unremoved hooks')
    require(result['full_forward_count']==p['binding']['segment_forward_budget'],'Incomplete segment forward count')
    names={a['action'] for a in core.actions()}
    for row in p['rows']:
        identity=row['target_id'];require(set(predictions[identity])==names and 'predecision_u' in frames[identity],'All78 actual masks and ordinary descriptors must be retained')
        path=a.output_dir/('H1_'+identity+'.npz');np.savez_compressed(path,**predictions[identity])
        frames[identity].update(prediction_file=path.name,prediction_file_sha256=sha(path))
    result.update(status='H1_SEGMENT_COMPLETE',completed_targets=2,forward_records=records,first_pair_dual_audit=pair,
                  frames=[frames[row['target_id']] for row in p['rows']],segment_summaries={k:m.summary() for k,m in meters.items()},
                  recovered_W_unchanged=True,BN_buffers_unchanged=True,original_modules_restored=True,all_hooks_removed=True,
                  wall_seconds=time.monotonic()-started,max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  max_reserved_bytes=torch.cuda.max_memory_reserved(device),flow_specific_increment_claim=False,hardware_benefit_claim=False,
                  interpretation='Two preassigned exploratory H8 frames only. Full H1 requires all4 segments. No oracle ranking, fitting, k4 or hardware claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('h0-result','h0-inputs'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--h0-cpu-checks',type=Path,required=True);parser.add_argument('--cpu-checks',type=Path,required=True)
    parser.add_argument('--segment',type=int,choices=(1,2,3,4),required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--run-inference',action='store_true')
    a=parser.parse_args();h0,h0runner,runtime=core.load_h0();parent=runtime[3];p=prepare(a,h0,h0runner,runtime)
    result={'schema':'flow_h1_finite_action_segment_v1','status':'H1_PREFLIGHT_COMPLETE','binding':p['binding'],'inference_executed':False}
    if not a.run_inference:
        a.output_dir.mkdir(parents=True,exist_ok=False);parent.write(a.output_dir/'results.json',result);return
    require(parent.GPU_LOCK.parent.is_dir(),'Shared GPU lock directory must already exist')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('H1 fixed <=30minute segment exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            result['inference_executed']=True;execute(a,p,result,h0,runtime)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
