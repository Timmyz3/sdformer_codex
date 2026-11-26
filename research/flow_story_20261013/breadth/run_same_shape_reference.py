#!/usr/bin/env python3
"""Two-target original-D numerical check; defaults to preflight, no training.

Compare all-tile masked reference with kept-only execution using replayed masks.
Separately measure native dense -> all-keep tiled dense AEE conversion.
"""
import argparse
import contextlib
import copy
import fcntl
import inspect
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROBE_SHA = '06672b84bd686d5cfd4a888fef7c2b6521b2d0c8f71361495447d1d59727bd57'
POLICIES_SHA = 'fe734513e488157458602cbd4ce50235e5db01883370039bacafae9d6bd8ffb7'
import hashlib


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, msg):
    if not ok: raise ValueError(msg)


require(sha(HERE/'probe_actual_skip.py') == PROBE_SHA, 'Pinned diagnostic loader changed')
import probe_actual_skip as probe


def compare(a, b, torch):
    delta = (a.double()-b.double()).abs()
    norm = float(torch.linalg.vector_norm(a.double()))
    error = float(torch.linalg.vector_norm(a.double()-b.double()))
    return {'numerically_equal': bool(torch.equal(a,b)), 'max_abs_error': float(delta.max()),
        'mean_abs_error': float(delta.mean()), 'different_elements': int((delta != 0).sum()),
        'elements': a.numel(), 'relative_L2_error': error/norm if norm else None,
        'allclose_atol1e_5_rtol1e_5': bool(torch.allclose(a,b,atol=1e-5,rtol=1e-5))}


def load_model(a, context, ev, torch, device):
    native, trainer, metrics, rows, hashes, config, run, side, regions, plan = context
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    require(saved.get('schema_version') == ev.SCHEMA and saved.get('contract') == run['contract'] and
        saved.get('contract_sha256') == side['contract_sha256'] and saved.get('provenance') == run['provenance'] and
        saved.get('effective_config') == run['effective_config'] and
        saved['training']['completed_optimizer_steps'] == side['completed_optimizer_steps'], 'Checkpoint payload differs')
    state = saved['model_state_dict']
    require(all(isinstance(v, torch.Tensor) and (not v.is_floating_point() or bool(torch.isfinite(v).all()))
                for v in state.values()), 'Invalid checkpoint tensor')
    sys.path.insert(0, str(a.source_root)); os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for obj in (MS_SpikingformerFlowNet_en4, PSN, configure_snn_backend):
        require(Path(inspect.getfile(obj)).resolve().is_relative_to(a.source_root), 'Unpinned native import')
    config = copy.deepcopy(config)
    config['swin_transformer']['input_size'] = [288,384] if a.geometry == 'crop_center9' else [480,640]
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(state, strict=True); del state, saved
    model.to(device); functional.set_step_mode(model,'m'); configure_snn_backend(model,device,config,PSN)
    ev.apply_bn(model,'D',torch)
    return model, functional


def run(a, context, ev):
    import torch
    import same_shape_tile_mlp as tile
    torch.set_num_threads(a.cpu_threads); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(a.device)
    require(device.type in ('cpu','cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    if device.type == 'cuda': torch.cuda.set_device(device)
    require(sha(HERE/'policies.py') == POLICIES_SHA, 'Pinned policies changed')
    policies = ev.import_file('same_shape_policies', HERE/'policies.py'); policies.configure_prior_root(a.prior_root)
    model, functional = load_model(a,context,ev,torch,device)
    native,trainer,metrics,rows,hashes,config,contract,side,regions,plan = context
    before = ev.bn_signature(model,torch); ys,xs = metrics.crop_slices(a.geometry)
    start = time.monotonic(); meters = {}; layer_count = sum(type(m).__name__ == 'MS_Spiking_Mlp' for m in model.modules())
    result = {'schema':'flow_same_shape_reference_v1','status':'RUNNING','native_preflight':plan,
        'source_sha256':{p.name:sha(p) for p in (Path(__file__),Path(tile.__file__),HERE/'policies.py',HERE/'probe_actual_skip.py')},
        'BN_policy':'D: original checkpoint running stats, no recalibration',
        'targets':[r['target_id'] for r in rows[:a.reference_targets]], 'keep_fraction':0.75,
        'pair_mask_contract':'reference online policy; actual replays every reference mask, does not rescore',
        'precision':'FP32; TF32 off; no autocast','all_keep_shortcut':False,
        'scope':'two-target train-side execution diagnosis; not quality conclusion, recovery or accelerator speed',
        'runtime':{'torch':str(torch.__version__),'device':str(device),'cuda':torch.version.cuda},
        'frames':[], 'summaries':{}, 'input_sha256':{}, 'GPU_inference_executed':device.type=='cuda'}
    def deadline(): require(time.monotonic()-start < a.wall_minutes*60, 'Wall budget exhausted')
    def forward(chunk, mode=None, policy=None):
        deadline(); functional.reset_net(model)
        cm = tile.same_shape_mlp_model(model,mode,policy,audit_hashes=True) if mode else contextlib.nullcontext([])
        with cm as wrappers:
            output = model(chunk)['flow'][-1]
            require(bool(torch.isfinite(output).all()), 'Nonfinite prediction')
            output = output.detach().cpu().clone()
            records = [r for w in wrappers for r in w.records]
        if mode: require(len(records)==layer_count, 'Missing/duplicate native layer execution')
        return output,records
    def measure(label, row, output, gt, masks):
        require(tuple(output.shape)==(1,2)+gt.shape[1:], 'Unexpected flow shape')
        meter = meters.setdefault(label,metrics.MetricAccumulator())
        return meter.update(row['target_id'],row['sequence'],output[0].numpy(),gt,masks)
    ev.write_result(a.output,result)
    with torch.inference_mode():
        for row in rows[:a.reference_targets]:
            expected,evidence = {},{}
            for name,rel in native.target_files(row).items():
                native.checked_path(a.data_root,rel); expected['events' if name=='input' else name]=hashes[rel]['sha256']
            arrays,_ = metrics.load_arrays(a.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
            result['input_sha256'][row['target_id']] = evidence
            masks = metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions) if regions else {'all_valid':arrays['mask']}
            masks = {k:v[ys,xs] for k,v in masks.items()}; gt=arrays['gt'][:,ys,xs]
            chunk = ev.prepare_input(arrays['events'][:,ys,xs],device,torch)
            dense,_ = forward(chunk)
            all_policy = policies.make_policy('uniform',1.0,target_id=row['target_id'])
            tiled,all_records = forward(chunk,'tiled_all_then_mask',all_policy)
            frame = {'target_id':row['target_id'],'native_dense':measure('native_dense',row,dense,gt,masks),
                'tiled_all_keep':measure('tiled_all_keep',row,tiled,gt,masks),
                'dense_to_all_tiled_conversion':compare(dense,tiled,torch),'all_keep_execution':all_records,'pairs':[]}
            result['frames'].append(frame); ev.write_result(a.output,result)
            for name in a.reference_policies:
                policy = policies.make_policy(name,0.75,seed=0,target_id=row['target_id'])
                reference,ref_records = forward(chunk,'tiled_all_then_mask',policy)
                replay = tile.ReplayPolicy(ref_records)
                actual,actual_records = forward(chunk,'tiled_kept_only',replay); replay.complete()
                pair = {'policy':name,'reference':measure(name+'_reference',row,reference,gt,masks),
                    'actual':measure(name+'_actual',row,actual,gt,masks),'output_comparison':compare(reference,actual,torch),
                    'same_masks':[r['keep'] for r in ref_records]==[r['keep'] for r in actual_records],
                    'same_layer_input_values':[r['input_sha256'] for r in ref_records]==[r['input_sha256'] for r in actual_records],
                    'same_layer_output_values':[r['output_sha256'] for r in ref_records]==[r['output_sha256'] for r in actual_records],
                    'same_layer_output_strides':[r['output_stride'] for r in ref_records]==[r['output_stride'] for r in actual_records],
                    'reference_execution':ref_records,'actual_execution':actual_records,
                    'reference_policy_records':ev.compact_policy_records(policy.records)}
                frame['pairs'].append(pair)
                print(__import__('json').dumps({'target':row['target_id'],'policy':name,'comparison':pair['output_comparison']}),flush=True)
                ev.write_result(a.output,result)
    result['BN_buffers_unchanged'] = before == ev.bn_signature(model,torch)
    require(result['BN_buffers_unchanged'], 'BN mutated')
    result['summaries'] = {k:m.summary() for k,m in meters.items()}
    result.update(status='REFERENCE_COMPLETE',wall_seconds=time.monotonic()-start,
        all_replayed_pairs_exact=all(p['output_comparison']['numerically_equal'] and p['same_layer_input_values'] and
            p['same_layer_output_values'] for f in result['frames'] for p in f['pairs']))
    ev.write_result(a.output,result)


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-targets',type=int,choices=(1,2),default=2)
    parser.add_argument('--reference-policies',default='uniform,seeded_random,tp_factored')
    parser.add_argument('--gpu-lock',type=Path)
    custom,remaining=parser.parse_known_args(argv); ev=probe.evaluator(); a=ev.parse_args(remaining)
    for key,value in vars(custom).items(): setattr(a,key,value)
    a.reference_policies=a.reference_policies.split(',')
    require(a.reference_policies and len(a.reference_policies)==len(set(a.reference_policies)) and
        set(a.reference_policies)<=set(('uniform','seeded_random','tp_factored','tp_literal')), 'Unknown/duplicate policy')
    require(not(a.run_inference and a.device.startswith('cuda')) or a.gpu_lock is not None,'Explicit shared GPU lock required')
    context=ev.preflight(a)
    if not a.run_inference:
        ev.write_result(a.output,{'status':'REFERENCE_PREFLIGHT_PASS_NO_INFERENCE','native_preflight':context[-1],
            'reference_targets':a.reference_targets,'policies':a.reference_policies,'script_sha256':sha(__file__)})
        return 0
    lock=a.gpu_lock.open('a') if a.gpu_lock else contextlib.nullcontext()
    try:
        with lock as handle:
            if a.gpu_lock: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(a,context,ev)
    except BaseException as exc:
        ev.write_result(a.output.with_suffix(a.output.suffix+'.failure.json'),{'status':'REFERENCE_INCOMPLETE',
            'error':str(exc),'traceback':traceback.format_exc()}); raise
    return 0


if __name__=='__main__': raise SystemExit(main())
