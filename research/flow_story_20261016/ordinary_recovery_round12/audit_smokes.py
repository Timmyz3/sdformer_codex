#!/usr/bin/env python3
"""Independent complete four-smoke service/state receipt audit; stdlib, no GPU.

Only completed raw files with explicit SHA are accepted. No quality inference.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ARMS=('native','FF','CC','fine_s2b2')
TARGET='sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
FREEZE_SHA='59a9cbb14fbaed1e6f2ed69a2878daad8a029522552d8ff37169fbca8dc23424'
GEOMETRY=HERE.parents[1]/'flow_story_20261014/structured_merge/audit_structured_merge_service.py'
GEOMETRY_SHA='abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def preflight():
    freeze=HERE/'freeze_smoke_v1.json';require(sha(freeze)==FREEZE_SHA,'Fixed execution freeze')
    value=json.loads(freeze.read_text())
    require(len(value['files'])==13,'Frozen file inventory')
    for path,digest in value['files'].items():require(sha(ROOT/path)==digest,'Frozen source differs: '+path)
    require(sha(GEOMETRY)==GEOMETRY_SHA,'Pure service source')
    spec=importlib.util.spec_from_file_location('_r12_independent_geometry',GEOMETRY)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    require(sha(g.MODULE)==g.MODULE_SHA,'Frozen original wrapper')
    constants=g.runner_constants()
    shapes={'sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp'.format(s,b):list(shape)
            for s,shape in constants['STAGE_SHAPES'].items() for b in range(constants['DEPTHS'][s])}
    cpu=json.loads((HERE/'runner_cpu_results.json').read_text())
    require(cpu['passed'] and cpu['tests_run']==8 and not cpu['GPU_initialized'] and not cpu['full_model_run']
            and not cpu['errors'] and not cpu['failures'],'Final CPU contract')
    for path,digest in cpu['source_sha256'].items():require(sha(path)==digest,'CPU source')
    pins=json.loads((HERE/'source_pins.json').read_text())['sources']
    for key,pin in pins.items():require(sha(ROOT/pin['path'])==pin['sha256']==cpu['inherited_sources'][key],'Inherited pin '+key)
    return g,g.extracted_source_ledger(),shapes,json.loads((HERE/'check_membership_v1.json').read_text())


def audit(paths,digests,output):
    require(len(paths)==len(digests)==4 and not output.exists(),'Four explicit raws and fresh output required')
    g,ledger,shapes,members=preflight()
    scope={n for n in shapes if '.layers.0.' in n or '.layers.2.' in n}
    require(len(scope)==8,'Stage02 inventory')
    counts=Counter();runs={};reference=None;raw_sources={}
    for path,digest in zip(paths,digests):
        require(len(digest)==64 and sha(path)==digest,'Explicit smoke raw SHA')
        raw_sources[str(path)]=digest
        raw=json.loads(path.read_text());arm=raw['arm']
        require(arm in ARMS and arm not in runs and raw['schema']=='flow_round12_fair_recovery_v1' and
                raw['status']=='ROUND12_SMOKE_COMPLETE' and raw['mode']=='smoke','Completed unique arm only')
        require(raw['inference_executed'] and raw['training_performed'] and raw['completed_optimizer_steps']==5 and
                raw['smoke_weights_discarded'] and not raw.get('evaluations') and 'weights_file' not in raw,'Smoke scope')
        require(all(raw[k] for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),
                'State/hook closure')
        require(not raw['hardware_benefit_claim'] and not raw['flow_specific_increment_claim'],'No quality/hardware claim')
        binding=raw['binding']
        for field,name in (('runner_sha256','run_recovery.py'),('wrapper_sha256','trainable_representative_mlp.py'),
            ('runner_checks_sha256','runner_cpu_results.json'),('source_pins_sha256','source_pins.json'),
            ('membership_sha256','check_membership_v1.json'),('selection_sha256','selection_manifest_v1.json')):
            require(binding[field]==sha(HERE/name),'Current source/member binding '+field)
        require(binding['arm']==arm and binding['mode']=='smoke' and binding['fit_ids']==members['groups']['fit'] and
                binding['check_ids']==[] and binding['allowed_parameter_count']==7252064,'Train/check/parameter binding')
        require(binding['checkpoint_sha256']=='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30' and
                binding['loss_sha256']=='92aa1451469690ad89c16b6954c972d3c460689f5cb765662b4eff7fc7dec122','Weights/loss')
        require(binding['expected_training_forwards']==5 and binding['expected_audit_forwards']==(4 if arm=='native' else 6)
                and binding['expected_evaluation_forwards']==0 and 0<binding['wall_minutes']<=30,'Bounded smoke budget')
        names=raw['allowed_parameter_names']
        require(len(names)==len(set(names))==80 and raw['allowed_parameter_count']==7252064 and
                {n.split('.mlp.')[0]+'.mlp' for n in names}==scope,'Full8MLP white list')
        suffixes={'fc1.weight','fc2.weight','sn1.spiking_neuron.weight','sn1.spiking_neuron.bias',
                  'sn2.spiking_neuron.weight','sn2.spiking_neuron.bias','bn1.norm_layer.weight','bn1.norm_layer.bias',
                  'bn2.norm_layer.weight','bn2.norm_layer.bias'}
        require(set(names)=={layer+'.'+key for layer in scope for key in suffixes},'Actual native parameter categories')
        require(set(raw['updated_parameter_names'])<=set(names) and
                all(any(n.startswith(layer+'.') for n in raw['updated_parameter_names']) for layer in scope),'All8 layers updated')
        bn=raw['BN_scope'];sig=bn['BN_signature_before_inference'];frozen={n for n in sig if '.mlp.' in n}
        require(bn['scope']=='MLP_CMA176' and bn['BN_count']==len(sig)==78 and bn['MLP_BN_count']==len(frozen)==24 and
                set(bn['frozen_BN_names'])==set(bn['MLP_BN_names'])==frozen and
                set(bn['current_stat_BN_names'])==set(sig)-frozen and len(set(sig)-frozen)==54,'BN partition')
        for name,s in sig.items():
            require(not s['training'] and s['tracking']==(name in frozen) and
                    all((s[k] is not None)==(name in frozen) for k in ('running_mean','running_var','num_batches_tracked')),'BN mode/buffers')
        runtime=raw['runtime']
        require(runtime['device_index']==0 and not runtime['TF32'] and not runtime['AMP'] and
                runtime['grad_enabled_training'] and runtime['threads']==2 and 'cudnn' in runtime,'Runtime graph')
        common=dict(initial=raw['initial_model_state_sha256'],frozen=raw['frozen_state_before_sha256'],names=names,
                    BN=bn['BN_signature_before_inference'],runtime=runtime,recipe=binding['recipe'],
                    input_evidence=raw['training_input_evidence'],loss=raw['native_loss'])
        if reference is None:reference=common
        else:require(common==reference,'Four-arm common start/config/data differs')
        require(set(raw['training_input_evidence'])==set(members['groups']['fit'][:5]),'First five fit-only inputs')
        perrun=Counter();byid={};training=[]
        for rec in raw['forward_records']:
            fid=rec['forward_id'];require(fid==len(byid)+1,'Contiguous real forward ids');byid[fid]=rec
            ex=rec['execution'];purpose=rec['purpose'];isnew=purpose=='training' or purpose.startswith('audit_new_')
            mode=ex.get('mode','actual')
            if not isnew and arm!='native':mode=next(iter(ex['layers'].values()))['mode']
            expected_layers=set() if arm=='native' else scope
            require(set(ex['layers'])==expected_layers and set(ex['all12_observed_fc_rows'])==set(shapes),'All12FC/eight wrapper inventory')
            total=0
            for name,shape in shapes.items():
                t,b,h,w,c=shape;rows=t*b*h*w
                if name in expected_layers:
                    layer=ex['layers'][name];fine=arm=='FF' or (arm=='fine_s2b2' and name==TARGET)
                    mask=[[fine]*math.ceil(w/9) for _ in range(math.ceil(h/9))]
                    require(layer['shape_TBHWC']==shape and layer['fine_mask']==mask and layer['mode']==mode,'Whole fixed mask/real shape')
                    if isnew:
                        require(mode=='actual' and not layer['temporal_pooling'] and not layer['external_residual_added_here'],'New selected producer graph')
                        geom=g.geometry(shape,mask);rows=t*b*(geom['fine']+geom['reps'])
                        require(layer['native_mlp_calls']==math.ceil(h/9)*math.ceil(w/9),'Actual complete tile calls')
                        require(layer['executed_fc_rows_per_layer']==rows,'New independently derived producer rows')
                    else:
                        g.audit_record(layer,shape,4*c,ledger);rows=layer['executed_fc_rows_per_layer']
                    counts['wrapper_records']+=1
                require(ex['all12_observed_fc_rows'][name]==dict(fc1_rows=rows,fc2_rows=rows),'Actual FC hooks')
                total+=rows;counts['all12_paired_FC_records']+=1
            expected=609600 if mode=='dual_reference' else {'native':558000,'FF':558000,'CC':153600,'fine_s2b2':164200}[arm]
            require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total==expected,'PerFC total')
            counts['FC_rows_each_total']+=total;counts['full_forwards']+=1;perrun[purpose]+=1
            if isnew:require(ex['full_flow_count']==4 and ex['gradient_enabled']==(purpose=='training'),'New complete flow/grad context')
            if purpose=='training':training.append(rec)
        require(len(byid)==raw['full_forward_count']==(9 if arm=='native' else 11) and len(training)==5,'Smoke actual forward count')
        require(perrun['training']==5 and all(perrun[p+'_'+e]==1 for p in ('audit_new','audit_old_actual') for e in ('initial','final')),
                'Two audit endpoints')
        if arm!='native':require(all(perrun['audit_old_dual_'+e]==1 for e in ('initial','final')),'Two old dual audits')
        require(len(perrun)==(5 if arm=='native' else 7),'No hidden forward purpose')
        require([s['step'] for s in raw['steps']]==list(range(1,6)) and
                [s['target_id'] for s in raw['steps']]==members['groups']['fit'][:5],'Exact five optimizer updates')
        layers=Counter();categories=Counter()
        for step,rec in zip(raw['steps'],training):
            require(step['target_id']==rec['target_id'] and step['service']==rec['execution'] and step['valid_GT_pixels']>0,'Step/forward binding')
            ga=step['gradient_audit']
            require(ga['all_allowed_gradients_present_finite'] and ga['frozen_gradients_None'] and
                    set(ga['per_layer_squared_L2'])==scope and set(ga['category_squared_L2'])=={'FC','PSN','BN_affine'},'Gradient coverage fields')
            require(all(math.isfinite(v) and v>=0 for v in list(ga['per_layer_squared_L2'].values())+list(ga['category_squared_L2'].values())),
                    'Finite gradient diagnostics')
            for key,value in ga['per_layer_squared_L2'].items():layers[key]+=int(value>0)
            for key,value in ga['category_squared_L2'].items():categories[key]+=int(value>0)
            require(math.isfinite(step['task_loss']) and math.isfinite(step['preclip_grad_norm']) and step['preclip_grad_norm']>=0,'Finite optimizer loss/norm')
            counts['optimizer_updates']+=1
        coverage=raw['gradient_coverage']
        require(dict(layers)==coverage['layer_steps_nonzero'] and dict(categories)==coverage['category_steps_nonzero'] and
                all(layers.values()) and all(categories.values()) and coverage['complete'] and coverage['per_step_zero_is_allowed'],'Real five-step nonzero coverage')
        for endpoint in ('initial','final'):
            pair=raw[endpoint+'_numerical_pair']
            require(pair['all_flow_outputs_exact'] and pair['flow_count']==4 and pair['all8_inputs_exact'] and pair['all8_branches_exact'],
                    'Full4flow/all8 numerical audit')
            new=next(r['execution'] for r in raw['forward_records'] if r['purpose']=='audit_new_'+endpoint)
            old=next(r['execution'] for r in raw['forward_records'] if r['purpose']=='audit_old_actual_'+endpoint)
            require(pair['new_execution']==new and pair['old_actual_execution']==old,'Actual numerical audit binding')
            require(set(new['capture_input_sha256'])==scope,'All8 captured input hashes')
            if arm=='native':require(pair['native_has_no_tiled_dual'] and 'old_dual_execution' not in pair,'Native dual boundary')
            else:
                dual=next(r['execution'] for r in raw['forward_records'] if r['purpose']=='audit_old_dual_'+endpoint)
                require(pair['old_dual_execution']==dual and pair['actual_dual_all_flows_branches_exact'],'Dual numerical binding')
                for name in scope:
                    require(old['layers'][name]['input_sha256']==dual['layers'][name]['input_sha256']==new['capture_input_sha256'][name] and
                            old['layers'][name]['output_sha256']==dual['layers'][name]['output_sha256'],'Actual old dual/branch hashes')
        require(raw['full_entry_peak_bytes']['allocated']>=raw['max_allocated_bytes'] and
                raw['full_entry_peak_bytes']['reserved']>=raw['max_reserved_bytes'],'Peak scope')
        times=[s['synchronized_train_step_seconds'] for s in raw['steps'][1:]]
        require(all(math.isfinite(v) and v>0 for v in times),'Finite synchronized step times')
        ordered=sorted(times)
        runs[arm]=dict(raw_sha256=digest,full_forwards=raw['full_forward_count'],training_updates=5,
            full_entry_peak_bytes=raw['full_entry_peak_bytes'],training_peak_bytes=dict(allocated=raw['max_allocated_bytes'],reserved=raw['max_reserved_bytes']),
            post_first_four_step_seconds=dict(min=min(times),median=(ordered[1]+ordered[2])/2,max=max(times)),
            training_seconds=raw['training_seconds'],wall_seconds=raw['wall_seconds'],gradient_coverage=coverage,
            updated_parameter_tensors=len(raw['updated_parameter_names']),all4flows_8branches_actual_reference_exact=True)
    require(set(runs)==set(ARMS) and counts==Counter(full_forwards=42,optimizer_updates=20,
            wrapper_records=264,all12_paired_FC_records=504,FC_rows_each_total=16561800),'Complete four-smoke service counts')
    receipt=dict(schema='flow_round12_independent_four_smoke_audit_v1',status='PASS_ENGINEERING_SMOKES_ONLY',
        source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'freeze_smoke_v1.json',HERE/'runner_cpu_results.json',GEOMETRY)},
        raw_sha256=raw_sources,counts=dict(counts),runs=runs,common_start_data_recipe_state_exact=True,
        no_GPU_or_model_execution_by_reviewer=True,new_check_quality_used=False,full_recovery_performed=False,
        quality_or_hardware_benefit_claim=False,
        boundary='Verified fixed runner completed4 real smokes, full row hooks and numerical/gradient/state assertions. No checkpoint saved for independent tensor reinspection; no AEE/new-check or net-cycle conclusion.')
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return dict(status=receipt['status'],counts=dict(counts),output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,action='append',default=[])
    parser.add_argument('--results-sha256',action='append',default=[])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.results:
        require(not args.results_sha256 and args.output is None,'Preflight args')
        preflight();print('PREFLIGHT_SOURCE_CONTRACT_PASS_NO_SMOKE_RESULT_READ')
    else:
        require(args.output,'Output required');print(json.dumps(audit(args.results,args.results_sha256,args.output)))


if __name__=='__main__':main()
