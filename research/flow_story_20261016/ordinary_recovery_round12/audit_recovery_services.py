#!/usr/bin/env python3
"""Independent complete four-recovery service/state audit; stdlib, no GPU.

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
import zipfile
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


def audit(paths,digests,output,partial=False):
    require(len(paths)==len(digests) and (1<=len(paths)<4 if partial else len(paths)==4) and not output.exists(),
            'Explicit raws and fresh output required; partial accepts one to three arms only')
    g,ledger,shapes,members=preflight()
    scope={n for n in shapes if '.layers.0.' in n or '.layers.2.' in n}
    require(len(scope)==8,'Stage02 inventory')
    counts=Counter();runs={};reference=None;raw_sources={};archives=[]
    helper=HERE.parents[1]/'flow_story_20261014/damage_predictor_round6/audit_complete_collection.py'
    require(sha(helper)=='bf37de4584054a86cdc6f7480596b9d75948c0b665d494b5947be396cb3986a0','Archive helper pin')
    spec=importlib.util.spec_from_file_location('_r12_archive_reader',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    for path,digest in zip(paths,digests):
        require(len(digest)==64 and sha(path)==digest,'Explicit recovery raw SHA')
        raw_sources[str(path)]=digest
        raw=json.loads(path.read_text());arm=raw['arm']
        require(arm in ARMS and arm not in runs and raw['schema']=='flow_round12_fair_recovery_v1' and
                raw['status']=='ROUND12_FAIR_RECOVERY_COMPLETE' and raw['mode']=='recover','Completed unique arm only')
        require(raw['inference_executed'] and raw['training_performed'] and raw['completed_optimizer_steps']==200 and
                raw['smoke_weights_discarded'] is False and set(raw['evaluations'])=={'initial','endpoint'} and
                raw['weights_file']=='weights.pt','Recovery scope')
        require(all(raw[k] for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),
                'State/hook closure')
        require(not raw['hardware_benefit_claim'] and not raw['flow_specific_increment_claim'],'No quality/hardware claim')
        binding=raw['binding']
        for field,name in (('runner_sha256','run_recovery.py'),('wrapper_sha256','trainable_representative_mlp.py'),
            ('runner_checks_sha256','runner_cpu_results.json'),('source_pins_sha256','source_pins.json'),
            ('membership_sha256','check_membership_v1.json'),('selection_sha256','selection_manifest_v1.json')):
            require(binding[field]==sha(HERE/name),'Current source/member binding '+field)
        require(binding['arm']==arm and binding['mode']=='recover' and binding['fit_ids']==members['groups']['fit'] and
                binding['check_ids']==members['groups']['check'] and binding['allowed_parameter_count']==7252064,'Train/check/parameter binding')
        require(binding['checkpoint_sha256']=='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30' and
                binding['loss_sha256']=='92aa1451469690ad89c16b6954c972d3c460689f5cb765662b4eff7fc7dec122','Weights/loss')
        require(binding['expected_training_forwards']==200 and binding['expected_audit_forwards']==(4 if arm=='native' else 6)
                and binding['expected_evaluation_forwards']==40 and 0<binding['wall_minutes']<=30,'Bounded recovery budget')
        expected_smokes={'results.json:'+a:sha(HERE/('smoke_'+a+'_v1/results.json')) for a in ARMS}
        require(binding['smoke_receipts_sha256']==expected_smokes,'Allfour accepted smokes bound')
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
        require(set(raw['training_input_evidence'])==set(members['groups']['fit']),'Exact fit20 inputs')
        evaluation_arms={};evaluation_records=[r for r in raw['forward_records'] if r['purpose'].startswith('evaluation_')]
        cursor=0
        for label in ('initial','endpoint'):
            evaluation=raw['evaluations'][label];modes=(arm,) if label=='initial' else ARMS
            require([f['target_id'] for f in evaluation['frames']]==members['groups']['check'] and
                    set(evaluation['summaries'])==set(modes),'Eight fixed check frames and all modes')
            for frame in evaluation['frames']:
                require(set(frame['arms'])==set(modes),'Complete evaluation matrix cell')
                h.archive_check(path.parent,frame,'prediction_file',archives)
                with zipfile.ZipFile(path.parent/frame['prediction_file']) as z:
                    require(set(z.namelist())=={mode+'.npy' for mode in modes},'NPZ exact mode inventory')
                    for mode in modes:
                        record=evaluation_records[cursor];cursor+=1;item=frame['arms'][mode]
                        require(record['purpose']=='evaluation_'+label and record['target_id']==frame['target_id'] and
                                record['execution']==item['execution'],'Every endpoint binds a real full forward')
                        evaluation_arms[record['forward_id']]=mode
                        with z.open(mode+'.npy') as stream:
                            header=h.read_header(stream)
                            require(header==dict(descr='<f4',fortran_order=False,shape=(2,480,640)),'Complete dense prediction header')
                            hashed=hashlib.sha256();size=0
                            for block in iter(lambda:stream.read(1024*1024),b''):hashed.update(block);size+=len(block)
                            require(size==2*480*640*4 and hashed.hexdigest()==item['prediction_sha256'],'Opaque prediction payload SHA')
                        counts['evaluated_prediction_payloads']+=1
        require(cursor==len(evaluation_records)==40,'Exactly40 actual evaluated endpoints per state')
        weights=path.parent/raw['weights_file']
        require(weights.resolve().parent==path.parent.resolve() and sha(weights)==raw['weights_sha256'],'Local complete saved checkpoint SHA')
        perrun=Counter();byid={};training=[]
        for rec in raw['forward_records']:
            fid=rec['forward_id'];require(fid==len(byid)+1,'Contiguous real forward ids');byid[fid]=rec
            ex=rec['execution'];purpose=rec['purpose'];isnew=purpose=='training' or purpose.startswith('audit_new_')
            mode=ex.get('mode','actual')
            if not isnew and ex['layers']:mode=next(iter(ex['layers'].values()))['mode']
            exec_arm=evaluation_arms.get(fid,arm)
            expected_layers=set() if exec_arm=='native' else scope
            require(set(ex['layers'])==expected_layers and set(ex['all12_observed_fc_rows'])==set(shapes),'All12FC/eight wrapper inventory')
            total=0
            for name,shape in shapes.items():
                t,b,h,w,c=shape;rows=t*b*h*w
                if name in expected_layers:
                    layer=ex['layers'][name];fine=exec_arm=='FF' or (exec_arm=='fine_s2b2' and name==TARGET)
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
            expected=609600 if mode=='dual_reference' else {'native':558000,'FF':558000,'CC':153600,'fine_s2b2':164200}[exec_arm]
            require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total==expected,'PerFC total')
            counts['FC_rows_each_total']+=total;counts['full_forwards']+=1;perrun[purpose]+=1
            if isnew:require(ex['full_flow_count']==4 and ex['gradient_enabled']==(purpose=='training'),'New complete flow/grad context')
            if purpose=='training':training.append(rec)
        require(len(byid)==raw['full_forward_count']==(244 if arm=='native' else 246) and len(training)==200,'Recovery actual forward count')
        require(perrun['training']==200 and all(perrun[p+'_'+e]==1 for p in ('audit_new','audit_old_actual') for e in ('initial','final')),
                'Two audit endpoints')
        if arm!='native':require(all(perrun['audit_old_dual_'+e]==1 for e in ('initial','final')),'Two old dual audits')
        require(len(perrun)==(7 if arm=='native' else 9) and perrun['evaluation_initial']==8 and perrun['evaluation_endpoint']==32,
                'No hidden forward purpose/endpoint count')
        require([s['step'] for s in raw['steps']]==list(range(1,201)) and
                [s['target_id'] for s in raw['steps']]==members['groups']['fit']*10,'Exact fit20 repeated10/200 updates')
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
                coverage['complete']==bool(all(layers.values()) and all(categories.values())) and coverage['per_step_zero_is_allowed'],
                'Real200-step coverage, no per-step nonzero requirement')
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
        runs[arm]=dict(raw_sha256=digest,full_forwards=raw['full_forward_count'],training_updates=200,
            full_entry_peak_bytes=raw['full_entry_peak_bytes'],training_peak_bytes=dict(allocated=raw['max_allocated_bytes'],reserved=raw['max_reserved_bytes']),
            post_first_199_step_seconds=dict(min=min(times),median=ordered[len(ordered)//2],max=max(times)),
            training_seconds=raw['training_seconds'],wall_seconds=raw['wall_seconds'],gradient_coverage=coverage,
            updated_parameter_tensors=len(raw['updated_parameter_names']),all4flows_8branches_actual_reference_exact=True,
            weights_file=str(weights),weights_sha256=raw['weights_sha256'],weights_bytes=weights.stat().st_size)
    expected=Counter()
    for arm in runs:
        expected.update(full_forwards=244 if arm=='native' else 246,optimizer_updates=200,
            wrapper_records=192 if arm=='native' else 1904,all12_paired_FC_records=(244 if arm=='native' else 246)*12,
            FC_rows_each_total={'native':129766400,'FF':130985600,'CC':45252800,'fine_s2b2':47500000}[arm],
            evaluated_prediction_payloads=40)
    require((partial or set(runs)==set(ARMS)) and counts==expected,'Actual arm service counts')
    if not partial:
        require(counts==Counter(full_forwards=982,optimizer_updates=800,wrapper_records=5904,all12_paired_FC_records=11784,
                FC_rows_each_total=353504800,evaluated_prediction_payloads=160),'Complete four-recovery service counts')
    require(len(archives)==16*len(runs),'All prediction NPZs for audited arms')
    receipt=dict(schema='flow_round12_independent_four_recovery_service_audit_v1',
        status='PASS_PARTIAL_ACTUAL_RECOVERY_SERVICES_ONLY' if partial else 'PASS_ACTUAL_RECOVERY_SERVICES_ONLY',partial=partial,
        source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'freeze_smoke_v1.json',HERE/'runner_cpu_results.json',GEOMETRY)},
        raw_sha256=raw_sources,counts=dict(counts),runs=runs,common_start_data_recipe_state_exact_within_audited_arms=True,
        prediction_archives=dict(files=len(archives),bytes=sum(a['bytes'] for a in archives),records=archives),
        no_GPU_or_model_execution_by_reviewer=True,new_check_quality_used=False,all_four_recoveries_audited=not partial,
        quality_or_hardware_benefit_claim=False,
        boundary='Verified only explicitly supplied completed200-step runs and their real endpoints, archived predictions and checkpoint bytes. Four-arm closure requires partial=false. Tensor/optimizer checkpoint inspection is separate; no AEE comparison or net-cycle conclusion.')
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return dict(status=receipt['status'],counts=dict(counts),output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,action='append',default=[])
    parser.add_argument('--results-sha256',action='append',default=[])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--partial',action='store_true',help='Audit one to three completed arms without asserting four-arm closure')
    args=parser.parse_args()
    if not args.results:
        require(not args.results_sha256 and args.output is None and not args.partial,'Preflight args')
        preflight();print('PREFLIGHT_SOURCE_CONTRACT_PASS_NO_RECOVERY_RESULT_READ')
    else:
        require(args.output,'Output required');print(json.dumps(audit(args.results,args.results_sha256,args.output,args.partial)))


if __name__=='__main__':main()
