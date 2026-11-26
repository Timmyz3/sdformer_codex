#!/usr/bin/env python3
"""Independent R11 real-service/alias audit, stdlib only; no quality or GPU.

Default checks pinned prelaunch contracts. Full audit requires completed raw SHA.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parents[1] / 'flow_story_20261014'
GEOMETRY = OLD/'structured_merge/audit_structured_merge_service.py'
HELPERS = OLD/'damage_predictor_round6/audit_complete_collection.py'
BN = OLD/'damage_allocation_probe/hardware_contract_complete_bn_audit.json'
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
PREFIX = tuple('sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp'.format(s,b)
               for s,b in ((0,0),(0,1),(2,0),(2,1),(2,2)))
POLICIES = ('B5_task','B5_MSE','rich37_task','rich37_MSE','PCA11_task','PCA11_MSE','flow11_task','flow11_MSE',
            'B5_small4_task','B5_small4_MSE','fit_position_mean','fixed_uniform_index3','amplitude_abs_mean',
            'pooling_residual_mse','psn1_activity','psn1_spatial_neighbor_hamming')
ANCHORS = ('native','FF','CC','fine_s2b2')
PINS = {
    HERE/'run_joint.py':'b4b4ee3dd24923bb8cb6d171ac8ab975ca7f76964b5858bb058f747f2a1a9044',
    HERE/'source_pins.json':'9d53ce8df09227a2da2dade4925b5799ec0a07134ec0aef9297936889f23c205',
    HERE/'membership_v1.json':'f1f893714d3d4428f36e029a5ea0cc601ddf948241d3b34668c0b35b887b85c2',
    HERE/'inference_models_v1.json':'c56af012cf4ed212a9c50131ee3e82d046604ff1debbac9afcf16445b253edbd',
    GEOMETRY:'abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10',
    HELPERS:'bf37de4584054a86cdc6f7480596b9d75948c0b665d494b5947be396cb3986a0',
    BN:'9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567',
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def preflight():
    for path, expected in PINS.items(): require(sha(path)==expected,'Prelaunch pin changed: '+str(path))
    pins = json.loads((HERE/'source_pins.json').read_text())['sources']
    for item in pins.values(): require(sha(ROOT/item['path'])==item['sha256'],'Inherited source pin')
    cpu = json.loads((HERE/'runner_cpu_results.json').read_text())
    require(cpu['passed'] and cpu['tests_run']==7 and not cpu['GPU_initialized'] and not cpu['full_model_run']
            and not cpu['errors'] and not cpu['failures'],'CPU service contract')
    for path, expected in cpu['source_sha256'].items(): require(sha(path)==expected,'CPU current source binding')
    for field,name in (('source_pins_sha256','source_pins.json'),('membership_sha256','membership_v1.json'),
                       ('models_sha256','inference_models_v1.json')):
        require(cpu[field]==sha(HERE/name),'CPU '+field)
    membership = json.loads((HERE/'membership_v1.json').read_text())
    members = membership['groups']['check']
    require(len(members)==len(set(members))==8 and not set(members)&set(membership['excluded_ids']), 'Fresh identity membership')
    g = load(GEOMETRY,'_r11_independent_geometry'); h = load(HELPERS,'_r11_archive_helpers')
    require(sha(g.MODULE)==g.MODULE_SHA,'Original wrapper source')
    constants = g.runner_constants()
    shapes = {'sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp'.format(s,b):list(shape)
              for s,shape in constants['STAGE_SHAPES'].items() for b in range(constants['DEPTHS'][s])}
    require(len(shapes)==12 and shapes[TARGET]==[10,1,30,40,384],'Actual geometry')
    return g,h,g.extracted_source_ledger(),shapes,members,pins


def audit(path, digest, output):
    require(not output.exists(),'Fresh output required')
    require(len(digest)==64 and sha(path)==digest,'Explicit complete raw SHA')
    g,h,ledger,shapes,members,pins = preflight()
    raw = json.loads(path.read_text())
    require(raw['schema']=='flow_round11_frozen_joint_budget_v1' and
            raw['status']=='ROUND11_FROZEN_JOINT_BUDGET_COMPLETE' and raw['completed_targets']==len(raw['frames'])==8,'Completed R11 only')
    require(raw['inference_executed'] and not raw['training_performed'] and not raw['hardware_benefit_claim'],'Run scope')
    require(all(raw[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'State closure')
    binding = raw['binding']
    for field,name in (('runner_sha256','run_joint.py'),('runner_checks_sha256','runner_cpu_results.json'),
                       ('source_pins_sha256','source_pins.json'),('membership_sha256','membership_v1.json'),
                       ('inference_models_sha256','inference_models_v1.json')):
        require(binding[field]==sha(HERE/name),'Runtime '+field)
    require(binding['target']==TARGET and binding['target_ids']==members and binding['full_forward_maximum']==293 and
            binding['main_logical_arm_count']==288 and binding['audit_forward_count']==5 and binding['budgets']==[2,4]
            and binding['policy_names']==list(POLICIES) and binding['anchors']==list(ANCHORS),'Fixed runtime contract')
    require(binding['checkpoint_sha256']=='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30','Checkpoint')
    old = json.loads((ROOT/pins['r10_result']['path']).read_text())
    require(binding['reference_r10_sha256']==pins['r10_result']['sha256'] and
            binding['origin_decisions_sha256']==pins['r10_decisions']['sha256'],'R10 bindings')
    require(all(raw['runtime'].get(k)==v for k,v in old['runtime'].items()) and
            set(raw['runtime'])==set(old['runtime'])|{'NumPy','selector_device'} and
            isinstance(raw['runtime']['NumPy'],str) and bool(raw['runtime']['NumPy']) and
            raw['runtime']['selector_device']=='CPU NumPy folded FP32' and
            raw['original_state_sha256']==old['original_state_sha256'],'FP32 runtime/state')
    sig = raw['BN_scope']['BN_signature_before_inference']
    require(sig==old['BN_scope']['BN_signature_before_inference'],'BN signature')
    frozen = {n for n in sig if '.mlp.' in n}; external=set(sig)-frozen; bn=raw['BN_scope']
    require(len(frozen)==24 and len(external)==54 and bn['scope']=='MLP_CMA176' and bn['BN_count']==78 and
            bn['MLP_BN_count']==24 and set(bn['frozen_BN_names'])==frozen and set(bn['current_stat_BN_names'])==external
            and bn['non_bn_state_unchanged'],'Private/current BN partition')
    for name,s in sig.items():
        require(not s['training'] and s['tracking']==(name in frozen) and
                (s['running_mean'] is not None)==(name in frozen) and (s['running_var'] is not None)==(name in frozen),'BN mode')
    scope={n for n in shapes if '.layers.0.' in n or '.layers.2.' in n}
    counts=Counter(); snapshots=[]; frames={f['target_id']:f for f in raw['frames']}
    require([f['target_id'] for f in raw['frames']]==members,'Completed member order')
    records=raw['forward_records']
    require([r['forward_id'] for r in records]==list(range(1,len(records)+1)) and len(records)==raw['full_forward_count']<=293,'Unique contiguous actual forward IDs')
    byid={r['forward_id']:r for r in records}

    def prefix(ex, baseline):
        for name in PREFIX:
            require(ex['layers'][name]['input_sha256']==baseline['layers'][name]['input_sha256'],'Current CC five input hashes')
            counts['checked_prefix_input_hashes']+=1
        for name in PREFIX[:-1]:
            require(ex['layers'][name]['output_sha256']==baseline['layers'][name]['output_sha256'],'Current CC four earlier output hashes')
            counts['checked_prefix_output_hashes']+=1

    for rec in records:
        arm,mode,ex=rec['arm'],rec['mode'],rec['execution']
        native=arm=='native'; selected=rec['selected_tiles']
        require(mode in ('actual','dual_reference') and (mode=='actual' or not native),'Actual/dual mode')
        if selected is not None:
            selected=tuple(tuple(t) for t in selected)
            require(len(selected) in (0,2,4) and len(set(selected))==len(selected) and
                    set(selected)<=set((r,c) for r in range(3) for c in range(4)),'Legal complete joint tiles')
        expected_layers=set() if native else scope
        require(set(ex['layers'])==expected_layers and set(ex['all12_observed_fc_rows'])==set(shapes),'Observed layer inventory')
        total=0; weighted=0
        for name,shape in shapes.items():
            rows=10*shape[2]*shape[3]
            if name in expected_layers:
                layer=ex['layers'][name]
                mask=[[arm=='FF' or (name==TARGET and (arm=='fine_s2b2' or (r,c) in (selected or ())))
                       for c in range((shape[3]+8)//9)] for r in range((shape[2]+8)//9)]
                require(layer['mode']==mode and layer['fine_mask']==mask,'Complete T10/C mask or tail mismatch')
                g.audit_record(layer,shape,4*shape[4],ledger)
                rows=layer['executed_fc_rows_per_layer']; counts['wrapper_records']+=1
            require(ex['all12_observed_fc_rows'][name]==dict(fc1_rows=rows,fc2_rows=rows),'Actual paired FC hooks')
            total+=rows; weighted+=8*rows*shape[4]**2; counts['all12_paired_FC_records']+=1
        expected=609600 if mode=='dual_reference' else (558000 if arm in ('native','FF') else
                  164200 if arm=='fine_s2b2' else 153600+720*len(selected))
        require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total==expected,'Actual perFC total')
        if selected is not None and len(selected) in (2,4) and mode=='actual':
            require(weighted=={2:71415889920,4:73114583040}[len(selected)],'BothFC nominal arithmetic')
        counts['FC_rows_each_total']+=total
        counts['actual_forwards' if mode=='actual' else 'dual_forwards']+=1
        counts['purpose_'+rec['purpose']]+=1
        counts['rows_'+str(expected)]+=1
        if rec['target_id'] in frames and (selected is not None or arm=='fine_s2b2'):
            prefix(ex,frames[rec['target_id']]['arms']['CC']['execution'])
        if ex.get('batch_norm_inputs'): snapshots.append(ex['batch_norm_inputs'])
    require(counts['dual_forwards']==4 and counts['purpose_dual_audit']==4 and counts['purpose_archive_replay']==1 and
            counts['purpose_main']==raw['main_actual_forward_count'],'Actual purpose counts')
    require(len(records)==raw['main_actual_forward_count']+5 and raw['main_actual_forward_count']+raw['alias_logical_arm_count']==288,'Actual versus logical denominator')
    arms=set(ANCHORS)|{p+'_k'+str(k) for p in POLICIES for k in (2,4)}
    archives=[]; aliases=0; main_ids=[]
    for frame in raw['frames']:
        require(set(frame['arms'])==arms and set(frame['decisions'])==set(POLICIES) and
                frame['decision_archived_before_GT_load'],'All logical arms and prior decisions')
        decision_path=path.parent/frame['decision_file']
        require(decision_path.resolve().parent==path.parent.resolve() and sha(decision_path)==frame['decision_file_sha256'],'Decision archive')
        decision=json.loads(decision_path.read_text())
        require(not decision['GT_used'] and not decision['fitting_performed'] and decision['decisions']==frame['decisions'] and
                decision['target_id']==frame['target_id'] and decision['predecision_u_sha256']==frame['predecision_u_sha256'],'No-GT decision binding')
        seen={}
        for policy in POLICIES:
            dec=frame['decisions'][policy]
            require(sorted(dec['ranking_indices'])==list(range(12)),'Complete frozen ranking')
            for k in (2,4):
                arm=policy+'_k'+str(k); item=frame['arms'][arm]
                selected=tuple(tuple(t) for t in dec['selected'][str(k)])
                expected_selected=tuple(sorted((i//4,i%4) for i in dec['ranking_indices'][:k]))
                require(selected==expected_selected and item['selected_tiles']==[list(t) for t in selected],'Decision/mask binding')
                require(item['alias_of']==seen.get(selected),'Exactly same-frame first equivalent mask alias')
                if selected not in seen: seen[selected]=arm
        for arm,item in frame['arms'].items():
            rec=byid[item['actual_forward_id']]
            require(rec['purpose']=='main' and rec['target_id']==frame['target_id'] and rec['mode']=='actual' and
                    rec['execution']==item['execution'] and rec['selected_tiles']==item['selected_tiles'],'Logical arm actual-forward binding')
            if item['alias_of'] is None:
                require(rec['arm']==arm,'Unique execution arm name');main_ids.append(rec['forward_id'])
            else:
                aliases+=1; prior=frame['arms'][item['alias_of']]
                require(prior['alias_of'] is None and rec['arm']==item['alias_of'] and
                        prior['actual_forward_id']==item['actual_forward_id'] and prior['prediction_sha256']==item['prediction_sha256'] and
                        prior['selected_tiles']==item['selected_tiles'],'Alias changed action or endpoint')
            if arm!='native' and arm!='FF':
                require(item['execution']['layers'][TARGET]['input_sha256']==frame['predecision_u_sha256'],'Same current CC u')
        require(frame['ordinary_score_cost']['diagnostic_only_extra_psn1_calls']==12 and
                frame['ordinary_score_cost']['diagnostic_only_extra_psn1_temporal_MACs_nominal']==37324800,'Extra PSN score service')
        for key in ('predecision_input_file','prediction_file'):
            h.archive_check(path.parent,frame,key,archives,raw_u=key=='predecision_input_file')
        with zipfile.ZipFile(path.parent/frame['prediction_file']) as z:
            require(set(z.namelist())=={a+'.npy' for a in arms},'All36 prediction payloads')
            for arm,item in frame['arms'].items():
                with z.open(arm+'.npy') as stream:
                    header=h.read_header(stream)
                    require(header==dict(descr='<f4',fortran_order=False,shape=(2,480,640)),'Prediction shape/dtype')
                    hashed=hashlib.sha256(); size=0
                    for block in iter(lambda:stream.read(1024*1024),b''): hashed.update(block);size+=len(block)
                    require(size==2*480*640*4 and hashed.hexdigest()==item['prediction_sha256'],'Opaque prediction payload SHA')
    require(aliases==raw['alias_logical_arm_count'] and len(main_ids)==len(set(main_ids))==raw['main_actual_forward_count'],'No reused execution counted twice')
    require(set(main_ids)=={r['forward_id'] for r in records if r['purpose']=='main'},'No unreported main execution')
    require(set(raw['dual_checks'])=={'FF','fine_s2b2','flow11_task_k2','flow11_task_k4'},'Four dual audit subjects')
    for arm,pair in raw['dual_checks'].items():
        actual=byid[pair['actual_forward_id']]; dual=byid[pair['dual_forward_id']]
        require(actual['execution']==pair['actual_execution'] and dual['execution']==pair['dual_execution'] and
                actual['target_id']==dual['target_id']==members[0] and dual['purpose']=='dual_audit' and
                actual['selected_tiles']==dual['selected_tiles'] and pair['flow_list_length']>0 and
                pair['all_flows_exact'] and pair['all8_inputs_exact'] and pair['all8_branches_exact'],'Actual/dual binding')
        if arm.startswith('flow11_task'):
            require(raw['frames'][0]['arms'][arm]['actual_forward_id']==actual['forward_id'],'Primary alias dual coverage')
    require({p['dual_forward_id'] for p in raw['dual_checks'].values()}==
            {r['forward_id'] for r in records if r['purpose']=='dual_audit'},'Every physical dual is accounted once')
    require(set(raw['anchor_replays'])=={'CC'},'Single archived replay')
    replay=raw['anchor_replays']['CC']; original=next(f for f in old['frames'] if f['target_id']==replay['target_id'])
    rec=byid[replay['forward_id']]
    require(rec['purpose']=='archive_replay' and rec['execution']==replay['execution'] and
            replay['prediction_sha256']==original['arms']['coarse_stage02']['prediction_sha256'] and
            replay['exact_archived_prediction'] and replay['all8_input_output_SHA_exact'],'R10 CC replay')
    for name in scope:
        for field in ('input_sha256','output_sha256'):
            require(replay['execution']['layers'][name][field]==original['arms']['coarse_stage02']['execution']['layers'][name][field],'R10 archived branch hashes')
    require(len(snapshots)==1 and set(snapshots[0])==set(sig),'One full78 BN snapshot')
    for record in json.loads(BN.read_text())['external_BN']['records']:
        hook=snapshots[0][record['name']]
        require(not hook['training'] and not hook['track_running_stats'] and not hook['running_mean_exists'] and
                len(hook['input_calls'])==1 and hook['input_calls'][0]['shape']==record['native_input_shape'] and
                hook['input_calls'][0]['stride']==record['observed']['input_calls'][0]['stride'] and
                hook['input_calls'][0]['dtype']=='torch.float32','Actual external current BN geometry')
    private_elements=0
    for name in frozen:
        hook=snapshots[0][name]
        require(not hook['training'] and hook['track_running_stats'] and hook['running_mean_exists'] and
                all(c['dtype']=='torch.float32' for c in hook['input_calls']),'Private BN mode')
        private_elements+=sum(h.product(c['shape']) for c in hook['input_calls'])
    require(private_elements==152064000,'Private CC BN elements')
    receipt=dict(schema='flow_round11_independent_service_audit_v1',status='PASS',
        source_sha256={str(p):sha(p) for p in list(PINS)+[path,Path(__file__),HERE/'runner_cpu_results.json']},
        full_forward_count=len(records),logical_endpoints=288,alias_logical_arm_count=aliases,counts=dict(counts),
        perFC_rows=dict(native=558000,FF=558000,CC=153600,fine_s2b2=164200,k2=155040,k4=156480,dual=609600),
        extra_diagnostic_PSN=dict(calls=96,temporal_MACs_nominal=8*37324800),
        BN=dict(private_frozen=24,external_current_stat=54,external_elements=976584960,private_CC_elements=private_elements,
                input_snapshot_frames=1,state_buffers_modules_hooks_restored=True),
        archives=dict(NPZ_files=len(archives),NPZ_bytes=sum(a['bytes'] for a in archives),
                      prediction_payloads_SHA_checked=288,raw_u_payloads_SHA_checked=8,decision_JSON_files_SHA_checked=8,records=archives),
        scope=dict(quality_values_analyzed=False,prediction_arrays_numerically_loaded=False,GPU_or_model_execution=False,
                   hardware_gain_claim=False,score_policy_latency_or_energy_measured=False),
        boundary='Actual joint producer services and same-mask alias accounting only; all policies still require their own score costs. Not measured PE cycles, deployable full-network inference savings, or PPA.')
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return dict(status='PASS',full_forward_count=len(records),aliases=aliases,counts=dict(counts),output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path);parser.add_argument('--results-sha256');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.results is None:
        require(args.results_sha256 is None and args.output is None,'Preflight has no result output')
        _,_,_,shapes,members,_=preflight()
        print(json.dumps(dict(status='PREFLIGHT_ONLY_PASS',raw_results_audited=False,members=8,target_shape=shapes[TARGET],
            logical_endpoints=288,maximum_actual_forwards=293,alias_scope='same frame, exact complete mask',
            actual_perFC_rows=[153600,155040,156480,164200,558000],dual_perFC_rows=609600)))
    else:
        require(args.results_sha256 and args.output,'Raw SHA and fresh output required')
        print(json.dumps(audit(args.results,args.results_sha256,args.output)))


if __name__=='__main__':main()
