#!/usr/bin/env python3
"""Build explicit H0 inputs only from completed frozen R13 evidence.

CPU/std-library only; no SSH, GPU, training, deployment or automatic execution.
This module does not read any experiment result merely by being imported.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
H0DIR=HERE.parent/'implementation_h0'
CONDITIONS=('reference','diversity')
ARMS=('native','FF','CC','fine_s2b2')
KEYS=tuple(c+'/'+a for c in CONDITIONS for a in ARMS)
CHECKPOINT_SHA='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
OWN=('build_h0_inputs.py','test_builder_cpu.py','source_pins.json')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_helpers():
    for name,digest in json.loads((HERE/'source_pins.json').read_text())['accepted_h0_sources'].items():
        require(sha(H0DIR/name)==digest,'Accepted H0 source differs: '+name)
    sys.path.insert(0,str(H0DIR));h0=importlib.import_module('h0_core')
    require(Path(h0.__file__).resolve()==H0DIR/'h0_core.py','Wrong H0 import')
    runner=h0.load_file(H0DIR/'run_h0.py','_h0_input_builder_accepted_validators')
    return h0,runner


def select_state(analysis):
    """Pure fixed gate priority. Only synthetic inputs used during preparation."""
    normal=analysis['ordinary_reference_normality'];gates=analysis['candidate_gates']
    require(set(normal)==set(gates)==set(CONDITIONS),'Both conditions required')
    for condition in CONDITIONS:
        require(type(normal[condition]) is bool and set(gates[condition])=={'CC','fine_s2b2'},'Complete bool gate tables required')
        for arm,gate in gates[condition].items():
            families=('normal_references','matched_recovered_references','own_checkpoint_references','shared_fine_preserved')
            require(all(set(gate[f])=={'native','FF'} and all(type(v['pass']) is bool for v in gate[f].values()) for f in families),
                    'Four complete fixed quality gate families required')
            conjunction=all(v['pass'] for f in families for v in gate[f].values())
            require(type(gate['ordinary_base_quality_pass']) is bool and gate['ordinary_base_quality_pass']==conjunction,
                    'Ordinary base gate is not the fixed conjunction')
            require(normal[condition]==all(v['pass'] for v in gate['normal_references'].values()),'Ordinary normality/table disagreement')
    for condition in CONDITIONS:
        for arm in ('CC','fine_s2b2'):
            if gates[condition][arm]['ordinary_base_quality_pass']:
                return {'condition':condition,'W_arm':arm,'scope':'qualified_ordinary_base','ordinary_N_F_stable':True,
                        'selected_base_passes_required_gates':True,'shared_fine_protection_passes':True,
                        'rationale':'First fully qualified ordinary base in fixed order reference/CC, reference/P, diversity/CC, diversity/P. All four frozen gate families pass.'}
    for condition in CONDITIONS:
        if normal[condition]:
            return {'condition':condition,'W_arm':'FF','scope':'action_only_screening','ordinary_N_F_stable':True,
                    'selected_base_passes_required_gates':False,'shared_fine_protection_passes':None,
                    'rationale':'No CC/P state passes all four frozen gate families. First stable ordinary N/F condition in reference,diversity order; use its FF state only for bounded action-space screening.'}
    return None


def mapped_asset(value):
    require(set(value)=={'local_path','remote_path','sha256'},'Mapping asset requires explicit local/remote paths and SHA')
    local=Path(value['local_path']);remote=Path(value['remote_path'])
    require(local.is_absolute() and local.is_file() and remote.is_absolute(),'Existing local and absolute remote source paths required')
    require(sha(local)==value['sha256'],'Local mapped source bytes differ: '+str(local))
    return local,{'path':str(remote),'sha256':value['sha256']}


def verify_completed_reports(analysis,audit,common,runner):
    require(analysis.get('schema')=='flow_round13_coverage_quality_analysis_v1' and analysis.get('script_sha256')==
            'b1997b55deea309088388a011b83272bafe438bedc43db5ee4440655bc5034ad','Frozen completed all-eight quality analysis required')
    require(analysis.get('analysis_protocol_sha256')=='5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b' and
            analysis.get('R12_V2_source_sha256')=='525ba2821d1e671b581f63e80c4bce02ef3e524c3e6af7d143a723914b001cd8','Frozen protocol/math differs')
    require((analysis.get('state_layout_count'),analysis.get('evaluation_endpoint_count'),analysis.get('region_count'),analysis.get('full_forward_count'))==
            (40,320,16,1964),'Incomplete quality analysis counts')
    require(analysis.get('initial_state_all_eight_exact') is True and analysis.get('initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact') is True,
            'Initial all-eight exact audit missing')
    require(set(analysis.get('inputs',{}))==set(analysis.get('prediction_artifact_audits',{}))==set(KEYS),'All-eight result/prediction sources required')
    require(all(v=={'prediction_files':16,'prediction_payloads':40} for v in analysis['prediction_artifact_audits'].values()),'Incomplete prediction artifact audit')
    require(all(analysis['common_contract'].get(k)==v for k,v in common.items()),'Common source state differs')
    # Empty states here means no selected state yet. This validator still checks
    # all16 runs, all8 recovered result identities and independent state audits.
    runner.validate_independent_execution(audit,analysis,{'states':{}},common)


def prepare(mapping,h0,runner):
    require(mapping.get('schema')=='flow_h0_remote_mapping_v1','Unknown deployment mapping')
    names={'analysis','independent_execution_audit','runtime_manifest','membership','bn_buffers','bn_receipt','h8_manifest'}
    require(set(mapping['assets'])==names,'Exactly seven explicit common/report assets required')
    locals_,assets={},{}
    for name,value in mapping['assets'].items():locals_[name],assets[name]=mapped_asset(value)
    analysis=json.loads(locals_['analysis'].read_text());audit=json.loads(locals_['independent_execution_audit'].read_text())
    membership=json.loads(locals_['membership'].read_text());h8=json.loads(locals_['h8_manifest'].read_text())
    require(assets['membership']['sha256']==h0.MEMBER_SHA and h8==h0.h8_manifest(membership,h0.MEMBER_SHA),'Frozen H8/membership differs')
    require(assets['h8_manifest']['sha256']==sha(H0DIR/'h8_manifest.json'),'H0 accepted H8 bytes differ')
    runtime=json.loads(locals_['runtime_manifest'].read_text());require(runtime.get('schema')=='flow_round13_runtime_v1','R13 runtime schema required')
    bn=json.loads(locals_['bn_receipt'].read_text())
    require(bn.get('schema')=='flow_round13_common_bn_v1' and bn.get('status')=='COMMON_BN_COMPLETE' and bn['BN_scope']=='MLP_R13_COMMON_CMA176',
            'Completed legal current BN required')
    require(bn['checkpoint_sha256']==CHECKPOINT_SHA and bn['membership_sha256']==h0.MEMBER_SHA and bn['buffers_sha256']==assets['bn_buffers']['sha256']
            and bn['bn_target_ids']==[r['target_id'] for r in membership['bn_members']] and bn['old_CMA_loaded'] is False and bn['GT_used'] is False,
            'New BN source/state/member lineage differs')
    require((bn.get('calibration_forwards'),bn.get('calibration_bn_count'),bn.get('retained_mlp_bn_count'))==(176,78,24)
            and bn.get('calibration_definition')=='all78_reset_CMA_train_nonBN_eval','Current BN calibration definition/counts differ')
    common={'checkpoint_sha256':CHECKPOINT_SHA,'membership_sha256':h0.MEMBER_SHA,'runtime_manifest_sha256':assets['runtime_manifest']['sha256'],
            'BN_buffers_sha256':assets['bn_buffers']['sha256'],'BN_receipt_sha256':assets['bn_receipt']['sha256']}
    verify_completed_reports(analysis,audit,common,runner)
    require(runtime['region_sha256']==analysis['common_contract']['region_sha256'],'Current training-side region contract differs')
    require(set(mapping['results'])==set(KEYS),'All eight result mappings required')
    recovered={(r['condition']+'/'+r['arm']):r for r in audit['results'] if r['mode']=='recover'}
    remote_results={};remote_weights={};raw={};local_weight_checks={}
    for key,item in mapping['results'].items():
        allowed={'local_result','remote_result','result_sha256','remote_weights','weights_sha256','local_weights'}
        require(set(item)<=allowed and set(item)>=(allowed-{'local_weights'}),'Result mapping fields incomplete/unrecognized')
        local=Path(item['local_result']);remote=Path(item['remote_result']);weights=Path(item['remote_weights'])
        require(local.is_absolute() and local.is_file() and remote.is_absolute() and weights.is_absolute() and weights.parent==remote.parent and weights.name=='weights.pt',
                'Explicit same-run remote result/weights paths required')
        require(sha(local)==item['result_sha256']==analysis['inputs'][key]['sha256']==recovered[key]['result_sha256'],'Result bytes differ from both completed audits')
        value=json.loads(local.read_text());condition,arm=key.split('/');binding=value['binding']
        require(value.get('schema')=='flow_round13_coverage_recovery_v1' and value.get('status')=='ROUND13_COVERAGE_RECOVERY_COMPLETE' and
                value.get('completed_optimizer_steps')==200 and value.get('training_performed') is True,'Complete actual recovery result required')
        require(binding['condition']==condition and binding['arm']==arm and binding['mode']=='recover' and
                binding['runner_sha256']==h0.R13_SHA and binding['execute_sha256']=='da4f460c9582dbe244218dd70f8b5928c6ede1495527d6cf355d1cbc01955541',
                'Actual recovered condition/arm/source differs')
        require(all(binding.get(k)==v for k,v in analysis['common_contract'].items() if k!='initial_state_sha256'),'Raw result common binding differs from analysis')
        require(value['initial_model_state_sha256']==analysis['common_contract']['initial_state_sha256']==audit['common_initial_model_sha256'],
                'Actual common initial state differs')
        require(value['weights_file']=='weights.pt' and value['weights_sha256']==item['weights_sha256']==recovered[key]['weights']['sha256'],
                'Actual recovered weight SHA differs')
        require(value['final_state_sha256']==recovered[key]['final_state_sha256'] and all(value.get(k) is True for k in
                ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'Independent final state/closure differs')
        if 'local_weights' in item:
            p=Path(item['local_weights']);require(p.is_absolute() and p.is_file() and sha(p)==item['weights_sha256'],'Optional local weight copy differs')
            local_weight_checks[key]=True
        else:local_weight_checks[key]=False
        remote_results[key]={'path':str(remote),'sha256':item['result_sha256']};remote_weights[key]={'path':str(weights),'sha256':item['weights_sha256']};raw[key]=value
    choice=select_state(analysis)
    receipt={'schema':'flow_h0_input_build_decision_v1','selection_rule':'qualified first: reference/CC,reference/P,diversity/CC,diversity/P; otherwise stable N/F reference,diversity -> FF; neither stable -> no runnable files',
             'completed_quality_and_execution_sources':{k:assets[k] for k in ('analysis','independent_execution_audit')},
             'source_common':common,'choice':choice,'local_weight_bytes_checked':local_weight_checks,'remote_path_existence_or_bytes_checked':False,
             'remote_verification_required':'Root deploys unchanged source evidence and uses accepted H0 preflight to verify every actual remote asset SHA.',
             'GPU_used':False,'SSH_used':False,'experiment_started':False,'hardware_benefit_claim':False,'flow_specific_increment_claim':False}
    if choice is None:return None,None,receipt
    directory=Path(mapping['output_remote_dir']);require(directory.is_absolute(),'Explicit absolute future remote bundle directory required')
    destinations={assets[k]['path'] for k in assets}|{v['path'] for v in remote_results.values()}|{v['path'] for v in remote_weights.values()}
    require(not {str(directory/'admission.json'),str(directory/'inputs.json')}&destinations,'New input bundle collides with a source asset')
    admission={'schema':'flow_h0_admission_v1','status':'READY_AFTER_R13_ALL_EIGHT_AUDIT',**choice,
               'audited_conditions_arms':[[c,a] for c in CONDITIONS for a in ARMS],
               'all_eight_audit_receipt':assets['analysis'],'independent_execution_audit':assets['independent_execution_audit']}
    inputs={'schema':'flow_h0_inputs_v1','condition':choice['condition'],'W_arm':choice['W_arm'],
            **{k:assets[k] for k in ('runtime_manifest','membership','bn_buffers','bn_receipt','h8_manifest')},'all_eight_results':remote_results,
            'states':{role:{'result':remote_results[choice['condition']+'/'+arm],'weights':remote_weights[choice['condition']+'/'+arm]}
                      for role,arm in (('W',choice['W_arm']),('N','native'),('F','FF'))}}
    runner.validate_admission(admission,choice['condition'],choice['W_arm'])
    runner.validate_completed_analysis(analysis,inputs,admission,common)
    runner.validate_independent_execution(audit,analysis,inputs,common)
    receipt.update(status='INPUTS_PREPARED_NOT_DEPLOYED_OR_EXECUTED',remote_bundle_dir=str(directory))
    return inputs,admission,receipt


def write_bundle(output_dir,remote_dir,inputs,admission,receipt):
    require(not output_dir.exists(),'Fresh local bundle output required')
    output_dir.mkdir(parents=True,exist_ok=False)
    def write(name,value):(output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    if inputs is None:
        receipt=dict(receipt,status='NO_QUALIFYING_STATE_NO_RUNNABLE_INPUTS')
        write('NO_QUALIFYING_STATE.json',receipt)
    else:
        write('admission.json',admission)
        inputs=dict(inputs,admission={'path':str(Path(remote_dir)/'admission.json'),'sha256':sha(output_dir/'admission.json')})
        write('inputs.json',inputs)
        receipt=dict(receipt,generated_files={name:sha(output_dir/name) for name in ('admission.json','inputs.json')})
        write('BUILD_RECEIPT.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping',type=Path,required=True);parser.add_argument('--mapping-sha256',required=True)
    parser.add_argument('--cpu-checks',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();require(sha(a.mapping)==a.mapping_sha256,'Explicit mapping SHA differs')
    h0,runner=load_helpers();checks=json.loads(a.cpu_checks.read_text())
    require(checks.get('passed') is True and checks.get('GPU_initialized') is False and checks.get('torch_imported') is False,'Pure CPU builder checks required')
    require(checks['source_sha256']=={name:sha(HERE/name) for name in OWN},'Builder source CPU receipt differs')
    mapping=json.loads(a.mapping.read_text());inputs,admission,receipt=prepare(mapping,h0,runner)
    receipt.update(builder_source_sha256={name:sha(HERE/name) for name in OWN},mapping_sha256=a.mapping_sha256,cpu_checks_sha256=sha(a.cpu_checks))
    receipt=write_bundle(a.output_dir,mapping.get('output_remote_dir',''),inputs,admission,receipt)
    print(json.dumps({'status':receipt['status'],'choice':receipt['choice'],'experiment_started':False}),flush=True)


if __name__=='__main__':main()
