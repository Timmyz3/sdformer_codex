#!/usr/bin/env python3
"""Fixed diversity/native LR stability diagnostic; preflight by default.

This experiment is separate from R13. It changes LR only, uses the original
initial state with the same legal BN, and never resumes recovered R13 weights.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import traceback
import types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
DATE=HERE.parent
R13DIR=DATE/'coverage_recovery_round13/implementation'
SCHEMA='flow_native_lr_stability_v1'
OWN=('run_native_lr.py','execute_recovery.py','test_native_lr_cpu.py','inherited_pins.json')
R13_SHA='9aefdaae4e203816f24415eb73c6c480be1529baba28f5c74bc10e9ca1764ef5'
R13_EXEC_SHA='da4f460c9582dbe244218dd70f8b5928c6ede1495527d6cf355d1cbc01955541'
MEMBER_SHA='eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_r13():
    require(sha(R13DIR/'run_coverage_recovery.py')==R13_SHA and sha(R13DIR/'execute_recovery.py')==R13_EXEC_SHA,'Frozen R13 code differs')
    pins=json.loads((HERE/'inherited_pins.json').read_text())
    for relative,digest in pins['sources'].items():require(sha(DATE/relative)==digest,'Frozen dependency differs: '+relative)
    return load_file(R13DIR/'run_coverage_recovery.py','_native_lr_frozen_r13')


R13=load_r13()
RECIPE=dict(R13.RECIPE,lr=2.5e-5)
BN_SCOPE=R13.BN_SCOPE
ARMS=('native',)
LOSS_SHA=R13.LOSS_SHA
tile=R13.tile
for _name in ('apply_fresh_bn','state_digest','select_parameters','forward_train','reference_forward','compare_all','gradient_audit','gradient_coverage'):
    globals()[_name]=getattr(R13,_name)


def budget(mode):
    require(mode in ('smoke','recover'),'Fixed smoke/recover only')
    return {'expected_training_forwards':5 if mode=='smoke' else 200,'expected_audit_forwards':4,
            'expected_evaluation_forwards':0 if mode=='smoke' else 24}


def baseline_contract(value,binding,manifest,checksum):
    require(value.get('schema')==R13.SCHEMA and value.get('status')=='ROUND13_COVERAGE_RECOVERY_COMPLETE'
            and value.get('completed_optimizer_steps')==200 and value.get('training_performed') is True,'Completed original-LR R13 reference required')
    source=value['binding']
    require((source['condition'],source['arm'],source['mode'])==('diversity','native','recover'),'Only R13 diversity/native reference allowed')
    require(source['runner_sha256']==R13_SHA and source['execute_sha256']==R13_EXEC_SHA and source['recipe']==R13.RECIPE,'Original fixed R13 recipe/source differs')
    common=('checkpoint_sha256','wrapper_sha256','membership_sha256','runtime_manifest_sha256','BN_buffers_sha256','BN_receipt_sha256',
            'loss_sha256','allowed_parameter_count','region_sha256','membership_audit')
    require(all(source[k]==binding[k] for k in common),'Original-LR reference common inputs differ')
    require(source['membership_sha256']==MEMBER_SHA and source['fit_ids']==[r['target_id'] for r in manifest['train_order_diversity']]
            and source['check_ids']==[r['target_id'] for r in manifest['check_members']],'Original-LR reference identities/order differ')
    require(all(value.get(k) is True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'Original-LR reference closure incomplete')
    frames=value['evaluations']['initial']['frames'];ids=source['check_ids']
    require(len(ids)==len(frames)==8 and [f['target_id'] for f in frames]==ids,'Eight paired native initial frames required')
    require(all(set(f['arms'])=={'native'} and f['sequence']==r['sequence'] for f,r in zip(frames,manifest['check_members'])),'Initial frame identity/layout differs')
    predictions={f['target_id']:f['arms']['native']['prediction_sha256'] for f in frames}
    digest=lambda x:isinstance(x,str) and len(x)==64 and set(x)<=set('0123456789abcdef')
    require(digest(value['initial_model_state_sha256']) and all(digest(x) for x in predictions.values()),'Valid initial state/prediction SHA required')
    return {'result_sha256':checksum,'initial_state_sha256':value['initial_model_state_sha256'],
            'initial_native_prediction_sha256':predictions,'baseline_lr':R13.RECIPE['lr'],'diagnostic_lr':RECIPE['lr'],
            'only_recipe_difference':['lr'],'R13_check8_now_development':True,'recovered_baseline_weights_used_as_initialization':False}


def validate_smoke(paths,binding,baseline_sha):
    require(len(paths)==1,'Exactly one new native lower-LR smoke required')
    path=Path(paths[0]);value=json.loads(path.read_text())
    require(value.get('schema')==SCHEMA and value.get('status')=='NATIVE_LR_SMOKE_COMPLETE' and value.get('completed_optimizer_steps')==5
            and value.get('training_performed') is True and value.get('full_forward_count')==9,'Complete new five-step/nine-forward smoke required')
    fields=('runner_sha256','execute_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256','loss_sha256',
            'membership_sha256','runtime_manifest_sha256','BN_buffers_sha256','BN_receipt_sha256','allowed_parameter_count','condition','arm','recipe')
    require(all(value['binding'][k]==binding[k] for k in fields) and value['binding']['mode']=='smoke','Single native smoke binding differs')
    require(value['binding']['baseline_comparison']['result_sha256']==baseline_sha,'Smoke paired baseline differs')
    require(all(value.get(k) is True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed',
                'smoke_weights_discarded','paired_initial_state_exact')),'Smoke source/state closure missing')
    require(value['initial_numerical_pair']['all_flow_outputs_exact'] and value['final_numerical_pair']['all_flow_outputs_exact']
            and value['gradient_coverage']['complete'],'Smoke numerical/gradient audit missing')
    require(value['actual_optimizer_hyperparameters']=={'lr':RECIPE['lr'],'weight_decay':RECIPE['weight_decay'],
                'betas':list(RECIPE['betas']),'eps':RECIPE['eps']},'Actual smoke optimizer differs')
    require(value['initial_model_state_sha256']==value['binding']['baseline_comparison']['initial_state_sha256'],'Smoke initial state not paired')
    return {'native':sha(path)}


def prepare(a,collector,probe,parent):
    require(a.condition=='diversity' and a.arm=='native' and a.membership_sha256==MEMBER_SHA,'Fixed D200/native membership only')
    require(sha(a.baseline_result)==a.baseline_result_sha256,'Explicit original-LR result SHA differs')
    require({k for k in RECIPE if RECIPE[k]!=R13.RECIPE[k]}=={'lr'} and RECIPE['lr']==2.5e-5,'Only lower LR may change')
    # Frozen R13 prepare performs metadata/data/source preflight only. Its original
    # all-four smoke gate is replaced by this experiment's single-native gate.
    namespace=dict(R13.prepare.__globals__)
    namespace.update(HERE=HERE,OWN=OWN,RECIPE=RECIPE,__file__=__file__,
                     validate_smokes=lambda paths,binding:validate_smoke(paths,binding,a.baseline_result_sha256))
    function=types.FunctionType(R13.prepare.__code__,namespace,'prepare_native_lr_inputs',R13.prepare.__defaults__)
    prepared=function(a,collector,probe,parent)
    inherited,fit,checks,binding,manifest=prepared
    require(len(fit)==200 and len({r['target_id'] for r in fit})==200,'Exactly D200 once required')
    require(len(checks)==(0 if a.mode=='smoke' else 8),'Fixed check support differs')
    reference=json.loads(a.baseline_result.read_text())
    binding.update(budget(a.mode))
    binding.update(experiment='native_lr_stability',baseline_result={'path':str(a.baseline_result),'sha256':a.baseline_result_sha256},
                   baseline_comparison=baseline_contract(reference,binding,manifest,a.baseline_result_sha256),
                   reused_prepare_source_sha256=R13_SHA,reused_execute_parent_sha256=R13_EXEC_SHA,
                   initial_layouts=[] if a.mode=='smoke' else ['native'],endpoint_layouts=[] if a.mode=='smoke' else ['native','FF'],
                   total_full_forward_budget=sum(budget(a.mode).values()),fresh_W0_and_optimizer=True,
                   scope='Single-axis ordinary stability diagnostic; original backbone saw training sequences; reused check8 is development. No flow-specific or hardware claim.')
    return inherited,fit,checks,binding,manifest


def parameter_drift(selected,initial,torch):
    groups={k:{'initial_squared_L2':0.,'delta_squared_L2':0.,'max_abs_delta':0.,'tensors':0} for k in ('FC','PSN','BN_affine')}
    layers={}
    for name,p in selected.items():
        before=initial[name].double();after=p.detach().cpu().double();difference=after-before
        category='FC' if '.fc1.' in name or '.fc2.' in name else 'PSN' if '.sn1.' in name or '.sn2.' in name else 'BN_affine'
        layer=name.split('.mlp.')[0]+'.mlp'
        if layer not in layers:layers[layer]={'initial_squared_L2':0.,'delta_squared_L2':0.,'max_abs_delta':0.,'tensors':0}
        for record in (groups[category],layers[layer]):
            record['initial_squared_L2']+=float(before.square().sum());record['delta_squared_L2']+=float(difference.square().sum())
            record['max_abs_delta']=max(record['max_abs_delta'],float(difference.abs().max()));record['tensors']+=1
    for record in list(groups.values())+list(layers.values()):
        record['relative_L2']=(record['delta_squared_L2']/record['initial_squared_L2'])**.5 if record['initial_squared_L2'] else None
    return {'categories':groups,'layers':layers,'scope':'Actual endpoint minus paired initial parameter values; CPU FP64 reduction, no extra model forward.'}


def execute_function():
    module=load_file(HERE/'execute_recovery.py','_native_lr_execution')
    namespace=dict(module.execute.__globals__)
    names=('require','RECIPE','BN_SCOPE','ARMS','LOSS_SHA','tile','apply_fresh_bn','state_digest','select_parameters',
           'forward_train','reference_forward','compare_all','gradient_audit','gradient_coverage','sha','parameter_drift')
    namespace.update({key:globals()[key] for key in names})
    return types.FunctionType(module.execute.__code__,namespace,module.execute.__name__,module.execute.__defaults__)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-manifest','membership','bn-buffers','bn-receipt','baseline-result'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--mode',choices=('smoke','recover'),required=True)
    parser.add_argument('--smoke-receipt',type=Path,action='append',default=[])
    parser.add_argument('--runner-checks',type=Path,default=HERE/'runner_cpu_results.json')
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args(argv);a.condition='diversity';a.arm='native'
    require(a.mode!='smoke' or not a.smoke_receipt,'Smoke starts from W0, takes no recovered/smoke artifact')
    collector,probe,parent,r9,r8=R13.load_runtime();prepared=prepare(a,collector,probe,parent)
    result={'schema':SCHEMA,'status':'PREFLIGHT_PASS_NO_INFERENCE','binding':prepared[3],'training_performed':False,'inference_executed':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight output required')
        parent.write(a.preflight_output,result);print(result['status']);return
    require(parent.GPU_LOCK.parent.is_dir(),'Existing shared lock directory required')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('Native LR diagnostic fixed30minute cap exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            result['inference_executed']=True;execute_function()(a,prepared,result,collector,probe,parent,r9,r8)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
