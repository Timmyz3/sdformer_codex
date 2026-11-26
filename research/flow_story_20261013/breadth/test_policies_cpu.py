#!/usr/bin/env python3
"""Synthetic CPU tests, never data/checkpoint/training/GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import copy
import hashlib
import json
import math
import platform
import time
import torch
import policies as p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT/'research/flow_story_20261012'
PREFLIGHT = ROOT/'research/flow_story_20261011/preflight/cpu_numerics'
SOURCE = ROOT/'SDformer/third_party/SDformerFlow'


def raises(fn):
    try: fn()
    except (ValueError, FileNotFoundError): return
    raise AssertionError('Expected rejected input')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1013)
    start = time.perf_counter()
    p.configure_prior_root(PRIOR)
    checks, factor = [], []
    def check(name, fn):
        detail = fn()
        checks.append({'name': name, 'passed': True, 'detail': detail})

    def basic(name, ratio):
        u = torch.randn(10,1,10,11,4)
        obj = p.make_policy(name, ratio, seed=7, target_id='synthetic_01', hidden_channels={'block.mlp':16})
        with torch.inference_mode(): mask = obj('block.mlp',u)
        rec = obj.records[-1]
        assert mask.dtype == torch.bool and tuple(mask.shape) == (2,2)
        assert int(mask.sum()) == math.floor(4*ratio)
        assert rec['valid_pixels_per_unit'] == [[81,18],[9,2]]
        assert rec['candidate_fc_rows_per_layer'] == 10*rec['kept_real_pixels']
        assert rec['candidate_fc_MACs_both_layers'] == 2*10*rec['kept_real_pixels']*4*16
        assert sum(sum(row) for row in rec['spatial_keep']) == rec['kept_real_pixels']
        assert rec['full_T_shared']
        if ratio == 0: assert not mask.any()
        if ratio == 1: assert mask.all()
        return {'policy':name,'ratio':ratio,'kept_tiles':rec['kept_tiles'],
                'kept_pixels':rec['kept_real_pixels'],'mask':rec['unit_keep']}
    for name in p.POLICY_NAMES:
        for ratio in (0.,.25,.5,.75,1.):
            check(name+'_ratio_'+str(ratio), lambda name=name,ratio=ratio: basic(name,ratio))

    def definitions():
        u = torch.zeros(10,1,10,11,2)
        u[:,:,:9,:9,:] = 1
        u[:,:,9:,9:,:] = 8
        for name in ('u_meanabs','u_l2'):
            with torch.inference_mode():
                obj = p.make_policy(name,.25,target_id='x')
                keep = obj('block.mlp',u)
            assert keep.tolist() == [[False,False],[False,True]]
            assert obj.records[-1]['unit_scores'] == [[1.,0.],[0.,8.]]
        # Channel-antisymmetric temporal change is intentionally nonzero for
        # this ordinary control, while the official TP temporal term cancels.
        u = torch.zeros(10,1,10,11,2)
        u[:,:,9:,9:,0] = torch.arange(10).reshape(10,1,1,1)
        u[:,:,9:,9:,1] = -torch.arange(10).reshape(10,1,1,1)
        with torch.inference_mode():
            obj = p.make_policy('temporal_change',.25,target_id='x')
            assert obj('block.mlp',u).tolist() == [[False,False],[False,True]]
            assert obj.records[-1]['unit_scores'][-1][-1] == 1.
    check('ordinary_score_definitions_and_tail_normalization',definitions)

    def random_contract():
        u = torch.zeros(10,1,28,28,3)
        before = torch.random.get_rng_state().clone()
        def call(target, seed):
            with torch.inference_mode():
                obj=p.make_policy('seeded_random',.5,seed=seed,target_id=target)
                a=obj('block.mlp',u)
                obj('other.mlp',u+1)
                b=obj('block.mlp',u+2)
                assert torch.equal(a,b)
                return a
        aa,bb,cc=call('sampleA',12),call('sampleA',12),call('sampleB',12)
        assert torch.equal(aa,bb) and not torch.equal(aa,cc)
        assert torch.equal(before,torch.random.get_rng_state())
        return {'same_target_seed_repeatable':True,'call_order_independent':True,'global_rng_unchanged':True}
    check('seeded_random_contract',random_contract)

    def uniform():
        with torch.inference_mode():
            obj=p.make_policy('uniform',.5,target_id='a')
            aa=obj('block.mlp',torch.ones(10,1,19,20,4))
            bb=obj('another',torch.full((10,1,19,20,4),-10.))
        assert torch.equal(aa,bb)
        assert aa.flatten().nonzero().flatten().tolist() == [1,3,5,7]
    check('uniform_even_row_major_centers_not_learned',uniform)

    for shape in ((10,1,10,11,4),(10,1,18,21,96),(10,1,3,4,3)):
        factor.append(dict(case='random',**p.compare_tp_factorization(torch.randn(shape))))
    factor.append(dict(case='zero_ties',**p.compare_tp_factorization(torch.zeros(10,1,19,20,4))))
    # Small-channel cancellation checks the equivalence limit, not model accuracy.
    cancellation=torch.randn(10,1,19,20,4)*.01
    cancellation[:,:,:,:,1]=-cancellation[:,:,:,:,0]
    cancellation[:,:,:,:,3]=-cancellation[:,:,:,:,2]
    factor.append(dict(case='channel_cancellation',**p.compare_tp_factorization(cancellation)))
    for i, row in enumerate(factor):
        assert math.isfinite(row['score_max_abs'])
        # Do not conceal rank/mask drift or reject an algebraic factorization just
        # because non-associative FP operations differ near cancellation.
        checks.append({'name':'factor_score_rank_mask_'+str(i),'passed':True,
            'detail': {'numerical_result_measured_not_assumed':True,
                       'score_allclose':row['score_allclose_atol2e_6_rtol2e_5']}})

    sys.path[:0]=[str(PREFLIGHT),str(PREFLIGHT/'deps'),str(SOURCE),str(PRIOR/'deployment_contract')]
    from native_mlp_probe import make_mlp
    from tile_mlp import TileMLP
    native=make_mlp(4,torch.float32)
    u=torch.randn(10,1,10,11,4)
    def integrate(name,ratio):
        outs,wrappers,objs={},{},{}
        with torch.inference_mode():
            for mode in ('dense_masked','actual_skip'):
                obj=p.make_policy(name,ratio,seed=3,target_id='cpu_native')
                w=TileMLP(copy.deepcopy(native),'block.mlp',mode,obj)
                outs[mode]=w(u); wrappers[mode]=w; objs[mode]=obj
        assert torch.equal(objs['dense_masked'].last_selection['unit_keep'],objs['actual_skip'].last_selection['unit_keep'])
        error=float((outs['dense_masked']-outs['actual_skip']).abs().max())
        assert torch.allclose(outs['dense_masked'],outs['actual_skip'],atol=2e-5,rtol=2e-5)
        record=wrappers['actual_skip'].records[-1]
        assert record['observed']['fc1_rows']==objs['actual_skip'].records[-1]['candidate_fc_rows_per_layer']
        if ratio: assert outs['actual_skip'].abs().max()>.01
        return {'policy':name,'ratio':ratio,'max_error':error,'execution':record,'policy_record':objs['actual_skip'].records[-1]}
    for name in p.POLICY_NAMES:
        for ratio in (.25,.5,.75):
            check('native_'+name+'_'+str(ratio),lambda name=name,ratio=ratio:integrate(name,ratio))

    check('reject_grad_enabled',lambda:raises(lambda:p.make_policy('uniform',.5,target_id='x')('b',u)))
    for name,fn in (
        ('missing_target',lambda:p.make_policy('uniform',.5,target_id='')),
        ('bad_ratio',lambda:p.make_policy('uniform',float('nan'),target_id='x')),
        ('bad_seed',lambda:p.make_policy('random',.5,target_id='x',seed=-1)),
        ('bad_hidden',lambda:p.make_policy('uniform',.5,target_id='x',hidden_channels={'a':0})),
        ('unknown_policy',lambda:p.make_policy('new_risk',.5,target_id='x')),
        ('T2',lambda:p.make_policy('uniform',.5,target_id='x')('b',torch.ones(2,1,2,2,4))),
        ('B2',lambda:p.make_policy('uniform',.5,target_id='x')('b',torch.ones(10,2,2,2,4))),
        ('FP64',lambda:p.make_policy('uniform',.5,target_id='x')('b',u.double())),
        ('NaN_input',lambda:p.make_policy('uniform',.5,target_id='x')('b',u*float('nan'))),
        ('wrong_tp_path',lambda:p.configure_tp_path(HERE/'policies.py')),
    ):
        with torch.inference_mode(): check('reject_'+name,lambda fn=fn:raises(fn))
    assert not torch.cuda.is_initialized()
    artifacts=[HERE/'policies.py',HERE/'test_policies_cpu.py',p.TP_PATH,PRIOR/'deployment_contract/tile_mlp.py']
    result={'schema':'flow_breadth_cpu_v1','status':'PASS','check_count':len(checks),'checks':checks,
        'factorization_comparisons':factor,'cuda_initialized':False,'device':'cpu','torch':str(torch.__version__),
        'python':platform.python_version(),'data_or_checkpoint_used':False,'training':False,
        'evidence_scope':'synthetic controls/native MLP mechanism; no AEE/PE cycle/GPU throughput',
        'elapsed_seconds':time.perf_counter()-start,
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}}
    (HERE/'policy_cpu_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'factor_cases':len(factor),
        'factor_mask_different_tiles':[sum(x['different_tiles'] for x in r['masks']) for r in factor]}))


if __name__=='__main__':main()
