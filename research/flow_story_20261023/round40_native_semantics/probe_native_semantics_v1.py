#!/usr/bin/env python3
"""Frozen tiny CPU semantic probe; refuses execution without reviewed hashes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'SDformer/third_party/SDformerFlow'
DEPS = ROOT / 'research/flow_story_20261011/preflight/cpu_numerics/deps'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, text):
    if not ok:
        raise AssertionError(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--admission', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    approval = json.loads(args.admission.read_text())
    require(approval['schema'] == 'flow_r40_native_semantics_admission_v1', 'admission schema')
    require(approval['status'] == 'PASS' and approval['reviewer'] != 'hardware-implementer', 'independent review')
    for name in ('PROTOCOL_V1.md', 'probe_native_semantics_v1.py', 'SOURCE_MANIFEST_V1.json'):
        require(sha(HERE / name) == approval['sha256'][name], 'reviewed hash: ' + name)
    manifest = json.loads((HERE / 'SOURCE_MANIFEST_V1.json').read_text())
    for name, digest in manifest['sources'].items():
        require(sha(ROOT / name) == digest, 'native source pin: ' + name)
    require(not args.output_dir.exists(), 'fresh output directory required')
    args.output_dir.mkdir(parents=True)
    sys.path[:0] = [str(DEPS), str(SOURCE)]
    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from models.STSwinNet_SNN.Spiking_modules import SpikingNormLayer
    from models.STSwinNet_SNN.Spiking_swin_transformer3D import (
        window_partition_v2, window_reverse, MS_Spiking_Mlp,
        MS_Spiking_SwinTransformerBlock3D)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(402023)
    raw = {}
    result = {'schema': 'flow_r40_native_semantics_result_v1', 'status': 'INCOMPLETE',
              'device': 'cpu', 'seed': 402023, 'torch': torch.__version__, 'numpy': np.__version__,
              'training_steps': 0, 'checkpoint_reads': 0, 'native_block_calls': 0,
              'admission_sha256': sha(args.admission), 'checks': {}}
    started = time.monotonic()

    def save(name, value):
        raw[name] = value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()

    def current_bn(wrapper):
        bn = wrapper.norm_layer
        bn.track_running_stats = False
        bn.running_mean = bn.running_var = bn.num_batches_tracked = None

    kwargs = dict(num_steps=10, spike_norm='BN', neuron_type='psn', v_th=.1,
                  v_reset=None, surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True)
    try:
        with torch.inference_mode():
            layouts = []
            for ci, shape in enumerate(((1, 10, 18, 18, 4), (2, 10, 9, 18, 4))):
                b, d, h, w, c = shape
                ids = torch.arange(np.prod(shape), dtype=torch.int64).reshape(shape)
                native = window_partition_v2(ids, (2, 9, 9))
                physical = []
                for bi in range(b):
                    for td in range(0, d, 2):
                        for yy in range(0, h, 9):
                            for xx in range(0, w, 9):
                                physical.append(ids[bi, td:td+2, yy:yy+9, xx:xx+9, :].clone())
                physical = torch.stack(physical)
                expected = physical.reshape(2, -1, 9, 9, c)
                require(torch.equal(native, expected), 'native layout vs independent physical enumeration')
                inverse = window_reverse(native.view(-1, 2, 9, 9, c), (2, 9, 9), b, d, h, w)
                require(torch.equal(inverse, ids), 'native inverse round trip')
                m = native.shape[1]
                provenance = []
                same_physical = 0
                for j in range(m):
                    members = []
                    for t in range(2):
                        scalar = int(native[t, j, 0, 0, 0])
                        coords = [int(v) for v in np.unravel_index(scalar, shape)]
                        members.append({'packed_t': t, 'physical_window': (t*m+j)//2,
                                        'local_time': (t*m+j)%2, 'origin': coords})
                    same_physical += members[0]['physical_window'] == members[1]['physical_window']
                    provenance.append({'slot': j, 'members': members})
                layouts.append({'shape': list(shape), 'slots': m, 'same_physical_pair_slots': same_physical,
                                'provenance': provenance, 'roundtrip_exact': True})
                save(f'L{ci}_input', ids); save(f'L{ci}_native', native)
                save(f'L{ci}_physical_time_major_counterfactual', physical.permute(1,0,2,3,4))
                if ci == 0:
                    packed = native.float()
            result['checks']['layout'] = layouts

            psn = PSN(2).eval().requires_grad_(False)
            psn.weight.copy_(torch.tensor([[1.,2.],[4.,8.]])); psn.bias.zero_()
            capture = {}
            handle = psn.surrogate_function.register_forward_pre_hook(lambda m, a: capture.update(h=a[0].clone()))
            spike = psn(packed); handle.remove()
            expected_h = torch.stack((packed[0]+2*packed[1], 4*packed[0]+8*packed[1])).flatten(1)
            require(torch.equal(capture['h'], expected_h), 'T2 native PSN weighted sums')
            require(torch.equal(spike, (expected_h >= 0).float().view_as(packed)), 'T2 PSN threshold')
            save('P2_input', packed); save('P2_prethreshold', capture['h']); save('P2_spike', spike)
            mlp = MS_Spiking_Mlp(4, hidden_features=8, norm_layer='BN', drop=0., **kwargs).eval()
            psn10 = mlp.sn1.spiking_neuron
            psn10.weight.zero_(); psn10.weight[:,9] = 1.; psn10.bias.fill_(-.5)
            p0 = torch.zeros(10,1,2,2,4); p1 = p0.clone(); p1[9,0,0,0,0] = 1.
            y0 = psn10(p0); y1 = psn10(p1)
            expected_change = torch.zeros_like(y0, dtype=torch.bool); expected_change[:,0,0,0,0] = True
            require(torch.equal(y0 != y1, expected_change), 'T10 native PSN complete-time witness')
            for name, value in [('P10_input0',p0),('P10_input1',p1),('P10_output0',y0),('P10_output1',y1)]: save(name,value)
            result['checks']['psn'] = {'T2_prethreshold_exact': True, 'T10_changed_outputs': int((y0 != y1).sum())}

            u = torch.arange(2*4*4*2*2, dtype=torch.float32).reshape(2,4,4,2,2)
            active = [0,2]
            bn = SpikingNormLayer(4, 2, 'BN').eval()
            functional.set_step_mode(bn, 'm'); current_bn(bn)
            full = bn(u); subset = bn(u[:,active])
            u_changed = u.clone(); u_changed[:,1] += 100.
            changed = bn(u_changed)
            arr = u.numpy().astype(np.float64)
            mean = arr.mean(axis=(0,1,3,4),keepdims=True); var = arr.var(axis=(0,1,3,4),keepdims=True)
            ref = (arr-mean) / np.sqrt(var+bn.norm_layer.eps)
            require(np.allclose(full.numpy(),ref,atol=2e-6,rtol=2e-6), 'BN population reference')
            subset_error = float((full[:,active]-subset).abs().max())
            perturb_error = float((full[:,active]-changed[:,active]).abs().max())
            require(subset_error > 1e-3 and perturb_error > 1e-3, 'current BN domain witness')
            fixed = SpikingNormLayer(4, 2, 'BN').eval(); functional.set_step_mode(fixed,'m')
            fixed.norm_layer.running_mean.copy_(torch.arange(4,dtype=torch.float32))
            fixed.norm_layer.running_var.copy_(torch.arange(1,5,dtype=torch.float32))
            fixed_full = fixed(u); fixed_subset = fixed(u[:,active]); fixed_changed = fixed(u_changed)
            require(torch.equal(fixed_full[:,active],fixed_subset), 'fixed-running subset control')
            require(torch.equal(fixed_full[:,active],fixed_changed[:,active]), 'fixed-running perturb control')
            for name,value in [('B_input',u),('B_changed_input',u_changed),('B_full',full),('B_subset',subset),
                               ('B_changed',changed),('B_fixed_full',fixed_full),('B_fixed_subset',fixed_subset),
                               ('B_fixed_changed',fixed_changed),('B_mean',mean),('B_population_variance',var),
                               ('B_subset_mean',u[:,active].numpy().mean(axis=(0,1,3,4))),
                               ('B_subset_population_variance',u[:,active].numpy().var(axis=(0,1,3,4)))]: save(name,value)
            result['checks']['bn'] = {'current_subset_max_abs':subset_error,'inactive_perturb_active_max_abs':perturb_error,
                                      'fixed_running_subset_exact':True,'fixed_running_inactive_perturb_exact':True,
                                      'native_calls':6,'current_population_per_channel':32,'subset_population_per_channel':16}

            torch.manual_seed(402023)
            block = MS_Spiking_SwinTransformerBlock3D(dim=4,input_resolution=(18,18),num_heads=2,
                window_size=(2,9,9),shift_size=(0,0,0),mlp_ratio=2.,norm_layer='BN',
                drop=0.,attn_drop=0.,drop_path=0.,use_checkpoint=False,**kwargs).eval().requires_grad_(False)
            functional.set_step_mode(block,'m')
            for module in block.modules():
                if isinstance(module,SpikingNormLayer): current_bn(module)
            require(type(block.attn).__name__=='Spiking_QK_WindowAttention3D','v2 attention class')
            require(type(block.mlp).__name__=='MS_Spiking_Mlp','v2 MLP class')
            observations = {}; handles = []
            def hook(name):
                def capture_module(module,a,y):
                    observations[name]={'input_shape':list(a[0].shape),'output_shape':list(y.shape)}
                    save('N_'+name.replace('.','_')+'_input',a[0]); save('N_'+name.replace('.','_')+'_output',y)
                return capture_module
            affine_names = ['attn.linear_q','attn.linear_k','attn.proj','mlp.fc1','mlp.fc2']
            selected_names = affine_names + ['attn.proj_sn','attn.sn_q','attn.sn_k','attn.sn2_q','attn.attn_sn',
                'mlp.sn1','mlp.sn2','attn.bn_q','attn.bn_k','attn.proj_bn','mlp.bn1','mlp.bn2']
            for name in selected_names: handles.append(block.get_submodule(name).register_forward_hook(hook(name)))
            nx = torch.randn(1,10,18,18,4)
            try:
                ny = block(nx, None); result['native_block_calls'] += 1
            finally:
                for h in handles: h.remove()
            require(ny.shape==nx.shape and bool(torch.isfinite(ny).all()),'native finite block output')
            for name in affine_names:
                shape = observations[name]['input_shape']
                observations[name]['rows'] = int(np.prod(shape[:-1]))
                require(observations[name]['rows']==3240,'native affine row count '+name)
            for name in ['attn.proj_sn','attn.sn_q','attn.sn_k','attn.sn2_q','attn.attn_sn']:
                require(observations[name]['input_shape'][0]==2,'native attention PSN T2')
            for name in ['mlp.sn1','mlp.sn2']: require(observations[name]['input_shape'][0]==10,'native MLP PSN T10')
            save('N_input',nx); save('N_output',ny)
            for name,value in block.state_dict().items(): save('N_state_'+name,value)
            result['checks']['native_block']={'observations':observations,'classes':[type(block.attn).__name__,type(block.mlp).__name__]}

            def union_axis(a,axis): return np.bitwise_or.reduce(a,axis=axis,keepdims=True)
            def mix_time(a): return np.broadcast_to(union_axis(a,0),a.shape).copy()
            def affine(a): return np.broadcast_to(union_axis(a,-1),a.shape).copy()
            def bn_dep(a,current):
                return np.broadcast_to(np.bitwise_or.reduce(a.reshape(-1,a.shape[-1]),axis=0),a.shape).copy() if current else a
            labels = np.empty((1,10,18,18,4),dtype=np.int64)
            for t in range(10):
                for y in range(18):
                    for x in range(18): labels[0,t,y,x,:]=1<<((t//2)*4+(y//9)*2+x//9)
            dependencies = {}
            for current in (False,True):
                z = window_partition_v2(torch.from_numpy(labels),(2,9,9)).numpy().astype(np.uint64)
                z=mix_time(z)
                q=mix_time(bn_dep(affine(z),current)); k=mix_time(bn_dep(affine(z),current))
                q=q.reshape(2,20,2,-1,2); q=mix_time(union_axis(q,-1)).reshape(20,2,-1,1)
                k=k.reshape(20,2,-1,2)
                a=(k|q).reshape(20,2,2,9,9,2).transpose(2,0,3,4,1,5).reshape(2,20,9,9,4)
                a=bn_dep(affine(a),current)  # attn_sn output is returned as diagnostic only, not used by proj.
                inv=window_reverse(torch.from_numpy(a.astype(np.int64)).reshape(-1,2,9,9,4),(2,9,9),1,10,18,18).numpy().astype(np.uint64)
                residual=inv|labels.astype(np.uint64)
                m=residual.transpose(1,0,2,3,4)
                m=bn_dep(affine(mix_time(m)),current); m=bn_dep(affine(mix_time(m)),current)
                final=residual|m.transpose(1,0,2,3,4)
                parents=list(range(20))
                def find(i):
                    while parents[i]!=i: i=parents[i]
                    return i
                for t in range(10):
                    for y in range(18):
                        for x in range(18):
                            target=(t//2)*4+(y//9)*2+x//9
                            bits=int(np.bitwise_or.reduce(final[0,t,y,x]))
                            for other in range(20):
                                if bits & (1<<other): parents[find(other)]=find(target)
                groups={}
                for i in range(20): groups.setdefault(find(i),[]).append(i)
                tag='current_BN' if current else 'packing_PSN_only_hypothetical_no_BN'
                dependencies[tag]={'components':list(groups.values()),'component_sizes':sorted(map(len,groups.values())),
                                   'interpretation':'structural support superset; not measured numerical influence'}
                save('D_'+tag,final)
            result['checks']['dependency_closure']=dependencies
            result['status']='SEMANTIC_CHECKS_COMPLETE'
    except Exception:
        result['exception']=traceback.format_exc()
    result['elapsed_seconds']=time.monotonic()-started
    result['imported_sources']={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and str(path).endswith('.py') and (str(path).startswith(str(SOURCE)) or str(path).startswith(str(DEPS))):
            result['imported_sources'][name]={'path':str(path),'sha256':sha(path)}
    np.savez_compressed(args.output_dir/'RAW_V1.npz',**raw)
    result['raw_sha256']=sha(args.output_dir/'RAW_V1.npz')
    (args.output_dir/'RESULTS_V1.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'native_block_calls':result['native_block_calls']}))
    if result['status']!='SEMANTIC_CHECKS_COMPLETE': raise SystemExit(2)


if __name__ == '__main__':
    main()
