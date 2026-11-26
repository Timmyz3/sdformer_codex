#!/usr/bin/env python3
"""Read-only v2 CPU layout/dependency audit; writes only its own JSON output.

No model imports, checkpoint loading, CUDA, training, or source edits. Actual AST
function bodies execute through a small NumPy tensor adapter. Neural operators
in the QK audit are identity/constant probes, NOT trained numerical inference.
The dependency audit uses a head-dimension quotient (32 -> 1) with full-channel
producer groups; a separate actual-width probe checks that quotient exactly.
Compatible with the host's Python 3.6 / NumPy 1.19.
"""
import argparse
import ast
import collections
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types
import urllib.request

import numpy as np

ROOT = pathlib.Path('/home/zhumd/work/sdformer_codex')
SOURCE = ROOT / 'SDformer/third_party/SDformerFlow'
HERE = pathlib.Path(__file__).resolve().parent
FILES = {
    'spiking_layout': 'models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
    'ann_layout': 'models/STSwinNet/swin_transformer3D_v2.py',
    'psn': 'models/STSwinNet_SNN/Spiking_submodules.py',
    'normalization': 'models/STSwinNet_SNN/Spiking_modules.py',
    'evaluation': 'eval_DSEC_flow_SNN.py',
    'training': 'train_flow_parallel_supervised_SNN.py',
    'config': 'configs/train_DSEC_supervised_SDformerFlow_en4.yml',
}


class Tensor:
    """Only the Tensor operations occurring in the extracted functions."""
    def __init__(self, value):
        self.a = np.asarray(value)

    @property
    def shape(self):
        return self.a.shape

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        # Source .view calls require contiguous tensors. All audit calls satisfy
        # this, except singleton-axis views which NumPy/PyTorch both support.
        return Tensor(self.a.reshape(shape))

    reshape = view

    def permute(self, *axes):
        return Tensor(self.a.transpose(axes))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.a))

    def float(self):
        # Preserve exact int64 IDs. The audit does not test floating-point error.
        return self

    def flatten(self, start_dim=0):
        return Tensor(self.a.reshape(self.a.shape[:start_dim] + (-1,)))

    def sum(self, dim=None, keepdim=False):
        return Tensor(self.a.sum(axis=dim, keepdims=keepdim))

    def mul(self, other):
        return Tensor(self.a * asarray(other))

    def __add__(self, other):
        return Tensor(self.a + asarray(other))


def asarray(x):
    return x.a if isinstance(x, Tensor) else np.asarray(x)


class TorchAdapter:
    @staticmethod
    def addmm(bias, weight, x):
        return Tensor(asarray(bias) + np.matmul(asarray(weight), asarray(x)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_node(text, name, class_name=None):
    tree = ast.parse(text)
    scope = tree.body
    if class_name:
        scope = next(x for x in tree.body
                     if isinstance(x, ast.ClassDef) and x.name == class_name).body
    return next(x for x in scope
                if isinstance(x, ast.FunctionDef) and x.name == name)


def extract(path, name, class_name=None):
    text = path.read_text()
    node = get_node(text, name, class_name)
    # Drop type annotations only (torch.Tensor is unavailable). Body is unchanged.
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    try:
        module = ast.Module(body=[node], type_ignores=[])
    except TypeError:
        module = ast.Module(body=[node])
    scope = {'torch': TorchAdapter}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope[name]


partition = extract(SOURCE / FILES['spiking_layout'], 'window_partition_v2')
reverse = extract(SOURCE / FILES['ann_layout'], 'window_reverse')
window_size = extract(SOURCE / FILES['ann_layout'], 'get_window_size')
native_qk = extract(SOURCE / FILES['spiking_layout'], 'forward',
                    'Spiking_QK_WindowAttention3D')
native_psn = extract(SOURCE / FILES['psn'], 'forward', 'PSN')


def qk_probe(x, heads):
    """Execute unmodified QK forward AST, exposing only K-path layout.

    All SN/linear/dropout operators are identity; sn2_q returns ones. Thus the
    real reshape/multiply/permute/reverse path carries IDs without changing them.
    norm_layer=None bypasses numerical normalization for layout inspection only.
    """
    identity = lambda value: value
    T, _, H, W, C = x.shape
    obj = types.SimpleNamespace(
        num_heads=heads, norm_layer=None,
        positional_encoding=Tensor(np.zeros((1, heads, T*H*W, C//heads),
                                            dtype=x.a.dtype)),
        proj_sn=identity, linear_q=identity, sn_q=identity,
        linear_k=identity, sn_k=identity,
        sn2_q=lambda value: Tensor(np.ones(value.shape, dtype=value.a.dtype)),
        attn_drop=identity, attn_sn=identity, proj=identity)
    return native_qk(obj, x)[0]


def prepare(B, D, H, W, shift, C=1):
    ws, shift = window_size((D, H, W), (2, 9, 9), shift)
    Dp, Hp, Wp = [((n+w-1)//w)*w for n, w in zip((D, H, W), ws)]
    ids = np.arange(B*D*H*W*C, dtype=np.int64).reshape(B, D, H, W, C)
    padded = np.pad(ids, ((0,0),(0,Dp-D),(0,Hp-H),(0,Wp-W),(0,0)),
                    mode='constant', constant_values=-1)
    shifted = np.roll(padded, tuple(-x for x in shift), axis=(1,2,3))
    return ids, Tensor(shifted), ws, shift, (Dp, Hp, Wp)


def reverse_to_original(x, B, D, H, W, ws, shift, padded):
    Dp, Hp, Wp = padded
    C = x.shape[-1]
    full = reverse(x.view(-1, *(ws+(C,))), ws, B, Dp, Hp, Wp).a
    full = np.roll(full, shift, axis=(1,2,3))
    return full[:, :D, :H, :W, :]


def coord(uid, D, H, W):
    if int(uid) < 0:
        return None
    return [int(x) for x in np.unravel_index(int(uid), (1000, D, H, W))]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def root(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.root(a), self.root(b)
        if ra != rb:
            self.p[rb] = ra


def closure_for_groups(deps, group_ids, B, H, W, heads, all_time_groups=True):
    """Connected consumers of shared full-channel Q/K producer pairs.

    Atomic requested output group = 9x9 spatial tile across ALL real T. Every
    group's full-channel projection is retained/cancelled together. Connected
    components guarantee no cancelled Q/K producer group is needed by a retained
    attention/MLP group, under frozen affine normalization and arbitrary weights.
    Input residual values are not cancelled by this test. Coarser whole-window
    hardware may require further merging; component membership need not be local.
    """
    actual = np.unique(group_ids)
    compressed = np.searchsorted(actual, group_ids)
    ng = len(actual)
    consumer = np.broadcast_to(compressed, deps.shape)
    codes = np.unique(deps.reshape(-1)*ng + consumer.reshape(-1))
    dsu = DSU(ng)
    last_pair = -1
    first = -1
    for code in codes:
        producer, group = divmod(int(code), ng)
        if producer != last_pair:
            last_pair, first = producer, group
        else:
            dsu.union(first, group)
    buckets = collections.defaultdict(list)
    for g in range(ng):
        buckets[dsu.root(g)].append(g)
    counts = sorted(len(v) for v in buckets.values())
    labels = np.asarray([dsu.root(g) for g in range(ng)], dtype=np.int64)
    time_space = labels[compressed[:, :, :, :, 0]]
    spatial = time_space[:,0] if all_time_groups else time_space
    pixels = collections.Counter(int(x) for x in spatial.reshape(-1))
    batch_components = collections.defaultdict(set)
    for b in range(B):
        for label in np.unique(spatial[b]):
            batch_components[int(label)].add(b)
    sizes = sorted(pixels.values())
    return {
        'requested_full_time_spatial_groups' if all_time_groups else 'requested_spatiotemporal_window_groups': ng,
        'connected_components': len(counts),
        'groups_per_component_histogram': dict(collections.Counter(counts)),
        'real_spatial_sites_per_component_min_max' if all_time_groups else 'real_spatiotemporal_sites_per_component_min_max': [sizes[0], sizes[-1]],
        'largest_fraction_of_batch_real_spatial_sites' if all_time_groups else 'largest_fraction_of_batch_real_spatiotemporal_sites': sizes[-1]/float(spatial.size),
        'components_crossing_batch_examples': sum(len(v)>1 for v in batch_components.values()),
        'all_requested_groups_independent': len(counts) == ng,
        'example_component_original_group_ids': [
            [int(actual[g]) for g in bucket[:40]]
            for bucket in list(buckets.values())[:3]],
        'example_membership_truncated_at_40': any(len(v)>40 for v in list(buckets.values())[:3]),
    }


def run_case(label, stage, B, D, H, W, C, heads, shift):
    ids, padded_ids, ws, actual_shift, padded = prepare(B,D,H,W,shift)
    parts = partition(padded_ids, ws)
    roundtrip = reverse_to_original(parts, B,D,H,W,ws,actual_shift,padded)
    assert np.array_equal(roundtrip, ids)
    lanes = parts.a[:, :, :, :, 0]
    a, b = lanes[0], lanes[1]
    valid = (a>=0) & (b>=0)
    ac = np.array(np.unravel_index(a[valid], (B,D,H,W)))
    bc = np.array(np.unravel_index(b[valid], (B,D,H,W)))
    same_batch = ac[0] == bc[0]
    same_spatial = (ac[2]==bc[2]) & (ac[3]==bc[3])
    same_nominal_tpair = (ac[1]//2)==(bc[1]//2)
    examples = []
    for idx in np.argwhere(valid)[:3]:
        i,j,k = [int(z) for z in idx]
        examples.append({'partition_slot_y_x':[i,j,k],
                         'input_tlane0_original_b_t_y_x':coord(a[i,j,k],D,H,W),
                         'input_tlane1_original_b_t_y_x':coord(b[i,j,k],D,H,W)})

    # Run the ACTUAL PSN forward AST with a time-swap matrix and identity
    # surrogate. This tests pairing, not the checkpoint's spike values.
    psn = types.SimpleNamespace(weight=Tensor(np.array([[0,1],[1,0]],dtype=np.int64)),
                                bias=Tensor(np.zeros((2,1),dtype=np.int64)),
                                surrogate_function=lambda x:x)
    swapped = native_psn(psn, parts)
    assert np.array_equal(swapped.a[0], parts.a[1])
    assert np.array_equal(swapped.a[1], parts.a[0])

    T, N, wh, ww, _ = parts.shape
    # Full-channel Q/K projections before their PSN share all output consumers
    # of the paired T lanes. Therefore pair IDs omit the T dimension.
    pair = np.arange(N*wh*ww, dtype=np.int64).reshape(1,N,wh,ww,1)
    pair = np.broadcast_to(pair, (T,N,wh,ww,heads)).copy()
    deps = reverse_to_original(qk_probe(Tensor(pair), heads),
                               B,D,H,W,ws,actual_shift,padded)
    yy, xx = np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    Hp, Wp = padded[1:]
    ny, nx = Hp//wh, Wp//ww
    results = {}
    for grid_name, sy, sx in [('unshifted_output_tiles',0,0),
                              ('current_shift_aligned_output_tiles',actual_shift[1],actual_shift[2])]:
        local = ((yy-sy)%Hp//wh)*nx + ((xx-sx)%Wp//ww)
        group = np.stack([local+bidx*ny*nx for bidx in range(B)],axis=0)
        group = np.broadcast_to(group[:,None,:,:,None], (B,D,H,W,1))
        results[grid_name] = closure_for_groups(deps,group,B,H,W,heads)

    # Attention-only alternative: no MLP T=10 linkage. Output atoms follow
    # current shifted 2x9x9 windows, still full channel and full producer pairs.
    Dp=padded[0]
    temporal=((np.arange(D)-actual_shift[0])%Dp)//ws[0]
    local=((yy-actual_shift[1])%Hp//wh)*nx + ((xx-actual_shift[2])%Wp//ww)
    group=temporal[None,:,None,None]*ny*nx + local[None,None,:,:]
    group=group+np.arange(B)[:,None,None,None]*(Dp//ws[0])*ny*nx
    attention_only=closure_for_groups(deps,group[:,:,:,:,None],B,H,W,heads,False)

    # Trace K-path identities with one quotient element per native 32-channel
    # chunk. This is NOT an assertion that native attention is identity.
    token_ids = np.broadcast_to(parts.a, (T,N,wh,ww,heads)).copy()
    qk_ids = reverse_to_original(qk_probe(Tensor(token_ids),heads),
                                B,D,H,W,ws,actual_shift,padded)
    intended = np.broadcast_to(ids, qk_ids.shape)
    mismatch = qk_ids != intended
    return {
        'input':label,'stage':stage,'batch':B,'T':D,'feature_HWC':[H,W,C],
        'heads':heads,'requested_shift':list(shift),'actual_shift':list(actual_shift),
        'actual_window':list(ws),'padded_DHW':list(padded),
        'partition_shape_C1':list(parts.shape),
        'partition_reverse_roundtrip_exact':True,
        'actual_psn_ast_swap_probe_exact':True,
        'psn_pairs_both_real':int(valid.sum()),
        'psn_pairs_cross_batch':int((~same_batch).sum()),
        'psn_pairs_different_spatial_site':int((~same_spatial).sum()),
        'psn_pairs_same_batch_spatial_adjacent_nominal_time':int((same_batch & same_spatial & same_nominal_tpair).sum()),
        'psn_pair_examples':examples,
        'qk_identity_operator_probe_fraction_mismatched_headchunks':float(mismatch.mean()),
        'qk_identity_operator_probe_meaning':'K-path layout only; nonlinear/linear operations replaced by probes, so this is not a trained-output error',
        'frozen_affine_bn_joint_attention_mlp_closure':results,
        'frozen_affine_bn_attention_only_current_shift_window_closure':attention_only,
        'current_stat_bn_joint_contract':'global BN statistics retain all required producer values; local closed components above do not apply',
    }


def actual_width_quotient_test():
    records=[]
    for heads in [3,6,12,24]:
        B,D,H,W=2,4,9,18
        _, x, ws, shift, padded=prepare(B,D,H,W,(0,0,0))
        p=partition(x,ws)
        small=np.broadcast_to(p.a,p.shape[:-1]+(heads,)).copy()
        actual=np.repeat(small,32,axis=-1)
        small_out=qk_probe(Tensor(small),heads).a
        actual_out=qk_probe(Tensor(actual),heads).a
        exact=np.array_equal(actual_out,np.repeat(small_out,32,axis=-1))
        assert exact
        records.append({'heads':heads,'native_C':heads*32,
                        'shape_BDHW':[B,D,H,W],'quotient_layout_exact':bool(exact)})
    return records


def minimal_counterexample():
    # Two physical time windows, one spatial site: PSN pairs t=0 with t=2.
    x=Tensor(np.arange(4,dtype=np.int64).reshape(1,4,1,1,1))
    p=partition(x,(2,1,1)).a
    # Two batch samples: source reshaping pairs b=0 with b=1.
    xb=Tensor(np.arange(4,dtype=np.int64).reshape(2,2,1,1,1))
    pb=partition(xb,(2,1,1)).a
    return {'temporal_input_shape':[1,4,1,1,1],
            'temporal_partition_T_by_slot':p[:,:,0,0,0].tolist(),
            'expected_physical_adjacent_T_by_slot':[[0,2],[1,3]],
            'batch_input_shape':[2,2,1,1,1],
            'batch_partition_T_by_slot':pb[:,:,0,0,0].tolist(),
            'batch_ID_legend':{'0':'b0t0','1':'b0t1','2':'b1t0','3':'b1t1'}}


def mlp_numpy_probe():
    """Numerical structure test of dense T-PSN -> FC -> BN repeated twice.

    Random synthetic weights only, threshold forward (not surrogate gradients),
    no learned-model accuracy claim. Spatial subsetting is exact for frozen BN.
    """
    rng=np.random.RandomState(20261011)
    T,B,H,W,C,M=10,2,18,18,3,7
    x=rng.randn(T,B,H,W,C)
    A1=rng.randn(T,T); A2=rng.randn(T,T)
    W1=rng.randn(C,M); W2=rng.randn(M,C)
    bias1=rng.randn(T,1,1,1,1); bias2=rng.randn(T,1,1,1,1)
    def normalize(z,policy):
        if policy=='frozen':
            return z*0.7+0.13
        axes=tuple(range(z.ndim-1))
        return (z-z.mean(axis=axes,keepdims=True))/np.sqrt(z.var(axis=axes,keepdims=True)+1e-5)
    def f(inp,policy):
        # Retain dimensions when selecting rows, hence bias shape is stable.
        z=(np.tensordot(A1,inp,axes=(1,0))+bias1>0).astype(float)
        z=normalize(np.matmul(z,W1),policy)
        z=(np.tensordot(A2,z,axes=(1,0))+bias2>0).astype(float)
        return normalize(np.matmul(z,W2),policy)
    full=f(x,'frozen'); part=f(x[:,:,:9,:9,:],'frozen')
    max_err=float(np.max(np.abs(full[:,:,:9,:9,:]-part)))
    assert max_err==0.0
    current=f(x,'current'); current_part=f(x[:,:,:9,:9,:],'current')
    current_err=float(np.max(np.abs(current[:,:,:9,:9,:]-current_part)))
    assert current_err>0.0
    two_steps_missing=x.copy();two_steps_missing[:2]=0
    time_err=float(np.max(np.abs(full[2:]-f(two_steps_missing,'frozen')[2:])))
    assert time_err>0
    return {'shape_TBHWC':[T,B,H,W,C],'all_time_9x9_tile_frozen_BN_max_abs_error':max_err,
            'all_time_9x9_tile_current_BN_max_abs_error':current_err,
            'removing_2_time_inputs_changes_retained_time_outputs_max_abs':time_err,
            'interpretation':'synthetic operator-structure CPU probe only; no checkpoint and no AEE result',
            'current_BN_axes_in_probe':'all leading axes T,B,H,W; actual SpikingJelly version domain requires environment pinning; spatial coupling holds even for per-time BN'}


def command(args):
    return subprocess.check_output(args,cwd=str(SOURCE),universal_newlines=True).strip()


def provenance(check_upstream):
    records={}
    for key,rel in FILES.items():
        p=SOURCE/rel
        records[key]={'path':str(p),'sha256':sha(p.read_bytes())}
    meta={'local_git_head':command(['git','rev-parse','HEAD']),
          'local_git_top':command(['git','rev-parse','--show-toplevel']),
          'local_origin':command(['git','config','--get','remote.origin.url']),
          'targeted_git_status':command(['git','status','--short','--']+list(FILES.values())),
          'files':records,
          'official_repository':'https://github.com/yitian97/SDformerFlow',
          'upstream_status':'not requested'}
    if check_upstream:
        try:
            try:
                req=urllib.request.Request('https://api.github.com/repos/yitian97/SDformerFlow/commits?per_page=1',
                                           headers={'User-Agent':'flow-cpu-preflight-audit'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    latest=json.loads(response.read().decode())[0]
                commit=latest['sha'];date=latest['commit']['committer']['date']
                meta['upstream_pin_method']='GitHub commits API'
            except Exception as exc:
                meta['github_api_error']=repr(exc)
                remote=subprocess.check_output(['git','ls-remote',
                    'https://github.com/yitian97/SDformerFlow.git','HEAD'],
                    universal_newlines=True,timeout=30)
                commit=remote.split()[0];date='not fetched'
                meta['upstream_pin_method']='read-only git ls-remote HEAD after API failure'
            matches={}
            for key in ['spiking_layout','ann_layout','psn','normalization']:
                rel=FILES[key]
                url='https://raw.githubusercontent.com/yitian97/SDformerFlow/{}/{}'.format(commit,rel)
                req=urllib.request.Request(url,headers={'User-Agent':'flow-cpu-preflight-audit'})
                with urllib.request.urlopen(req,timeout=15) as response:
                    raw=response.read()
                matches[key]={'url':url,'sha256':sha(raw),'whole_file_matches_local':sha(raw)==records[key]['sha256']}
                funcs={
                    'spiking_layout':[('window_partition_v2',None),('forward','Spiking_QK_WindowAttention3D'),('forward','MS_Spiking_Mlp')],
                    'ann_layout':[('window_reverse',None),('get_window_size',None)],
                    'psn':[('forward','PSN')],
                    'normalization':[('__init__','SpikingNormLayer')],
                }[key]
                matches[key]['function_AST_checks']={}
                for name,cls in funcs:
                    local_ast=ast.dump(get_node((SOURCE/rel).read_text(),name,cls))
                    upstream_ast=ast.dump(get_node(raw.decode(),name,cls))
                    matches[key]['function_AST_checks'][(cls+'.' if cls else '')+name]={
                        'same_AST':local_ast==upstream_ast,
                        'local_AST_sha256':sha(local_ast.encode()),
                        'upstream_AST_sha256':sha(upstream_ast.encode())}
            meta.update(upstream_status='checked pinned raw sources',upstream_commit=commit,
                        upstream_commit_date=date,upstream_files=matches)
        except Exception as exc:
            meta.update(upstream_status='verification incomplete',upstream_error=repr(exc))
    return meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-upstream',action='store_true')
    parser.add_argument('--output',default=str(HERE/'v2_window_layout_results.json'))
    args=parser.parse_args()
    prov=provenance(args.check_upstream)
    cases=[]
    for label,hs,ws in [('crop_288x384',[72,36,18,9],[96,48,24,12]),
                        ('full_480x640',[120,60,30,15],[160,80,40,20])]:
        for stage,(H,W,C,heads) in enumerate(zip(hs,ws,[96,192,384,768],[3,6,12,24])):
            for B in [1,2]:
                for shift in [(0,0,0),(1,4,4)]:
                    case=run_case(label,stage,B,10,H,W,C,heads,shift)
                    cases.append(case)
                    closure=case['frozen_affine_bn_joint_attention_mlp_closure']['unshifted_output_tiles']
                    print('{} s{} B{} shift{}: roundtrip OK; cross_batch {}; joint groups {} -> {} components'.format(
                        label,stage,B,case['actual_shift'],case['psn_pairs_cross_batch'],
                        closure['requested_full_time_spatial_groups'],closure['connected_components']),flush=True)
    ledger=json.loads((ROOT/'research/flow_story_20261010/FLOW_V2_STATIC_COST_SPLIT_20261011.json').read_text())['compute']
    mlp=sum(v for k,v in ledger['detail'].items() if k.endswith('_mlp'))
    attention=sum(v for k,v in ledger['detail'].items() if k.endswith('_attn'))
    total=ledger['dense_MAC_PE_cycles_UPPER']
    result={
        'audit_utc':datetime.datetime.utcnow().isoformat()+'Z',
        'execution':{'python':sys.version,'numpy':np.__version__,'device':'CPU NumPy only',
                     'torch_imported':False,'GPU_used':False,'old_source_modified':False,
                     'no_checkpoint':True,'script_sha256':sha(pathlib.Path(__file__).read_bytes())},
        'provenance':prov,
        'method':{'AST_extracted_exact_bodies':['window_partition_v2','window_reverse','get_window_size','Spiking_QK_WindowAttention3D.forward','PSN.forward'],
                  'adapter_limitations':'only layout and PSN pairing are original AST execution; QK neural ops replaced by explicit identity/one probes; dependency labels are full-channel producer groups; no trained numerical forward',
                  'closure_definition':'per-block full-time spatial output groups merged when they share a full-channel paired-T Q/K producer; frozen affine BN and disabled dropout required; no claim of minimal arbitrary fine-grained scheduling',
                  'padding':'-1 unique-ID sentinel, excluded as real output; padded intermediates may still be needed by real consumers',
                  'batch_warning':'B2 is tested to detect cross-example coupling, not proposed as deployment batch'},
        'minimal_counterexample':minimal_counterexample(),
        'actual_width_quotient_checks':actual_width_quotient_test(),
        'mlp_operator_probe':mlp_numpy_probe(),
        'MLP_only_static_upper_bound':{'MLP_dense_work':mlp,'full_dense_work':total,'fraction':mlp/float(total),
            'ideal_max_speedup_zero_MLP':1/(1-mlp/float(total)),
            'MLP_work_fraction_to_cancel_for_net_10pct_without_overhead':.10/(mlp/float(total)),
            'scope':'old crop ledger only, frozen-BN deployment variant required for whole local MLP producer cancellation; not cycles, latency or accuracy measurement'},
        'attention_only_static_upper_bound':{'attention_dense_work':attention,'full_dense_work':total,
            'fraction':attention/float(total),
            'can_reach_10pct_full_model_work_reduction_under_this_ledger_alone':attention/float(total)>=0.10,
            'scope':'old crop dense ledger; not a full-resolution hardware measurement'},
        'case_count':len(cases),'cases':cases,
    }
    assert len(cases)==32
    with open(args.output,'w') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('Wrote {}'.format(args.output),flush=True)


if __name__=='__main__':
    main()
