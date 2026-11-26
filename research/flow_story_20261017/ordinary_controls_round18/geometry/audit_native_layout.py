"""Read-only CPU provenance/dependency audit; no model inference or GT.

The label-union pass is a structural dependency analysis with current BN
deliberately excluded. It is not an activation Jacobian or measured pruning
opportunity. Literal native partition/reverse functions are AST-extracted.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py'
REVERSE = ROOT/'SDformer/third_party/SDformerFlow/models/STSwinNet/swin_transformer3D_v2.py'
CHECKPOINT = ROOT/'research/flow_story_20261016/baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.attn.'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def native_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def partition_np(x):
    b, d, h, w, c = x.shape
    return x.reshape(b, d//2, 2, h//9, 9, w//9, 9, c).transpose(0,1,3,5,2,4,6,7).copy().reshape(2,-1,9,9,c)


def reverse_np(x, b, d, h, w):
    return x.reshape(b,d//2,h//9,w//9,2,9,9,-1).transpose(0,1,4,2,5,3,6,7).copy().reshape(b,d,h,w,-1)


def bits(v):
    return [i for i in range(20) if int(v) & (1 << i)]


def temporal_union(x):
    return np.broadcast_to(np.bitwise_or.reduce(x, axis=0, keepdims=True), x.shape).copy()


def channel_union(x):
    return np.broadcast_to(np.bitwise_or.reduce(x, axis=-1, keepdims=True), x.shape).copy()


def components(dep):
    # Components of the undirected structural tile dependence graph. The
    # residual identity connects each output tile to its same input tile.
    unseen = set(range(20)); out=[]
    while unseen:
        work=[min(unseen)]; group=set()
        while work:
            i=work.pop()
            if i in group: continue
            group.add(i)
            work.extend(j for j in range(20) if (j in dep[i] or i in dep[j]) and j not in group)
        unseen -= group
        out.append(sorted(group))
    return out


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);a=p.parse_args()
    assert not a.output.exists()
    part=native_function(SOURCE, 'window_partition_v2')
    rev=native_function(REVERSE, 'window_reverse')
    tests=[]
    for dims in [(1,2,9,9,1),(1,4,18,27,2),(2,4,18,18,1),(1,10,36,45,1)]:
        x=torch.arange(np.prod(dims), dtype=torch.int64).reshape(dims)
        got=part(x,(2,9,9)); ref=partition_np(x.numpy())
        assert np.array_equal(got.numpy(),ref)
        back=rev(got.reshape(-1,2,9,9,dims[-1]),(2,9,9),*dims[:4])
        assert torch.equal(back,x)
        # Independent scalar unravel for every value in the small ID fixtures.
        b,d,h,w,c=dims; nw=b*(d//2)*(h//9)*(w//9)
        for t in range(2):
            for wi in range(nw):
                outer,inner_t=divmod(t*nw+wi,2)
                bi,dt,hr,wc=np.unravel_index(outer,(b,d//2,h//9,w//9))
                assert torch.equal(got[t,wi],x[bi,2*dt+inner_t,9*hr:9*(hr+1),9*wc:9*(wc+1)])
        tests.append({'shape':dims,'native_equals_numpy':True,'scalar_index_check':True,'partition_reverse_identity':True})

    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    state=ck['model_state_dict']
    pos=state[TARGET+'positional_encoding']
    assert tuple(pos.shape)==(1,12,162,32)
    support={}
    for suffix in ['proj_sn.spiking_neuron.weight','sn_q.spiking_neuron.weight',
                   'sn_k.spiking_neuron.weight','sn2_q.spiking_neuron.weight',
                   'linear_q.weight','linear_k.weight','proj.weight']:
        v=state[TARGET+suffix]
        n=int(torch.count_nonzero(v)); assert n==v.numel()
        support[suffix]={'shape':list(v.shape),'nonzero':n,'elements':v.numel()}
    del ck,state

    b,d,h,w,c=1,10,30,40,384; hp,wp=36,45
    labels=np.zeros((b,d,hp,wp,c),dtype=np.uint32)
    for r in range(4):
        for col in range(5):
            labels[:,:,r*9:min((r+1)*9,h),col*9:min((col+1)*9,w),:]=np.uint32(1<<(r*5+col))
    x=partition_np(labels)
    # Actual active QK forward; temporal PSN and dense Linear are unioned.
    # BN is treated as a local affine operation here ONLY to isolate layout.
    x=temporal_union(x)  # proj_sn
    q=temporal_union(channel_union(x))
    k=temporal_union(channel_union(x))
    t,nw,wh,ww,channels=x.shape; heads=12
    q=q.reshape(t,nw,heads,-1,channels//heads)
    k=k.reshape(nw,heads,-1,channels//heads)
    att_token=temporal_union(np.bitwise_or.reduce(q,axis=-1,keepdims=True))
    attn=np.bitwise_or(k,att_token.reshape(nw,heads,-1,1))
    x=attn.reshape(nw,heads,t,wh,ww,channels//heads).transpose(2,0,3,4,1,5).copy().reshape(t,nw,wh,ww,channels)
    # native attn_sn(x) is computed but not used by returned x.
    x=channel_union(x)  # proj
    out=reverse_np(x.reshape(nw,-1,channels).reshape(-1,2,9,9,channels),b,d,hp,wp)[:,:,:h,:w]
    dep={}
    for r in range(4):
        for col in range(5):
            region=out[:,:,9*r:min(9*(r+1),h),9*col:min(9*(col+1),w)]
            dep[r*5+col]=bits(np.bitwise_or.reduce(region.reshape(-1)))

    # Human-readable exact PSN pair examples use unique input voxel IDs.
    ids=torch.arange(d*hp*wp).reshape(1,d,hp,wp,1)
    packed=part(ids,(2,9,9))
    examples=[]
    for wi in [0,1,18,19,40,41,98,99]:
        coordinates=[list(map(int,np.unravel_index(int(packed[ti,wi,0,0,0]),(d,hp,wp)))) for ti in range(2)]
        examples.append({'packed_batch_column':wi,'two_PSN_time_inputs_at_local_h0w0':coordinates})
    result={'schema':'flow_r18_native_window_layout_audit_v1','status':'PASS_READONLY_CPU',
            'sources':{str(SOURCE):sha(SOURCE),str(REVERSE):sha(REVERSE)},
            'checkpoint_sha256':sha(CHECKPOINT),'script_sha256':sha(Path(__file__)),
            'tests':tests,'actual_target_weight_support':support,'actual_heads':12,
            'original_shape':[b,d,h,w,c],'padded_shape':[b,d,hp,wp,c],
            'literal_partition_shape':[t,nw,wh,ww,channels], 'PSN_pair_examples':examples,
            'tile_input_dependencies_excluding_current_BN':{str(i):j for i,j in dep.items()},
            'undirected_components_excluding_current_BN':components(dep),
            'counts':{'GPU_forwards':0,'GT_numeric_reads':0,'training_steps':0},
            'scope':'Structural potential dependencies from nonzero weights and literal index algebra; not activation sensitivity, current-BN independence, quality, cycles, or a proof of the minimal skip group. Native code unchanged.',
            'current_BN_note':'Actual q/k/proj current BN also couples the full packed domain; the isolated analysis deliberately omits those extra dependencies.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'components_without_current_BN':result['undirected_components_excluding_current_BN'],'examples':examples[:2]}))


if __name__=='__main__':main()
