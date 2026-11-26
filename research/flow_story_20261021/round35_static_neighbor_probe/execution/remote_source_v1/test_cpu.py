"""Geometry and genuine sampled integer/dense parity, zero network forwards."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import static_contract as sc

def run(corepath,contractpath):
    import torch
    torch.set_num_threads(2);torch.manual_seed(35);assert not torch.cuda.is_initialized()
    assert sc.sha(corepath)=='ce07ae04c9f3d58738ae134cdc1c8276542a41a04584720b5d69359e220d1f9f'
    spec=importlib.util.spec_from_file_location('_r35_cpu_frozen_core',corepath);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    c=sc.validate(sc.read(contractpath),core);masks,record=sc.materialize(c,core,True)
    rows={}
    for name in core.TARGETS:
        g=c['geometries'][name];tail,high,meta=core.row_masks(10,1,g['H'],g['W'],torch.device('cpu'),'M',masks[core.group_name(name)])
        expected=291600 if name in core.CONVS else 0
        assert int((high&~tail).sum())==expected
        if name in core.CONVS:assert int(high.sum())==322500 and int((~high).sum())==445500
        rows[name]=dict(rows=len(high),tail=int(tail.sum()),high=int(high.sum()),low=int((~high).sum()),restored=int((high&~tail).sum()))
    name=core.CONVS[0];tail,high,_=core.row_masks(10,1,240,320,torch.device('cpu'),'M',masks[name])
    ids=torch.unique(torch.cat((torch.arange(0,len(tail),997),torch.nonzero(tail).flatten()[:8],torch.nonzero(high&~tail).flatten()[:8],torch.nonzero(~high).flatten()[:8])))
    x=torch.randint(0,2,(len(ids),16)).float();weight=torch.randn(16,16)
    q,s,lq,ls,off,rep=core.representations(weight,'M','rne4')
    kwargs=dict(high8=high[ids],low_q=lq,low_scale=ls,low_offset=off,representation=rep)
    out,_,integer,audit=core.integer_affine(x,q,s,tail[ids],'M',**kwargs)
    ref,_,refinteger,_=core.integer_affine(x,q,s,tail[ids],'M',reference=True,**kwargs)
    assert torch.equal(integer,refinteger) and torch.equal(out,ref)
    for bit in range(8):assert sum(v['rows'] for v in audit['kernel_calls'] if v['bit']==bit)==int(high[ids].sum())+(int((~high[ids]).sum()) if bit>=4 else 0)
    rejected=[]
    for kind in ['151_contract','fragment','tail']:
        try:
            if kind=='151_contract':
                bad=copy.deepcopy(c);bad['selected_complete_tiles_per_patch']=151;sc.validate(bad,core)
            else:
                bad=masks[name].clone();bad[0,0]=False if kind=='fragment' else bad[0,0]
                if kind=='tail':bad[239,319]=True
                core.row_masks(10,1,240,320,torch.device('cpu'),'M',bad)
        except ValueError:rejected.append(kind)
        else:raise AssertionError('Bad contract accepted '+kind)
    return dict(status='PASS',GPU_used=False,neural_network_forwards=0,core_sha256=sc.sha(corepath),contract_sha256=sc.sha(contractpath),
        all28_geometry_rows=rows,selectedmask=record,real_sampled_integer_rows=len(ids),real_integer_and_dense_equal=True,rejected=rejected)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path,required=True);ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists();result=run(a.core,a.contract);a.output.write_text(json.dumps(result,indent=2)+'\n');print('R35_CPU_PASS',result['real_sampled_integer_rows'])
