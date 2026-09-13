"""Spatial-neighbor M batches, original official full K/N, no APEC roots."""
import sys
sys.dont_write_bytecode=True
import json, numpy as np, torch
from execute import HERE,prior,capture,load_official_api

torch.set_num_threads(1);torch.set_num_interop_threads(1)
Accelerator,FC,Simulator,_=load_official_api()
result=dict(interface='same-t spatial M batches into original PPU; destination scatter restores original p,t coordinates',
    choice='Fixed architecture-derived T-major row permutation, no sample-dependent selection',
    preserves='M256/K16/N128, original weights and K order, all3000 destinations and all768 outputs',samples={})
for sample in (1,9):
    a,matrix=capture(sample)
    # Undo the official API entry transpose so its actual internal input has
    # neighboring spatial rows at a fixed timestep. No row value is changed.
    adapted=matrix.reshape(300,10,6912).transpose(1,0,2).reshape(3000,6912)
    relation=prior.Relation();acc=Accelerator('Prosperity',128,32,256,16,product_sparsity=True,issue_type=2,mem_if_width=1024)
    got,stdout,elapsed=prior.run(FC,Simulator,acc,adapted,10,300,768,relation)
    ref,counts,residency=prior.reference(matrix,acc,True,relation)
    assert got==ref
    p_all=[];r_all=[];o_all=[];comp=[];prep=[]
    for mi,m in enumerate(range(0,3000,256)):
        pp=[];rr=[];oo=[];cc=[];pc=[]
        for k in range(0,6912,16):
            tile=matrix[m:min(m+256,3000),k:k+16]
            residual,p,c=relation.entry(tile)
            pp.append(p.numpy().astype(np.int16))
            rr.append((residual.numpy().astype(np.uint16)*(1<<np.arange(16,dtype=np.uint16))).sum(1,dtype=np.uint16))
            oo.append(np.lexsort((np.arange(len(tile)),tile.sum(1))).astype(np.uint16))
            cc.append(c['residual_nnz']+c['equal_rows']);pc.append(c['preprocess'])
        p_all.append(np.stack(pp));r_all.append(np.stack(rr));o_all.append(np.stack(oo));comp.extend(cc*6);prep.extend(pc*6)
    row=dict(official_outer_counts=got,relation_counts_one_N_slice=counts,official_stdout=stdout,
        host_seconds_not_cycles=elapsed,finite_tile_precompute_pipeline_cycles=int(np.maximum(comp,prep).sum()+prep[0]),
        unchanged_full_output_bytes_INT8=3000*768,unchanged_full_output_bytes_FP32=3000*768*4,
        per_output_row_scatter_address_ops=3000*6,
        no_extra_weight_permutation_bytes=True,
        streaming_T10_completion_limit='Layer-level outputs can retain the original endpoint. A streaming noncausal consumer must retain at least its required completed timestep outputs; this model does not infer that store to be free.')
    result['samples'][str(sample)]=row
    if sample==1:
        ridx=(np.arange(3000)%300)*10+np.arange(3000)//300
        np.savez(HERE/'spatial_major_numeric_plan.npz',perm=np.arange(6912),row_index=ridx,
                 **{f'p{i}':p for i,p in enumerate(p_all)},**{f'r{i}':r for i,r in enumerate(r_all)},
                 **{f'o{i}':o for i,o in enumerate(o_all)})
    print(sample,row,flush=True)
    (HERE/'spatial_results.json').write_text(json.dumps(result,indent=2)+'\n')
