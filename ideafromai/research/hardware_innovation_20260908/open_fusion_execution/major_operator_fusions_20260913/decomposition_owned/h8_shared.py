"""One pair or omitted coordinate per physical H8, no per-output selectors."""
import itertools,json
import numpy as np
from prototype import HERE,load_capture,error
from count_basis import group,ungroup,execute_pair

def main():
    w,theta,bias,frames=load_capture();wg=group(w).reshape(12,8,216,4)
    pairs=np.array(list(itertools.combinations(range(4),2)))
    costs=np.stack([((wg[...,i]-wg[...,j])**2/2).sum(1) for i,j in pairs])
    chosen=np.argmin(costs,axis=0)
    pp=pairs[chosen]
    pair=np.repeat(pp[:,None],8,axis=1).reshape(96,216,2)
    z=group(w);aa=np.mean(np.take_along_axis(z,pair,axis=-1),axis=-1).astype(np.float32).astype(float)
    retained=z.copy();np.put_along_axis(retained,pair,0,axis=-1);retained=ungroup(retained)
    part=np.zeros_like(z);np.put_along_axis(part,pair,np.repeat(aa[:,:,None],2,axis=-1),axis=-1)
    wh=ungroup(part)+retained;signs=np.ones((216,4),dtype=np.int8)
    omitted=np.argmin(np.sum(wg*wg,axis=1),axis=-1)
    omit=np.repeat(omitted[:,None],8,axis=1).reshape(96,216,1)
    w34=z.copy();np.put_along_axis(w34,omit,0,axis=-1);w34=ungroup(w34)
    out=HERE/'count_results'
    np.savez_compressed(out/'unsigned_h8_shared_pair_w32.npz',a=aa.astype(np.float32),pair=pair.astype(np.int8),pair_h8=pp.astype(np.int8),
        signs=signs,retained=retained.astype(np.float32),weight=wh.astype(np.float32),theta=np.array(theta),bias=bias.astype(np.float32),coefficient_bits=np.array(32))
    np.savez_compressed(out/'plain34_h8_shared_w32.npz',weight=w34.astype(np.float32),omitted=omitted.astype(np.int8),bias=bias.astype(np.float32),coefficient_bits=np.array(32))
    with np.load(HERE.parent/'root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz') as zc:
        cur=('matched_dense_firstframe',zc['input'].reshape(-1,864).astype(float),zc['output'].reshape(-1,96).astype(float))
    rows=[]
    for name,ww in [('unsigned_h8_shared_pair_w32',wh),('plain34_h8_shared_w32',w34)]:
        vals=[]
        for n,x,y in frames+[cur]:
            yp=x@ww.reshape(96,-1).T+bias
            value=dict(file=n,**error(yp,y))
            if name.startswith('unsigned'):
                pe,ledger=execute_pair(x,aa,signs,retained,pair,theta,bias)
                err=float(np.max(np.abs(yp-pe)));assert err<1e-10
                value.update(ledger,factorized_max_abs=err)
                value['generic_private_decode_counter_not_h8_service']=value.pop('consumer_pair_decode_requests_per_position')
                value['hardware_decode_scope']='Generic execute_pair supplies numerical/AAC counts only. H8 selector reuse and control cost are in ../paid_schedule.json.'
            else:
                value['total_fullwidth_AAC_per_position']=float(np.mean((x!=0)@np.count_nonzero(ww,axis=0).reshape(-1)))
            vals.append(value)
        rows.append(dict(name=name,frames=vals,local_holdout2_relative_l2_mean=float(np.mean([v['relative_l2'] for v in vals[2:4]])),
            static_coefficient_bytes=62208*4+96*4,metadata_bytes=(12*216*(3 if name.startswith('unsigned') else 2)/8),
            group_scope='One shared pair/omitted-coordinate per H8 output tile and input-channel-4/spatial-tap group',fit='Weight-only exhaustive 6-pair/4-index choice. No GT or capture fitting.'))
        print(name,vals[-1],flush=True)
    (out/'h8_shared_summary.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
