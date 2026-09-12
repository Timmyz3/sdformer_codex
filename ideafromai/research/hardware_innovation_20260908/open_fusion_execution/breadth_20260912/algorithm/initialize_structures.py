"""Equal1024-step TRAIN-moment initialization from one common ordinary parent."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    base=args.root;chain=base/'algorithm/patch_probe/residual_consumer_probe/projection_chain'
    sys.path.insert(0,str(chain))
    import torch
    from torch import nn
    from fast_temporal_basis import MATCHINGS,fit_readout
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(912)
    op=base/'open_fusion_execution';parent=op/'stage_20260912/algorithm/hardware_exports/ordinary/deployed_constants.npz'
    with np.load(parent) as z:common={k:z[k].copy() for k in z.files}
    old=json.loads((chain/'fast_temporal_recovery/initialization.json').read_text())
    moments=old['source_moments'];mean=np.asarray(moments['mean']);cov=np.asarray(moments['covariance'])
    teacher=common['As_q16'].astype(np.float64)/(2**int(common['As_exponent']))
    tau=common['source_threshold'].astype(np.float64)/16384
    permutation=np.asarray(old['source_fit']['row_permutation'],np.int64);inverse=np.argsort(permutation)
    block=np.zeros((10,10));
    for ids in [[0,1,2],[3,4,5],[6,7,8,9]]:block[np.ix_(ids,ids)]=1
    order=torch.as_tensor(permutation);sigma=torch.as_tensor(cov);target=torch.as_tensor(teacher)
    class Structure(nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            if kind=='lifting40':
                self.coeff=nn.Parameter(torch.zeros(4,5,2))
                fit=fit_readout(mean,cov,teacher,np.zeros(10),np.eye(10),row_permutation=permutation)
                self.gain=nn.Parameter(torch.as_tensor(fit['row_gain']).clone())
            else:self.coeff=nn.Parameter(torch.as_tensor(teacher[inverse]*(block if kind=='contiguous34' else 1)).clone())
        def matrix(self):
            if self.kind!='lifting40':return (self.coeff*(torch.as_tensor(block) if self.kind=='contiguous34' else 1))[order]
            result=torch.eye(10)
            for layer,pair in enumerate(MATCHINGS):
                i,j=np.asarray(pair).T;x,y=result[i],result[j]
                a,b=self.coeff[layer,:,0,None],self.coeff[layer,:,1,None]
                u=x+a*y;v=y+b*u
                result=result.index_copy(0,torch.as_tensor(i),u).index_copy(0,torch.as_tensor(j),v)
            return self.gain[:,None]*result[order]
    out=HERE/'initialization';out.mkdir(parents=True,exist_ok=True)
    record=dict(complete=False,parent=str(parent),common_prior_GT_updates=320,
        initializer=dict(steps=1024,optimizer='Adam',lr=.01,seed=912,source_moments=str(chain/'fast_temporal_recovery/initialization.json'),
            scope='Existing complete TRAIN4 FP32 producer-input moments, shared by all structures; actual deployment starts from RNE I24 and is separately evaluated.',
            source_frames=[r['file'] for r in moments['frames']],
            bias='Each endpoint analytically matches teacher mean; no validation fitting or GT in this stage.'),
        common_output_permutation=permutation.tolist(),arms={})
    for kind in ['dense','contiguous34','lifting40']:
        torch.manual_seed(912);model=Structure(kind);opt=torch.optim.Adam(model.parameters(),lr=.01)
        history=[]
        for step in range(1024):
            opt.zero_grad();delta=model.matrix()-target;loss=torch.einsum('ti,ij,tj->',delta,sigma,delta)
            loss.backward();opt.step()
            if step in [0,1023]:history.append(dict(step=step+1,loss=float(loss.detach())))
        effective=model.matrix().detach().numpy();constants={k:v.copy() for k,v in common.items()}
        if kind=='lifting40':
            q=np.rint(model.coeff.detach().numpy()*4096)
            assert q.min()>=-32768 and q.max()<=32767
            constants.update(lifting_q12=q.astype(np.int16),lifting_fraction_bits=np.asarray(12),
                lifting_matchings=np.asarray(MATCHINGS,np.int64),source_permutation=permutation)
            gain=model.gain.detach().numpy();assert np.all(gain!=0)
            cutoff=(tau-(teacher-effective)@mean)/gain
            direction=np.sign(gain).astype(np.int8)
            constants['source_threshold']=np.where(direction>0,np.ceil(cutoff*16384),np.floor(cutoff*16384)).astype(np.int64)
            constants['source_direction']=direction;constants['source_constant']=np.full(10,-1,np.int8)
            constants['source_readout_gain_initial']=gain
        else:
            raw=np.rint(effective*(2**int(common['As_exponent'])));q=raw.clip(-32768,32767)
            constants['As_q16']=q.astype(np.int16)
            actual=q/(2**int(common['As_exponent']))
            cutoff=tau-(teacher-actual)@mean
            constants['source_threshold']=np.ceil(cutoff*16384).astype(np.int64)
            constants['source_mask']=(block[permutation] if kind=='contiguous34' else np.ones((10,10))).astype(np.uint8)
        constants['structure']=np.asarray(kind);constants['source_cutoff_trainable']=np.asarray(True)
        np.savez_compressed(out/(kind+'_constants.npz'),**constants)
        record['arms'][kind]=dict(history=history,prequantized_source_MSE=float(np.einsum('ti,ij,tj->',effective-teacher,cov,effective-teacher)),
            source_coefficients=40 if kind=='lifting40' else int(np.count_nonzero(constants['source_mask'])),
            moment_fit_parameters=sum(p.numel() for p in model.parameters()),
            nonzero_actual=int(np.count_nonzero(constants['lifting_q12'] if kind=='lifting40' else constants['As_q16'])),
            constants=kind+'_constants.npz',initial_source_threshold=constants['source_threshold'].tolist())
        print('INITIALIZED',kind,record['arms'][kind],flush=True)
    record['complete']=True;(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
