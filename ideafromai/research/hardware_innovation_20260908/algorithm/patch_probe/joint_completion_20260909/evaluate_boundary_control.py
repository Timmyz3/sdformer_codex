"""Ordinary shared Q16 shortcut/output control for the two W8 students.

The sn1 source producer keeps its original input. Only the residual identity
saved beside the already-produced sn1 gates is Q16. The block output is Q16.
This is a new numerical control, not a novelty or an unchanged FP model.
Calibration uses the union of the specified students on train16 only.
"""
import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_integer_bridge import IntegerKernel, DyadicConv2Control
from evaluate_network import BLOCK, TARGET, BlockReady, save_json


def exponent(peak):
    return torch.ceil(torch.log2(peak.clamp_min(2.**-20)/32767)).to(torch.int32)


class Boundary:
    def __init__(self, block, conv2, params=None):
        self.params, self.conv2 = params, conv2
        self.peak_identity = torch.zeros(96, device=conv2.weight.device)
        self.peak_output = torch.zeros_like(self.peak_identity)
        self.counts = {}
        self.handles = [block.register_forward_pre_hook(self.before),
                        block.register_forward_hook(self.after)]

    def before(self, module, inputs):
        self.identity = inputs[0]

    def affine(self, z):
        self.z2 = z
        return self.conv2.affine(z)

    def after(self, module, inputs, output):
        if self.params is None:
            self.peak_identity = torch.maximum(self.peak_identity, self.identity.abs().amax((0,1,3,4)))
            self.peak_output = torch.maximum(self.peak_output, output.abs().amax((0,1,3,4)))
            self.identity = self.z2 = None
            raise BlockReady()
        a = self.params
        si, so, sq, sw = [a[k][None,None,:,None,None] for k in ('identity_scale', 'output_scale', 'internal_scale', 'weight_scale')]
        offset = a['offset_int'][None,None,:,None,None]
        qi = torch.round(self.identity.double()/si)
        clip_i = (qi.abs()>32767).sum()
        qi = qi.clamp(-32767,32767)
        # Every quantity before the final rounding is an exact-range integer.
        numerator = qi*(si/sq)+self.z2.double()*(sw/sq)+offset
        qo = torch.round(numerator/(so/sq))
        clip_o = (qo.abs()>32767).sum()
        self.counts = dict(values=qo.numel(), identity_clipped=int(clip_i), output_clipped=int(clip_o),
                           internal_max_abs=float(numerator.abs().max()))
        result = (qo.clamp(-32767,32767)*so).float()
        self.identity = self.z2 = None
        return result

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-files',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--parameters',type=Path)
    p.add_argument('--split',choices=('diverse','valid'),default='diverse')
    p.add_argument('--count',type=int,default=10)
    args=p.parse_args()
    algorithm=args.root/'algorithm'
    sys.path[:0]=[str(algorithm/'nrv_cost_probe'),str(algorithm)]
    import run_probe as probe
    from run_bn_probe import input_frame,read_names
    from evaluate_stage2_deployment import CoarseReady,summarize
    from spikingjelly.activation_based import functional
    system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    fixed=torch.load(algorithm/'patch_probe/patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,v in fixed.items():
        bn=modules[name];bn.track_running_stats=True
        bn.running_mean=v['mean'].to(bn.weight);bn.running_var=v['var'].to(bn.weight)
    c1,n1,sn=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.norm1'],modules[TARGET]
    c2,n2=modules[BLOCK+'.conv2.0'],modules[BLOCK+'.norm2']
    bn2=modules[next(n for n in fixed if n.startswith(BLOCK+'.norm2'))]
    control=DyadicConv2Control.from_modules(c2,bn2,float(sn.thresh))
    original=c1.forward,n1.forward,sn.forward,c2.forward,n2.forward
    params=None
    if not args.calibrate:
        with np.load(args.parameters) as d:
            params={k:torch.as_tensor(d[k],device=c1.weight.device).double()
                    for k in ('identity_scale','output_scale','internal_scale','weight_scale','offset_int')}
    boundary=Boundary(modules[BLOCK],control,params)
    c2.forward=types.MethodType(lambda self,x:control.convolution(x,16),c2)
    n2.forward=types.MethodType(lambda self,z:boundary.affine(z),n2)
    if args.calibrate:
        capture=torch.load(algorithm/'patch_probe/partial_completion/capture.pt',map_location='cpu',weights_only=False)
        names=capture['metadata']['train'];del capture
    else:
        names=(json.loads((algorithm/'samples.json').read_text())['valid'] if args.split=='diverse'
               else read_names(args.data,'valid'))[:args.count]
    args.output.mkdir(parents=True,exist_ok=True)
    run=dict(complete=False,calibration=args.calibrate,files=names,parameters=str(args.parameters),results={},
             numeric='same W1/W2 dyadic W8; shared Q16 identity and Q16 output; BN2 offset integer with eight guard fractional bits',
             source='original source-neuron input and theta*g unchanged; only saved shortcut identity is quantized',
             scope='ordinary new-model numeric control, dense evaluation; no hardware cycles')
    save_json(args.output/'run.json',run)
    try:
        for i,file in enumerate(args.model_files):
            kernel=IntegerKernel(file,c1.weight.device)
            c1.forward=types.MethodType(lambda self,x,k=kernel:k.convolution(x,16),c1)
            n1.forward=types.MethodType(lambda self,x:x,n1)
            for mode in ('exact','conditional'):
                sn.forward=types.MethodType(lambda self,x,k=kernel,m=mode:k.network_output(x,m,256),sn)
                axis=f'{i:02d}_{file.stem}_{mode}'
                rows=[]
                for j,name in enumerate(names):
                    functional.reset_net(model)
                    x,label,mask=input_frame(args.data,name,targets=not args.calibrate)
                    try:model(x)
                    except BlockReady:
                        if not args.calibrate:raise
                    except CoarseReady:
                        pred=F.interpolate(current.pop('flow'),(480,640),mode='bilinear',align_corners=False)
                        error=torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask],dim=1)
                        total,n=float(error.double().sum()),error.numel()
                        rows.append(dict(file=name,valid_pixels=n,aee_sum=total,AEE=total/n,**boundary.counts))
                    if (j+1)%10==0 or j+1==len(names):
                        print('BOUNDARY',axis,j+1,'calibration' if args.calibrate else summarize(rows,False)['AEE_frame_mean'],flush=True)
                run['results'][axis]=(dict(frames=len(names),calibrated=True) if args.calibrate else summarize(rows,True))
                if not args.calibrate:save_json(args.output/(axis+'_frames.json'),rows)
                save_json(args.output/'run.json',run)
        if args.calibrate:
            ei,eo=exponent(boundary.peak_identity),exponent(boundary.peak_output)
            ew=torch.from_numpy(control.arrays['weight_scale_exponent']).to(ei)
            eq=torch.minimum(torch.minimum(ei,eo),ew)-8
            exps={k:v.cpu().numpy() for k,v in dict(identity_exponent=ei,output_exponent=eo,internal_exponent=eq).items()}
            scales={k:np.ldexp(np.ones(96),v.cpu().numpy()) for k,v in dict(identity_scale=ei,output_scale=eo,internal_scale=eq).items()}
            offset=np.rint(control.arrays['bn_bias_fp64']/scales['internal_scale']).astype(np.int64)
            bound=32767*scales['identity_scale']/scales['internal_scale']+np.maximum(abs(control.arrays['Y_lower']),abs(control.arrays['Y_upper']))*control.arrays['weight_scale']/scales['internal_scale']+abs(offset)
            np.savez(args.output/'parameters.npz',**exps,**scales,offset_int=offset,
                     weight_scale=control.arrays['weight_scale'],internal_legal_abs_bound=bound,
                     identity_train_peak=boundary.peak_identity.cpu().numpy(),output_train_peak=boundary.peak_output.cpu().numpy())
            run['internal_legal_signed_bits']=int(np.ceil(np.log2(bound.max()+1)))+1
            run['calibration_rule']='per-H union maximum of both students and both modes on train16; power-of-two symmetric Q16; no validation calibration'
        run['complete']=True;save_json(args.output/'run.json',run)
    finally:
        boundary.close()
        c1.forward,n1.forward,sn.forward,c2.forward,n2.forward=original


if __name__=='__main__':main()
