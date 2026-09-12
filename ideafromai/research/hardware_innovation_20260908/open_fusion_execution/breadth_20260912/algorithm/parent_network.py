"""Current frozen parent loader shared by the bounded breadth experiments."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


class ParentNetwork:
    def __init__(self,args):
        base=args.root;self.base=base;self.args=args;op=base/'open_fusion_execution'
        patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain'
        self.chain=chain;self.lift=chain/'fast_temporal_recovery_lifting40';self.op=op
        latent=patch/'factor_completion_20260909/latent_stage_train16';self.latent=latent
        for p in [chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe',
                  op/'new_interface_selection/aee_rebase',op/'stage_20260912/algorithm/onepass_math']:
            sys.path.insert(0,str(p))
        import torch
        import run_probe as probe
        from flow_backward_probe import TrainableLatentPair
        from capture import BLOCK,SOURCE_SN,PROJECT,CONSUMER_SN
        from evaluate_branch_control import mask_nonanchors
        self.BLOCK,self.SOURCE_SN,self.PROJECT,self.CONSUMER_SN=BLOCK,SOURCE_SN,PROJECT,CONSUMER_SN
        args.split='diverse';system=probe.load_system(args)
        self.model,self.modules,_,self.current,sources,_,_,_=system
        probe.install_sources(system,sources);self.current['count_codes']=False
        fixed=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
        for path,values in fixed.items():
            bn=self.modules[path];bn.track_running_stats=True
            bn.running_mean,bn.running_var=values['mean'].to(bn.weight),values['var'].to(bn.weight)
        self.model.eval();self.model.requires_grad_(False)
        flags=json.loads((chain/'affine_shared_temporal_control_diverse10/run.json').read_text())
        torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul'])
        torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
        parent=latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
        self.pair=TrainableLatentPair(arrays(parent),self.modules[SOURCE_SN].weight.device)
        self.pair.u.requires_grad_(False);self.pair.v.requires_grad_(False)
        self.conv1,self.sn2=self.modules[BLOCK+'.conv1.0'],self.modules[BLOCK+'.sn2.spiking_neuron']
        self.original_pair=(self.conv1.forward,self.sn2.forward)
        self.conv1.forward,self.sn2.forward=self.pair.conv_forward,self.pair.neuron_forward
        masks={}
        def nonanchor(module,inputs,output):
            key=(tuple(output.shape[-2:]),output.device)
            if key not in masks:
                m=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool);m[::2,::2]=True;masks[key]=m
            return mask_nonanchors(output,masks[key])
        self.anchor_hook=self.modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
        self.common=(self.modules,arrays(res/'rank_control_parameters.npz'),arrays(chain/'rank32_diverse10/parameters.npz'))
        self.controller=self.helper=None;self.original_bn=None

    def install(self,axis):
        import torch
        from train_shared_temporal_recovery import SharedTemporalControl
        from lifting_temporal_control import LiftingTemporalControl
        from fixed_temporal_coordinates import FixedTemporalForward
        from fixed_lifting_coordinates import FixedLiftingForward
        from ped_rebase_adapter import install
        from numeric import Arithmetic
        if axis=='ordinary':
            identity='identity_permuted_base';mode='original_ordered24'
            student=self.chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz'
            controller=SharedTemporalControl(*self.common,fit={});controller.load_saved(identity,arrays(student))
            helper=FixedTemporalForward(controller,self.pair.temporal.theta)
        else:
            identity='fast_raw_diagonal';mode='activation_whitened24';student=self.lift/'stage320'/f'{identity}.npz'
            init=json.loads((self.lift/'initialization.json').read_text())
            controller=LiftingTemporalControl(*self.common,source_fit=init['source_fit'],consumer_fits=init['consumer_fits'],basis_lifting=init['basis_lifting'])
            controller.load_saved(identity,arrays(student));helper=FixedLiftingForward(controller,self.pair.temporal.theta)
        install(helper,arrays(self.op/'new_interface_selection'/(axis+'_rebase_parameters.npz')),mode,helper.export_constants())
        for p in controller.params.values():p.requires_grad_(False)
        bn=self.modules[self.PROJECT+'.norm_layer'];self.original_bn=bn.forward
        self.bn=bn;self.bn_gamma=bn.weight.detach().cpu().numpy().copy();self.bn_beta=bn.bias.detach().cpu().numpy().copy()
        self.math=Arithmetic()
        def onepass(x):
            p=x.detach().permute(0,2,3,1).contiguous().cpu().numpy()
            st=self.math.statistics(p,self.bn_gamma,self.bn_beta,bn.eps)
            out=x*torch.as_tensor(st[3],device=x.device).reshape(1,96,1,1)
            return out+torch.as_tensor(st[4],device=x.device).reshape(1,96,1,1)
        bn.forward=onepass;self.controller,self.helper=controller,helper
        return helper

    def release_axis(self):
        import torch
        from spikingjelly.activation_based import functional
        if self.original_bn is not None:self.bn.forward=self.original_bn;self.original_bn=None
        if self.helper is not None:self.helper.restore();self.helper=None
        if self.controller is not None:self.controller.restore();self.controller=None
        functional.reset_net(self.model);self.current.pop('flow',None);torch.cuda.empty_cache()

    def close(self):
        self.release_axis();self.anchor_hook.remove()
        self.conv1.forward,self.sn2.forward=self.original_pair
