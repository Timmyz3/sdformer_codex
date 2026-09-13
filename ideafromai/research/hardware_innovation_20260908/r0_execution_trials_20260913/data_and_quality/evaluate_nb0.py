"""Fresh local original PSN/SDSA NB0, ten identical remote-mirrored frames."""
import json,sys,copy,random
import numpy as np
from model_access import HERE,BASE,REPO
sys.path[:0]=[str(BASE/'algorithm'),str(REPO/'SDformer/neuron_experiments/H9_bipolar_self_attention/overlay'),str(REPO/'SDformer/third_party/SDformerFlow')]
from run_bn_probe import input_frame,set_bn_mode
import torch,yaml
from configs.parser import YAMLParser
from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
from models.STSwinNet.load_pretrained import load_pretrained_interpolate
from spikingjelly.activation_based import functional

def main():
    config=BASE/'open_fusion_execution/accuracy_baseline/source_nb0_config.yml'
    cfg=YAMLParser.combine_entries(yaml.safe_load(config.read_text()))
    cfg['data']['path']=str(HERE/'data_mirror');cfg['swin_transformer']['input_size']=[480,640]
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.allow_tf32=bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.benchmark=bool(cfg['runtime']['cudnn_benchmark'])
    model=MS_SpikingformerFlowNet_en4(copy.deepcopy(cfg['model']),copy.deepcopy(cfg['swin_transformer'])).cuda()
    model.init_weights()
    ckpt=torch.load(HERE/'nb0_checkpoint_epoch29.pth',map_location='cpu',weights_only=False)
    source_state=ckpt.state_dict() if hasattr(ckpt,'state_dict') else ckpt['model_state_dict']
    state={k.replace('module.',''):v for k,v in source_state.items()}
    oldkeys=set(state);load_pretrained_interpolate(model,state)
    audit=model.load_state_dict(state,strict=False)
    assert not audit.unexpected_keys
    assert all('relative_position_index' in k or 'relative_coords_table' in k or 'attn_mask' in k for k in audit.missing_keys)
    functional.set_step_mode(model,'m');model.eval();model.requires_grad_(False);set_bn_mode(model)
    assert not any('ATLIFTernaryPSN' in type(m).__name__ or 'Shiftmax' in type(m).__name__ for m in model.modules())
    names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
    report=dict(complete=False,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        checkpoint='nb0_checkpoint_epoch29.pth',config=str(config),data_root=str(HERE/'data_mirror'),
        window=cfg['swin_transformer']['window_size'],remap='v1',BN_no_running_count=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),
        missing_keys=list(audit.missing_keys),unexpected_keys=list(audit.unexpected_keys),remap_removed_keys=sorted(oldkeys-set(state)),
        ATLIF_count=0,Shiftmax_count=0,head='original complete model flow[-1]',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        historical_gate=1.45460286107,rows=[])
    def save(): (HERE/'nb0_diverse10.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x,label,valid=input_frame(HERE/'data_mirror',name)
            pred=model(x)['flow'][-1]
            error=torch.linalg.vector_norm(pred.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid],dim=1)
            total=float(error.double().sum());count=error.numel()
            report['rows'].append(dict(file=name,valid_pixels=count,aee_sum=total,AEE=total/count))
            report['AEE_frame_mean']=float(np.mean([r['AEE'] for r in report['rows']]))
            save();print('NB0_FRESH',len(report['rows']),report['AEE_frame_mean'],flush=True)
    report['complete']=True;report['valid_pixels']=sum(r['valid_pixels'] for r in report['rows']);save()
if __name__=='__main__':main()
