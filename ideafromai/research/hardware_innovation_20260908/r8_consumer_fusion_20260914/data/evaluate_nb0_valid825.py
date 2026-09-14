"""Same-A800 original PSN/SDSA NB0, complete official local valid825; no student helper."""
import json,sys,copy,random,time
import numpy as np
from model_access import HERE,BASE,REPO
sys.path[:0]=[str(BASE/'algorithm'),str(REPO/'SDformer/neuron_experiments/H9_bipolar_self_attention/overlay'),str(REPO/'SDformer/third_party/SDformerFlow')]
from run_bn_probe import input_frame,set_bn_mode,read_names
import torch,yaml
from configs.parser import YAMLParser
from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
from models.STSwinNet.load_pretrained import load_pretrained_interpolate
from spikingjelly.activation_based import functional

def main():
    config=HERE/'source_nb0_config.yml'
    data=REPO/'SDformer/data/Datasets/DSEC/saved_flow_data'
    checkpoint=REPO/'SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth'
    cfg=YAMLParser.combine_entries(yaml.safe_load(config.read_text()))
    cfg['data']['path']=str(data);cfg['swin_transformer']['input_size']=[480,640]
    assert cfg['swin_transformer']['window_size']==[2,15,15]
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.allow_tf32=bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.benchmark=bool(cfg['runtime']['cudnn_benchmark'])
    model=MS_SpikingformerFlowNet_en4(copy.deepcopy(cfg['model']),copy.deepcopy(cfg['swin_transformer'])).cuda()
    model.init_weights()
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source_state=ckpt.state_dict() if hasattr(ckpt,'state_dict') else ckpt['model_state_dict']
    state={k.replace('module.',''):v for k,v in source_state.items()}
    oldkeys=set(state);load_pretrained_interpolate(model,state)
    audit=model.load_state_dict(state,strict=False)
    assert not audit.unexpected_keys
    assert all('relative_position_index' in k or 'relative_coords_table' in k or 'attn_mask' in k for k in audit.missing_keys)
    functional.set_step_mode(model,'m');model.eval();model.requires_grad_(False);set_bn_mode(model)
    assert sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())==78
    assert not any('ATLIFTernaryPSN' in type(m).__name__ or 'Shiftmax' in type(m).__name__ for m in model.modules())
    names=read_names(data,'valid')
    assert len(names)==825 and len(set(names))==825
    started=time.monotonic()
    report=dict(complete=False,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        checkpoint=str(checkpoint),config=str(config),data_root=str(data),
        window=cfg['swin_transformer']['window_size'],remap='v1',BN_no_running_count=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),
        missing_keys=list(audit.missing_keys),unexpected_keys=list(audit.unexpected_keys),remap_removed_keys=sorted(oldkeys-set(state)),
        ATLIF_count=0,Shiftmax_count=0,head='original complete model flow[-1]',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        historical_gate=1.44535253468097,frames=names,training=False,student_installed=False,rows=[])
    def save(): (HERE/'nb0_valid825.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x,label,valid=input_frame(data,name)
            pred=model(x)['flow'][-1]
            error=torch.linalg.vector_norm(pred.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid],dim=1)
            total=float(error.double().sum());count=error.numel()
            report['rows'].append(dict(file=name,valid_pixels=count,aee_sum=total,AEE=total/count))
            report['AEE_frame_mean']=float(np.mean([r['AEE'] for r in report['rows']]))
            save()
            if len(report['rows'])%50==0 or len(report['rows'])==825:print('NB0_SAME_A800',len(report['rows']),report['AEE_frame_mean'],flush=True)
    report['complete']=True;report['valid_pixels']=sum(r['valid_pixels'] for r in report['rows']);report['wall_seconds']=time.monotonic()-started
    report['AEE_pixel_mean']=sum(r['aee_sum'] for r in report['rows'])/report['valid_pixels'];save()
if __name__=='__main__':main()
