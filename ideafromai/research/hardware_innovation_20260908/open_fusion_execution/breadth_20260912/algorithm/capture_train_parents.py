"""Actual two-parent train-only updatedI24/proj-g capture at4096 fixed anchors."""
from pathlib import Path
import argparse,json
import numpy as np
from parent_network import ParentNetwork,arrays
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();args.output=HERE/'train_capture';args.output.mkdir(parents=True,exist_ok=True)
    net=ParentNetwork(args)
    import torch
    from train_consumer_recovery import read_train16
    from run_bn_probe import input_frame,read_names
    from evaluate_stage2_deployment import CoarseReady
    from spikingjelly.activation_based import functional
    train=read_train16(net.latent/'flow_train_list.json');name=train[0]
    assert name in set(read_names(args.data,'train')) and name not in set(read_names(args.data,'valid'))
    yy=np.linspace(0,119,64,dtype=np.int64);xx=np.linspace(0,159,64,dtype=np.int64)
    y,x=np.meshgrid(yy,xx,indexing='ij');positions=np.stack([y.ravel(),x.ravel()],1)
    record=dict(complete=False,file=name,split='train',train_source=str(net.latent/'flow_train_list.json'),
        positions=4096,layout='T10,C96,N4096; positions_yx refer to anchor-grid120x160; original image coordinates are2*positions.',
        sampling='Fixed64x64 linspace grid, no activation/motion/validation selection.',axes={})
    try:
        for axis in ['ordinary','lifting_raw']:
            helper=net.install(axis);values={}
            gold=arrays(net.op/'stage_20260912/algorithm/hardware_exports'/axis/'deployed_constants.npz')
            actual=helper.export_constants();assert all(np.array_equal(actual[k],gold[k]) for k in gold)
            yt=torch.as_tensor(positions[:,0],device=helper.device);xt=torch.as_tensor(positions[:,1],device=helper.device)
            def ped(module,inputs,output):
                values['updated_I24']=helper.updated[:,:,::2,::2][:,:,yt,xt].detach().to(torch.int32).cpu().numpy()
            def source(module,inputs,output):
                values['source_I24']=helper.i[:,:,::2,::2][:,:,yt,xt].detach().to(torch.int32).cpu().numpy()
            def gate(module,inputs,output):
                values['projection_g']=output[:,0,:,::2,::2][:,:,yt,xt].ne(0).cpu().numpy()
            hooks=[net.modules[net.SOURCE_SN].register_forward_hook(source),net.modules[net.PROJECT+'.conv_res'].register_forward_hook(ped),net.modules[net.CONSUMER_SN].register_forward_hook(gate)]
            try:
                functional.reset_net(net.model);voxel,_,_=input_frame(args.data,name,targets=False)
                with torch.no_grad():
                    try:net.model(voxel)
                    except CoarseReady:pass
                assert values['updated_I24'].shape==values['projection_g'].shape==(10,96,4096)
                values.update(positions_yx=positions,frame=np.asarray(name),parent_axis=np.asarray(axis))
                np.savez_compressed(args.output/(axis+'.npz'),**values)
                record['axes'][axis]=dict(file=axis+'.npz',shape=[10,96,4096],
                    actual_proj_gate_ones=int(values['projection_g'].sum()),parent_constants_exact=True,
                    updated_min=int(values['updated_I24'].min()),updated_max=int(values['updated_I24'].max()))
                print('TRAIN_CAPTURE',axis,name,record['axes'][axis],flush=True)
            finally:
                for h in hooks:h.remove()
                net.release_axis()
        record['complete']=True;(args.output/'index.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:net.close()


if __name__=='__main__':main()
