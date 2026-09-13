"""Same-parent diverse10 forward for fixed sparse/decomposition candidates."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'aee')
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    configs=json.loads(args.config.read_text())
    stage=args.root/'open_fusion_execution/major_operator_fusions_20260913'
    algorithm=args.root/'open_fusion_execution/breadth_20260912/algorithm'
    sys.path.insert(0,str(algorithm))
    from parent_network import ParentNetwork,arrays
    net=ParentNetwork(args)
    from fixed_structure import LiteralForward
    from evaluate_branch_control import evaluate_axis
    spec=importlib.util.spec_from_file_location('spatial_adapter',stage/'decomposition_owned/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    names=json.loads((args.root/'algorithm/samples.json').read_text())['valid']
    base=arrays(algorithm/'matched_training/dense/stage320/deployed_constants.npz')
    result=dict(complete=False,optimizer_updates=0,
        parent='matched dense stage320',frames=names,NB0_diverse10=1.45460286107,
        scope='Exploratory exact same full-forward/AEE protocol; dense PyTorch kernels are not hardware speed.',axes={})
    try:
        for config in configs:
            key=config['name'];structure=config.get('structure','dense')
            source=(base if structure=='dense' else arrays(algorithm/'matched_training'/structure/'stage320/deployed_constants.npz'))
            q={k:v.copy() for k,v in source.items()}
            if config.get('sparse'):
                z=arrays(stage/config['sparse'])
                q['U_conv2_theta_q16']=z['U_conv2_theta_q16']
                assert np.array_equal(q['U_conv2_theta_exponent'],z['U_conv2_theta_exponent'])
            net.install('ordinary');net.helper.restore()
            net.helper=LiteralForward(net.controller,net.pair.temporal.theta,q,structure)
            restore=None
            if config.get('decomposition'):
                _,restore=adapter.install_conv(net.modules[adapter.TARGET],stage/config['decomposition'])
            try:
                summary=evaluate_axis(args,net.model,net.current,names,key,progress_tag='MAJOR_FUSION_AEE')
                result['axes'][key]=dict(config=config,summary=summary,
                    below_same_protocol_NB0=summary['AEE_frame_mean']<result['NB0_diverse10'])
                (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            finally:
                if restore:restore()
                net.release_axis()
        result['complete']=True
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('MAJOR_FUSION_AEE_COMPLETE',len(result['axes']),flush=True)
    finally:net.close()


if __name__=='__main__':main()
