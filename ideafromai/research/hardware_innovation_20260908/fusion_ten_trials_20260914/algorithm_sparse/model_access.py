"""Read-only matched-dense loader, with explicit local/remote repository roots."""
from pathlib import Path
from types import SimpleNamespace
import sys,os
assert sys.version_info[:2]==(3,12)
HERE=Path(__file__).resolve().parent
BASE=Path(os.environ.get('R0_STREAM_BASE',str(HERE.parents[1])))
REPO=Path(os.environ.get('R0_STREAM_REPO',str(HERE.parents[4])))
ALGORITHM=BASE/'open_fusion_execution/breadth_20260912/algorithm'

def load_parent(output,data_root=None):
    for p in [ALGORITHM,BASE/'algorithm',BASE/'algorithm/nrv_cost_probe']:sys.path.insert(0,str(p))
    import run_bn_probe
    original=run_bn_probe.build_model
    incoming=REPO/'SDformer/hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
    def relocated(args):
        args.code_root=REPO/'SDformer'
        args.config=incoming/'dsec_c12_alpha0125_ep29_resume5_20260830.yml'
        args.checkpoint=incoming/'checkpoint_epoch34.pth'
        args.data=Path(data_root) if data_root else REPO/'SDformer/data/Datasets/DSEC/saved_flow_data'
        for p in [args.config,args.checkpoint,args.data]:
            if not p.exists():raise FileNotFoundError(p)
        return original(args)
    run_bn_probe.build_model=relocated
    from parent_network import ParentNetwork,arrays
    args=SimpleNamespace(root=BASE,output=Path(output),split='diverse')
    args.output.mkdir(parents=True,exist_ok=True)
    try:net=ParentNetwork(args)
    finally:run_bn_probe.build_model=original
    from fixed_structure import LiteralForward
    net.install('ordinary');net.helper.restore()
    constants=arrays(ALGORITHM/'matched_training/dense/stage320/deployed_constants.npz')
    net.helper=LiteralForward(net.controller,net.pair.temporal.theta,constants,'dense')
    return args,net
