"""CPU contract checks for temporary whole-network MLP replacement/guards."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import copy
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'),
               str(ROOT/'SDformer/third_party/SDformerFlow')]
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
from spikingjelly.activation_based import functional
from tile_mlp import TileMLP, pattern_mask, tiled_mlp_model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(814)
    m = MS_Spiking_Mlp(16, hidden_features=32, norm_layer='BN', drop=0.,
        num_steps=10, spike_norm='BN', neuron_type='psn', v_th=.1,
        v_reset=None, surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True).double().eval()
    functional.set_step_mode(m, 'm')
    with torch.no_grad():
        for sn in (m.sn1.spiking_neuron, m.sn2.spiking_neuron):
            sn.weight.normal_(0,.3); sn.bias.fill_(.02)
    x = torch.randn(10,1,11,13,16,dtype=torch.float64)
    records = []
    with torch.inference_mode():
        assert m(x).abs().max().item() > .01, 'trivial zero branch invalidates test'
        for label in ('all_keep','all_skip','checkerboard','last_edge_only'):
            policy = lambda name, v, label=label: pattern_mask(label,v.shape[2],v.shape[3],device=v.device)
            d,a = TileMLP(m,'block.mlp','dense_masked',policy),TileMLP(m,'block.mlp','actual_skip',policy)
            ref,actual = x+d(x),x+a(x)
            error = (ref-actual).abs().max().item()
            assert torch.allclose(ref,actual,atol=1e-11,rtol=1e-11)
            if label=='all_skip':
                assert a.records[-1]['observed']=={'fc1_rows':0,'fc2_rows':0,'native_mlp_calls':0}
                assert torch.equal(actual,x)
            records.append({'case':label,'max_absolute_residual_output_error':error,
                            'record':a.records[-1]})
        current=copy.deepcopy(m); current.bn1.norm_layer.running_mean=None
        try: TileMLP(current,'block.mlp','actual_skip',policy)
        except ValueError: records.append({'guard':'current-stat BN rejected','passed':True})
        else: raise AssertionError('current-stat accepted')
        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__();self.mlp=m;self.cnf='ADD';self.use_checkpoint=False
        graph=torch.nn.Module();graph.add_module('block',Parent());graph.eval()
        original=graph.block.mlp
        try:
            with tiled_mlp_model(graph,'actual_skip',policy) as wrapped:
                assert graph.block.mlp is wrapped[0]
                raise RuntimeError('injected downstream failure')
        except RuntimeError: pass
        assert graph.block.mlp is original
        records.append({'guard':'original object restored after downstream failure','passed':True})
    probe=TileMLP(m,'block.mlp','actual_skip',lambda n,v:pattern_mask('all_skip',v.shape[2],v.shape[3]))
    try: probe(x)
    except ValueError: records.append({'guard':'autograd/training use rejected','passed':True})
    else: raise AssertionError('autograd accepted')
    result={'scope':'CPU synthetic integration/guards; no trained quality, GPU speed or cycle claim',
            'passed':True,'cuda_initialized':torch.cuda.is_initialized(),'records':records}
    (HERE/'tile_mlp_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(records),'cuda_initialized':torch.cuda.is_initialized()}))


if __name__=='__main__': main()
