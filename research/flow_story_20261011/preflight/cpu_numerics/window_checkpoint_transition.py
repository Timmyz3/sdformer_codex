#!/usr/bin/env python3
"""Check the existing 9->15 positional-weight loader on synthetic CPU state."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import hashlib
import json
from pathlib import Path
from native_mlp_probe import SOURCE
import torch
from models.STSwinNet_SNN.Spiking_swin_transformer3D import Spiking_QK_WindowAttention3D
from models.STSwinNet.load_pretrained import load_pretrained_interpolate


def build(size):
    return Spiking_QK_WindowAttention3D(
        dim=96, window_size=(2,size,size), pretrained_window_size=(0,0,0),
        num_heads=3, version='swinv1', norm='BN', num_steps=10, spike_norm='BN',
        neuron_type='psn', v_th=0.1, v_reset=None,
        surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True).cpu()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(91315)
    crop, full = build(9), build(15)
    with torch.no_grad():
        crop.positional_encoding.copy_(torch.linspace(-1,1,crop.positional_encoding.numel()).reshape_as(crop.positional_encoding))
    original = copy.deepcopy(crop.state_dict())
    rejected = False
    try: copy.deepcopy(full).load_state_dict(original,strict=False)
    except RuntimeError as exc:
        rejected = 'size mismatch for positional_encoding' in str(exc)
    assert rejected
    remapped = copy.deepcopy(original)
    load_pretrained_interpolate(full,remapped)
    incompatible = full.load_state_dict(remapped,strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    untouched = all(torch.equal(original[k],remapped[k]) for k in original if k!='positional_encoding')
    source_unchanged = all(torch.equal(original[k],crop.state_dict()[k]) for k in original)
    assert untouched and source_unchanged
    assert list(remapped['positional_encoding'].shape)==[1,3,450,32]
    assert torch.isfinite(remapped['positional_encoding']).all()
    source=SOURCE/'models/STSwinNet/load_pretrained.py'
    result={'scope':'Synthetic state-dict shape/loader check; no real checkpoint or AEE',
        'raw_strict_false_load_rejected_size_mismatch':rejected,
        'source_position_shape':list(original['positional_encoding'].shape),
        'target_position_shape':list(full.positional_encoding.shape),
        'existing_loader':'load_pretrained_interpolate; selected by remap=v1 for architecture swinv1 even in SDformerFlow-v2',
        'resampling':'existing trilinear interpolation, not a new method',
        'remapped_strict_load_passed':True,'other_state_tensors_unchanged':untouched,
        'source_model_unchanged':source_unchanged,'cuda_initialized':torch.cuda.is_initialized(),
        'loader_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    assert not result['cuda_initialized']
    (Path(__file__).parent/'window_checkpoint_transition_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
