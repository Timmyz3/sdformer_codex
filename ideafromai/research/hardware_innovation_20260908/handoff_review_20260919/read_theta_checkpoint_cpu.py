"""Read only scalar theta values from PyTorch zip checkpoints, no torch/GPU import."""
import collections
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

ROOT = Path('/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results')

class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch' and name in ('FloatStorage', 'LongStorage'):
            return name
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda storage, offset, size, stride, requires_grad, hooks: {
                'storage': storage, 'offset': offset, 'size': size, 'stride': stride}
        raise ValueError((module, name))
    def persistent_load(self, pid):
        return pid

def read(path):
    with zipfile.ZipFile(path) as archive:
        name = next(x for x in archive.namelist() if x.endswith('data.pkl'))
        root = name.removesuffix('data.pkl')
        data = MetadataReader(io.BytesIO(archive.read(name))).load()
        state = data.get('model_state_dict', data.get('state_dict', data))
        out = {}
        for k, v in state.items():
            if not k.endswith('.thresh'):
                continue
            assert v['size'] in ((), (1,)), (k, v['size'])
            typ, dtype, key, device, count = v['storage']
            assert dtype == 'FloatStorage'
            payload = archive.read(root + 'data/' + key)
            out[k] = struct.unpack_from('<f', payload, v['offset'] * 4)[0]
        opt = data.get('optimizer_state_dict', data.get('optimizer', {}))
        return {'keys': list(data.keys()), 'theta': out,
                'epoch': data.get('epoch'), 'scaler': data.get('scaler'),
                'optimizer_groups': opt.get('param_groups', []) if isinstance(opt, dict) else []}

base = read(ROOT/'t57_grad_ctrl/warmstart/checkpoint_epoch34.pth')
out = {'base_count': len(base['theta']), 'base_checkpoint_keys': base['keys'], 'arms': {}}
for arm in ('t57_grad_ctrl', 't57_grad_fix'):
    warm = read(ROOT/arm/'warmstart/checkpoint_epoch34.pth')
    end = read(ROOT/arm/'checkpoint_epoch35.pth')
    shared = sorted(set(warm['theta']) & set(end['theta']))
    rows = [{'name': k, 'before': warm['theta'][k], 'after': end['theta'][k],
             'delta': end['theta'][k] - warm['theta'][k]} for k in shared]
    out['arms'][arm] = {
        'end_checkpoint_keys': end['keys'], 'epoch': end['epoch'],
        'scaler': end['scaler'],
        'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in end['optimizer_groups']],
        'warm_count': len(warm['theta']), 'end_count': len(end['theta']),
        'shared_count': len(shared),
        'warm_equals_ctrl': warm['theta'] == base['theta'],
        'changed': sum(r['delta'] != 0 for r in rows),
        'positive': sum(r['delta'] > 0 for r in rows),
        'negative': sum(r['delta'] < 0 for r in rows),
        'max_abs_delta': max(abs(r['delta']) for r in rows),
        'mean_delta': sum(r['delta'] for r in rows)/len(rows),
        'unchanged_names': [r['name'] for r in rows if r['delta'] == 0],
        'rows': rows,
    }
print(json.dumps(out, indent=2))
