"""Read this verified tensor-only torch ZIP using NumPy, without executing globals.

Research parameter access only. This does not emulate PyTorch arithmetic.
"""
from collections import OrderedDict
from pathlib import Path
import io
import pickle
import zipfile
import numpy as np


def read_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        roots = [x[:-len('data.pkl')] for x in archive.namelist() if x.endswith('/data.pkl')]
        assert len(roots) == 1
        root = roots[0]
        assert archive.read(root + 'byteorder') == b'little'
        buffers = {}

        def rebuild(storage, offset, size, stride, requires_grad, backward_hooks, metadata=None):
            dtype, raw = storage
            assert offset >= 0 and len(size) == len(stride)
            assert all(x >= 0 for x in size) and all(x >= 0 for x in stride)
            extent = offset + sum((s - 1) * t for s, t in zip(size, stride)) + 1
            if all(size):
                assert extent * dtype.itemsize <= len(raw)
            return np.ndarray(size, dtype=dtype, buffer=raw,
                              offset=offset * dtype.itemsize,
                              strides=tuple(x * dtype.itemsize for x in stride))

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {
                    ('collections', 'OrderedDict'): OrderedDict,
                    ('torch', 'FloatStorage'): np.dtype('<f4'),
                    ('torch', 'LongStorage'): np.dtype('<i8'),
                    ('torch._utils', '_rebuild_tensor_v2'): rebuild,
                }
                if (module, name) not in allowed:
                    raise ValueError(('unsupported pickle global', module, name))
                return allowed[module, name]

            def persistent_load(self, pid):
                assert len(pid) == 5 and pid[0] == 'storage'
                _, dtype, key, location, count = pid
                assert dtype in (np.dtype('<f4'), np.dtype('<i8'))
                assert str(key).isdigit() and count >= 0
                if key not in buffers:
                    buffers[key] = archive.read(root + 'data/' + key)
                raw = buffers[key]
                assert len(raw) == count * dtype.itemsize
                return dtype, raw

        return Reader(io.BytesIO(archive.read(root + 'data.pkl'))).load()


if __name__ == '__main__':
    path = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth')
    data = read_checkpoint(path)
    print('TOP', list(data)[:20])
    for name in ('state_dict', 'model', 'model_state_dict'):
        if name in data and isinstance(data[name], dict):
            data = data[name]
            break
    print('KEYS', len(data))
    for key, value in data.items():
        if any(s in key for s in ('layers.0.swin_blocks.0.mlp', 'layers.3.swin_blocks.0.mlp')):
            print(key, value.shape if hasattr(value, 'shape') else type(value).__name__,
                  value.tolist() if hasattr(value, 'size') and value.size <= 12 else '')
