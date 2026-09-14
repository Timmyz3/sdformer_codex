"""Check the GPU evaluator's phase3 arithmetic against independent tile gold."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'spatial_r16_integer'))
from phase3_endpoint import Phase3Integer


def main():
    torch.set_num_threads(1)
    folder = HERE.parent / 'spatial_winograd_pruning/unconstrained'
    model = Phase3Integer(folder / 'factors.npz')
    results = []
    for filename in ('gold_tiles.npz', 'gold_sequences.npz'):
        with np.load(folder / filename) as gold:
            for index, words in enumerate(gold['source_words']):
                source = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.float32) * model.theta
                identity = gold['identity_fp32_bits'][index].copy().view(np.float32)
                p, z = model.raw_p(torch.from_numpy(source), tile=True, return_z=True)
                actual = model.consume(p, torch.from_numpy(identity), return_intermediates=True)
                for key, field in [('p', 'p_int'), ('J', 'J_q20'), ('wide', 'wide_int64'), ('i24', 'i24')]:
                    assert np.array_equal(actual[key].numpy(), gold[field][index]), (filename, index, key)
                assert np.array_equal(z.numpy(), gold['z_halo_int'][index]), (filename, index, 'Z')
                assert torch.equal((model.reader_value(actual['i24']).double() * 16384).round().long(), actual['i24'])
            result5 = model(torch.from_numpy(source)[:, None], torch.from_numpy(identity)[:, None], tile=True)
            assert torch.equal(result5[:, 0], actual['i24'])
            results.append(dict(file=filename, tiles=len(gold['source_words']), raw_values=int(gold['p_int'].size)))
    report = dict(passed=True, device='cpu', torch=torch.__version__, sets=results,
                  checked=['Z', 'p2', 'J', 'wide', 'I24', 'reader', 'T_B_layout'], differences=0)
    out = HERE / 'unconstrained'
    out.mkdir(exist_ok=True)
    (out / 'torch_cpu_stats.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
