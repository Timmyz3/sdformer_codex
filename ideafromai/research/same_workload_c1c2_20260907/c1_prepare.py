"""Reuse the sealed complete-input reconstruction without importing Torch."""
import sys
sys.dont_write_bytecode = True
assert sys.version_info[:2] == (3, 12)
import ast
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'complete_transfer_20260907/c1_full_layer.py'
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
LEDGER = HW / 'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'
LEDGER_SHA = 'daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835'


def main():
    target = ROOT / 'c1_input.bin'
    receipt = ROOT / 'c1_input_identity.json'
    assert not target.exists() and not receipt.exists()
    original = OLD.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == '1467c07c38712200cd77680cd46a5d733d744e1a3524847dbc50c05883e43da4'
    tree = ast.parse(original)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['sha', 'restore']]
    assert len(selected) == 2
    # Exact archived function bodies execute with the same required globals.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(OLD), 'exec'), globals())
    plan = json.loads((ROOT / 'c1_plan.json').read_text())
    matrix, identity = restore(plan['input'])
    source = matrix[:, 4::9].copy()
    assert source.shape == (3000, 768)
    support = np.packbits(source, axis=1, bitorder='little')
    masks = np.sum(matrix.reshape(3000, 432, 16).astype(np.uint16)
                   * (1 << np.arange(16, dtype=np.uint16)), axis=2, dtype=np.uint16)
    theta = Fraction(identity['threshold_amplitude_from_sealed_contract']['nonzero_float32'])
    assert theta.denominator & (theta.denominator - 1) == 0
    assert theta.numerator * 128 * 6912 < (1 << 47)
    header = struct.pack('<8s8I2Q', b'C1UNION1', 10, 15, 20, 768, 768, 3000, 6912, 16,
                         theta.numerator, theta.denominator)
    with target.open('xb') as f:
        f.write(header)
        f.write(support.tobytes())
        f.write(masks.astype('<u2', copy=False).tobytes())
    result = {'status': 'COMPLETE_INPUT_RESTORED_AND_FROZEN', 'identity': identity,
              'reused_reconstruction_source': str(OLD), 'reused_source_sha256': sha(OLD),
              'plan_sha256': sha(ROOT / 'c1_plan.json'), 'prepare_script_sha256': sha(Path(__file__)),
              'binary_sha256': sha(target), 'binary_bytes': target.stat().st_size,
              'theta_exact_numerator': theta.numerator, 'theta_exact_denominator': theta.denominator,
              'diagnostic_weight_contract': 'signedINT8 W; exact frozen theta rational carried into48bit numerators; not trained FP32 W',
              'binary_format': 'LE header8s/8uint32/2uint64;3000*96 support bytes;3000*432 uint16 original lowered masks'}
    with receipt.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'binary_bytes', 'theta_exact_numerator', 'theta_exact_denominator']}))


if __name__ == '__main__':
    main()
