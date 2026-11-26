"""E0 independent NumPy quantizer/int64 golden; no original data or GPU."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author-receipt", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    source = ROOT / "research/flow_story_20261021/e0_ordinary"
    source_hashes = {n: sha(source / n) for n in
                     ("mixed_integer_core.py", "mask_contract.py", "run_joint_recovery.py", "test_cpu.py")}
    receipt = json.loads(args.author_receipt.read_text())
    assert receipt["status"] == "PASS" and receipt["source_sha256"] == source_hashes
    spec = importlib.util.spec_from_file_location("_independent_e0", source / "mixed_integer_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(20261021)
    w = rng.normal(size=(16, 16)).astype(np.float32)
    w[0] = 0
    w[1, :8] = np.array([-7, 7, -.5, .5, -1.5, 1.5, -2.5, 2.5], dtype=np.float32)
    w[1, 8:] = 0
    x = rng.integers(0, 2, (1100, 16)).astype(np.float32)
    x[:4] = 0
    x[4:8] = 1
    tail = np.ones((10, 11), dtype=bool)
    tail[:9, :9] = False
    tail = np.tile(tail.reshape(-1), 10)
    chosen = np.zeros((10, 11), dtype=bool)
    # Geometry has one complete tile. Q4 tests low domain; mixed additionally
    # uses a second separate synthetic row partition to exercise both scales.
    chosen[:9, :9] = True
    maxabs = np.max(np.abs(w), axis=1)
    s8, s4 = maxabs / np.float32(127), maxabs / np.float32(7)
    q8 = np.rint(w / np.where(s8 == 0, np.float32(1), s8)[:, None]).clip(-127, 127).astype(np.int64)
    q4 = np.rint(w / np.where(s4 == 0, np.float32(1), s4)[:, None]).clip(-8, 7).astype(np.int64)
    assert q4[1, :8].tolist() == [-7, 7, 0, 0, -2, 2, -2, 2]
    cases = []
    for variant in core.VARIANTS:
        for mode in core.MODES:
            high = np.ones(len(x), dtype=bool) if mode == "Q8" else tail.copy()
            if mode == "M":
                high[:110] = True
            q, s, lq, ls, off, rep = core.representations(torch.from_numpy(w), mode, variant)
            np.testing.assert_array_equal(q.numpy(), q8.astype(np.int8))
            np.testing.assert_array_equal(s.numpy(), s8)
            expected_low = q4 * 16 if variant == "rne4" else np.floor_divide(q8, 16) * 16
            expected_scale = s4 / np.float32(16) if variant == "rne4" else s8
            expected_offset = 8 if variant == "midpoint" else 0
            if mode == "Q8":
                expected_low, expected_scale, expected_offset = q8, s8, 0
            np.testing.assert_array_equal(lq.numpy(), expected_low.astype(np.int8))
            np.testing.assert_array_equal(ls.numpy(), expected_scale)
            assert off == expected_offset
            y, _, actual, work = core.integer_affine(torch.from_numpy(x), q, s, torch.from_numpy(tail),
                mode, high8=torch.from_numpy(high), low_q=lq, low_scale=ls, low_offset=off, representation=rep)
            gold = x.astype(np.int64) @ q8.T
            gold[~high] = x[~high].astype(np.int64) @ (expected_low + expected_offset).T
            expected_y = gold.astype(np.float32) * s8
            expected_y[~high] = gold[~high].astype(np.float32) * expected_scale
            np.testing.assert_array_equal(actual.numpy(), gold.astype(np.int32))
            np.testing.assert_array_equal(y.numpy(), expected_y)
            assert work["logical_row_plane_work"] == int(high.sum()) * 8 + int((~high).sum()) * 4
            assert work["midpoint_correction"]["rows"] == (int((~high).sum()) if off else 0)
            cases.append(dict(mode=mode, variant=variant, independent_quantizer_and_int64_exact=True))
    assert source_hashes == {n: sha(source / n) for n in source_hashes}
    assert not torch.cuda.is_initialized()
    out = dict(schema="flow_e0_independent_cpu_v1", status="PASS", cases=cases,
               source_hashes=source_hashes, script_sha256=sha(__file__),
               original_author_CPU_receipt_sha256=sha(args.author_receipt),
               GPU_calls=0, real_model_forwards=0, GT_arrays_read=False,
               limitations="Synthetic arithmetic and source checks; STE/backbone GPU parity remains runtime evidence.")
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(cases), output=str(args.output))))


if __name__ == "__main__":
    main()
