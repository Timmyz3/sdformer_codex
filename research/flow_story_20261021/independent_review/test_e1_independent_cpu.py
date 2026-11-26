"""Independent E1 synthetic arithmetic/dispatch/metadata checks; no data arrays/GPU."""
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


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    base = ROOT / "research"
    paths = {
        "family": base / "flow_story_20261021/e1_localization/family_core.py",
        "runner": base / "flow_story_20261021/e1_localization/run_localization.py",
        "q6": base / "flow_story_20261019/round27/implementation/recovery_integer_core.py",
        "q4": base / "flow_story_20261019/round28/implementation/recovery_integer_core.py",
        "roles": base / "flow_story_20261021/selections_v1.json",
        "members": base / "flow_story_20261019/round27/independent_review/membership_v1.json",
    }
    source_hashes = {k: sha(p) for k, p in paths.items()}
    q6, q4 = load(paths["q6"], "_ie1q6"), load(paths["q4"], "_ie1q4")
    family = load(paths["family"], "_ie1family")
    roles, members = [json.loads(paths[k].read_text()) for k in ("roles", "members")]
    loc, e2, conf = [], [], []
    for seq in sorted(members["sequences"]):
        ids = [x["target_id"] for x in members["dev_members"] if x["sequence"] == seq]
        ids.sort(key=lambda tid: hashlib.sha256(("flow-e01-20261021|" + tid).encode()).hexdigest())
        assert len(ids) == 8
        loc += ids[:2]
        e2 += ids[2:4]
        conf += ids[4:]
    assert loc == roles["localization_ids"]
    assert e2 == roles["e2_reserved_ids"] and conf == roles["confirmation_reserved_ids"]
    assert len(set(loc + e2 + conf)) == 64
    assert not set(loc + e2 + conf) & {x["target_id"] for x in members["train_members"]}

    model = torch.nn.Module()
    generator = torch.Generator().manual_seed(1921)
    originals = {}
    # Exact production target names, deliberately small operators, real padded halo.
    for name in q6.TARGETS:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        op = (torch.nn.Linear(8, 8, bias=False) if name in q6.LINEARS
              else torch.nn.Conv2d(8, 8, 3, padding=1, bias=False))
        values = torch.randint(-127, 128, op.weight.shape, generator=generator).float()
        values.reshape(8, -1)[:, 0] = -127
        with torch.no_grad():
            op.weight.copy_(values)
        parent.add_module(parts[-1], op)
        originals[name] = op
    adapter = family.FamilyCore(q6, q4)
    expected_counts = dict(Q6=0, patch_Q4=4, fc1_Q4=12, fc2_Q4=12, mlp_Q4=24)
    checks = []
    for mode in family.MODES:
        count = 0
        with adapter.installed(model, mode, ste=False) as wrappers:
            assert len(wrappers) == 28
            for name, op in wrappers.items():
                low = ((mode == "patch_Q4" and name in q6.CONVS)
                       or (mode == "fc1_Q4" and name in q6.LINEARS and name.endswith(".fc1"))
                       or (mode == "fc2_Q4" and name in q6.LINEARS and name.endswith(".fc2"))
                       or (mode == "mlp_Q4" and name in q6.LINEARS))
                count += int(low)
                assert op.mode == ("Q4" if low else "Q6")
                assert op.weight is originals[name].weight
                is_conv = name in q6.CONVS
                shape = (10, 1, 8, 10, 11) if is_conv else (10, 1, 10, 11, 8)
                x = torch.randint(0, 2, shape, generator=generator).float()
                xn = x.numpy().astype(np.int64)
                if is_conv:
                    padded = np.pad(xn, ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1)))
                    fields = [padded[:, :, :, iy:iy+10, ix:ix+11] for iy in range(3) for ix in range(3)]
                    rows = np.stack(fields, axis=3).transpose(0, 1, 4, 5, 2, 3).reshape(-1, 72)
                else:
                    rows = xn.reshape(-1, 8)
                w = originals[name].weight.detach().numpy().reshape(8, -1)
                scale = np.max(np.abs(w), axis=1) / np.float32(127)
                integers = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int64)
                reduced = np.floor_divide(integers, 16 if low else 4) * (16 if low else 4)
                tail2d = np.ones((10, 11), dtype=bool)
                tail2d[:9, :9] = False
                tail = np.tile(tail2d.reshape(-1), 10)
                golden = rows @ reduced.T
                golden[tail] = rows[tail] @ integers.T
                op.capture_integer = True
                with torch.no_grad():
                    y = op(x)
                assert np.array_equal(op.last_integer.numpy(), golden.astype(np.int32))
                yp = y.permute(0, 1, 3, 4, 2).numpy() if is_conv else y.numpy()
                assert np.array_equal(yp.reshape(-1, 8), golden.astype(np.float32) * scale)
                work = op.last_audit
                assert work["rows"] == 1100 and work["complete_rows"] == 810 and work["tail_rows"] == 290
                perbit = {bit: sum(c["rows"] for c in work["kernel_calls"] if c["bit"] == bit)
                          for bit in range(8)}
                assert perbit == {bit: 290 if bit < (4 if low else 2) else 1100 for bit in range(8)}
                assert all(c["coefficient"] == (-128 if c["bit"] == 7 else 1 << c["bit"])
                           for c in work["kernel_calls"])
            assert count == expected_counts[mode]
        assert all(model.get_submodule(n) is op for n, op in originals.items())
        checks.append({"mode": mode, "Q4_targets": count, "operators_independent_int64_exact": 28})
    try:
        with adapter.installed(model, "patch_Q4", ste=False):
            raise RuntimeError("intentional restoration check")
    except RuntimeError:
        pass
    assert all(model.get_submodule(n) is op for n, op in originals.items())
    assert source_hashes == {k: sha(p) for k, p in paths.items()}
    assert not torch.cuda.is_initialized()
    output = {
        "schema": "flow_e1_independent_synthetic_cpu_v1",
        "status": "PASS",
        "sources": {k: {"path": str(paths[k]), "sha256": v} for k, v in source_hashes.items()},
        "script_sha256": sha(__file__),
        "mode_checks": checks,
        "independent_integer_operator_cases": 140,
        "metadata_roles_reconstructed": [16, 16, 32],
        "restoration_after_exception": True,
        "GPU_calls": 0,
        "real_model_forwards": 0,
        "new_quality_arrays_read": False,
        "scope": "Synthetic 28-target dispatch and int64 arithmetic; no full backbone or GPU deployment claim",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "operators": 140, "output": str(args.output)}))


if __name__ == "__main__":
    main()
