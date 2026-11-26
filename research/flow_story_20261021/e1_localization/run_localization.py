#!/usr/bin/env python3
"""Fixed 16 x 5 family screen, same recovered Q6-W, no quality selection.

Four extra dense-reference forwards on first fixed identity are engineering
checks, not new quality samples. All 16 Q6 predictions match old saved payloads.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path
import family_core

sys.dont_write_bytecode = True
Q6_CORE_SHA = "19623636229b4dc5a85132e4acc8bf217f9f62a5e51fcab0fdd7394ee186f788"
WEIGHT_SHA = "7fd8cb4ab172ca7f21bc786c8b0eae6f508cb2e5a6f5fb34653ac8607552e0aa"
STATE_SHA = "cffd154767fc6b01de1a88c24f8eebf06c32f097703df61b0cf1e54af2782747"
SELECTION_SHA = "60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def run(a, result):
    import torch
    import numpy as np
    root = a.r27_root / "implementation"
    helper = load(root / "run_recovery.py", "_e1_old_helper")
    smoke = load(root / "run_smoke.py", "_e1_old_smoke")
    assert sha(root / "recovery_integer_core.py") == Q6_CORE_SHA
    q6 = load(root / "recovery_integer_core.py", "_e1_q6")
    q4 = load(a.r28_root / "implementation/recovery_integer_core.py", "_e1_q4")
    assert sha(a.weights) == WEIGHT_SHA and sha(a.selections) == SELECTION_SHA
    old_result = json.loads(a.old_result.read_text())
    assert old_result["weights_sha256"] == WEIGHT_SHA
    roles = json.loads(a.selections.read_text())
    pre = copy.copy(a)
    pre.runner_sha256 = sha(root / "run_smoke.py")
    p, members, _, _ = smoke.prepare(pre)
    rows = roles["localization_members"]
    assert len(rows) == 16 and all(x in members["dev_members"] for x in rows)
    assert [r["target_id"] for r in rows] == roles["localization_ids"]
    sources = {str(x): sha(x) for x in
               [Path(__file__), Path(family_core.__file__), root / "run_smoke.py",
                root / "run_recovery.py", root / "recovery_integer_core.py",
                a.r28_root / "implementation/recovery_integer_core.py", a.selections,
                a.old_result, a.members]}
    result.update(sources=sources, weights_sha256=WEIGHT_SHA, state_sha256=STATE_SHA,
                  selection_scope=roles["scope"], expected_quality_predictions=80,
                  expected_engineering_forwards=4, training=False,
                  development_GT_arrays_read=False, quality_selection=False)
    if not a.execute:
        write(a.preflight, dict(result, status="CPU_PREFLIGHT_PASS"))
        return
    lock = p["modules"][3].GPU_LOCK.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not torch.cuda.is_initialized()
        a.output.mkdir(parents=True, exist_ok=False)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("E1 30min cap")))
        signal.alarm(1800)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261019)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(20261019)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model, functional, bn = smoke.make_model(p, "full9", torch.device("cuda:0"), torch)
        cp = torch.load(a.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(cp["model_state_dict"], strict=True)
        del cp
        model.eval().requires_grad_(False)
        assert p["modules"][0].state_digest(model) == STATE_SHA
        bn_sig = p["base"].bn_signature(model, torch)
        core = family_core.FamilyCore(q6, q4)
        old_frames = {x["target_id"]: x for x in old_result["evaluations"]["endpoint"]["frames"]}
        result.update(status="RUNNING", runtime={"torch": str(torch.__version__),
                      "device": torch.cuda.get_device_name(0), "TF32": False, "AMP": False},
                      evaluations={}, parity={}, model_forwards=0)
        write(a.output / "results.json", result)
        for mode in family_core.MODES:
            frames = []
            folder = a.output / "predictions" / mode
            folder.mkdir(parents=True)
            for index, row in enumerate(rows):
                tid = row["target_id"]
                ev = helper.source_evidence(members, tid)
                events = helper.event_only(p, row, ev, np)
                chunk = p["base"].prepare_input(events, torch.device("cuda:0"), torch)
                if index == 0 and mode != "Q6":
                    actual, record = smoke.capture_full(model, functional, chunk, mode, core, torch, ste=False)
                    result["model_forwards"] += 1
                    reference, ref_record = smoke.capture_full(model, functional, chunk, mode, core, torch,
                                                               ste=False, reference=True)
                    result["model_forwards"] += 1
                    cmp = smoke.compare_full(actual, reference, torch)
                    assert cmp["pass"], "Actual integer/PSN/head reference parity"
                    result["parity"][mode] = dict(target_id=tid, comparison=cmp)
                    heads = {f"head{i}": helper.tensor_record(actual[f"head{i}"][0]) for i in range(4)}
                    prediction = actual["head3"][0].numpy()
                    work = helper.work_summary(record["work"])
                    bns = record["actual_BN_calls"]
                    del reference, actual
                else:
                    functional.reset_net(model)
                    with core.installed(model, mode, ste=False) as wraps:
                        with helper.observed(model, core, torch) as (bns, calls):
                            with torch.no_grad():
                                flows = model(chunk)["flow"]
                        work = helper.work_summary({n: copy.deepcopy(w.last_audit) for n, w in wraps.items()})
                    result["model_forwards"] += 1
                    assert len(flows) == 4
                    heads = {f"head{i}": helper.tensor_record(v[0]) for i, v in enumerate(flows)}
                    prediction = flows[3][0].detach().cpu().numpy()
                    del flows
                assert prediction.shape == (2, 480, 640) and prediction.dtype == np.float32
                assert np.isfinite(prediction).all()
                for name, w in work.items():
                    expected_mode = "Q4" if family_core.selected(name, mode, q6) else "Q6"
                    assert w["mode"] == expected_mode and w["full_T_and_channels"]
                if mode == "Q6":
                    assert heads == old_frames[tid]["heads"], "Exact recovered Q6 baseline replay"
                path = folder / (tid + ".npz")
                np.savez_compressed(path, head3=prediction)
                frames.append(dict(target_id=tid, sequence=row["sequence"], input_evidence=ev,
                                   prediction_file=str(path.relative_to(a.output)),
                                   prediction_file_sha256=sha(path), prediction_sha256=heads["head3"]["sha256"],
                                   heads=heads, actual_BN_calls=bns, work=work))
                del chunk, events, prediction
                print(json.dumps(dict(mode=mode, completed=index + 1)), flush=True)
            result["evaluations"][mode] = dict(frames=frames, model_forwards=16)
            write(a.output / "results.json", result)
        assert result["model_forwards"] == 84
        assert p["modules"][0].state_digest(model) == STATE_SHA
        assert p["base"].bn_signature(model, torch) == bn_sig
        result.update(status="LOCALIZATION80_AND4_REFERENCE_COMPLETE", wall_seconds=time.monotonic() - start,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(),
                      max_reserved_bytes=torch.cuda.max_memory_reserved(),
                      background_replay_bitwise_equal_all16=True)
        write(a.output / "results.json", result)
    except BaseException as e:
        if a.output.exists():
            result.update(status="FAILED", error=str(e), traceback=traceback.format_exc())
            write(a.output / "results.json", result)
        raise
    finally:
        signal.alarm(0)
        lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("r27-root", "r28-root", "r26-root", "members", "weights", "old-result",
              "selections", "output"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--preflight", type=Path)
    a = ap.parse_args()
    assert not a.output.exists()
    if not a.execute:
        assert a.preflight and not a.preflight.exists()
    run(a, {"schema": "flow_e1_family_screen_20261021_v1"})


if __name__ == "__main__":
    main()
