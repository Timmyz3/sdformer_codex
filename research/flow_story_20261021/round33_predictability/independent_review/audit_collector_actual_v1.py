"""Execution-only R33 audit; raw GT/mask bytes may be hashed, never array-loaded."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
BASE = ROUND.parent
ROOT = BASE.parents[1]
MEMBERS_SHA = "d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"
COLLECTOR_SHA = "2829d5303d65260fe1c8b3849b39aab780cfd2543769e50884b998c46cb4687d"
E3_SHA = "fe993b75e46a4a1426810d370516dcad33ebb524a94d1c0b0dae0ddef1baf274"
STATE_SHA = "7be9fed7c55332a150b4c9f67e863d5e68db2bb0d1829a5344703427a48ff018"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def child(root, rel):
    p = Path(rel)
    assert not p.is_absolute() and ".." not in p.parts
    return root/p


def hash_record(x, shape, dtype):
    assert x["shape"] == shape and x["dtype"] == dtype
    assert re.fullmatch("[0-9a-f]{64}", x["sha256"])


def work_sha(work):
    return hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest()


def audit_heads(freeze_path, actual, membership):
    assert freeze_path is not None and sha(freeze_path) == actual["heads_freeze_sha256"]
    f = read(freeze_path)
    assert f["schema"] == "flow_r33_heads_freeze_v1" and f["membership_sha256"] == MEMBERS_SHA
    fit_ids = [r["target_id"] for r in membership["records"] if r["role"] == "fit"]
    confirm_ids = [r["target_id"] for r in membership["records"] if r["role"] == "confirm"]
    assert f["fit_ids"] == fit_ids and f["confirm_ids"] == confirm_ids
    assert not set(fit_ids) & set(confirm_ids)
    assert {"MODEL_V1.json", "FIT_RECEIPT_V1.json"}.issubset(f["files"])
    for name, digest in f["files"].items():
        assert Path(name).name == name and sha(freeze_path.parent/name) == digest
    model = read(freeze_path.parent/"MODEL_V1.json")
    receipt = read(freeze_path.parent/"FIT_RECEIPT_V1.json")
    assert model["schema"] == "flow_r33_voxel_small_predictors_v1" and model["alpha"] == .1
    assert model["metadata"]["fit_ids"] == fit_ids and model["metadata"]["membership_sha256"] == MEMBERS_SHA
    assert model["metadata"]["alpha_search"] is False and model["metadata"]["no_upgrade_threshold"] == 0
    assert receipt["schema"] == "flow_r33_fit_receipt_v1" and receipt["alpha"] == .1
    assert receipt["fit_ids"] == fit_ids and receipt["membership_sha256"] == MEMBERS_SHA
    assert receipt["model_sha256"] == f["files"]["MODEL_V1.json"] and receipt["tune_or_confirm_read"] is False
    assert receipt["fit_pack_sha256"] == model["metadata"]["fit_pack_sha256"]
    for key, source in [("predictor_source_sha256", "predictors.py"), ("runner_source_sha256", "run_predictors.py")]:
        assert receipt[key] == model["metadata"][key] == sha(ROUND/"algorithm"/source)
    return {"heads_freeze_sha256": sha(freeze_path), "model_sha256": f["files"]["MODEL_V1.json"],
            "fit_receipt_sha256": f["files"]["FIT_RECEIPT_V1.json"], "fit_identities": len(fit_ids),
            "scope": "Collector preflight requires this already-existing frozen model/receipt before CUDA; no confirmation labels are inputs to fitting."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Local source root containing saved_flow_data")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--heads-freeze", type=Path)
    ap.add_argument("--launch-receipt", type=Path)
    a = ap.parse_args()
    assert not a.output.exists()
    mpath = ROUND/"data/MEMBERSHIP_V1.json"
    assert sha(mpath) == MEMBERS_SHA
    membership = read(mpath)
    mreview = read(HERE/"MEMBERSHIP_INDEPENDENT_AUDIT_V1.json")
    assert mreview["status"] == "PASS_METADATA_SCOPE" and mreview["membership_sha256"] == MEMBERS_SHA
    assert sha(ROUND/"collect.py") == COLLECTOR_SHA
    input_source = ROOT/"research/flow_story_20261013/breadth/evaluate_breadth.py"
    assert sha(input_source) == "1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628"
    sreview = read(HERE/"COLLECTOR_SOURCE_REVIEW_V1.json")
    assert sreview["status"] == "PASS_FOR_FIT_TUNE" and sreview["collector_sha256"] == COLLECTOR_SHA
    contractpath = BASE/"e3_rne_probe/ENDPOINT_CONTRACT_V1.json"
    contract = read(contractpath)
    assert sha(contractpath) == "1be30c237a24fd7173b6c5678e41aa56093301ad5f76f57b96a5a55f82df8718"
    priorpath = BASE/"actual/e3_rne_probe_v1/results.json"
    assert sha(priorpath) == E3_SHA
    prior = read(priorpath)
    prior_audit = read(BASE/"independent_review/E3_ACTUAL_AUDIT_V1.json")
    assert prior_audit["status"] == "PASS" and prior_audit["result_sha256"] == E3_SHA
    assert sha(BASE/"actual/e0_recovery_v1/recovered_step512.pt") == contract["endpoint"]["weights_sha256"]
    resultpath = a.run/"results.json"
    d = read(resultpath)
    assert d["schema"] == "flow_r33_collector_v1" and d["status"] == "R33_COMPLETE"
    assert d["phase"] in ("fit_tune", "confirm")
    roles = {"fit", "tune"} if d["phase"] == "fit_tune" else {"confirm"}
    rows = [r for r in membership["records"] if r["role"] in roles]
    ids = [r["target_id"] for r in rows]
    n, dense = (80, 6) if d["phase"] == "fit_tune" else (40, 0)
    assert len(rows) == len(set(ids)) == n
    assert d["selected_ids"] == ids and d["roles"] == [r["role"] for r in rows]
    assert d["membership_sha256"] == MEMBERS_SHA and d["runner_sha256"] == COLLECTOR_SHA
    assert d["endpoint"] == contract["endpoint"] and d["final_state_sha256"] == STATE_SHA
    assert d["action_contract_sha256"] == sha(contractpath)
    assert d["completed_optimizer_steps"] == 0 and d["training_performed"] is False
    assert d["GT_arrays_read"] is False and d["quality_computed"] is False and d["resume_allowed"] is False
    assert (d["model_forwards"], d["formal_model_forwards"], d["engineering_model_forwards"]) == (7*n+dense, 7*n, dense)
    assert d["expected_formal_forwards"] == 7*n and d["expected_engineering_forwards"] == dense
    assert d["wall_cap_minutes"] == (75 if dense else 40)
    assert math.isfinite(d["wall_seconds"]) and 0 < d["wall_seconds"] <= 60*d["wall_cap_minutes"]
    assert 0 < d["max_allocated_bytes"] <= d["max_reserved_bytes"]
    assert d["runtime"]["AMP"] is False and d["runtime"]["TF32"] is False and d["runtime"]["device"]
    assert d["BN_scope"] == prior["BN_scope"]
    heads_binding = audit_heads(a.heads_freeze, d, membership) if not dense else None
    if dense:
        assert d["heads_freeze_sha256"] is None
    launch_path = a.launch_receipt
    if launch_path is None and dense:
        launch_path = ROUND/"FIT_TUNE_LAUNCH_RECEIPT_V1.json"
    if launch_path is not None:
        launch = read(launch_path); args = launch["args"]
        assert "-O" not in args and args[args.index("--phase")+1] == d["phase"]
        assert args[args.index("--runner-sha256")+1] == COLLECTOR_SHA
        assert Path(args[args.index("--output")+1]).name == a.run.name
        if not dense:
            assert args[args.index("--heads-freeze-sha256")+1] == d["heads_freeze_sha256"]

    # Hash all actual source files without loading GT/mask (or any input) values.
    for row in rows:
        assert {f["category"] for f in row["source_files"]} == {"input", "gt", "mask"}
        for f in row["source_files"]:
            p = child(a.source, f["relative_path"])
            assert sha(p) == f["sha256"] and p.stat().st_size == int(f["bytes"])
    actions = ["background_Q4"] + [f"upgrade_{i}" for i in range(6)]
    assert set(d["evaluations"]) == set(actions)
    assert set(d["parity"]) == (set(actions[1:]) if dense else set())
    inputs, observed_predictions, work_pins = {}, set(), {}
    for action, state in enumerate(actions, start=-1):
        ev, pev = d["evaluations"][state], prior["evaluations"][state]
        exemplar = ev["first_frame_full_work"]
        assert exemplar == pev["frames"][0]["work"]  # Frozen W/shape/action, already independently audited.
        wdigest = work_sha(exemplar); work_pins[state] = wdigest
        assert ev["mask_layout"] == pev["mask_layout"] and ev["action"] == action
        assert ev["state_sha256"] == STATE_SHA and ev["model_forwards"] == n and ev["GT_arrays_read"] is False
        assert [f["target_id"] for f in ev["frames"]] == ids and len(ev["frames"]) == n
        for row, frame in zip(rows, ev["frames"]):
            tid = row["target_id"]
            assert frame["sequence"] == row["sequence"] and frame["role"] == row["role"] and frame["action"] == action
            source = next(f for f in row["source_files"] if f["category"] == "input")
            assert frame["input_source"] == source
            hash_record(frame["actual_input"], [1, 10, 2, 480, 640], "torch.float32")
            if state == "background_Q4":
                inputs[tid] = frame["actual_input"]
            assert frame["actual_input"] == inputs[tid]
            assert frame["actual_BN_calls"] == pev["frames"][0]["actual_BN_calls"]
            assert frame["actual_affine_call_counts"] == pev["frames"][0]["actual_affine_call_counts"]
            assert len(frame["actual_BN_calls"]) == 78 and len(frame["actual_affine_call_counts"]) == 28
            assert frame["work_sha256"] == wdigest and frame["work_checks"] == pev["frames"][0]["work_checks"]
            assert set(frame["heads"]) == {f"head{i}" for i in range(4)}
            for value in frame["heads"].values():
                hash_record(value, [2, 480, 640], "torch.float32")
            predfile = child(a.run, frame["prediction_file"])
            assert predfile not in observed_predictions and sha(predfile) == frame["prediction_file_sha256"]
            observed_predictions.add(predfile)
            with np.load(predfile, allow_pickle=False) as z:
                assert z.files == ["head3"]
                x = z["head3"]
                assert x.shape == (2, 480, 640) and x.dtype == np.float32 and np.isfinite(x).all()
                payload = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            assert payload == frame["prediction_sha256"] == frame["heads"]["head3"]["sha256"]
        if action >= 0 and dense:
            parity = d["parity"][state]
            assert parity["target_id"] == ids[0] and rows[0]["role"] == "fit"
            assert parity["comparison"]["pass"] is True
            checks = parity["comparison"]["checks"]
            assert set(checks) == set(prior["parity"][state]["comparison"]["checks"])
            assert len(checks) == 137 and sum(k.startswith("integer::") for k in checks) == 28
            assert sum(k.startswith("PSN::") for k in checks) == 105
            for key, value in checks.items():
                assert value["bitwise_equal"] and value["different_bytes"] == 0
                assert value["actual_sha256"] == value["reference_sha256"]
                if key.startswith("head"):
                    assert value["actual_sha256"] == ev["frames"][0]["heads"][key]["sha256"]
            for record_name, is_dense in [("actual_record", False), ("dense_record", True)]:
                rec = parity[record_name]
                assert (rec["PSN_count"], rec["integer_count"], rec["head_count"]) == (105, 28, 4)
                assert rec["ste_entry"] is False and rec["dense_integer_reference"] is is_dense
                assert rec["actual_BN_calls"] == ev["frames"][0]["actual_BN_calls"]
                assert rec["actual_target_calls"] == ev["frames"][0]["actual_affine_call_counts"]
                if not is_dense:
                    assert rec["work"] == exemplar
    assert len(observed_predictions) == 7*n
    assert set((a.run/"predictions").rglob("*.npz")) == observed_predictions
    out = dict(schema="flow_r33_actual_execution_audit_v1", status="PASS", phase=d["phase"],
               source_result_sha256=sha(resultpath), collector_sha256=COLLECTOR_SHA,
               membership_sha256=MEMBERS_SHA, endpoint=d["endpoint"], model_forwards=d["model_forwards"],
               formal_predictions=7*n, dense_references=dense, optimizer_updates=0,
               actual_source_files_rehashed=3*n, prediction_files_and_payloads_rehashed=7*n,
               all_seven_action_input_records_identical_per_identity=True,
               frozen_prepare_input_source_sha256=sha(input_source),
               action_work_exemplar_sha256=work_pins, prior_E3_result_sha256=E3_SHA,
               source_arrays_loaded=0, GT_values_read=False, quality_computed=False,
               heads_binding=heads_binding, recorded_wall_seconds=d["wall_seconds"],
               max_allocated_bytes=d["max_allocated_bytes"], max_reserved_bytes=d["max_reserved_bytes"],
               runtime=d["runtime"], launch_receipt_sha256=sha(launch_path) if launch_path else None,
               quality_analysis_allowed=True, auditor_sha256=sha(__file__),
               limitations=["Intermediate parity and head0..2 are runtime hash receipts, not separately archived raw tensors.",
                            "Source bytes and frozen prepare_input are bound; CUDA minmax input is checked identical across seven actions. CPU/GPU float32 normalization bitwise bridge is not claimed.",
                            "Memory and wall time are process receipts, not deployment latency/energy or rental time."])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({k: out[k] for k in ["status", "phase", "source_result_sha256", "model_forwards", "formal_predictions", "dense_references", "actual_source_files_rehashed", "GT_values_read"]}, indent=2))


if __name__ == "__main__":
    main()
