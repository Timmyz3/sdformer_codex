"""E0 diagnostic/recovery raw-array quality; actual bindings precede GT values."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as shared
import audit_e0_work_v1 as work

ROOT, R27 = shared.ROOT, shared.R27
BASE = HERE.parent / "e0_ordinary"
R30 = ROOT / "research/flow_story_20261020/round30"
W0 = "b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798"
sha, read, child, prediction = shared.sha, shared.read, shared.child, shared.prediction
raw, mathcore = shared.raw, shared.mathcore


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(a):
    result = read(a.run / "results.json")
    assert result["schema"] == "flow_e0_ordinary_quantizer_v1"
    freeze = read(BASE / "SOURCE_FREEZE_V1.json")
    assert all(sha(BASE / k) == v for k, v in freeze["files"].items())
    for field, name in (("runner", "run_joint_recovery.py"), ("core", "mixed_integer_core.py"),
                        ("mask_helper", "mask_contract.py")):
        assert result["sources"][field] == freeze["files"][name]
    members_path = R27 / "independent_review/membership_v1.json"
    members = read(members_path)
    assert result["sources"]["members"] == sha(members_path)
    records = {x["target_id"]: x for x in members["records"]}
    regions_path = ROOT / "research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
    assert sha(regions_path) == shared.REGIONS_SHA
    regions = read(regions_path)
    contract = read(R30 / "hardware/MASK_CONTRACT_V1.json")
    assert result["sources"]["mask_contract"] == sha(R30 / "hardware/MASK_CONTRACT_V1.json")
    assert not result["development_GT_read"] and not result["quality_selection"] and not result["quality_computed"]
    assert not result["runtime"]["AMP"] and not result["runtime"]["TF32"]
    old_audit = read(R27 / "actual_execution_audit_v1.json")
    assert old_audit["status"] == "PASS" and old_audit["quality_analysis_allowed"]
    native_run, q8_run = R27 / "actual_v1/formal_native_v1", R27 / "actual_v1/formal_Q8_v1"
    native, q8 = read(native_run / "results.json"), read(q8_run / "results.json")
    assert old_audit["source_results"]["native"] == sha(native_run / "results.json")
    assert old_audit["source_results"]["Q8"] == sha(q8_run / "results.json")
    if a.kind == "diagnostic":
        roles_path = HERE.parent / "selections_v1.json"
        roles = read(roles_path)
        rows = roles["localization_members"]
        stages = ("original_Q8", "original_M_drop4", "original_M_midpoint", "original_M_rne4")
        assert result["status"] == "FOUR_ORDINARY_STATES_16_IDENTITIES_COMPLETE"
        assert result["task"] == "diagnostic" and result["training_performed"] is False
        assert result["model_forwards"] == result["expected_model_forwards"] == 68
        assert result["quality_model_forwards"] == 64 and result["engineering_model_forwards"] == 4
        assert result["diagnostic_selection_sha256"] == result["sources"]["selections"] == sha(roles_path)
        assert result["diagnostic_ids"] == roles["localization_ids"]
        assert result["initial_state_sha256"] == result["final_state_sha256"] == W0
        assert set(result["diagnostic_parity"]) == {"original_M_midpoint", "original_M_rne4"}
        old_work = load(R30 / "audit_mixed_work_v1.py", "_e0_old_geometry_work")
        generic = load(ROOT / "research/flow_story_20261019/round28/audit_actual_recovery_v1.py",
                       "_e0_generic_byte_parity")
        oldm_run = R30 / "actual_v1/formal_joint_v1"
        oldm_result = read(oldm_run / "results.json")
        q8frames = {x["target_id"]: x for x in q8["evaluations"]["initial"]["frames"]}
        mframes = {x["target_id"]: x for x in oldm_result["evaluations"]["initial_M"]["frames"]}
        for stage in ("original_M_midpoint", "original_M_rne4"):
            parity = result["diagnostic_parity"][stage]
            assert parity["input_target_id"] == rows[0]["target_id"]
            assert set(parity["comparisons"]) == {"training", "dense_integer"}
            records_parity = [parity["deployment_record"]]
            for route in ("training", "dense_integer"):
                generic.parity(parity["comparisons"][route], True)
                records_parity.append(parity["comparisons"][route]["record"])
            for rec, dense in zip(records_parity, (False, False, True)):
                assert len(rec["actual_BN_calls"]) == 78 and set(rec["actual_BN_calls"].values()) == {1}
                assert set(rec["actual_target_calls"]) == work.TARGETS and set(rec["actual_target_calls"].values()) == {1}
                work.check_calls(q8frames[rows[0]["target_id"]]["actual_affine_calls"], rec["work"], "M",
                                 (480, 640), contract, dense=dense,
                                 variant="midpoint" if stage.endswith("midpoint") else "rne4")
            pp = child(a.run, parity["heads_file"])
            assert sha(pp) == parity["heads_sha256"]
            with np.load(pp, allow_pickle=False) as z:
                assert set(z.files) == {"head0", "head1", "head2", "head3"}
                for name in z.files:
                    assert z[name].shape == (1, 2, 480, 640) and np.isfinite(z[name]).all()
                    assert raw.payload_sha(z[name]) == parity["comparisons"]["training"]["checks"][name]["actual_sha256"]
        execution = dict(status="PASS", model_forwards=68, quality_predictions=64,
                         engineering_model_forwards=4, actual_result_sha256=sha(a.run / "results.json"),
                         scope="Complete no-training diagnosis, source/operator work/parity/payload checks; runtime intermediates not separately saved.")
    else:
        assert a.audit
        execution = read(a.audit)
        assert execution["schema"] == "flow_e0_independent_actual_execution_audit_v1"
        assert execution["status"] == "PASS" and execution["quality_analysis_allowed"]
        assert execution["source_result_sha256"] == sha(a.run / "results.json")
        assert execution["counts"] == dict(optimizer_updates=512, quality_predictions=320, formal_model_forwards=1353)
        rows = members["dev_members"]
        stages = ("initial_Q4", "initial_M", "endpoint_Q4", "endpoint_Q8", "endpoint_M")
        audited = {(x["state"], x["target_id"]): x for x in execution["predictions"]}
        labels = {"initial_Q4": "initial_rne_Q4", "initial_M": "initial_rne_M",
                  **{"endpoint_" + m: "recovered_rne_" + m for m in ("Q4", "Q8", "M")}}
        assert len(audited) == 320
    ids = [x["target_id"] for x in rows]
    assert len(set(ids)) == len(ids) == (16 if a.kind == "diagnostic" else 64)
    assert not set(ids) & set(regions["fit_target_ids"] + members["sealed_ids"] + members["H8_historical_ids"])
    assert set(result["evaluations"]) == set(stages)
    sources, evidence, bound = {}, {}, {}
    for tid in ids:
        sources[tid], evidence[tid] = {}, {}
        for rec in records[tid]["source_files"]:
            kind = {"input": "events", "gt": "gt", "mask": "mask"}[rec["category"]]
            p = child(R27 / "dev64_source_snapshot", rec["relative_path"])
            assert sha(p) == rec["sha256"] and p.stat().st_size == int(rec["bytes"])
            sources[tid][kind] = p
            evidence[tid][kind] = dict(relative_path=rec["relative_path"], sha256=rec["sha256"])
    for stage in stages:
        ev = result["evaluations"][stage]
        assert ev["model_forwards"] == len(ids) and [x["target_id"] for x in ev["frames"]] == ids
        for row in ev["frames"]:
            tid = row["target_id"]
            assert row["input_evidence"] == evidence[tid]
            p = prediction(a.run, row)
            if a.kind == "diagnostic":
                assert ev["state_sha256"] == W0
                assert row["actual_BN_calls"] == q8frames[tid]["actual_BN_calls"]
                assert set(row["actual_affine_call_counts"]) == work.TARGETS
                assert set(row["actual_affine_call_counts"].values()) == {1}
                mode = "Q8" if stage == "original_Q8" else "M"
                checker = old_work if stage in stages[:2] else work
                kwargs = {} if stage in stages[:2] else {"variant": "midpoint" if stage.endswith("midpoint") else "rne4"}
                checker.check_calls(q8frames[tid]["actual_affine_calls"], row["work"], mode, (480, 640), contract, **kwargs)
                if stage == "original_Q8":
                    assert row["heads"] == q8frames[tid]["heads"]
                if stage == "original_M_drop4":
                    assert row["heads"] == mframes[tid]["heads"]
            else:
                av = audited[(labels[stage], tid)]
                assert av["file_sha256"] == row["prediction_file_sha256"]
                assert av["head3_sha256"] == row["prediction_sha256"]
                assert av["input_evidence"] == row["input_evidence"]
            bound[(stage, tid)] = p
    refs = [("original_native", native_run, native, "initial")]
    if a.kind == "recovery":
        refs += [("recovered_native", native_run, native, "endpoint"),
                 ("original_Q8", q8_run, q8, "initial")]
    for state, run, result_ref, phase in refs:
        byid = {x["target_id"]: x for x in result_ref["evaluations"][phase]["frames"]}
        for tid in ids:
            assert byid[tid]["input_evidence"] == evidence[tid]
            bound[(state, tid)] = prediction(run, byid[tid])
    return rows, stages + tuple(x[0] for x in refs), sources, bound, regions, execution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("diagnostic", "recovery"), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    rows, states, sources, bound, regions, execution = bind(a)
    frames = []
    for row in rows:
        tid = row["target_id"]
        original = {k: np.load(p, allow_pickle=False) for k, p in sources[tid].items()}
        masks = raw.region_masks(original["gt"], original["mask"], original["events"],
                                 regions["boundary_threshold_pixels"], regions["large_displacement_threshold_pixels"])
        measurements = {}
        for state in states:
            with np.load(bound[(state, tid)], allow_pickle=False) as z:
                measurements[state] = raw.measure(z["head3"], original["gt"], masks)
        frames.append(dict(target_id=tid, sequence=row["sequence"], states=measurements))
    seqs = list(dict.fromkeys(x["sequence"] for x in rows))
    summaries = {state: {scope: {region: mathcore.aggregate(
        [x["states"][state][region] for x in frames if scope == "overall" or x["sequence"] == scope])
        for region in mathcore.REGIONS} for scope in ["overall"] + seqs} for state in states}
    if a.kind == "diagnostic":
        pairs = [(s, r) for s in states[:4] for r in ("original_native", "original_Q8") if s != r]
        pairs += [(s, "original_M_drop4") for s in ("original_M_midpoint", "original_M_rne4")]
    else:
        pairs = [(s, r) for s in states[:5] for r in ("original_native", "recovered_native", "original_Q8")]
        pairs += [("endpoint_Q4", "initial_Q4"), ("endpoint_M", "initial_M"), ("endpoint_M", "endpoint_Q8")]
    comparisons = {c + "__vs__" + r: mathcore.comparison(frames, summaries, c, r, seqs) for c, r in pairs}
    output = dict(schema="flow_e0_independent_raw_quality_v1", status="COMPLETE", kind=a.kind,
                  execution_review=execution, states=list(states), summaries=summaries, comparisons=comparisons, frames=frames,
                  new_quality_predictions=64 if a.kind == "diagnostic" else 320,
                  reused_reference_predictions=16 if a.kind == "diagnostic" else 192,
                  region_definition_sha256=shared.REGIONS_SHA, reducer_sha256=sha(__file__),
                  raw_math_sha256=sha(raw.__file__), scalar_math_sha256=sha(mathcore.__file__),
                  scope="Previously exposed training-domain development; ordinary quantization only, no task-specific scorer or official DSEC result.",
                  legacy_quality_lines="3% allGT/5% B,L are fully reported diagnostic lines; no new mandatory quality gate.",
                  selector_learning_demonstrated=False)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    table = {s: {r: summaries[s]["overall"][r]["pixel_weighted_AEE"] for r in ("all_valid", "B", "L")} for s in states}
    print(json.dumps(dict(status="COMPLETE", pixel_AEE=table, output=str(a.output)), indent=2))


if __name__ == "__main__":
    main()
