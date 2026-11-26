"""E1 fixed 16x5 original-array review, with two reused native references.

All result/artifact/source bindings precede numerical GT access. No model or GPU.
R27 raw region mathematics is reused, but no hard-coded 64-frame analyze() call.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
R27 = ROOT / "research/flow_story_20261019/round27"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(R27 / "independent_review"))
import analysis_core as mathcore
import raw_quality_math as raw

MODES = ("Q6", "patch_Q4", "fc1_Q4", "fc2_Q4", "mlp_Q4")
STATE_SHA = "cffd154767fc6b01de1a88c24f8eebf06c32f097703df61b0cf1e54af2782747"
WEIGHT_SHA = "7fd8cb4ab172ca7f21bc786c8b0eae6f508cb2e5a6f5fb34653ac8607552e0aa"
REGIONS_SHA = "2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def child(root, name):
    p = Path(name)
    assert not p.is_absolute() and ".." not in p.parts
    return Path(root) / p


def prediction(run, row):
    p = child(run, row["prediction_file"])
    assert sha(p) == row["prediction_file_sha256"]
    with np.load(p, allow_pickle=False) as data:
        assert data.files == ["head3"]
        x = data["head3"]
        assert x.shape == (2, 480, 640) and x.dtype == np.float32 and np.isfinite(x).all()
        assert raw.payload_sha(x) == row["prediction_sha256"] == row["heads"]["head3"]["sha256"]
    return p


def low(name, mode):
    return ((mode == "patch_Q4" and ".patch_embed.residual_encoding." in name)
            or (mode == "fc1_Q4" and name.endswith(".fc1"))
            or (mode == "fc2_Q4" and name.endswith(".fc2"))
            or (mode == "mlp_Q4" and ".mlp." in name))


def bind(run):
    result = read(run / "results.json")
    assert result["schema"] == "flow_e1_family_screen_20261021_v1"
    assert result["status"] == "LOCALIZATION80_AND4_REFERENCE_COMPLETE"
    assert result["model_forwards"] == 84 and result["expected_quality_predictions"] == 80
    assert result["expected_engineering_forwards"] == 4
    assert result["training"] is False and result["development_GT_arrays_read"] is False
    assert result["quality_selection"] is False and result["background_replay_bitwise_equal_all16"] is True
    assert result["state_sha256"] == STATE_SHA and result["weights_sha256"] == WEIGHT_SHA
    assert result["runtime"]["TF32"] is False and result["runtime"]["AMP"] is False
    assert set(result["evaluations"]) == set(MODES)
    cpu = read(HERE / "E1_INDEPENDENT_CPU_V1.json")
    assert cpu["status"] == "PASS"
    for key in ("family", "runner", "q6", "q4", "roles", "members"):
        desc = cpu["sources"][key]
        assert sha(desc["path"]) == desc["sha256"]
        assert desc["sha256"] in result["sources"].values()
    roles_path = Path(cpu["sources"]["roles"]["path"])
    roles = read(roles_path)
    members = read(R27 / "independent_review/membership_v1.json")
    regions_path = ROOT / "research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
    assert sha(regions_path) == REGIONS_SHA
    regions = read(regions_path)
    ids = roles["localization_ids"]
    assert len(ids) == len(set(ids)) == 16
    assert not set(ids) & set(regions["fit_target_ids"])
    assert not set(ids) & set(members["sealed_ids"] + members["H8_historical_ids"])
    records = {x["target_id"]: x for x in members["records"]}
    q6_result_path = R27 / "actual_v1/formal_Q6_v1/results.json"
    q6_old = read(q6_result_path)
    assert sha(q6_result_path) in result["sources"].values()
    assert q6_old["final_state_sha256"] == STATE_SHA and q6_old["weights_sha256"] == WEIGHT_SHA
    oldframes = {x["target_id"]: x for x in q6_old["evaluations"]["endpoint"]["frames"]}
    sources, evidence = {}, {}
    for tid in ids:
        sources[tid], evidence[tid] = {}, {}
        for record in records[tid]["source_files"]:
            kind = {"input": "events", "gt": "gt", "mask": "mask"}[record["category"]]
            p = child(R27 / "dev64_source_snapshot", record["relative_path"])
            assert sha(p) == record["sha256"] and p.stat().st_size == int(record["bytes"])
            sources[tid][kind] = p
            evidence[tid][kind] = dict(relative_path=record["relative_path"], sha256=record["sha256"])
    bound, costs = {}, {}
    expected_count = dict(Q6=0, patch_Q4=4, fc1_Q4=12, fc2_Q4=12, mlp_Q4=24)
    for mode in MODES:
        frames = result["evaluations"][mode]["frames"]
        assert result["evaluations"][mode]["model_forwards"] == 16
        assert [x["target_id"] for x in frames] == ids
        mode_costs = []
        for row in frames:
            tid = row["target_id"]
            assert row["sequence"] == records[tid]["sequence"] and row["input_evidence"] == evidence[tid]
            assert row["actual_BN_calls"] == oldframes[tid]["actual_BN_calls"]
            assert len(row["actual_BN_calls"]) == 78 and set(row["actual_BN_calls"].values()) == {1}
            assert set(row["work"]) == set(oldframes[tid]["work"]) and len(row["work"]) == 28
            saved = 0
            assert sum(low(n, mode) for n in row["work"]) == expected_count[mode]
            for name, w in row["work"].items():
                oldw = oldframes[tid]["work"][name]
                for field in ("rows", "Cin", "Cout", "tail_rows", "complete_rows"):
                    assert w[field] == oldw[field]
                islow = low(name, mode)
                drop = 4 if islow else 2
                assert w["mode"] == ("Q4" if islow else "Q6") and w["reference"] is False
                assert w["logical_row_plane_work"] == 8 * w["rows"] - drop * w["complete_rows"]
                assert w["full_T_and_channels"] and w["binary_verified"] and w["int32_accumulation"]
                assert w["scale_applications"] == 1 and w["spatial_rows_retained"]
                assert w["low_bit_rule"] == f"arithmetic_shift_drop{drop}_no_reclamp_no_offset"
                if islow:
                    saved += 2 * w["complete_rows"] * math.ceil(w["Cin"] / 128) * math.ceil(w["Cout"] / 16)
            mode_costs.append(saved)
            if mode == "Q6":
                assert row["heads"] == oldframes[tid]["heads"]
            bound[(mode, tid)] = prediction(run, row)
        assert len(set(mode_costs)) == 1
        costs[mode] = dict(additional_cancelled_core_ticks_vs_Q6=mode_costs[0],
                           fraction_of_native_arithmetic=mode_costs[0] / 3652955059,
                           modeled_upper_bound_only=True, scorer_and_supply_costs_deducted=False)
    assert set(result["parity"]) == set(MODES[1:])
    for mode in MODES[1:]:
        p = result["parity"][mode]
        assert p["target_id"] == ids[0] and p["comparison"]["pass"]
        checks = p["comparison"]["checks"]
        assert sum(k.startswith("integer::") for k in checks) == 28
        assert sum(k.startswith("PSN::") for k in checks) == 105
        assert sum(k.startswith("head") for k in checks) == 4
        assert all(v["bitwise_equal"] and v["different_bytes"] == 0
                   and v["actual_sha256"] == v["reference_sha256"] for v in checks.values())
    native_run = R27 / "actual_v1/formal_native_v1"
    native = read(native_run / "results.json")
    old_audit = read(R27 / "actual_execution_audit_v1.json")
    assert old_audit["status"] == "PASS" and old_audit["quality_analysis_allowed"]
    assert old_audit["source_results"]["native"] == sha(native_run / "results.json")
    assert old_audit["source_results"]["Q6"] == sha(q6_result_path)
    for stage, state in (("initial", "original_native"), ("endpoint", "recovered_native")):
        rows = {x["target_id"]: x for x in native["evaluations"][stage]["frames"]}
        for tid in ids:
            assert rows[tid]["input_evidence"] == evidence[tid]
            bound[(state, tid)] = prediction(native_run, rows[tid])
    audit = dict(status="PASS", actual_result_sha256=sha(run / "results.json"),
                 same_W_state_sha256=STATE_SHA, fixed_selection_sha256=sha(roles_path),
                 new_quality_predictions=80, engineering_reference_forwards=4,
                 reused_native_reference_predictions=32, total_distinct_frame_identities=16,
                 original_source_arrays_bound=48, source_byte_binding_before_GT_values=True,
                 four_full_graph_reference_comparisons=True, all16_Q6_background_heads_exact=True,
                 source_CPU_review_sha256=sha(HERE / "E1_INDEPENDENT_CPU_V1.json"),
                 limitation="Intermediate integer/PSN arrays are runtime comparisons, not saved offline replay.")
    return result, roles, regions, sources, bound, costs, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    result, roles, regions, sources, bound, costs, audit = bind(args.run)
    states = MODES + ("original_native", "recovered_native")
    frames = []
    for row in roles["localization_members"]:
        tid = row["target_id"]
        # First numerical source-array access after all complete execution bindings.
        arrays = {k: np.load(p, allow_pickle=False) for k, p in sources[tid].items()}
        masks = raw.region_masks(arrays["gt"], arrays["mask"], arrays["events"],
                                 regions["boundary_threshold_pixels"],
                                 regions["large_displacement_threshold_pixels"])
        metrics = {}
        for state in states:
            with np.load(bound[(state, tid)], allow_pickle=False) as data:
                metrics[state] = raw.measure(data["head3"], arrays["gt"], masks)
        frames.append(dict(target_id=tid, sequence=row["sequence"], states=metrics))
    seqs = list(dict.fromkeys(x["sequence"] for x in frames))
    summaries = {state: {scope: {region: mathcore.aggregate(
        [x["states"][state][region] for x in frames if scope == "overall" or x["sequence"] == scope])
        for region in mathcore.REGIONS} for scope in ["overall"] + seqs} for state in states}
    pairs = [(m, ref) for m in MODES for ref in ("Q6", "original_native", "recovered_native") if m != ref]
    comparisons = {c + "__vs__" + r: mathcore.comparison(frames, summaries, c, r, seqs) for c, r in pairs}
    output = dict(schema="flow_e1_independent_raw_quality_v1", status="COMPLETE",
                  execution_review=audit, states=list(states), summaries=summaries,
                  comparisons=comparisons, frames=frames, family_cost_diagnostic=costs,
                  scope=roles["scope"], region_definition_sha256=REGIONS_SHA,
                  interpretation="Static family localization, unequal compute; no dynamic scorer or unseen confirmation demonstrated.",
                  legacy_quality_lines="R27 3% allGT/5% B,L retained as diagnostic reference lines, not new mandatory eligibility.",
                  selector_learning_demonstrated=False, new_quality_predictions=80,
                  old_native_reference_predictions_reused=32, frame_identities=16,
                  reducer_sha256=sha(__file__), raw_math_sha256=sha(raw.__file__),
                  scalar_math_sha256=sha(mathcore.__file__))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    table = {s: {r: summaries[s]["overall"][r]["pixel_weighted_AEE"] for r in ("all_valid", "B", "L")} for s in states}
    print(json.dumps(dict(status="COMPLETE", pixel_AEE=table, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
