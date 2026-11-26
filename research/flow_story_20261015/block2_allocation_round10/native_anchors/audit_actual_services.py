#!/usr/bin/env python3
"""Independent stdlib-only completed native-anchor service audit; no quality/model/GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R10 = HERE.parent
RAW_SHA = "1d5fe408ffd3a603f5c28093780bcafef74efba5bcf6c58ea8294fdbe1a88523"
FREEZE_SHA = "a47bbaf5d30d9253dfc8f09f0229ca3a7a535d0ca7db12230df3205b8391b376"
PARENT_SHA = "c31d18f5eca5d4343c02f347d0fbed0c9604120dfb874524f8caf6404b1b005d"
PARENT_AUDITOR_SHA = "02cfd6dcd1cee69c6905ea055b6c6805118047d8e5e9cc47e10df9092360dd52"
ARMS = ("native", "FF", "fine_s2b2")
ROWS = dict(native=558000, FF=558000, fine_s2b2=164200, CC=153600)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_wall(value):
    if isinstance(value, dict):
        return {k: without_wall(v) for k, v in value.items() if k != "wall_minutes"}
    if isinstance(value, list):
        return [without_wall(v) for v in value]
    return value


def audit(results, output):
    require(not output.exists(), "Fresh output required")
    require(sha(results) == RAW_SHA, "Complete raw SHA differs")
    require(sha(HERE/"freeze_v1.json") == FREEZE_SHA, "Freeze SHA differs")
    freeze = json.loads((HERE/"freeze_v1.json").read_text())
    require(set(freeze["files_sha256"]) == {"run_anchors.py", "test_anchors_cpu.py", "runner_cpu_results.json",
            "source_pins.json", "PROTOCOL.md", "launch.json"}, "Six-file freeze inventory")
    sources = {str(results): RAW_SHA, str(HERE/"freeze_v1.json"): FREEZE_SHA}
    for name, digest in freeze["files_sha256"].items():
        path = HERE/name
        require(path.resolve().parent == HERE and sha(path) == digest, "Frozen file differs: "+name)
        sources[str(path)] = digest
    pins = json.loads((HERE/"source_pins.json").read_text())
    require(pins["round10_result_sha256"] == freeze["reference_result_sha256"] == PARENT_SHA,
            "Parent digest differs")
    for name, pin in pins["sources"].items():
        path = R10/pin["relative_to_round10"]
        require(sha(path) == pin["sha256"], "Inherited helper differs: "+name)
        sources[str(path.resolve())] = pin["sha256"]
    cpu = json.loads((HERE/"runner_cpu_results.json").read_text())
    require(cpu["passed"] and cpu["tests_run"] == 4 and not cpu["GPU_initialized"] and not cpu["full_model_run"]
            and not cpu["errors"] and not cpu["failures"] and cpu["source_pins_sha256"] == sha(HERE/"source_pins.json"),
            "Own CPU receipt invalid")
    for path, digest in cpu["source_sha256"].items():
        require(sha(path) == digest, "CPU fixture/source changed")
        sources[str(path)] = digest
    parent_auditor = R10/"audit_services.py"
    require(sha(parent_auditor) == PARENT_AUDITOR_SHA, "Parent pure auditor changed")
    pa = load(parent_auditor, "_anchor_pure_service_auditor")
    g, helpers, ledger, shapes, groups, _ = pa.preflight()
    sources.update({str(p): pin for p, pin in pa.PINS.items()})
    sources[str(parent_auditor)] = PARENT_AUDITOR_SHA
    parent_path = R10/"collect_v1/results.json"
    require(sha(parent_path) == PARENT_SHA, "Parent raw changed")
    sources[str(parent_path)] = PARENT_SHA
    parent = json.loads(parent_path.read_text())
    raw = json.loads(results.read_text())
    require(raw["schema"] == "flow_round10_same_check_native_anchors_v1" and
            raw["status"] == "ROUND10_SAME_CHECK_NATIVE_ANCHORS_COMPLETE" and
            raw["full_forward_count"] == 27 and raw["completed_targets"] == len(raw["frames"]) == 8,
            "Complete 27-forward/eight-member result required")
    require(raw["inference_executed"] and not raw["training_performed"] and not raw["hardware_benefit_claim"]
            and all(raw[k] for k in ("original_model_unchanged", "BN_buffers_unchanged", "original_modules_restored")),
            "Completion/state closure differs")
    binding = raw["binding"]
    for field, path in (("runner_sha256", HERE/"run_anchors.py"), ("runner_checks_sha256", HERE/"runner_cpu_results.json"),
                        ("source_pins_sha256", HERE/"source_pins.json"), ("reference_round10_sha256", parent_path),
                        ("membership_sha256", R10/"membership_v1.json")):
        require(binding[field] == sha(path), "Runtime binding differs: "+field)
    require(binding["full_forward_budget"] == freeze["full_forward_budget"] == 27 and binding["actual_anchor_forwards"] == 24
            and binding["arms"] == list(ARMS) and binding["wall_minutes"] == 10 and not freeze["training"]
            and freeze["design_after_R10_quality"], "Frozen budget/scope")
    require(without_wall(binding["inherited_data_and_numerical_contract"]) == without_wall(parent["binding"]),
            "Inherited data/numerics differ beyond changed wall cap")
    require(binding["checkpoint_sha256"] == parent["binding"]["checkpoint_sha256"] and
            raw["original_state_sha256"] == parent["original_state_sha256"] and
            all(parent["runtime"][k] == v for k, v in raw["runtime"].items()), "Numerical state/runtime differs")
    bn = raw["BN_scope"]
    require({k: v for k, v in bn.items() if k != "execution"} ==
            {k: v for k, v in parent["BN_scope"].items() if k != "execution"}, "BN scope/signatures differ")
    sig = bn["BN_signature_before_inference"]
    private = {n for n in sig if ".mlp." in n}
    require(len(sig) == 78 and len(private) == 24 and len(set(sig)-private) == 54, "24/54 BN count")
    for name, state in sig.items():
        require(not state["training"] and state["tracking"] == (name in private) and
                (state["running_mean"] is not None) == (name in private) and
                (state["running_var"] is not None) == (name in private), "BN current/frozen mode")
    ids = groups["check"]
    require([f["target_id"] for f in raw["frames"]] == binding["target_ids"] == freeze["check_ids"] == ids,
            "All eight original check identities/order required")
    refs = {f["target_id"]: f for f in parent["frames"] if f["group"] == "check"}
    require(list(refs) == ids, "Parent check order")
    scope = {n for n in shapes if ".layers.0." in n or ".layers.2." in n}
    counts = dict(forwards=0, wrapper_records=0, all12_FC_records=0, FC_rows_each_total=0,
                  archived_prefix_input_hashes=0, archived_prefix_output_hashes=0)
    per_arm = {}

    def execution_check(ex, arm, mode="actual"):
        require(set(ex["all12_observed_fc_rows"]) == set(shapes) and
                set(ex["layers"]) == (set() if arm == "native" else scope), "Native/wrapper inventory")
        totals = {k: 0 for k in g.counts(next(iter(shapes.values())), 384, 0)}
        pool = {k: 0 for k in g.added_counts(next(iter(shapes.values())), dict(coarse=0, reps=0))}
        for name, shape in shapes.items():
            points = shape[2]*shape[3]
            if name in ex["layers"]:
                rec = ex["layers"][name]
                fine = arm == "FF" or (arm == "fine_s2b2" and name == pa.TARGET)
                mask = [[fine]*((shape[3]+8)//9) for _ in range((shape[2]+8)//9)]
                require(rec["fine_mask"] == mask and rec["mode"] == mode, "Whole-layer mask/tails/mode")
                g.audit_record(rec, shape, 4*shape[4], ledger)
                geo = g.geometry(shape, mask)
                if mode == "dual_reference":
                    geo = dict(fine=shape[2]*shape[3], coarse=shape[2]*shape[3],
                               reps=g.geometry(shape, [[False]*len(row) for row in mask])["reps"])
                points = geo["fine"]+geo["reps"]
                for key, value in g.added_counts(shape, geo).items():
                    pool[key] += value
                counts["wrapper_records"] += 1
            nominal = g.counts(shape, 4*shape[4], points)
            require(ex["all12_observed_fc_rows"][name] == dict(fc1_rows=nominal["fc_rows_each"], fc2_rows=nominal["fc_rows_each"]),
                    "Independent per-layer FC hooks")
            for key, value in nominal.items():
                totals[key] += value
            counts["all12_FC_records"] += 1
        require(totals["fc_rows_each"] == ex["summed_fc1_rows"] == ex["summed_fc2_rows"] ==
                (609600 if mode == "dual_reference" else ROWS[arm]), "All12 rows")
        counts["forwards"] += 1
        counts["FC_rows_each_total"] += totals["fc_rows_each"]
        key = arm if mode == "actual" else "dual_reference"
        record = dict(all12_nominal=totals, stage02_wrapper_added_work=pool)
        require(key not in per_arm or per_arm[key] == record, "Fixed same-arm service varies")
        per_arm[key] = record

    archives = []
    for frame in raw["frames"]:
        tid = frame["target_id"]
        require(frame["group"] == "post_result_same_check_diagnostic" and frame["sequence"] == refs[tid]["sequence"]
                and frame["input_evidence"] == refs[tid]["input_evidence"] and frame["timestamps"] == refs[tid]["timestamps"]
                and set(frame["arms"]) == set(ARMS), "Frame identities/data/time/arms")
        for arm in ARMS:
            ex = frame["arms"][arm]["execution"]
            execution_check(ex, arm)
            if arm == "fine_s2b2":
                baseline = refs[tid]["arms"]["coarse_stage02"]["execution"]["layers"]
                require(ex["prefix_input_SHA_exact_to_archived_CC"] == list(pa.PREFIX) and
                        ex["earlier_prefix_output_SHA_exact_to_archived_CC"] == list(pa.PREFIX[:-1]), "Prefix declaration")
                for name in pa.PREFIX:
                    require(ex["layers"][name]["input_sha256"] == baseline[name]["input_sha256"], "Archived prefix input")
                    counts["archived_prefix_input_hashes"] += 1
                for name in pa.PREFIX[:-1]:
                    require(ex["layers"][name]["output_sha256"] == baseline[name]["output_sha256"], "Archived prefix output")
                    counts["archived_prefix_output_hashes"] += 1
        helpers.archive_check(results.parent, frame, "prediction_file", archives)
        with zipfile.ZipFile(results.parent/frame["prediction_file"]) as z:
            require(set(z.namelist()) == {a+".npy" for a in ARMS}, "Prediction archive members")
            for arm in ARMS:
                with z.open(arm+".npy") as stream:
                    header = helpers.read_header(stream)
                    require(header["shape"] == (2, 480, 640) and header["descr"] == "<f4" and not header["fortran_order"],
                            "Prediction shape/dtype")
                    digest = hashlib.sha256(); size = 0
                    for part in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(part); size += len(part)
                    require(size == 4*2*480*640 and digest.hexdigest() == frame["arms"][arm]["prediction_sha256"],
                            "Prediction payload hash")
    require(set(raw["anchor_replays"]) == {"CC"}, "Exactly one CC replay")
    cc = raw["anchor_replays"]["CC"]
    oldcc = refs[ids[0]]["arms"]["coarse_stage02"]
    require(cc["target_id"] == ids[0] and cc["exact_archived_prediction"] and cc["all8_input_output_SHA_exact"] and
            cc["prediction_sha256"] == oldcc["prediction_sha256"], "CC archived prediction exact replay")
    execution_check(cc["execution"], "CC")
    for name in scope:
        for key in ("input_sha256", "output_sha256"):
            require(cc["execution"]["layers"][name][key] == oldcc["execution"]["layers"][name][key], "CC all8 hash replay")
    require(set(raw["dual_checks"]) == {"FF", "fine_s2b2"}, "Exactly two duals")
    for arm, pair in raw["dual_checks"].items():
        require(pair["flow_list_length"] == 4 and all(pair[k] for k in
                ("all_flows_exact", "all8_inputs_exact", "all8_branches_exact")), "Complete dual comparison flags")
        require(pair["actual_execution"] == raw["frames"][0]["arms"][arm]["execution"], "Dual actual side duplicates first arm")
        execution_check(pair["dual_execution"], arm, "dual_reference")
        for name in scope:
            for key in ("input_sha256", "output_sha256"):
                require(pair["actual_execution"]["layers"][name][key] == pair["dual_execution"]["layers"][name][key],
                        "Dual recorded input/branch hashes")
    require(counts == dict(forwards=27, wrapper_records=152, all12_FC_records=324, FC_rows_each_total=11614400,
                          archived_prefix_input_hashes=40, archived_prefix_output_hashes=32), "Full service inventory")
    sources[str(Path(__file__).resolve())] = sha(__file__)
    receipt = dict(schema="flow_round10_native_anchors_service_audit_v1", status="PASS", source_sha256=sources,
        counts=counts, members=ids, actual_anchor_forwards=24, CC_replay_forwards=1, dual_reference_forwards=2,
        native_wrappers=0, nonnative_stage02_wrappers=8, per_arm_services=per_arm,
        exact_checks=dict(CC_archived_prediction=True, CC_archived_input_output_pairs=8,
                          dual_full_flow_outputs=8, dual_input_output_hash_pairs=16),
        archives=dict(files=len(archives), bytes=sum(a["bytes"] for a in archives), records=archives,
                      prediction_payloads_hashed=24, prediction_values_interpreted=False, GT_values_loaded=False),
        BN=dict(private_frozen=24, external_current_stat=54, signature_matches_R10=True,
                all_model_state_matches_R10=True, completion_state_buffers_modules_restored=True,
                new_full_BN_input_shape_snapshot=False),
        scope=dict(quality_fields_analyzed=False, GPU_or_model_executed_by_auditor=False, fullnet_cycles_or_PPA_claim=False,
                   R10_failure_unchanged=True, same_member_post_result_diagnostic=True),
        numerical_boundary="Final NPZ payload SHA rechecked; other four-flow exact comparisons are source-bound runner assertions. "
                           "Operation totals are logical arithmetic, not measured PE issue cycles or hardware latency.")
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    return dict(status="PASS", counts=counts, archive_files=len(archives), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"run_v1/results.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.output)))


if __name__ == "__main__":
    main()
