#!/usr/bin/env python3
"""Read-only ROUND7 execution audit; no quality fields or model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R6 = HERE.parent / "damage_predictor_round6"
GEOMETRY = HERE.parent / "structured_merge/audit_structured_merge_service.py"
PINS = {
    HERE / "run_v1/results.json": "0410c6f05047146b432e186fcde863d8c1476495ee113ffc58cfba45a4bee1d3",
    HERE / "freeze_v1.json": "7baa7950745902638eba6320d493be87dddf85f90f79d26bad9ea6b1aff75272",
    GEOMETRY: "abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10",
    R6 / "complete_collection_service_audit.json": "0abeae09c3b20f520c04529bfb89d4470473ef00129ce617e6f84dbbd9f92da2",
    R6 / "matched_fine_service_audit.json": "b023a206e4ffb9fd4ab6d0ee63ea9179f27ed1d7a7396bd6ad4b51e34f614903",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for path, digest in PINS.items():
        require(sha(path) == digest, "Pinned input changed: " + str(path))
    freeze = json.loads((HERE / "freeze_v1.json").read_text())
    for path, digest in freeze["files"].items():
        require(sha(ROOT / path) == digest, "Frozen file changed")
    cpu = json.loads((HERE / "runner_cpu_results.json").read_text())
    require(cpu["passed"] and cpu["tests_run"] == 3 and not cpu["GPU_initialized"]
            and not cpu["full_model_run"] and not cpu["errors"] and not cpu["failures"], "CPU checks")
    for path, digest in cpu["source_sha256"].items():
        require(sha(path) == digest, "CPU source changed")
    spec = importlib.util.spec_from_file_location("_round7_service_geometry", GEOMETRY)
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    require(sha(g.MODULE) == g.MODULE_SHA, "Wrapper source")
    constants, ledger = g.runner_constants(), g.extracted_source_ledger()
    shapes = {"sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s,b): list(shape)
              for s,shape in constants["STAGE_SHAPES"].items() for b in range(constants["DEPTHS"][s])}
    scope = {name for name in shapes if ".layers.0." in name or ".layers.2." in name}
    raw = json.loads((HERE / "run_v1/results.json").read_text())
    coarse = json.loads((R6 / "collect_v1/results.json").read_text())
    fine = json.loads((R6 / "matched_fine_check_v1/results.json").read_text())
    refs = {"coarse_stage02": {f["target_id"]: f for f in coarse["frames"] if f["group"] == "check"},
            "fine_stage02": {f["target_id"]: f for f in fine["frames"]}}
    targets = json.loads((R6 / "selection_manifest_v1.json").read_text())["groups"]["check"]
    require(raw["status"] == "ROUND7_MIXED_SCOPE_DIAGNOSTIC_COMPLETE" and
            raw["full_forward_count"] == 20 and raw["completed_targets"] == len(raw["frames"]) == 8, "Completion")
    require(raw["inference_executed"] and not raw["training_performed"] and not raw["hardware_benefit_claim"] and
            all(raw[k] for k in ("original_model_unchanged", "BN_buffers_unchanged", "original_modules_restored")), "Model restoration")
    binding = raw["binding"]
    require(binding["runner_sha256"] == sha(HERE / "run_repair_scope.py") and
            binding["runner_checks_sha256"] == sha(HERE / "runner_cpu_results.json") and
            binding["collector_source_sha256"] == sha(R6 / "collect_singletons.py") and
            binding["native_forward_source_sha256"] == sha(HERE.parent / "structured_merge/scope_extension_v1/run_scope_extension.py"), "Source binding")
    require(binding["reference_coarse_sha256"] == sha(R6 / "collect_v1/results.json") and
            binding["reference_fine_sha256"] == sha(R6 / "matched_fine_check_v1/results.json"), "Anchor binding")
    require(binding["arms"] == [["C0_F2", [0]], ["F0_C2", [2]]] and
            binding["full_forward_budget"] == freeze["full_forward_budget"] == 20 and
            binding["target_ids"] == targets == [f["target_id"] for f in raw["frames"]], "Fixed arms/budget/membership")
    for reference in (coarse, fine):
        require(binding["checkpoint_sha256"] == reference["binding"]["checkpoint_sha256"] and
                raw["runtime"] == reference["runtime"] and raw["original_state_sha256"] == reference["original_state_sha256"] and
                raw["BN_scope"]["BN_signature_before_inference"] == reference["BN_scope"]["BN_signature_before_inference"], "Reference state/runtime/BN")
    bn = raw["BN_scope"]; signatures = bn["BN_signature_before_inference"]
    frozen = {name for name in signatures if ".mlp." in name}; external = set(signatures) - frozen
    require(len(frozen) == 24 and len(external) == 54 and bn["scope"] == "MLP_CMA176" and
            bn["BN_count"] == 78 and bn["MLP_BN_count"] == 24 and set(bn["frozen_BN_names"]) == frozen and
            set(bn["current_stat_BN_names"]) == external and bn["non_bn_state_unchanged"], "BN scope")
    for name, signature in signatures.items():
        require(not signature["training"] and signature["tracking"] == (name in frozen) and
                (signature["running_mean"] is not None) == (name in frozen) and
                (signature["running_var"] is not None) == (name in frozen), "BN mode")
    counts = dict(forwards=0, wrapper_records=0, all12_FC_hooks=0, FC_rows_each_total=0)

    def audit_execution(execution, coarse_stages, mode):
        require(set(execution["layers"]) == scope and set(execution["all12_observed_fc_rows"]) == set(shapes), "Layer inventory")
        total = 0; arithmetic = {}; added = {}
        for name, shape in shapes.items():
            rows = 10 * shape[2] * shape[3]
            if name in scope:
                stage = 0 if ".layers.0." in name else 2
                mask = [[stage not in coarse_stages] * ((shape[3] + 8)//9) for _ in range((shape[2] + 8)//9)]
                record = execution["layers"][name]
                require(record["fine_mask"] == mask and record["mode"] == mode, "Routing/mode/tails")
                g.audit_record(record, shape, 4*shape[4], ledger)
                rows = record["executed_fc_rows_per_layer"]
                geo = g.geometry(shape, mask)
                services = g.counts(shape, 4*shape[4], geo["fine"] + geo["reps"])
                work = g.added_counts(shape, geo)
                for key, value in services.items(): arithmetic[key] = arithmetic.get(key,0) + value
                for key, value in work.items(): added[key] = added.get(key,0) + value
                counts["wrapper_records"] += 1
            require(execution["all12_observed_fc_rows"][name] == dict(fc1_rows=rows,fc2_rows=rows), "Independent FC hooks")
            total += rows; counts["all12_FC_hooks"] += 1
        require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == total, "Executed total rows")
        counts["forwards"] += 1; counts["FC_rows_each_total"] += total
        return dict(all12_FC_rows_each=total, selected_scope_services=arithmetic,
                    selected_scope_added=added, note="Selected-path arithmetic; dual executed row total includes both producers")

    arms = {"C0_F2": (0,), "F0_C2": (2,)}
    expected_rows = {"C0_F2": 217200, "F0_C2": 494400}
    arm_services = {}; archives = []; prefix_checks = 0
    first0 = "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp"
    first2 = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"
    for frame in raw["frames"]:
        tid = frame["target_id"]
        require(frame["group"] == "post_result_development" and set(frame["arms"]) == set(arms), "Frame grouping")
        for reference in refs.values():
            for key in ("input_evidence", "timestamps", "sequence"):
                require(frame[key] == reference[tid][key], "Reference input binding")
        for arm, coarse_stages in arms.items():
            a = frame["arms"][arm]; result = audit_execution(a["execution"], coarse_stages, "actual")
            require(result["all12_FC_rows_each"] == expected_rows[arm], "Mixed row expectation")
            if arm in arm_services: require(arm_services[arm] == result, "Member-varying service")
            else: arm_services[arm] = result
            ref_arm = "coarse_stage02" if arm == "C0_F2" else "fine_stage02"
            require(a["same_prefix_input_matches"] == ref_arm, "Prefix label")
            for name in (first0, first2):
                require(a["execution"]["layers"][name]["input_sha256"] ==
                        refs[ref_arm][tid]["arms"][ref_arm]["execution"]["layers"][name]["input_sha256"], "Same-prefix input hash")
                prefix_checks += 1
        path = HERE / "run_v1" / frame["prediction_file"]
        require(path.resolve().parent == (HERE / "run_v1").resolve() and sha(path) == frame["prediction_file_sha256"], "Archive integrity")
        archives.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    require(set(raw["anchor_replays"]) == {"coarse_stage02", "fine_stage02"} and set(raw["dual_checks"]) == set(arms), "Extra forward inventory")
    for arm, stages in (("coarse_stage02", (0,2)), ("fine_stage02", ())):
        replay = raw["anchor_replays"][arm]
        require(replay["target_id"] == targets[0] and replay["exact_archived_prediction"] and
                replay["prediction_sha256"] == refs[arm][targets[0]]["arms"][arm]["prediction_sha256"], "Anchor prediction hash replay")
        audit_execution(replay["execution"], stages, "actual")
    for arm, stages in arms.items():
        pair = raw["dual_checks"][arm]
        require(pair["flow_exact"] and pair["all8_branches_exact"] and
                pair["actual_execution"] == raw["frames"][0]["arms"][arm]["execution"], "Actual/dual guard")
        require(audit_execution(pair["dual_execution"], stages, "dual_reference")["all12_FC_rows_each"] == 609600, "Dual both-producer rows")
    require(counts == dict(forwards=20,wrapper_records=160,all12_FC_hooks=240,FC_rows_each_total=7623600), "Overall execution inventory")
    receipt = dict(schema="flow_round7_execution_service_audit_v1",status="PASS",
        source_sha256={str(p):sha(p) for p in tuple(PINS) + tuple(ROOT / p for p in freeze["files"]) + (Path(__file__),)},
        checked=counts, members=targets, mixed_actual_forwards=16,anchor_replays=2,dual_forwards=2,
        same_prefix_hash_checks=prefix_checks, actual_services_by_arm=arm_services,
        BN=dict(private_frozen=24,external_current_stat=54,signatures_and_original_state_exact_to_R6=True,
                buffers_unchanged=True,modules_restored=True,new_full_BN_input_shape_hooks=False),
        archive_integrity=dict(files=8,bytes=sum(x["bytes"] for x in archives),records=archives,prediction_or_GT_arrays_loaded=False),
        scope=dict(quality_fields_analyzed=False,model_or_GPU_run_by_auditor=False,cycle_or_net_PE_claim=False,
                   new_validation_population=False,post_result_diagnostic=True),
        boundary="Fullstage repair endpoints have different service budgets. These guards prove completed declared execution, not quality, model generalization, dynamic allocation novelty, or same-graph PE savings.")
    out = HERE / "service_audit.json"; out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="PASS",checked=counts,prefix_hash_checks=prefix_checks,arms=arm_services,output=out.name)))


if __name__ == "__main__":
    main()
