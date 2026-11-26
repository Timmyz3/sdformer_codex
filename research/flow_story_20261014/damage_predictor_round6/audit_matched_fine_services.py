#!/usr/bin/env python3
"""Execution-only audit of the separate ROUND6 matched all-fine check8.

Stdlib only: source/membership/state/BN/service metadata and opaque NPZ hashes.
No prediction, GT, quality summaries, model execution, or timing inference.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "structured_merge/audit_structured_merge_service.py"
RAW = HERE / "matched_fine_check_v1/results.json"
PINS = {
    RAW: "19f01c573ddd5a8e3383d379a262987cbc17423bad80dbdc594f12bb9f9de9f9",
    HERE / "matched_fine_freeze_v1.json": "4a0453b87db03740707992831fb4f3e26c996d1ba72f02c7de8f0dc6b4e55344",
    HERE / "native_check_v1/results.json": "48af99c1984d089ef0618789dc38898da0dbf4314320cce5fc18cce5114c4540",
    HERE / "complete_collection_service_audit.json": "0abeae09c3b20f520c04529bfb89d4470473ef00129ce617e6f84dbbd9f92da2",
    GEOMETRY: "abbcf5d93e0062cbb2ca128fde184a51e91cbcd9aa5947cc6efb2880bc704d10",
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
    for path, pin in PINS.items():
        require(sha(path) == pin, "Pinned input changed: " + str(path))
    freeze = json.loads((HERE / "matched_fine_freeze_v1.json").read_text())
    for name, pin in freeze["source_sha256"].items():
        require(sha(HERE / name) == pin, "Runner freeze changed: " + name)
    cpu = json.loads((HERE / "matched_fine_cpu_results.json").read_text())
    require(cpu["passed"] and cpu["tests_run"] == 3 and not cpu["GPU_initialized"]
            and not cpu["full_model_run"] and not cpu["errors"] and not cpu["failures"], "CPU receipt")
    for path, pin in cpu["source_sha256"].items():
        require(sha(path) == pin, "CPU source receipt changed")
    spec = importlib.util.spec_from_file_location("_matched_pure_geometry", GEOMETRY)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    require(sha(geometry.MODULE) == geometry.MODULE_SHA, "Frozen wrapper source")
    constants, ledger = geometry.runner_constants(), geometry.extracted_source_ledger()
    shapes = {
        "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s, b): list(shape)
        for s, shape in constants["STAGE_SHAPES"].items()
        for b in range(constants["DEPTHS"][s])
    }
    scoped = {name for name in shapes if ".layers.0." in name or ".layers.2." in name}
    require(len(shapes) == 12 and len(scoped) == 8, "Native/scope layer inventory")
    raw = json.loads(RAW.read_text())
    native = json.loads((HERE / "native_check_v1/results.json").read_text())
    manifest = json.loads((HERE / "selection_manifest_v1.json").read_text())
    targets = manifest["groups"]["check"]
    require(len(targets) == len(set(targets)) == 8, "Unique check8")
    require(raw["status"] == "ROUND6_MATCHED_FINE_CHECK8_COMPLETE" and
            raw["completed_targets"] == raw["full_forward_count"] == len(raw["frames"]) == 8, "Completion")
    require(raw["inference_executed"] and not raw["training_performed"] and
            not raw["hardware_benefit_claim"] and all(raw[k] for k in
            ("original_model_unchanged", "BN_buffers_unchanged", "original_modules_restored")), "State restoration")
    binding = raw["binding"]
    require(binding["runner_sha256"] == sha(HERE / "collect_matched_fine_check.py") and
            binding["runner_checks_sha256"] == sha(HERE / "matched_fine_cpu_results.json") and
            binding["collector_source_sha256"] == sha(HERE / "collect_singletons.py") and
            binding["native_forward_source_sha256"] == sha(HERE.parent / "structured_merge/scope_extension_v1/run_scope_extension.py"), "Runtime source binding")
    require(binding["checkpoint_sha256"] == native["binding"]["checkpoint_sha256"] ==
            "d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30", "Checkpoint")
    require(binding["full_forward_budget"] == freeze["full_forward_budget"] == 8 and
            binding["arms"] == ["fine_stage02"] and binding["target_ids"] == targets ==
            freeze["fixed_check_ids"] == native["binding"]["target_ids"] ==
            [f["target_id"] for f in raw["frames"]], "Fixed check8/arm/budget")
    require(raw["runtime"] == native["runtime"] and raw["original_state_sha256"] ==
            native["original_state_sha256"], "Native runtime/state")
    signatures = raw["BN_scope"]["BN_signature_before_inference"]
    require(signatures == native["BN_scope"]["BN_signature_before_inference"], "Native BN signatures")
    frozen = {name for name in signatures if ".mlp." in name}
    external = set(signatures) - frozen
    require(len(frozen) == 24 and len(external) == 54, "BN partition")
    scope = raw["BN_scope"]
    require(scope["scope"] == "MLP_CMA176" and scope["BN_count"] == 78 and
            scope["MLP_BN_count"] == 24 and set(scope["frozen_BN_names"]) == frozen and
            set(scope["current_stat_BN_names"]) == external and scope["non_bn_state_unchanged"], "BN scope")
    for name, signature in signatures.items():
        require(not signature["training"] and signature["tracking"] == (name in frozen) and
                (signature["running_mean"] is not None) == (name in frozen) and
                (signature["running_var"] is not None) == (name in frozen), "BN mode/buffers")
    native_frames = {f["target_id"]: f for f in native["frames"]}
    wrapper_count = hook_count = 0
    archives, frame_records = [], []
    scope_counts = {}
    for name in sorted(scoped):
        shape = shapes[name]
        count = geometry.counts(shape, 4 * shape[4], shape[2] * shape[3])
        for key, value in count.items():
            scope_counts[key] = scope_counts.get(key, 0) + value
    require(scope_counts["fc_rows_each"] == 456000, "Scope full rows")
    for frame in raw["frames"]:
        identity = frame["target_id"]
        native_frame = native_frames[identity]
        require(frame["group"] == "check" and set(frame["arms"]) == {"fine_stage02"}, "Frame arm")
        for key in ("sequence", "timestamps", "input_evidence"):
            require(frame[key] == native_frame[key], "Native input receipt: " + key)
        execution = frame["arms"]["fine_stage02"]["execution"]
        require(set(execution["layers"]) == scoped and set(execution["all12_observed_fc_rows"]) == set(shapes), "Layer inventory")
        total_rows = 0
        for name, shape in shapes.items():
            rows = 10 * shape[2] * shape[3]
            if name in scoped:
                record = execution["layers"][name]
                mask = [[True] * ((shape[3] + 8) // 9) for _ in range((shape[2] + 8) // 9)]
                require(record["mode"] == "actual" and record["fine_mask"] == mask, "All-fine including real tails")
                geometry.audit_record(record, shape, 4 * shape[4], ledger)
                require(record["observed"]["coarse"]["native_mlp_calls"] == 0 and all(
                    record[key] == 0 for key in ("executed_pool_input_elements", "executed_pool_reduction_additions_logical",
                                                "executed_pool_divisions_logical", "executed_repeat_write_elements")), "No coarse/pool/repeat work")
                wrapper_count += 1
            require(execution["all12_observed_fc_rows"][name] == dict(fc1_rows=rows, fc2_rows=rows), "Full native FC hook")
            hook_count += 1
            total_rows += rows
        require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == total_rows == 558000, "All12 full rows")
        path = RAW.parent / frame["prediction_file"]
        require(path.resolve().parent == RAW.parent.resolve(), "Archive path escapes run")
        digest = sha(path)
        require(digest == frame["prediction_file_sha256"], "Opaque archive SHA")
        archives.append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size))
        frame_records.append(dict(target_id=identity, wrappers=8, full_FC_rows_each=558000,
                                  stage02_FC_rows_each=456000, coarse_calls=0, pool_and_repeat_elements=0))
    receipt = dict(schema="flow_round6_matched_fine_execution_audit_v1", status="PASS",
        source_sha256={str(p): sha(p) for p in tuple(PINS) + tuple(HERE / name for name in freeze["source_sha256"]) +
                       (HERE / "collect_singletons.py", geometry.MODULE, Path(__file__))},
        execution=dict(status=raw["status"], frames=8, full_forwards=8, wrapper_records_checked=wrapper_count,
                       all12_FC_hook_records_checked=hook_count, each_FC_total_rows=558000,
                       stage02_dense_services_per_frame=scope_counts, coarse_native_calls=0,
                       logical_pool_or_repeat_work=0, per_frame=frame_records),
        BN=dict(private_frozen=24, external_current_stat=54, signatures_exact_to_native=True,
                checkpoint_and_native_state_exact=True, buffers_unchanged=True,
                BN_input_shapes_newly_observed_in_this_run=False),
        archive_integrity=dict(files=len(archives), bytes=sum(x["bytes"] for x in archives), records=archives,
                               prediction_or_GT_arrays_loaded=False),
        audit_scope=dict(quality_fields_analyzed=False, model_or_GPU_executed_by_auditor=False,
                         numerical_native_tiled_equivalence_claim=False, cycle_or_net_hardware_gain_claim=False),
        metadata_errata=[dict(field="BN_scope.execution", issue="Inherited prose incorrectly says no tile wrappers",
                              execution_evidence="Eight stage02 actual all-fine wrappers per frame, independently audited"),
                         dict(field="binding.scope", issue="Inherited prose says not numerically equivalent to tiled arithmetic although this arm is tiled fine",
                              valid_boundary="Native-versus-tiled numerical equivalence is not claimed by this service audit")],
        boundary="Separate result-triggered eight-forward diagnostic; not appended to original 366 collection or eight native anchor forwards. All-fine tiled arithmetic is not asserted equal to native arithmetic. Services are nominal arithmetic and observed tensor execution, not PE cycles. Earlier receipts unchanged.")
    output = HERE / "matched_fine_service_audit.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="PASS", wrappers=wrapper_count, all12_FC_hooks=hook_count,
                          archives=len(archives), stage02=scope_counts, output=output.name)))


if __name__ == "__main__":
    main()
