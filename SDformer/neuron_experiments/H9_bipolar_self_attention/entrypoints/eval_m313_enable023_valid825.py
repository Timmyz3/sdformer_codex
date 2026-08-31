#!/usr/bin/env python3
"""Run one fail-closed M313r2 enable023 candidate against M312r2."""

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
WRAPPER = (REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
           "eval_m284_near_match_residual_elision.py")
SELECTIVE_MODULE = (
    REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/near_match_residual_elision_selective.py")
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
CANONICAL_CONTRACT = (
    REPO / "hw_autoresearch_nts07/contracts/"
    "m313r2_enable023_running_bn_valid825_contract_r1_20260825.json")
EXPECTED_CONTRACT_SHA256 = (
    "6eedc33ad6020b9f95d41e761acf5678c1e8b9d60dffb830d6eb8bc49a1844a2")
REQUIRED_OPTIONS = (
    "--contract", "--config", "--checkpoint", "--path-results",
    "--distance-threshold", "--max-samples", "--bn-policy",
    "--dump-per-frame", "--baseline-profile", "--baseline-per-frame",
    "--enabled-operator-indices",
)
FORWARDED_OPTIONS = REQUIRED_OPTIONS[:-1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "M313r2 duplicate JSON key: " + key)
            value[key] = item
        return value

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(
            handle, object_pairs_hook=pairs, parse_float=Decimal,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError("M313r2 non-finite JSON token: " + token)))

    def reject_nonfinite(item):
        if isinstance(item, Decimal):
            require(item.is_finite(), "M313r2 non-finite JSON number")
        elif isinstance(item, dict):
            for nested in item.values():
                reject_nonfinite(nested)
        elif isinstance(item, list):
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(value)
    return value


def finite_decimal(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RuntimeError("M313r2 invalid decimal: " + label)
    require(number.is_finite(), "M313r2 non-finite decimal: " + label)
    return number


def canonical_values():
    tokens = sys.argv[1:]
    require(len(tokens) == 2 * len(REQUIRED_OPTIONS),
            "M313r2 requires exactly eleven --option value pairs")
    values = {}
    for pair_index, expected in enumerate(REQUIRED_OPTIONS):
        option = tokens[2 * pair_index]
        value = tokens[2 * pair_index + 1]
        require(option == expected and "=" not in option,
                "M313r2 non-canonical option/order: " + option)
        require(not value.startswith("--"),
                "M313r2 missing or option-like value for " + option)
        require(option not in values,
                "M313r2 duplicate option: " + option)
        values[option] = value
    return values


def bound_input(contract, name):
    spec = contract["inputs"][name]
    path = (REPO / spec["path"]).resolve()
    require(path.is_file() and sha256(path) == spec["sha256"],
            "M313r2 bound input SHA drift: " + name)
    return path


def main():
    launcher_start = sha256(Path(__file__).resolve())
    values = canonical_values()
    require(int(values["--distance-threshold"]) == 1 and
            int(values["--max-samples"]) == 0 and
            values["--bn-policy"] == "running" and
            values["--enabled-operator-indices"] == "0,2,3",
            "M313 requires tau1, full valid825, running BN and enable023")

    contract_path = Path(values["--contract"]).resolve()
    require(contract_path == CANONICAL_CONTRACT.resolve(),
            "M313r2 contract must use its canonical repository path")
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M313r2 contract is not the launcher-pinned root of trust")
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m284_near_match_modified_forward_contract_v1" and
            contract.get("milestone") == "M313R2_ENABLE023_VALID825",
            "M313r2 contract schema/milestone drift")
    runtime = contract["runtime_identity"]
    require(sha256(WRAPPER) == runtime["wrapper_sha256"] and
            sha256(SELECTIVE_MODULE) == runtime["module_sha256"],
            "M313r2 wrapper/selective-module SHA drift")
    config_path = Path(values["--config"]).resolve()
    checkpoint_path = Path(values["--checkpoint"]).resolve()
    result_dir = Path(values["--path-results"]).resolve()
    per_frame_path = Path(values["--dump-per-frame"]).resolve()
    require(config_path.is_file() and
            sha256(config_path) == runtime["config_sha256"] and
            checkpoint_path.is_file() and
            sha256(checkpoint_path) == runtime["checkpoint_sha256"] and
            EVALUATOR.is_file() and
            sha256(EVALUATOR) == runtime["evaluator_sha256"],
            "M313r2 config/checkpoint/evaluator SHA drift")
    require(not result_dir.exists(),
            "M313r2 refuses a pre-existing result root")
    require(per_frame_path == result_dir / "per_frame.csv",
            "M313r2 per-frame CSV must be result_root/per_frame.csv")

    bound_inputs = {
        name: bound_input(contract, name) for name in contract["inputs"]
    }
    freeze_path = bound_inputs["selection_freeze"]
    freeze = strict_json(freeze_path)
    correction = freeze["selection_rule_correction"]
    require(correction["unique_enabled_operator_indices"] == [0, 2, 3] and
            correction["dropped_operator_index"] == 1 and
            correction["further_s10_combination_search_allowed"] is False,
            "M313 selection-freeze semantic drift")

    baseline_receipt_path = bound_inputs["baseline_receipt"]
    baseline_profile_path = bound_inputs["baseline_profile"]
    baseline_per_frame_path = bound_inputs["baseline_per_frame"]
    baseline_launch_path = bound_inputs["baseline_launch_receipt"]
    baseline_manifest_path = bound_inputs["baseline_manifest"]
    baseline_seal_path = bound_inputs["baseline_manifest_seal"]
    require(Path(values["--baseline-profile"]).resolve() == baseline_profile_path and
            Path(values["--baseline-per-frame"]).resolve() == baseline_per_frame_path,
            "M313 CLI baseline paths differ from frozen contract")
    baseline_root = baseline_receipt_path.parent
    require(baseline_profile_path.parent == baseline_root and
            baseline_per_frame_path == baseline_root / "per_frame.csv" and
            baseline_launch_path.parent == baseline_root and
            baseline_manifest_path.parent == baseline_root and
            baseline_seal_path.parent == baseline_root and
            baseline_root != result_dir,
            "M313r2 baseline artifacts are not one distinct canonical root")
    seal_tokens = baseline_seal_path.read_text(encoding="utf-8").split()
    require(seal_tokens == [sha256(baseline_manifest_path),
                            baseline_manifest_path.name],
            "M313r2 baseline manifest seal mismatch")

    baseline_expected_entries = {
        baseline_receipt_path.name: sha256(baseline_receipt_path),
        baseline_profile_path.name: sha256(baseline_profile_path),
        baseline_per_frame_path.name: sha256(baseline_per_frame_path),
        baseline_launch_path.name: sha256(baseline_launch_path),
    }
    baseline_manifest_entries = {}
    for line in baseline_manifest_path.read_text(
            encoding="utf-8").splitlines():
        tokens = line.split("  ", 1)
        require(len(tokens) == 2 and
                tokens[1] not in baseline_manifest_entries and
                Path(tokens[1]).name == tokens[1] and
                tokens[1] not in (".", ".."),
                "M313r2 malformed baseline manifest entry")
        baseline_manifest_entries[tokens[1]] = tokens[0]
    require(baseline_manifest_entries == baseline_expected_entries and
            all(sha256(baseline_root / name) == digest
                for name, digest in baseline_manifest_entries.items()),
            "M313r2 baseline manifest artifact-map replay failed")

    baseline_receipt = strict_json(baseline_receipt_path)
    baseline_profile = strict_json(baseline_profile_path)
    baseline_launch = strict_json(baseline_launch_path)
    require(baseline_receipt["status"] ==
            "PASS_EXACT_BASELINE_REQUIRES_PAIRED_CANDIDATE" and
            baseline_receipt["scope"] == {
                "samples": 825, "bn_policy": "running",
                "distance_threshold": 0} and
            baseline_receipt["paired_baseline"] is None and
            baseline_receipt["admission"] == {
                "modified_network_forward": True,
                "s10_screen": False,
                "valid825": True,
                "accuracy_promoted": False,
                "system_speedup": False,
                "ppa": False,
                "headline": False,
            },
            "M313 baseline receipt/profile identity drift")
    baseline_identity = baseline_receipt["identity"]
    require(baseline_identity["contract_path"] ==
                str(bound_inputs["m312r2_contract"]) and
            baseline_identity["contract_sha256"] ==
                contract["inputs"]["m312r2_contract"]["sha256"] and
            baseline_identity["wrapper_sha256"] == runtime["wrapper_sha256"] and
            baseline_identity["module_sha256"] ==
                contract["baseline_runtime_identity"]["module_sha256"] and
            baseline_identity["evaluator_sha256"] == runtime["evaluator_sha256"] and
            baseline_identity["config_sha256"] == runtime["config_sha256"] and
            baseline_identity["checkpoint_sha256"] == runtime["checkpoint_sha256"] and
            baseline_identity["catalog_sha256"] ==
                contract["inputs"]["catalog"]["sha256"] and
            baseline_identity["profile_sha256"] == sha256(baseline_profile_path) and
            baseline_identity["per_frame_sha256"] == sha256(baseline_per_frame_path) and
            baseline_identity["docs359_sha256"] == runtime["docs359_sha256"],
            "M313r2 baseline nested receipt identity drift")
    require(baseline_profile["samples"] == 825 and
            baseline_profile["eval_protocol"]["bn_policy"] == "running" and
            baseline_profile["metric_aggregation_audit"]["frame_count"] == 825 and
            baseline_profile["artifact_identity"]["config_sha256"] ==
                runtime["config_sha256"] and
            baseline_profile["artifact_identity"]["checkpoint_sha256"] ==
                runtime["checkpoint_sha256"] and
            baseline_profile["checkpoint_load_audit"]["missing_count"] == 0 and
            baseline_profile["checkpoint_load_audit"]["unexpected_count"] == 0 and
            baseline_profile["checkpoint_load_audit"]["overlay_missing_count"] == 0 and
            baseline_profile["checkpoint_load_audit"]["overlay_unexpected_count"] == 0,
            "M313r2 baseline profile population/protocol/identity drift")
    baseline_aee = finite_decimal(
        baseline_profile["metrics"]["AEE"], "baseline profile AEE")
    require(finite_decimal(baseline_receipt["metrics"]["AEE"],
                           "baseline receipt AEE") == baseline_aee,
            "M313r2 baseline receipt/profile AEE drift")
    require(baseline_launch["schema"] ==
            "m312r2_tau0_valid825_baseline_launch_receipt_v1" and
            baseline_launch["status"] ==
            "PASS_EXACT_TAU0_VALID825_BASELINE_SEALED" and
            baseline_launch["contract_sha256"] ==
                contract["inputs"]["m312r2_contract"]["sha256"] and
            baseline_launch["contract_path"] ==
                str(bound_inputs["m312r2_contract"]) and
            baseline_launch["launcher_sha256"] ==
                contract["inputs"]["m312r2_launcher"]["sha256"] and
            baseline_launch["wrapper_sha256"] == runtime["wrapper_sha256"] and
            baseline_launch["module_sha256"] ==
                contract["baseline_runtime_identity"]["module_sha256"] and
            baseline_launch["evaluator_sha256"] == runtime["evaluator_sha256"] and
            baseline_launch["config_sha256"] == runtime["config_sha256"] and
            baseline_launch["checkpoint_sha256"] == runtime["checkpoint_sha256"] and
            baseline_launch["receipt_sha256"] == sha256(baseline_receipt_path) and
            baseline_launch["receipt_path"] == str(baseline_receipt_path) and
            baseline_launch["profile_sha256"] == sha256(baseline_profile_path) and
            baseline_launch["profile_path"] == str(baseline_profile_path) and
            baseline_launch["per_frame_sha256"] == sha256(baseline_per_frame_path) and
            baseline_launch["per_frame_path"] == str(baseline_per_frame_path) and
            baseline_launch["samples"] == 825 and
            baseline_launch["distance_threshold"] == 0 and
            baseline_launch["bn_policy"] == "running" and
            finite_decimal(baseline_launch["aee"], "baseline launch AEE") ==
                baseline_aee and
            baseline_launch["claim_boundary"] == {
                "exact_baseline": True,
                "lossy_accuracy": False,
                "system_speedup": False,
                "headline": False,
            },
            "M313 baseline launch receipt drift")
    with baseline_per_frame_path.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    require(len(baseline_rows) == 825 and
            len(baseline_receipt["ordered_population"]) == 825,
            "M313 baseline is not ordered valid825")
    baseline_ordered_rows = [
        {"file": row["file"], "sequence": row["sequence"]}
        for row in baseline_rows
    ]
    require(baseline_receipt["ordered_population"] == baseline_ordered_rows,
            "M313r2 baseline receipt/CSV order drift")

    spec = importlib.util.spec_from_file_location("m313_pinned_m284_wrapper",
                                                  str(WRAPPER))
    require(spec is not None and spec.loader is not None,
            "M313 cannot import pinned M284 wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODULE = SELECTIVE_MODULE
    saved_argv = list(sys.argv)
    saved_enabled = os.environ.get("M306_ENABLED_OPERATOR_INDICES")
    sys.argv = [str(WRAPPER)]
    for option in FORWARDED_OPTIONS:
        sys.argv.extend((option, values[option]))
    os.environ["M306_ENABLED_OPERATOR_INDICES"] = "0,2,3"
    try:
        module.main()
    finally:
        sys.argv = saved_argv
        if saved_enabled is None:
            os.environ.pop("M306_ENABLED_OPERATOR_INDICES", None)
        else:
            os.environ["M306_ENABLED_OPERATOR_INDICES"] = saved_enabled

    receipt_path = result_dir / "m284_near_match_receipt_r1.json"
    profile_path = result_dir / "spike_profile.json"
    require(receipt_path.is_file() and profile_path.is_file() and
            per_frame_path.is_file(), "M313 candidate artifacts missing")
    receipt = strict_json(receipt_path)
    profile = strict_json(profile_path)
    require(receipt["scope"] == {
                "samples": 825, "bn_policy": "running",
                "distance_threshold": 1} and
            receipt["paired_baseline"]["sha256"] ==
                sha256(baseline_profile_path) and
            receipt["paired_baseline"]["baseline_per_frame_sha256"] ==
                sha256(baseline_per_frame_path) and
            Path(receipt["paired_baseline"]["path"]).resolve() ==
                baseline_profile_path and
            Path(receipt["paired_baseline"]["baseline_per_frame_path"]).resolve() ==
                baseline_per_frame_path and
            receipt["paired_baseline"]["ordered_population_identical"] is True,
            "M313 paired baseline linkage drift")
    candidate_identity = receipt["identity"]
    require(candidate_identity == {
                "contract_path": str(contract_path),
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "wrapper_sha256": runtime["wrapper_sha256"],
                "module_sha256": runtime["module_sha256"],
                "evaluator_sha256": runtime["evaluator_sha256"],
                "config_sha256": runtime["config_sha256"],
                "checkpoint_sha256": runtime["checkpoint_sha256"],
                "catalog_sha256": contract["inputs"]["catalog"]["sha256"],
                "profile_sha256": sha256(profile_path),
                "per_frame_sha256": sha256(per_frame_path),
                "docs359_sha256": runtime["docs359_sha256"],
            },
            "M313r2 candidate nested receipt identity drift")
    require(profile["samples"] == 825 and
            profile["eval_protocol"]["bn_policy"] == "running" and
            profile["metric_aggregation_audit"]["frame_count"] == 825 and
            profile["artifact_identity"]["config_sha256"] ==
                runtime["config_sha256"] and
            profile["artifact_identity"]["checkpoint_sha256"] ==
                runtime["checkpoint_sha256"] and
            profile["checkpoint_load_audit"]["missing_count"] == 0 and
            profile["checkpoint_load_audit"]["unexpected_count"] == 0 and
            profile["checkpoint_load_audit"]["overlay_missing_count"] == 0 and
            profile["checkpoint_load_audit"]["overlay_unexpected_count"] == 0,
            "M313r2 candidate profile population/protocol/identity drift")
    candidate_aee = finite_decimal(profile["metrics"]["AEE"],
                                   "candidate profile AEE")
    require(finite_decimal(receipt["metrics"]["AEE"],
                           "candidate receipt AEE") == candidate_aee,
            "M313r2 candidate receipt/profile AEE drift")
    independent_delta = candidate_aee - baseline_aee
    independent_gate = independent_delta <= Decimal("0.02")
    recorded_delta = finite_decimal(
        receipt["paired_baseline"]["candidate_minus_baseline_aee"],
        "recorded candidate-baseline AEE")
    require(finite_decimal(receipt["paired_baseline"]["aee"],
                           "paired baseline AEE") == baseline_aee and
            finite_decimal(receipt["paired_baseline"]["candidate_aee"],
                           "paired candidate AEE") == candidate_aee and
            finite_decimal(receipt["paired_baseline"]
                           ["absolute_aee_increase_gate"],
                           "paired AEE gate") == Decimal("0.02") and
            abs(recorded_delta - independent_delta) <= Decimal("1e-12") and
            receipt["paired_baseline"]["accuracy_gate_pass"] is
                independent_gate and
            receipt["status"] == (
                "PASS_MODIFIED_FORWARD_ACCURACY_GATE" if independent_gate
                else "NO_GO_MODIFIED_FORWARD_ACCURACY_GATE") and
            receipt["admission"] == {
                "modified_network_forward": True,
                "s10_screen": False,
                "valid825": True,
                "accuracy_promoted": independent_gate,
                "system_speedup": False,
                "ppa": False,
                "headline": False,
            },
            "M313r2 independent AEE/admission recomputation drift")
    with per_frame_path.open("r", encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidate_ordered_rows = [
        {"file": row["file"], "sequence": row["sequence"]}
        for row in candidate_rows
    ]
    require(len(candidate_rows) == 825 and
            candidate_ordered_rows == baseline_ordered_rows and
            receipt["ordered_population"] == candidate_ordered_rows,
            "M313r2 candidate/baseline ordered population drift")
    runtime_receipt = receipt["runtime"]
    total = int(runtime_receipt["aggregate_snapped_partition_vectors"])
    exact = int(runtime_receipt[
        "aggregate_exact_hit_snapped_partition_vectors"])
    positive = int(runtime_receipt[
        "aggregate_positive_distance_snapped_partition_vectors"])
    require(total == exact + positive and positive > 0,
            "M313 exact/positive snapped-count decomposition drift")
    disabled_name = runtime_receipt["operator_names"][1]
    for key in ("calls", "partition_vectors", "snapped_partition_vectors",
                "exact_hit_snapped_partition_vectors",
                "positive_distance_snapped_partition_vectors"):
        require(int(runtime_receipt[key][disabled_name]) == 0,
                "M313 disabled operator executed: " + key)

    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256 and
            sha256(Path(__file__).resolve()) == launcher_start and
            sha256(WRAPPER) == runtime["wrapper_sha256"] and
            sha256(SELECTIVE_MODULE) == runtime["module_sha256"] and
            sha256(EVALUATOR) == runtime["evaluator_sha256"] and
            sha256(config_path) == runtime["config_sha256"] and
            sha256(checkpoint_path) == runtime["checkpoint_sha256"],
            "M313r2 runtime trust boundary changed before seal")
    for name, input_path in bound_inputs.items():
        require(sha256(input_path) == contract["inputs"][name]["sha256"],
                "M313r2 bound input changed before seal: " + name)

    launch_receipt = {
        "schema": "m313r2_enable023_valid825_launch_receipt_v1",
        "status": receipt["status"],
        "contract_sha256": sha256(contract_path),
        "launcher_sha256": launcher_start,
        "wrapper_sha256": sha256(WRAPPER),
        "module_sha256": sha256(SELECTIVE_MODULE),
        "evaluator_sha256": sha256(EVALUATOR),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "candidate_receipt_sha256": sha256(receipt_path),
        "candidate_profile_sha256": sha256(profile_path),
        "candidate_per_frame_sha256": sha256(per_frame_path),
        "baseline_receipt_sha256": sha256(baseline_receipt_path),
        "baseline_profile_sha256": sha256(baseline_profile_path),
        "baseline_per_frame_sha256": sha256(baseline_per_frame_path),
        "baseline_manifest_seal_sha256": sha256(baseline_seal_path),
        "samples": 825,
        "distance_threshold": 1,
        "enabled_operator_indices": [0, 2, 3],
        "baseline_aee": str(baseline_aee),
        "candidate_aee": str(candidate_aee),
        "candidate_minus_baseline_aee": str(independent_delta),
        "accuracy_gate_pass": independent_gate,
        "aggregate_total_snapped": total,
        "aggregate_exact_hit_snapped": exact,
        "aggregate_positive_distance_snapped": positive,
        "claim_boundary": {
            "accuracy_promoted": independent_gate,
            "hardware_cycles_promoted": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    launch_path = result_dir / "m313r2_enable023_launch_receipt_r1.json"
    launch_path.write_text(json.dumps(
        launch_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = result_dir / "M313R2_CANDIDATE_MANIFEST.sha256"
    entries = [receipt_path, profile_path, per_frame_path, launch_path]
    expected_entries = {path.name: sha256(path) for path in entries}
    manifest_path.write_text("".join(
        "{}  {}\n".format(expected_entries[path.name], path.name)
        for path in entries), encoding="utf-8")
    replayed_entries = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split("  ", 1)
        require(len(tokens) == 2 and tokens[1] not in replayed_entries and
                Path(tokens[1]).name == tokens[1] and
                tokens[1] not in (".", ".."),
                "M313r2 malformed or duplicate manifest entry")
        replayed_entries[tokens[1]] = tokens[0]
    require(replayed_entries == expected_entries and
            all(sha256(result_dir / name) == digest
                for name, digest in replayed_entries.items()),
            "M313r2 candidate manifest artifact-map replay failed")
    seal_path = result_dir / "M313R2_CANDIDATE_MANIFEST.seal.sha256"
    seal_path.write_text("{}  {}\n".format(
        sha256(manifest_path), manifest_path.name), encoding="utf-8")
    seal_tokens = seal_path.read_text(encoding="utf-8").split()
    require(seal_tokens == [sha256(manifest_path), manifest_path.name] and
            replayed_entries == expected_entries and
            all(sha256(result_dir / name) == digest
                for name, digest in replayed_entries.items()),
            "M313r2 candidate manifest/seal final replay failed")
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256 and
            sha256(Path(__file__).resolve()) == launcher_start,
            "M313r2 launcher/contract changed during execution")
    for name, input_path in bound_inputs.items():
        require(sha256(input_path) == contract["inputs"][name]["sha256"],
                "M313r2 bound input changed during sealing: " + name)
    print("M313R2_DONE status={} delta_aee={} gate={} positive={} seal_sha256={}".format(
        receipt["status"],
        independent_delta, independent_gate, positive,
        sha256(seal_path)), flush=True)


if __name__ == "__main__":
    main()
