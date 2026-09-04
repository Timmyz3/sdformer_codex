#!/usr/bin/env python3
"""Fail-closed launcher that produces and seals the M312r2 tau0 baseline."""

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
WRAPPER = (REPO / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
           "eval_m284_near_match_residual_elision.py")
EVALUATOR = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
BASE_MODULE = (
    REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/near_match_residual_elision.py")
EXPECTED_CONTRACT_SHA256 = (
    "682766544d6cf879805452d15d30b9fa617171136b8eaad45dcf6ac8acf3b325")
REQUIRED_OPTIONS = (
    "--contract", "--config", "--checkpoint", "--path-results",
    "--distance-threshold", "--max-samples", "--bn-policy",
    "--dump-per-frame",
)


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
            require(key not in value, "M312r2 duplicate JSON key: " + key)
            value[key] = item
        return value

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(
            handle, object_pairs_hook=pairs, parse_float=Decimal,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError("M312r2 non-finite JSON token: " + token)))

    def reject_nonfinite(item):
        if isinstance(item, Decimal):
            require(item.is_finite(), "M312r2 non-finite JSON number")
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
        raise RuntimeError("M312r2 invalid decimal: " + label)
    require(number.is_finite(), "M312r2 non-finite decimal: " + label)
    return number


def canonical_values():
    """Accept only the exact, ordered ``--option value`` launch grammar."""
    tokens = sys.argv[1:]
    require(len(tokens) == 2 * len(REQUIRED_OPTIONS),
            "M312r2 requires exactly eight --option value pairs")
    values = {}
    for pair_index, expected in enumerate(REQUIRED_OPTIONS):
        option = tokens[2 * pair_index]
        value = tokens[2 * pair_index + 1]
        require(option == expected and "=" not in option,
                "M312r2 non-canonical option/order: " + option)
        require(not value.startswith("--"),
                "M312r2 missing or option-like value for " + option)
        require(option not in values,
                "M312r2 duplicate option: " + option)
        values[option] = value
    return values


def main():
    launcher_start = sha256(Path(__file__).resolve())
    values = canonical_values()
    require(int(values["--distance-threshold"]) == 0 and
            int(values["--max-samples"]) == 0 and
            values["--bn-policy"] == "running",
            "M312 requires tau0, full valid825, and running BN")

    contract_path = Path(values["--contract"]).resolve()
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256,
            "M312r2 contract is not the launcher-pinned root of trust")
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m284_near_match_modified_forward_contract_v1" and
            contract.get("milestone") == "M312R2_TAU0_VALID825_BASELINE",
            "M312r2 contract schema/milestone drift")
    runtime = contract["runtime_identity"]
    require(sha256(WRAPPER) == runtime["wrapper_sha256"] and
            sha256(BASE_MODULE) == runtime["module_sha256"],
            "M312r2 wrapper/base-module SHA drift")
    bound_inputs = {}
    for name, item in contract["inputs"].items():
        input_path = (REPO / item["path"]).resolve()
        require(input_path.is_file() and sha256(input_path) == item["sha256"],
                "M312r2 bound input SHA drift: " + name)
        bound_inputs[name] = (input_path, item["sha256"])
    config_path = Path(values["--config"]).resolve()
    checkpoint_path = Path(values["--checkpoint"]).resolve()
    require(config_path.is_file() and
            sha256(config_path) == runtime["config_sha256"],
            "M312r2 config SHA drift")
    require(checkpoint_path.is_file() and
            sha256(checkpoint_path) == runtime["checkpoint_sha256"],
            "M312r2 checkpoint SHA drift")
    require(EVALUATOR.is_file() and
            sha256(EVALUATOR) == runtime["evaluator_sha256"],
            "M312r2 evaluator SHA drift")
    result_dir = Path(values["--path-results"]).resolve()
    per_frame_path = Path(values["--dump-per-frame"]).resolve()
    require(not result_dir.exists(),
            "M312r2 refuses a pre-existing result root")
    require(per_frame_path == result_dir / "per_frame.csv",
            "M312r2 per-frame CSV must be result_root/per_frame.csv")

    spec = importlib.util.spec_from_file_location("m312_pinned_m284_wrapper",
                                                  str(WRAPPER))
    require(spec is not None and spec.loader is not None,
            "M312 cannot import pinned M284 wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved_argv = list(sys.argv)
    sys.argv = [str(WRAPPER)]
    for option in REQUIRED_OPTIONS:
        sys.argv.extend((option, values[option]))
    try:
        module.main()
    finally:
        sys.argv = saved_argv

    receipt_path = result_dir / "m284_near_match_receipt_r1.json"
    profile_path = result_dir / "spike_profile.json"
    require(receipt_path.is_file() and profile_path.is_file() and
            per_frame_path.is_file(), "M312 baseline artifacts missing")
    receipt = strict_json(receipt_path)
    profile = strict_json(profile_path)
    require(receipt["status"] ==
            "PASS_EXACT_BASELINE_REQUIRES_PAIRED_CANDIDATE" and
            receipt["scope"] == {
                "samples": 825, "bn_policy": "running",
                "distance_threshold": 0} and
            receipt["paired_baseline"] is None and
            receipt["admission"] == {
                "modified_network_forward": True,
                "s10_screen": False,
                "valid825": True,
                "accuracy_promoted": False,
                "system_speedup": False,
                "ppa": False,
                "headline": False,
            },
            "M312 exact baseline receipt semantic drift")
    identity = receipt["identity"]
    require(identity == {
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
            "M312r2 nested receipt identity drift")
    runtime_receipt = receipt["runtime"]
    require(runtime_receipt["aggregate_partition_vectors"] == 0 and
            runtime_receipt["aggregate_snapped_partition_vectors"] == 0 and
            runtime_receipt["aggregate_exact_hit_snapped_partition_vectors"] == 0 and
            runtime_receipt["aggregate_positive_distance_snapped_partition_vectors"] == 0,
            "M312 tau0 baseline unexpectedly executed snapping path")
    with per_frame_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 825 and
            len(receipt["ordered_population"]) == 825,
            "M312 baseline ordered population is not valid825")
    ordered_rows = [
        {"file": row["file"], "sequence": row["sequence"]}
        for row in rows
    ]
    require(receipt["ordered_population"] == ordered_rows,
            "M312r2 receipt/CSV ordered population drift")
    require(receipt["identity"]["per_frame_sha256"] == sha256(per_frame_path) and
            receipt["identity"]["profile_sha256"] == sha256(profile_path),
            "M312 baseline nested artifact SHA drift")
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
            "M312r2 profile population/protocol/identity drift")
    baseline_aee = finite_decimal(profile["metrics"]["AEE"], "profile AEE")
    require(finite_decimal(receipt["metrics"]["AEE"], "receipt AEE") ==
            baseline_aee,
            "M312r2 receipt/profile AEE drift")

    # Rehash the complete trust boundary immediately before committing a
    # receipt.  The second replay below detects concurrent output replacement.
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256 and
            sha256(Path(__file__).resolve()) == launcher_start and
            sha256(WRAPPER) == runtime["wrapper_sha256"] and
            sha256(BASE_MODULE) == runtime["module_sha256"] and
            sha256(EVALUATOR) == runtime["evaluator_sha256"] and
            sha256(config_path) == runtime["config_sha256"] and
            sha256(checkpoint_path) == runtime["checkpoint_sha256"],
            "M312r2 runtime trust boundary changed before seal")
    for name, (input_path, expected_sha) in bound_inputs.items():
        require(sha256(input_path) == expected_sha,
                "M312r2 bound input changed before seal: " + name)

    launch_receipt = {
        "schema": "m312r2_tau0_valid825_baseline_launch_receipt_v1",
        "status": "PASS_EXACT_TAU0_VALID825_BASELINE_SEALED",
        "contract_path": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "launcher_sha256": launcher_start,
        "wrapper_sha256": sha256(WRAPPER),
        "module_sha256": sha256(BASE_MODULE),
        "evaluator_sha256": sha256(EVALUATOR),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "profile_path": str(profile_path),
        "profile_sha256": sha256(profile_path),
        "per_frame_path": str(per_frame_path),
        "per_frame_sha256": sha256(per_frame_path),
        "samples": 825,
        "distance_threshold": 0,
        "bn_policy": "running",
        "aee": str(baseline_aee),
        "claim_boundary": {
            "exact_baseline": True,
            "lossy_accuracy": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    launch_receipt_path = result_dir / "m312r2_tau0_baseline_launch_receipt_r1.json"
    launch_receipt_path.write_text(
        json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest_path = result_dir / "M312R2_BASELINE_MANIFEST.sha256"
    entries = [receipt_path, profile_path, per_frame_path, launch_receipt_path]
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
                "M312r2 malformed or duplicate manifest entry")
        replayed_entries[tokens[1]] = tokens[0]
    require(replayed_entries == expected_entries and
            all(sha256(result_dir / name) == digest
                for name, digest in replayed_entries.items()),
            "M312r2 manifest artifact-map replay failed")
    seal_path = result_dir / "M312R2_BASELINE_MANIFEST.seal.sha256"
    seal_path.write_text("{}  {}\n".format(
        sha256(manifest_path), manifest_path.name), encoding="utf-8")
    seal_tokens = seal_path.read_text(encoding="utf-8").split()
    require(seal_tokens == [sha256(manifest_path), manifest_path.name] and
            replayed_entries == expected_entries and
            all(sha256(result_dir / name) == digest
                for name, digest in replayed_entries.items()),
            "M312r2 manifest/seal final replay failed")
    require(sha256(contract_path) == EXPECTED_CONTRACT_SHA256 and
            sha256(Path(__file__).resolve()) == launcher_start,
            "M312r2 launcher/contract changed during execution")
    for name, (input_path, expected_sha) in bound_inputs.items():
        require(sha256(input_path) == expected_sha,
                "M312r2 bound input changed during sealing: " + name)
    print("M312R2_DONE status=PASS_EXACT_TAU0_VALID825_BASELINE_SEALED "
          "samples=825 aee={} receipt_sha256={} seal_sha256={}".format(
              receipt["metrics"]["AEE"], sha256(receipt_path),
              sha256(seal_path)), flush=True)


if __name__ == "__main__":
    main()
