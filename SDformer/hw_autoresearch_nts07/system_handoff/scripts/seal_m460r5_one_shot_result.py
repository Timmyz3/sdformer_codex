#!/usr/bin/env python3
"""Validate and double-seal one M460R5 post-compute oracle result."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def verify_manifest(directory, filename):
    directory = Path(directory).resolve()
    path = directory / filename
    verified = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, relative = line.split("  ", 1)
            leaf = directory / relative
            require(leaf.is_file() and not leaf.is_symlink(),
                    "sealed leaf absent/symlink: " + str(leaf))
            require(sha256(leaf) == expected,
                    "sealed leaf SHA drift: " + str(leaf))
            verified.append(relative)
    return verified


def read_idle(path, minimum_rows):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) >= minimum_rows, "idle receipt population incomplete")
    require(all(int(row["gpu_contexts"]) == 0 and
                int(row["ml_processes"]) == 0 for row in rows),
            "GPU/ML activity present in idle receipt")
    return rows


def validate_capture_payload(capture_dir):
    capture_dir = Path(capture_dir).resolve()
    summary_path = capture_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    summary = strict_json(summary_path)
    require(summary.get("schema") ==
            "m460r5_h67_g8_one_shot_capture_v1",
            "M460R5 capture summary schema drift")
    require(summary.get("status") ==
            "PASS_M460R5_H67_EP35_NO_RUNNING_S10_ONE_SHOT_POSTCOMPUTE_ORACLE",
            "M460R5 capture summary status drift")
    audit = summary["identity"]["checkpoint_load_audit"]
    require(int(audit["missing_count"]) == 0 and
            int(audit["unexpected_count"]) == 0,
            "M460R5 checkpoint load audit mismatch")
    require(summary["identity"]["capture_bn_policy"] ==
            "no_running/current-batch", "M460R5 BN policy drift")
    population = summary["population"]
    require(int(population["samples"]) == 10 and
            population["sequence_keys"] == ["zurich_city_09_a"] and
            int(population["ffn_modules"]) == 12 and
            int(population["sample_module_records"]) == 120 and
            int(population["tokens"]) ==
            int(population["expected_tokens"]) == 5580000,
            "M460R5 population drift")
    require(summary["strict_runtime_state_machine"][
        "sn2_fc2_sn1_attack_accepted"] is False,
        "M460R5 strict hook order drift")
    require(summary["semantics"]["full_tensor_dumped"] is False,
            "M460R5 full tensor dump forbidden")
    admission = summary["admission"]
    for name in ("executable_skip", "delta_aee", "valid825_accuracy",
                 "cycle_speedup", "energy", "ppa", "system_speedup",
                 "headline", "training"):
        require(admission[name] is False,
                "M460R5 forbidden admission true: " + name)
    require(admission["checkpoint_bound_s10_postcompute_oracle"] is True and
            admission["postcompute_opportunity_counts"] is True,
            "M460R5 post-compute oracle admission absent")
    npz = sorted(capture_dir.glob("*.npz"))
    require(len(npz) == 120, "M460R5 reduction NPZ count drift")
    records = strict_json(capture_dir / "per_sample_module_manifest.json")
    require(len(records["records"]) == 120,
            "M460R5 per-sample/module record population drift")
    inner_leaves = verify_manifest(capture_dir, "manifest.sha256")
    outer_leaves = verify_manifest(
        capture_dir, "manifest.sha256.outer.seal.sha256")
    require(len(inner_leaves) == 123 and outer_leaves == ["manifest.sha256"],
            "M460R5 capture payload seal population drift")
    return summary, summary_path


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists(), "refusing author receipt overwrite")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--launch-outer-seal", required=True, type=Path)
    parser.add_argument("--consumed-marker", required=True, type=Path)
    args = parser.parse_args()

    stage = args.stage_root.resolve()
    capture = stage / "capture_payload"
    preflight_dir = stage / "fresh_preflight"
    summary, summary_path = validate_capture_payload(capture)
    inventory_path = preflight_dir / "package_build_inventory.json"
    preflight_path = preflight_dir / "preflight_receipt.json"
    preflight = strict_json(preflight_path)
    inventory = strict_json(inventory_path)
    require(preflight["status"] ==
            "PASS_M460R5_FRESH_EXACT_PREFLIGHT_BEFORE_ONE_SHOT" and
            preflight["cuda_initialized"] is False and
            preflight["capture_launched"] is False,
            "M460R5 fresh preflight receipt drift")
    require(inventory["status"] ==
            "PASS_M460R4_EXACT_PACKAGE_BUILD_AND_IMPORT_INVENTORY" and
            inventory["cuda_initialized"] is False,
            "M460R5 fresh inventory drift")
    pre_idle = read_idle(preflight_dir / "pre_capture_idle_receipt.csv", 4)
    post_idle = read_idle(stage / "post_capture_idle_receipt.csv", 1)
    marker = args.consumed_marker.resolve()
    marker_copy = stage / "one_shot_consumed.marker"
    require(marker.is_file() and marker_copy.is_file() and
            sha256(marker) == sha256(marker_copy),
            "M460R5 one-shot consumed marker drift")

    contract = strict_json(args.contract)
    fields = contract["post_capture_receipt_fields"]
    require(fields == [
        "launch_outer_seal_sha256", "capture_summary_sha256",
        "capture_inner_manifest_sha256",
        "capture_outer_seal_file_sha256"],
        "M460R5 receipt field names drift")
    author = {
        "schema": "m460r5_one_shot_capture_author_receipt_v1",
        "status": "PASS_M460R5_ONE_SHOT_S10_POSTCOMPUTE_ORACLE_CAPTURE",
        "contract_sha256": sha256(args.contract),
        "launch_outer_seal_sha256": sha256(args.launch_outer_seal),
        "capture_summary_sha256": sha256(summary_path),
        "capture_inner_manifest_sha256": sha256(
            capture / "manifest.sha256"),
        "capture_outer_seal_file_sha256": sha256(
            capture / "manifest.sha256.outer.seal.sha256"),
        "fresh_inventory_sha256": sha256(inventory_path),
        "fresh_preflight_receipt_sha256": sha256(preflight_path),
        "pre_capture_idle_receipt_sha256": sha256(
            preflight_dir / "pre_capture_idle_receipt.csv"),
        "post_capture_idle_receipt_sha256": sha256(
            stage / "post_capture_idle_receipt.csv"),
        "one_shot_consumed_marker_sha256": sha256(marker_copy),
        "population": summary["population"],
        "checkpoint_load_audit": {
            "missing_count": 0, "unexpected_count": 0},
        "strict_hook_order": True,
        "reduction_npz": 120,
        "full_tensor_dumped": False,
        "pre_capture_idle_snapshots": len(pre_idle),
        "post_capture_idle_snapshots": len(post_idle),
        "one_shot_attempts_consumed": 1,
        "postcompute_oracle_only": True,
        "executable_skip": False,
        "delta_aee": False,
        "cycle_speedup": False,
        "energy": False,
        "ppa": False,
        "system_speedup": False,
        "headline": False,
        "training": False,
        "claim_boundary": (
            "One frozen S10 reduction-only post-compute opportunity/oracle "
            "capture. No executable skip, Delta-AEE, accuracy, cycle, "
            "energy, PPA, system speedup or headline is admitted."),
    }
    author_path = stage / "m460r5_one_shot_capture_author_receipt.json"
    write_new_json(author_path, author)

    inner = stage / "manifest.sha256"
    outer = stage / "manifest.sha256.outer.seal.sha256"
    require(not inner.exists() and not outer.exists(),
            "M460R5 top-level seal already exists")
    leaves = sorted(path for path in stage.rglob("*")
                    if path.is_file() and path not in (inner, outer))
    require(all(not path.is_symlink() for path in leaves),
            "M460R5 result symlink forbidden")
    inner.write_text("".join(
        "{}  {}\n".format(sha256(path), str(path.relative_to(stage)))
        for path in leaves), encoding="utf-8")
    outer.write_text("{}  {}\n".format(sha256(inner), inner.name),
                     encoding="utf-8")
    verify_manifest(stage, outer.name)
    verify_manifest(stage, inner.name)
    print(json.dumps({
        "status": author["status"],
        "author_receipt": str(author_path),
        "top_manifest_sha256": sha256(inner),
        "top_outer_seal_file_sha256": sha256(outer),
        "one_shot_attempts_consumed": 1,
        "postcompute_oracle_only": True,
        "system_speedup": False,
    }, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
