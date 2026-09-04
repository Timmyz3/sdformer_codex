#!/usr/bin/env python3
"""Fresh fail-closed M460R5 preflight immediately before one-shot capture."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


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


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import file " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifest(manifest_path, root):
    manifest_path = Path(manifest_path).resolve()
    root = Path(root).resolve()
    verified = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, name = line.split("  ", 1)
            path = root / name
            require(path.is_file(), "sealed manifest leaf absent: " + str(path))
            actual = sha256(path)
            require(actual == expected,
                    "sealed manifest leaf SHA drift: " + str(path))
            verified.append(name)
    return verified


def verify_prior_authorization_roots(code_repo, contract):
    code_repo = Path(code_repo).resolve()
    roots = contract["review_authorization"]
    r4_launch_outer = code_repo / roots["r4_launch_outer_seal_path"]
    r4_result_outer = code_repo / roots["r4_remote_result_outer_seal_path"]
    review_outer = code_repo / roots["review_outer_seal_path"]
    require(sha256(r4_launch_outer) == roots[
        "r4_launch_outer_seal_file_sha256"], "R4 launch outer root drift")
    require(sha256(r4_result_outer) == roots[
        "r4_remote_result_outer_seal_file_sha256"],
        "R4 remote-result outer root drift")
    require(sha256(review_outer) == roots[
        "outer_seal_file_sha256"], "R4 review outer root drift")

    r4_launch_manifest = code_repo / roots["r4_launch_manifest_path"]
    r4_result_manifest = code_repo / roots["r4_remote_result_manifest_path"]
    review_manifest = code_repo / roots["review_manifest_path"]
    require(sha256(r4_launch_manifest) == roots["r4_launch_manifest_sha256"],
            "R4 launch manifest drift")
    require(sha256(r4_result_manifest) == roots[
        "r4_remote_result_manifest_sha256"], "R4 result manifest drift")
    require(sha256(review_manifest) == roots["review_manifest_sha256"],
            "R4 review manifest drift")
    launch_leaves = verify_manifest(r4_launch_manifest, code_repo)
    result_leaves = verify_manifest(r4_result_manifest,
                                    r4_result_manifest.parent)
    review_leaves = verify_manifest(review_manifest, review_manifest.parent)
    return {
        "r4_launch_outer_seal_file_sha256": sha256(r4_launch_outer),
        "r4_remote_result_outer_seal_file_sha256": sha256(r4_result_outer),
        "review_outer_seal_file_sha256": sha256(review_outer),
        "launch_leaves": len(launch_leaves),
        "result_leaves": len(result_leaves),
        "review_leaves": len(review_leaves),
    }


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists(), "refusing R5 preflight receipt overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--code-repo", required=True, type=Path)
    parser.add_argument("--git-worktree-root", required=True, type=Path)
    parser.add_argument("--immutable-data-repo", required=True, type=Path)
    parser.add_argument("--immutable-data-git-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--launch-outer-seal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m460r5_h67_g8_one_shot_capture_contract_v1",
            "M460R5 preflight contract schema drift")
    require(contract["authorization"]["one_shot_g8_s10_capture"] is True and
            int(contract["authorization"]["maximum_capture_attempts"]) == 1,
            "M460R5 one-shot authorization absent")
    freeze = strict_json(args.freeze)
    require(freeze.get("schema") ==
            "m460r4_remote_package_build_environment_freeze_v1",
            "M460R5 environment freeze drift")
    remote = contract["execution_roots"]["remote"]
    require(str(args.code_repo.resolve()) == remote["code_repo"] and
            str(args.git_worktree_root.resolve()) == remote[
                "git_worktree_root"] and
            str(args.immutable_data_repo.resolve()) == remote[
                "immutable_data_repo"] and
            str(args.immutable_data_git_root.resolve()) == remote[
                "immutable_data_git_root"],
            "M460R5 remote root argv drift")
    require(sys.flags.isolated == 1 and
            os.environ.get("PYTHONNOUSERSITE") == "1" and
            "PYTHONPATH" not in os.environ,
            "M460R5 isolated Python environment drift")

    r4_preflight = load_file_module(
        Path(args.code_repo) / "hw_autoresearch_nts07/system_handoff/scripts/"
        "preflight_m460r4_code_data_environment.py",
        "m460r5_frozen_r4_preflight")
    inventory_builder = load_file_module(
        Path(args.code_repo) / "hw_autoresearch_nts07/system_handoff/scripts/"
        "build_m460r4_package_inventory.py",
        "m460r5_inventory_revalidator")

    identities = r4_preflight.verify_contract_identities(
        contract, args.code_repo, args.immutable_data_repo)
    git = r4_preflight.git_identity(
        args.git_worktree_root, freeze["code_identity"]["git_commit"],
        freeze["code_identity"]["git_tree"])
    clean_untracked = r4_preflight.untracked_files(args.git_worktree_root)
    immutable_untracked = r4_preflight.untracked_files(
        args.immutable_data_git_root)
    shadow = r4_preflight.verify_no_shadow(
        args.git_worktree_root, args.immutable_data_git_root, freeze)
    data = r4_preflight.verify_data_roots(
        args.code_repo, args.immutable_data_repo, freeze)
    workload_samples = r4_preflight.verify_workload(
        args.immutable_data_repo, freeze)
    authorization_roots = verify_prior_authorization_roots(
        args.code_repo, contract)

    sealed_inventory = strict_json(args.inventory)
    inventory_builder.validate_inventory(sealed_inventory, freeze)
    live_inventory = inventory_builder.collect_inventory(args.code_repo, freeze)
    require(live_inventory == sealed_inventory,
            "M460R5 live environment differs from fresh sealed inventory")
    original_root = remote["immutable_data_repo"] + os.sep
    require(all(original_root not in item
                for item in live_inventory["final_sys_path"]),
            "immutable dirty tree leaked into M460R5 sys.path")

    result = {
        "schema": "m460r5_fresh_one_shot_preflight_receipt_v1",
        "status": "PASS_M460R5_FRESH_EXACT_PREFLIGHT_BEFORE_ONE_SHOT",
        "contract_sha256": sha256(args.contract),
        "launch_manifest_sha256": sha256(args.launch_manifest),
        "launch_outer_seal_file_sha256": sha256(args.launch_outer_seal),
        "fresh_inventory_sha256": sha256(args.inventory),
        "authorization_roots": authorization_roots,
        "contract_identity_files": len(identities),
        "git": git,
        "shadow_scan": shadow,
        "accepted_untracked_paths": {
            "clean_worktree": clean_untracked,
            "immutable_data_worktree": immutable_untracked,
        },
        "immutable_data": data,
        "workload_samples": workload_samples,
        "environment_revalidated_live": True,
        "python_isolated": True,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": None,
        "original_tree_imported_or_executed": False,
        "cuda_initialized": False,
        "model_constructed": False,
        "checkpoint_deserialized": False,
        "capture_launched": False,
        "training": False,
        "maximum_capture_attempts": 1,
        "claim_boundary": (
            "Fresh identity/import/data preflight only. nvidia-smi telemetry "
            "and byte hashing are allowed; no CUDA context, model, checkpoint "
            "deserialization, capture, training or performance claim here."),
    }
    write_new_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "accepted_clean_untracked": len(clean_untracked),
        "accepted_immutable_untracked": len(immutable_untracked),
        "cuda_initialized": False,
        "capture_launched": False,
    }, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
