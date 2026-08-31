#!/usr/bin/env python3
"""Fail-closed M460R4 code/data/environment preflight without capture."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
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


def check_output(argv):
    return subprocess.check_output(argv, universal_newlines=True).strip()


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import file " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_identity(record, code_repo, immutable_data_repo):
    root = record.get("root")
    path = Path(record["path"])
    if root == "code_repo":
        return Path(code_repo).resolve() / path
    if root == "code_hw":
        return Path(code_repo).resolve() / "hw_autoresearch_nts07" / path
    if root == "immutable_data_repo":
        return Path(immutable_data_repo).resolve() / path
    if root == "absolute":
        require(path.is_absolute(), "absolute identity is relative")
        return path
    raise RuntimeError("unknown identity root: " + str(root))


def verify_contract_identities(contract, code_repo, immutable_data_repo):
    records = []
    for name, record in sorted(contract["identity"].items()):
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_identity(record, code_repo, immutable_data_repo).resolve()
        require(path.is_file(), "missing contract identity {}: {}".format(
            name, path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "contract identity SHA drift: " + name)
        records.append({"name": name, "path": str(path), "sha256": actual})
    return records


def git_identity(worktree, expected_commit, expected_tree):
    worktree = Path(worktree).resolve()
    head = check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    tree = check_output([
        "git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"])
    require(head == expected_commit, "clean code worktree HEAD drift")
    require(tree == expected_tree, "clean code worktree tree drift")
    roots = [
        "SDformer/third_party/SDformerFlow",
        "SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/"
        "profile_nts11_hardware_p0.py",
        "SDformer/neuron_experiments/H9_bipolar_self_attention/overlay",
        "SDformer/neuron_experiments/H9_bipolar_self_attention/configs/generated/"
        "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
        "SDformer/hw_autoresearch_nts07/system_simulator/scripts/"
        "trace_m40_bottleneck_packed_sources.py",
    ]
    require(subprocess.call(
        ["git", "-C", str(worktree), "diff", "--quiet", "HEAD", "--"] +
        roots) == 0, "clean code tracked runtime roots differ from HEAD")
    require(subprocess.call(
        ["git", "-C", str(worktree), "diff", "--cached", "--quiet",
         "HEAD", "--"] + roots) == 0,
        "clean code staged runtime roots differ from HEAD")
    return {"head": head, "tree": tree, "tracked_runtime_roots_clean": True}


def untracked_files(worktree):
    text = check_output([
        "git", "-C", str(Path(worktree).resolve()), "ls-files", "--others",
        "--exclude-standard", "--", "SDformer"])
    return [] if not text else text.splitlines()


def shadow_candidates(paths, freeze):
    code_runtime = set()
    for record in freeze["runtime_imports"]:
        if record["root"] == "code_repo" and record["origin"] is not None:
            code_runtime.add("SDformer/" + record["origin"])
    code_runtime.update([
        "SDformer/third_party/SDformerFlow/DSEC_dataloader/__init__.py",
        "SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/__init__.py",
        "SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/"
        "models/STSwinNet_SNN/__init__.py",
    ])
    top_names = set([
        "torch", "torchvision", "numpy", "yaml", "spikingjelly",
        "timm", "einops", "configs", "DSEC_dataloader", "models", "utils",
    ])
    rejected = []
    for item in paths:
        normalized = item.replace("\\", "/")
        if normalized in code_runtime:
            rejected.append(normalized)
            continue
        prefix = "SDformer/"
        if not normalized.startswith(prefix):
            continue
        tail = normalized[len(prefix):]
        first = tail.split("/", 1)[0]
        stem = first[:-3] if first.endswith(".py") else first
        if stem in top_names:
            rejected.append(normalized)
    return sorted(set(rejected))


def verify_no_shadow(clean_worktree, immutable_git_root, freeze):
    clean_untracked = untracked_files(clean_worktree)
    immutable_untracked = untracked_files(immutable_git_root)
    clean_rejected = shadow_candidates(clean_untracked, freeze)
    immutable_rejected = shadow_candidates(immutable_untracked, freeze)
    require(not clean_rejected,
            "clean tree untracked import shadow: " + ",".join(clean_rejected))
    require(not immutable_rejected,
            "immutable-data tree untracked import shadow: " +
            ",".join(immutable_rejected))
    return {
        "clean_untracked_total": len(clean_untracked),
        "immutable_untracked_total": len(immutable_untracked),
        "clean_rejected": clean_rejected,
        "immutable_rejected": immutable_rejected,
        "policy": "exact imported modules plus top-level import names",
    }


def verify_data_roots(code_repo, immutable_data_repo, freeze):
    code_repo = Path(code_repo).resolve()
    immutable_data_repo = Path(immutable_data_repo).resolve()
    require(code_repo != immutable_data_repo,
            "code repo and immutable data repo must differ")
    require(not Path(code_repo).is_symlink() and
            not Path(immutable_data_repo).is_symlink(),
            "R4 root symlink forbidden")
    require(not str(code_repo).startswith(str(immutable_data_repo) + os.sep),
            "clean code repo nested inside immutable data repo")

    assets = []
    for record in freeze["immutable_data_identity"]["assets"]:
        path = immutable_data_repo / record["path"]
        require(path.is_file() and not path.is_symlink(),
                "immutable regular asset absent/symlink: " + str(path))
        if "bytes" in record:
            require(path.stat().st_size == int(record["bytes"]),
                    "immutable asset byte drift: " + str(path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "immutable asset SHA drift: " + str(path))
        assets.append({"role": record["role"], "path": str(path),
                       "sha256": actual})

    manifest_record = next(item for item in freeze[
        "immutable_data_identity"]["assets"]
        if item["role"] == "m40_dataset_manifest")
    manifest = strict_json(immutable_data_repo / manifest_record["path"])
    dataset_records = manifest["identity"]["dataset_input_files"]
    require(len(dataset_records) ==
            freeze["immutable_data_identity"]["dataset_files"] == 30,
            "S10 dataset population drift")
    dataset_root = (immutable_data_repo /
                    freeze["immutable_data_identity"]["dataset_root"])
    verified = []
    for record in dataset_records:
        path = dataset_root / record["relative_path"]
        require(path.is_file() and not path.is_symlink(),
                "S10 input absent/symlink: " + str(path))
        require(path.stat().st_size == int(record["bytes"]),
                "S10 input byte drift: " + str(path))
        require(sha256(path) == record["sha256"],
                "S10 input SHA drift: " + str(path))
        verified.append(record["relative_path"])
    return {"asset_files": assets, "dataset_root": str(dataset_root),
            "dataset_files": len(verified), "symlinks_used": False,
            "large_files_copied_by_m460r4": False}


def verify_workload(immutable_data_repo, freeze):
    record = next(item for item in freeze["immutable_data_identity"]["assets"]
                  if item["role"] == "sample_workload")
    path = Path(immutable_data_repo).resolve() / record["path"]
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10 and
            [int(row["sample_id"]) for row in rows] == list(range(10)) and
            all(row["sequence_key"] == "zurich_city_09_a" for row in rows),
            "frozen S10 workload identity/order drift")
    return len(rows)


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists(), "refusing preflight receipt overwrite")
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
            "m460r4_h67_g8_environment_preflight_contract_v1",
            "R4 contract schema drift")
    freeze = strict_json(args.freeze)
    require(freeze.get("schema") ==
            "m460r4_remote_package_build_environment_freeze_v1",
            "R4 environment freeze schema drift")
    remote = freeze["remote"]
    require(str(args.code_repo.resolve()) == remote["code_repo"],
            "remote clean code repo argv drift")
    require(str(args.git_worktree_root.resolve()) == remote[
        "git_worktree_root"], "remote Git worktree argv drift")
    require(str(args.immutable_data_repo.resolve()) == remote[
        "immutable_data_repo"], "immutable data repo argv drift")
    require(str(args.immutable_data_git_root.resolve()) == remote[
        "immutable_data_git_root"], "immutable data Git root argv drift")
    require(sys.flags.isolated == 1 and
            os.environ.get("PYTHONNOUSERSITE") == "1" and
            "PYTHONPATH" not in os.environ,
            "R4 preflight Python isolation drift")

    identities = verify_contract_identities(
        contract, args.code_repo, args.immutable_data_repo)
    git = git_identity(args.git_worktree_root,
                       freeze["code_identity"]["git_commit"],
                       freeze["code_identity"]["git_tree"])
    shadow = verify_no_shadow(args.git_worktree_root,
                              args.immutable_data_git_root, freeze)
    data = verify_data_roots(args.code_repo, args.immutable_data_repo, freeze)
    workload_samples = verify_workload(args.immutable_data_repo, freeze)

    inventory_builder = load_file_module(
        Path(args.code_repo) / "hw_autoresearch_nts07/system_handoff/scripts/"
        "build_m460r4_package_inventory.py", "m460r4_inventory_revalidator")
    sealed_inventory = strict_json(args.inventory)
    inventory_builder.validate_inventory(sealed_inventory, freeze)
    live_inventory = inventory_builder.collect_inventory(args.code_repo, freeze)
    require(live_inventory == sealed_inventory,
            "live environment differs from sealed package inventory")
    original_root = remote["immutable_data_repo"] + os.sep
    require(all(original_root not in item
                for item in sealed_inventory["final_sys_path"]),
            "immutable data repo leaked into runtime sys.path")

    result = {
        "schema": "m460r4_remote_code_data_environment_preflight_receipt_v1",
        "status": "PASS_M460R4_SEALED_REMOTE_PREFLIGHT_NO_CAPTURE",
        "contract_sha256": sha256(args.contract),
        "launch_outer_seal_sha256": sha256(args.launch_outer_seal),
        "launch_manifest_sha256": sha256(args.launch_manifest),
        "package_build_inventory_sha256": sha256(args.inventory),
        "contract_identity_files": len(identities),
        "git": git,
        "shadow_scan": shadow,
        "immutable_data": data,
        "workload_samples": workload_samples,
        "environment_revalidated_live": True,
        "python_isolated": True,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": None,
        "original_tree_imported": False,
        "cuda_initialized": False,
        "gpu_capture": False,
        "model_constructed": False,
        "checkpoint_read_by_model": False,
        "training": False,
        "cycle_speedup": False,
        "system_speedup": False,
        "headline": False,
        "decision_ceiling": "GO_INDEPENDENT_HAMMER_OF_REMOTE_PREFLIGHT_ONLY",
    }
    write_new_json(args.output, result)
    print(json.dumps({"status": result["status"],
                      "output": str(args.output),
                      "cuda_initialized": False,
                      "gpu_capture": False}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
