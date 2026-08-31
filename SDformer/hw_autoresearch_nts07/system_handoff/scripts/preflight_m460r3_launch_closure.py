#!/usr/bin/env python3
"""Verify the sealed M460R3 launch/runtime closure without GPU work.

Python-3.6-compatible by construction.  The caller must authenticate the
detached launch outer seal before invoking this script; this script then checks
the frozen contract, Git tree, critical runtime leaves and all 30 S10 inputs.
"""

import argparse
import csv
import hashlib
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


def resolve_identity(repo, path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith(("neuron_experiments/", "third_party/")):
        return repo / path
    return repo / "hw_autoresearch_nts07" / path


def verify_identity(repo, contract):
    verified = []
    for name, record in sorted(contract["identity"].items()):
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_identity(repo, record["path"])
        require(path.is_file(), "missing contract identity {}: {}".format(
            name, path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "contract identity SHA drift {} expected={} observed={}".format(
                    name, record["sha256"], actual))
        verified.append({"name": name, "path": str(path), "sha256": actual})
    return verified


def verify_git(repo, dependency):
    tracked = dependency["tracked_python_closure"]
    head = check_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    tree = check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"])
    require(head == tracked["git_commit"], "remote Git HEAD drift")
    require(tree == tracked["git_tree"], "remote Git tree drift")
    roots = [
        "third_party/SDformerFlow",
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
        "profile_nts11_hardware_p0.py",
        "neuron_experiments/H9_bipolar_self_attention/overlay",
        "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
        "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
        "hw_autoresearch_nts07/system_simulator/scripts/"
        "trace_m40_bottleneck_packed_sources.py",
    ]
    dirty = subprocess.call(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"] + roots)
    require(dirty == 0, "tracked runtime worktree differs from frozen HEAD")
    staged = subprocess.call(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet", "HEAD", "--"] + roots)
    require(staged == 0, "staged runtime worktree differs from frozen HEAD")
    return {"head": head, "tree": tree, "tracked_roots_clean": True}


def verify_critical(repo, dependency):
    rows = []
    for record in dependency["critical_files"]:
        path = repo / record["path"]
        require(path.is_file(), "missing critical runtime file: " + str(path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "critical runtime SHA drift: " + record["path"])
        rows.append({"path": record["path"], "sha256": actual})
    return rows


def verify_dataset(repo, dependency, mode):
    closure = dependency["dataset_closure"]
    manifest_path = repo / closure["source_manifest_path"]
    require(sha256(manifest_path) == closure["source_manifest_sha256"],
            "M40 dataset source manifest SHA drift")
    manifest = strict_json(manifest_path)
    records = manifest["identity"]["dataset_input_files"]
    require(len(records) == closure["files"] == 30,
            "dataset input population drift")
    if mode == "remote":
        dataset_root = Path(dependency["remote"]["dataset_root"])
    else:
        dataset_root = repo / "data/Datasets/DSEC/saved_flow_data"
    verified = []
    for record in records:
        path = dataset_root / record["relative_path"]
        require(path.is_file(), "missing S10 dataset input: " + str(path))
        require(path.stat().st_size == int(record["bytes"]),
                "S10 dataset byte-count drift: " + str(path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "S10 dataset SHA drift: " + str(path))
        verified.append({
            "relative_path": record["relative_path"],
            "bytes": int(record["bytes"]),
            "sha256": actual,
        })
    return {"root": str(dataset_root), "files": len(verified),
            "records": verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--outer-seal", required=True, type=Path)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    require(repo.is_dir(), "M460R3 repository absent")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m460r3_h67_g8_ffn_token_residual_s10_capture_contract_v1",
            "M460R3 preflight contract schema drift")
    dependency_path = resolve_identity(
        repo, contract["identity"]["runtime_dependency_closure"]["path"])
    dependency = strict_json(dependency_path)
    require(dependency.get("schema") == "m460r3_runtime_dependency_closure_v1",
            "M460R3 runtime dependency schema drift")

    remote = contract["remote_execution"]
    require(remote["host"] == "ssh.sd5ai.scnet.cn" and
            int(remote["port"]) == 10037 and remote["user"] == "root",
            "M460R3 remote SSH identity drift")
    require(remote["repo"] ==
            "/root/private_data/work/sdformer_codex/SDformer" and
            remote["python"] == "/opt/conda/envs/sdformerflow/bin/python",
            "M460R3 remote repo/python drift")
    if args.mode == "remote":
        require(str(repo) == remote["repo"], "remote repo argv drift")
        require(os.path.realpath(sys.executable) ==
                os.path.realpath(remote["python"]),
                "remote Python executable drift")
    require(sys.version_info[:2] >= (3, 6), "M460R3 requires Python >=3.6")

    launch_manifest = args.launch_manifest.resolve()
    outer_seal = args.outer_seal.resolve()
    require(launch_manifest.is_file() and outer_seal.is_file(),
            "M460R3 detached launch seal files absent")
    require(str(launch_manifest) == str(
        repo / remote["launch_manifest_relative_path"]),
        "launch manifest argv/path drift")
    require(str(outer_seal) == str(
        repo / remote["launch_outer_seal_relative_path"]),
        "launch outer seal argv/path drift")

    identity = verify_identity(repo, contract)
    git_receipt = verify_git(repo, dependency)
    critical = verify_critical(repo, dependency)
    dataset = verify_dataset(repo, dependency, args.mode)
    workload_path = resolve_identity(
        repo, contract["identity"]["sample_workload"]["path"])
    with workload_path.open("r", newline="", encoding="utf-8") as handle:
        workload = list(csv.DictReader(handle))
    require(len(workload) == 10 and
            [int(row["sample_id"]) for row in workload] == list(range(10)) and
            all(row["sequence_key"] == "zurich_city_09_a" for row in workload),
            "M460R3 frozen workload identity drift")

    result = {
        "schema": "m460r3_remote_dependency_preflight_v1",
        "status": "PASS_M460R3_REMOTE_CLOSURE_PREFLIGHT" if
                  args.mode == "remote" else
                  "PASS_M460R3_LOCAL_CLOSURE_STATIC_TEST",
        "mode": args.mode,
        "repo": str(repo),
        "python": {"executable": sys.executable,
                   "version": list(sys.version_info[:3]),
                   "python36_compatible_entry": True},
        "contract_identity_files": len(identity),
        "critical_runtime_files": len(critical),
        "git": git_receipt,
        "dataset_files": dataset["files"],
        "workload_samples": len(workload),
        "launch_manifest": str(launch_manifest),
        "launch_outer_seal": str(outer_seal),
        "gpu_touched": False,
        "remote_contacted_by_script": False,
        "training": False,
        "capture_launched": False,
        "system_speedup": False,
        "headline": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
