#!/usr/bin/env python3
"""Independently rehash and decode the M511 exact S10 bitpack payload.

This verifier authorizes only the exact activity payload.  It does not model
cycles, PGPR/TDR, speedup, energy, PPA, RTL, or full-network performance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import uuid

import numpy as np


POPCOUNT8 = np.asarray([bin(value).count("1") for value in range(256)],
                       dtype=np.uint8)
EXPECTED_CONTRACT_SHA256 = \
    "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e"
EXPECTED_RUNNER_SHA256 = \
    "fddf6a0fc06685fa87f94248c6f48776e59142e0111db3aee2cab38691b7f2d6"
EXPECTED_WRAPPER_SHA256 = \
    "feaeb6247aaf10644bfe7088049f7ab9471dc2d54d928c3fe42210e74265269e"
EXPECTED_PYTHON_SHA256 = \
    "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
EXPECTED_HOSTNAME_TOOL_SHA256 = \
    "c1f8c2c26baa42a5896989353aa7330cd41693435b5fe08386a8b7aa998629dc"
EXPECTED_NVIDIA_SMI_TOOL_SHA256 = \
    "6b8be04c92bf327401faa99d6c7aa7da351b0d4aca8531b422efe2e58b456886"
EXPECTED_INPUT_NAMES = {
    "capture_script", "profile", "config", "checkpoint", "snn_models",
    "spiking_stswinnet", "spiking_modules", "bsa_attention", "atlif_init",
    "atlif_impl", "utils", "m51_manifest", "m510_result",
    "m510_result_manifest", "m510_result_seal",
    "m510_result_review_manifest", "m510_result_review_seal",
    "m512_review", "m512_review_manifest", "m512_review_seal", "docs359",
}
EXPECTED_HOST_GPU_IDENTITY = {
    "hostname": "ic.ismd-nemo",
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "gpu_uuid": "GPU-2b9bf62c-21f9-6c5e-8ace-ee867d88a037",
    "gpu_driver": "575.64",
    "gpu_memory_total_mib": "24576",
}


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
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and
            path.parts and path.parts[0] not in ("", "."),
            "unsafe sealed member: " + name)
    return path


def verify_seal(directory):
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            seal.is_file() and outer.is_file() and
            not seal.is_symlink() and not outer.is_symlink(),
            "missing/symlinked capture seal")
    sealed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        member = safe_member(name)
        require(name not in sealed and name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            "duplicate/recursive sealed member: " + name)
        path = directory.joinpath(*member.parts)
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected,
                "sealed payload mismatch: " + name)
        sealed[name] = expected
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(seal) == expected,
            "capture outer seal mismatch")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    require(actual == set(sealed), "capture sealed/actual file-set mismatch")
    return {
        "manifest_sha256": sealed.get("manifest.json"),
        "sha256sums_sha256": sha256(seal),
        "seal_file_sha256": sha256(outer),
        "sealed_member_count": len(sealed),
        "sealed_names": sorted(sealed),
    }


def write_seal(directory):
    members = sorted(
        path.relative_to(directory) for path in directory.rglob("*")
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256"))
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / member), member.as_posix())
        for member in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(seal)), encoding="utf-8")


def key_values(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        require(key not in result, "duplicate receipt key: " + key)
        result[key] = value
    return result


def verify_runner_attempt(attempt_dir, hw_root, capture_dir):
    canonical = (hw_root / "results" /
        ".m511_h67_ep35_convtranspose_binary_input_capture_r1_attempt_consumed").resolve()
    require(attempt_dir == canonical and attempt_dir.is_dir() and
            not attempt_dir.is_symlink(),
            "missing/noncanonical runner attempt receipt")
    expected_attempt_files = {
        "SHA256SUMS", "SHA256SUMS.seal.sha256", "POSTCAPTURE_PASS.txt",
        "initial/ATTEMPT_CONSUMED.txt", "initial/identity.sha256",
        "initial/resource_preflight.log", "initial/SHA256SUMS",
        "initial/SHA256SUMS.seal.sha256",
    }
    actual_attempt_files = {
        path.relative_to(attempt_dir).as_posix()
        for path in attempt_dir.rglob("*") if path.is_file()
    }
    actual_attempt_dirs = {
        path.relative_to(attempt_dir).as_posix()
        for path in attempt_dir.rglob("*") if path.is_dir()
    }
    require(actual_attempt_files == expected_attempt_files and
            actual_attempt_dirs == {"initial"} and
            all(not path.is_symlink() for path in attempt_dir.rglob("*")),
            "runner attempt exact file/symlink population drift")
    initial = attempt_dir / "initial"
    initial_identity = verify_seal(initial)
    require(set(initial_identity["sealed_names"]) == {
        "ATTEMPT_CONSUMED.txt", "identity.sha256", "resource_preflight.log"
    }, "runner initial receipt member-set drift")

    top_seal = attempt_dir / "SHA256SUMS"
    top_outer = attempt_dir / "SHA256SUMS.seal.sha256"
    require(top_seal.is_file() and top_outer.is_file() and
            not top_seal.is_symlink() and not top_outer.is_symlink(),
            "runner final seal missing/symlinked")
    listed = {}
    for line in top_seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name in {
            "initial/SHA256SUMS.seal.sha256", "POSTCAPTURE_PASS.txt"
        } and name not in listed,
                "runner final member-set drift")
        path = attempt_dir.joinpath(*safe_member(name).parts)
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected,
                "runner final sealed member mismatch: " + name)
        listed[name] = expected
    require(set(listed) == {
        "initial/SHA256SUMS.seal.sha256", "POSTCAPTURE_PASS.txt"
    }, "runner final seal population drift")
    expected, name = top_outer.read_text(encoding="utf-8").strip().split(
        "  ", 1)
    require(name == "SHA256SUMS" and sha256(top_seal) == expected,
            "runner final outer seal drift")

    initial_receipt = key_values(initial / "ATTEMPT_CONSUMED.txt")
    require(set(initial_receipt) == {
                "status", "timestamp", "superseded_static_reviews",
                "authorized_static_review", "repo_root", "output",
                "quarantine", "hostname", "gpu_name", "gpu_uuid",
                "gpu_driver", "gpu_memory_total_mib", "cgroup_failcnt_start",
                "cgroup_under_oom_start", "cgroup_oom_kill_start",
            } and initial_receipt["status"] ==
            "CONSUMED_IMMEDIATELY_BEFORE_M511_ONE_SHOT" and
            initial_receipt["authorized_static_review"] == "r4" and
            initial_receipt["superseded_static_reviews"] == "r1,r2,r3" and
            all(initial_receipt[key] == value for key, value in
                EXPECTED_HOST_GPU_IDENTITY.items()) and
            Path(initial_receipt["output"]).resolve() == capture_dir and
            not Path(initial_receipt["quarantine"]).exists() and
            int(initial_receipt["cgroup_failcnt_start"]) == 0 and
            int(initial_receipt["cgroup_under_oom_start"]) == 0,
            "runner initial receipt semantic drift")
    preflight = (initial / "resource_preflight.log").read_text(
        encoding="utf-8").splitlines()
    require(len(preflight) == 6, "runner preflight line population drift")
    preflight_snapshots = []
    for index in range(3):
        timestamp_line = preflight[2 * index]
        require(timestamp_line.startswith("timestamp=") and
                timestamp_line.endswith(" sample={}".format(index + 1)),
                "runner preflight sample order drift")
        snapshot = dict(token.split("=", 1)
                        for token in preflight[2 * index + 1].split())
        require(set(snapshot) == {
            "commit_headroom_kib", "mem_available_kib", "swap_free_kib",
            "gpu_free_mib", "cgroup_failcnt", "cgroup_under_oom",
            "cgroup_oom_kill"
        } and int(snapshot["commit_headroom_kib"]) >= 8388608 and
                int(snapshot["mem_available_kib"]) >= 8388608 and
                int(snapshot["swap_free_kib"]) >= 8388608 and
                int(snapshot["gpu_free_mib"]) >= 20480 and
                int(snapshot["cgroup_failcnt"]) == 0 and
                int(snapshot["cgroup_under_oom"]) == 0,
                "runner preflight host/GPU/cgroup drift")
        preflight_snapshots.append(snapshot)
    require(int(preflight_snapshots[-1]["cgroup_oom_kill"]) ==
            int(initial_receipt["cgroup_oom_kill_start"]),
            "runner preflight/start oom_kill drift")

    identity_lines = (initial / "identity.sha256").read_text(
        encoding="utf-8").splitlines()
    require(len(identity_lines) == 9, "runner identity population drift")
    repo_root = Path(initial_receipt["repo_root"]).resolve()
    require(repo_root == hw_root.parent.resolve(),
            "runner receipt repo root drift")
    expected_identity_paths = {
        (hw_root / "system_handoff/scripts/run_m511_h67_ep35_convtranspose_binary_input_capture_r1_exact_sha.sh").resolve(),
        (hw_root / "system_handoff/scripts/run_m632_m511_local_rtx3090_capture_exact_sha.sh").resolve(),
        Path("/usr/bin/hostname").resolve(),
        Path("/usr/bin/nvidia-smi").resolve(),
        Path("/opt/anaconda3/envs/pytorch310/bin/python3.10").resolve(),
        (repo_root / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m511_h67_convtranspose_binary_inputs.py").resolve(),
        (hw_root / "contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json").resolve(),
        (hw_root / "reviews/m511_capture_static_hammer_r4_20260827/SHA256SUMS.seal.sha256").resolve(),
        (hw_root / "docs/359_DATE终局冻结_20260813.md").resolve(),
    }
    runner_seen = wrapper_seen = python_seen = False
    hostname_tool_seen = nvidia_smi_tool_seen = False
    observed_identity_paths = set()
    for line in identity_lines:
        expected, name = line.split("  ", 1)
        path = Path(name)
        require(path.is_absolute() and path.is_file() and
                not path.is_symlink() and
                sha256(path) == expected,
                "runner start/end identity drift: " + name)
        observed_identity_paths.add(path.resolve())
        if path.name == (
                "run_m511_h67_ep35_convtranspose_binary_input_capture_r1_exact_sha.sh"):
            require(expected == EXPECTED_RUNNER_SHA256,
                    "runner SHA is not the reviewed local identity")
            runner_seen = True
        elif path.name == "run_m632_m511_local_rtx3090_capture_exact_sha.sh":
            require(expected == EXPECTED_WRAPPER_SHA256,
                    "launch-wrapper SHA is not the reviewed identity")
            wrapper_seen = True
        elif path == Path(
                "/opt/anaconda3/envs/pytorch310/bin/python3.10"):
            require(expected == EXPECTED_PYTHON_SHA256,
                    "Python SHA is not the reviewed identity")
            python_seen = True
        elif path == Path("/usr/bin/hostname"):
            require(expected == EXPECTED_HOSTNAME_TOOL_SHA256,
                    "hostname-tool SHA is not the reviewed identity")
            hostname_tool_seen = True
        elif path == Path("/usr/bin/nvidia-smi"):
            require(expected == EXPECTED_NVIDIA_SMI_TOOL_SHA256,
                    "nvidia-smi-tool SHA is not the reviewed identity")
            nvidia_smi_tool_seen = True
    require(runner_seen and wrapper_seen and python_seen and
            hostname_tool_seen and nvidia_smi_tool_seen and
            observed_identity_paths == expected_identity_paths,
            "runner exact nine-file identity set drift")

    final_receipt = key_values(attempt_dir / "POSTCAPTURE_PASS.txt")
    require(set(final_receipt) == {
                "status", "timestamp", "capture_manifest_sha256",
                "capture_seal_file_sha256", "cgroup_failcnt_end",
                "hostname", "gpu_name", "gpu_uuid", "gpu_driver",
                "gpu_memory_total_mib",
                "cgroup_under_oom_end", "cgroup_oom_kill_end",
                "claim_boundary",
            } and final_receipt["status"] ==
            "PASS_EXACT_CAPTURE_AND_RUNNER_REHASH" and
            final_receipt["capture_manifest_sha256"] ==
            sha256(capture_dir / "manifest.json") and
            final_receipt["capture_seal_file_sha256"] ==
            sha256(capture_dir / "SHA256SUMS.seal.sha256") and
            all(final_receipt[key] == value for key, value in
                EXPECTED_HOST_GPU_IDENTITY.items()) and
            int(final_receipt["cgroup_failcnt_end"]) ==
            int(initial_receipt["cgroup_failcnt_start"]) and
            int(final_receipt["cgroup_under_oom_end"]) == 0 and
            int(final_receipt["cgroup_oom_kill_end"]) ==
            int(initial_receipt["cgroup_oom_kill_start"]) and
            final_receipt["claim_boundary"] ==
            "CAPTURE_ONLY_NO_CYCLES_SPEEDUP_RTL_ENERGY_PPA_OR_HEADLINE",
            "runner final receipt semantic/capture/cgroup drift")
    return {
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "launch_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "python_sha256": EXPECTED_PYTHON_SHA256,
        "hostname_tool_sha256": EXPECTED_HOSTNAME_TOOL_SHA256,
        "nvidia_smi_tool_sha256": EXPECTED_NVIDIA_SMI_TOOL_SHA256,
        "initial_sha256sums_sha256": initial_identity["sha256sums_sha256"],
        "final_sha256sums_sha256": sha256(top_seal),
        "final_seal_file_sha256": sha256(top_outer),
    }


def rehash_runtime_sources(manifest, contract):
    sources = manifest["raw_validation_sources"]
    sequence = sources["sequence_list"]
    sequence_path = Path(sequence["path"])
    require(sequence_path.is_file() and
            sequence_path.stat().st_size == sequence["bytes"] and
            sha256(sequence_path) == sequence["sha256"],
            "validation sequence-list drift")
    require(len(sources["samples"]) == 10, "raw source sample count drift")
    sequence_text = sequence_path.read_text(encoding="utf-8")
    for sample_id, sample in enumerate(sources["samples"]):
        expected = contract["samples"][sample_id]
        require(sample["sample_id"] == sample_id and
                sample["sample_key"] == expected["sample_key"] and
                expected["sample_key"] in sequence_text and
                set(sample["files"]) == {"event", "mask", "flow"},
                "raw source identity/order drift")
        for kind, entry in sample["files"].items():
            path = Path(entry["path"]).resolve()
            require(path.is_file() and path.stat().st_size == entry["bytes"] and
                    sha256(path) == entry["sha256"],
                    "raw source drift: {} {}".format(sample_id, kind))
            require(path.name == expected["sample_key"],
                    "raw source basename drift: {} {}".format(
                        sample_id, kind))
            if kind == "event":
                require(path.parent.name == expected["sequence_key"],
                        "event source sequence directory drift")


def rehash_contract_inputs(contract, manifest, repo_root):
    require(set(contract["inputs"]) == EXPECTED_INPUT_NAMES and
            len(contract["inputs"]) == 21 and
            set(manifest["identity"]["inputs"]) == EXPECTED_INPUT_NAMES,
            "capture frozen-input population drift")
    canonical_root = repo_root.resolve()
    observed = {}
    for name, entry in contract["inputs"].items():
        path = (repo_root / entry["path"]).resolve()
        try:
            path.relative_to(canonical_root)
        except ValueError:
            raise RuntimeError("contract input escapes repo root: " + name)
        captured = manifest["identity"]["inputs"][name]
        require(path.is_file() and sha256(path) == entry["sha256"] and
                captured["path"] == str(path) and
                captured["sha256"] == entry["sha256"],
                "frozen input file/path/hash drift: " + name)
        observed[name] = {"path": str(path), "sha256": entry["sha256"]}
    checkpoint = (repo_root /
                  contract["inputs"]["checkpoint"]["path"]).resolve()
    require(checkpoint.stat().st_size ==
            contract["checkpoint_identity"]["size_bytes"] and
            sha256(checkpoint) == contract["checkpoint_identity"]["sha256"],
            "checkpoint size/hash drift")
    return observed


def decode_record(capture_dir, record, expected, sample):
    require(record["sample_id"] == sample["sample_id"] and
            record["sample_key"] == sample["sample_key"] and
            record["sequence_key"] == sample["sequence_key"],
            "record sample identity drift")
    module_index = expected["module_index"]
    expected_relative = "calls/s{:02d}_d{}.activation.le.bitpack".format(
        sample["sample_id"], module_index)
    require(record["module_index"] == module_index and
            record["name"] == expected["name"] and
            record["operator"] == "ConvTranspose2d" and
            record["input_shape"] == expected["input_shape"] and
            record["output_shape"] == expected["output_shape"] and
            record["relative_path"] == expected_relative,
            "record module/path/shape drift")
    elements = int(np.prod(expected["input_shape"]))
    require(elements % 8 == 0 and record["elements"] == elements and
            record["packed_bytes"] == elements // 8 and
            record["tail_used_bits"] == 8,
            "record element/byte/tail drift")
    path = capture_dir.joinpath(*safe_member(expected_relative).parts)
    require(path.is_file() and not path.is_symlink() and
            path.stat().st_size == elements // 8 and
            sha256(path) == record["file_sha256"],
            "record file size/hash drift")
    packed = np.fromfile(path, dtype=np.uint8)
    active = int(POPCOUNT8[packed].sum(dtype=np.uint64))
    require(active == record["active"], "record active-count drift")
    per_t_elements = int(np.prod(expected["input_shape"][1:]))
    require(per_t_elements % 8 == 0, "per-timestep byte alignment drift")
    per_t_bytes = per_t_elements // 8
    per_timestep_active = [
        int(POPCOUNT8[packed[offset:offset + per_t_bytes]].sum(
            dtype=np.uint64))
        for offset in range(0, len(packed), per_t_bytes)
    ]
    require(len(per_timestep_active) == expected["input_shape"][0] and
            sum(per_timestep_active) == active,
            "per-timestep active population drift")
    return {
        "sample_id": sample["sample_id"],
        "module_index": module_index,
        "name": expected["name"],
        "relative_path": expected_relative,
        "elements": elements,
        "packed_bytes": elements // 8,
        "active": active,
        "activity_rate": active / float(elements),
        "per_timestep_active": per_timestep_active,
        "file_sha256": record["file_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--runner-attempt-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    hw_root = Path(__file__).resolve().parents[2]
    repo_root = hw_root.parent
    canonical_contract_lex = hw_root / "contracts" / \
        "m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
    canonical_capture_lex = hw_root / "system_handoff/outgoing" / \
        "m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"
    canonical_attempt_lex = hw_root / "results" / \
        ".m511_h67_ep35_convtranspose_binary_input_capture_r1_attempt_consumed"
    canonical_output_lex = hw_root / "results" / \
        "m511_h67_ep35_convtranspose_payload_verify_r1_20260827"
    contract_lex = Path(os.path.abspath(args.contract))
    capture_lex = Path(os.path.abspath(args.capture_dir))
    attempt_lex = Path(os.path.abspath(args.runner_attempt_dir))
    output_lex = Path(os.path.abspath(args.output_dir))
    require(contract_lex == canonical_contract_lex and
            capture_lex == canonical_capture_lex and
            attempt_lex == canonical_attempt_lex and
            output_lex == canonical_output_lex and
            not contract_lex.is_symlink() and
            not capture_lex.is_symlink() and
            not attempt_lex.is_symlink() and
            not output_lex.is_symlink(),
            "noncanonical/symlinked verifier path")
    contract_path = contract_lex.resolve()
    capture_dir = capture_lex.resolve()
    runner_attempt_dir = attempt_lex.resolve()
    output_dir = output_lex.resolve()
    require(contract_path.is_file() and capture_dir.is_dir(),
            "missing verifier input")
    require(output_dir.parent.is_dir() and not output_dir.exists(),
            "verifier output must be a new child of an existing directory")
    verifier_start = sha256(Path(__file__).resolve())
    contract_start = sha256(contract_path)
    require(contract_path == canonical_contract_lex.resolve() and
            contract_start == EXPECTED_CONTRACT_SHA256,
            "noncanonical/unreviewed M511 contract")
    capture_seal = verify_seal(capture_dir)
    runner_attempt = verify_runner_attempt(
        runner_attempt_dir, hw_root, capture_dir)
    contract = strict_json(contract_path)
    manifest = strict_json(capture_dir / "manifest.json")
    require((hw_root /
             contract["output"]["canonical_directory"]).resolve() ==
            capture_dir,
            "capture directory is not contract canonical output")
    expected_sealed = {"manifest.json", "RUN_COMPLETE.txt"} | {
        "calls/s{:02d}_d{}.activation.le.bitpack".format(
            sample["sample_id"], module["module_index"])
        for sample in contract["samples"] for module in contract["modules"]
    }
    require(set(capture_seal["sealed_names"]) == expected_sealed and
            capture_seal["manifest_sha256"] ==
            sha256(capture_dir / "manifest.json"),
            "capture sealed member population drift")
    require((capture_dir / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M511_EXACT_S10_CONVTRANSPOSE_INPUT_CAPTURE\n",
            "capture completion marker drift")
    require(contract["schema"] ==
            "m511_h67_ep35_convtranspose_binary_input_capture_contract_v1" and
            contract["status"] ==
            "LOCKED_STATIC_PREFLIGHT_REQUIRED_BEFORE_REMOTE_S10_CAPTURE",
            "contract schema/status drift")
    require(manifest["schema"] ==
            "m511_h67_ep35_convtranspose_binary_input_trace_v1" and
            manifest["status"] ==
            "PASS_EXACT_S10_FOUR_CONVTRANSPOSE_BINARY_INPUT_BITPACKS",
            "capture manifest schema/status drift")
    require(manifest["identity"]["contract_path"] == str(contract_path) and
            manifest["identity"]["contract_sha256"] == contract_start and
            manifest["population"]["samples"] == 10 and
            manifest["population"]["modules"] == 4 and
            manifest["population"]["records"] == 40 and
            manifest["population"]["input_elements"] == 696240000 and
            manifest["population"]["packed_bytes"] == 87030000,
            "capture identity/population drift")
    frozen_inputs = rehash_contract_inputs(
        contract, manifest, repo_root)
    load_audit = manifest["identity"]["checkpoint_load_audit"]
    require(int(load_audit["missing_count"]) == 0 and
            int(load_audit["unexpected_count"]) == 0,
            "capture checkpoint load was not exact")
    require(manifest["eval_protocol"] == contract["eval_protocol"],
            "capture evaluation protocol drift")
    require(contract["expected_population"]["samples"] == 10 and
            contract["expected_population"]["modules"] == 4 and
            contract["expected_population"]["records"] == 40 and
            contract["expected_population"]["input_elements"] == 696240000 and
            contract["expected_population"]["packed_bytes"] == 87030000 and
            contract["packing"] == {
                "layout": "C_ORDER_FLAT_T_B_C_H_W",
                "bit_order": "little",
                "accepted_values": [0, 1],
                "nonbinary_policy": "FAIL_CLOSED_NO_THRESHOLD_OR_COERCION",
                "whole_call_contiguous_copy_allowed": False,
                "chunk_elements": 8388608,
            }, "contract population/packing drift")
    require(contract["claim_boundary"] == {
        "h67_ep35": True,
        "zurich_valid_s10": True,
        "exact_convtranspose_input_bitpacks": True,
        "exact_convtranspose_outputs": False,
        "multi_sequence": False,
        "cycles": False,
        "speedup": False,
        "rtl": False,
        "vcs": False,
        "synopsys": False,
        "energy": False,
        "ppa": False,
        "system_speedup": False,
        "date_headline": False,
    }, "contract claim boundary drift")
    require(manifest["packing"] == {
        "order": "C_ORDER_FLAT",
        "bit_order": "little",
        "binary_values": [0, 1],
        "whole_call_contiguous_copy_allowed": False,
    }, "capture packing policy drift")
    require(manifest["cuda_synchronization"] == {
        "before_capture": 1,
        "per_sample_post_forward": 10,
        "final_pre_manifest": 1,
    }, "capture CUDA synchronization drift")
    require(manifest["claim_boundary"] == {
        "exact_binary_input_bitpacks": True,
        "convtranspose_outputs": False,
        "cycles": False,
        "speedup": False,
        "rtl": False,
        "energy": False,
        "ppa": False,
        "system_speedup": False,
        "date_headline": False,
    }, "capture claim boundary drift")
    rehash_runtime_sources(manifest, contract)
    require(set(manifest["module_identities"]) == {
        item["name"] for item in contract["modules"]
    }, "runtime module identity population drift")
    for module in contract["modules"]:
        runtime = manifest["module_identities"][module["name"]]
        for key in ("operator", "in_channels", "out_channels", "kernel_size",
                    "stride", "padding", "output_padding", "dilation",
                    "groups"):
            require(runtime[key] == module[key],
                    "runtime module property drift: {} {}".format(
                        module["name"], key))
        weight_sha = runtime["weight"]["content_sha256"]
        try:
            valid_weight_sha = len(weight_sha) == 64 and int(weight_sha, 16) >= 0
        except (TypeError, ValueError):
            valid_weight_sha = False
        require(runtime["weight"]["shape"] == module["weight_shape"] and
                runtime["weight"]["dtype"] == "torch.float32" and
                runtime["weight"]["content_bytes"] ==
                int(np.prod(module["weight_shape"])) * 4 and
                valid_weight_sha and runtime["weight"]["byte_order"] ==
                "little" and runtime["weight"]["layout"] ==
                "C_ORDER_CONTIGUOUS" and
                runtime["bias"] is None,
                "runtime module weight-shape/bias drift")

    records = manifest["records"]
    require(len(records) == 40, "capture record count drift")
    decoded = []
    cursor = 0
    for sample in contract["samples"]:
        for module in contract["modules"]:
            decoded.append(decode_record(
                capture_dir, records[cursor], module, sample))
            cursor += 1
    require(sum(row["elements"] for row in decoded) == 696240000 and
            sum(row["packed_bytes"] for row in decoded) == 87030000 and
            sum(row["active"] for row in decoded) ==
            manifest["population"]["active_elements"],
            "decoded aggregate population drift")
    require(sha256(Path(__file__).resolve()) == verifier_start and
            sha256(contract_path) == contract_start,
            "verifier/contract mutated during verification")
    require(rehash_contract_inputs(contract, manifest, repo_root) ==
            frozen_inputs,
            "frozen inputs mutated during verification")
    rehash_runtime_sources(manifest, contract)
    require(verify_seal(capture_dir) == capture_seal,
            "capture seal identity mutated during verification")
    require(verify_runner_attempt(
        runner_attempt_dir, hw_root, capture_dir) == runner_attempt,
        "runner attempt identity mutated during verification")

    output = {
        "schema": "m511_h67_convtranspose_payload_independent_verify_v1",
        "status": "PASS_EXACT_REHASH_AND_FULL_BITPACK_POPCOUNT",
        "identity": {
            "verifier_sha256": verifier_start,
            "contract_sha256": contract_start,
            "capture": {key: value for key, value in capture_seal.items()
                        if key != "sealed_names"},
            "runner_attempt": runner_attempt,
        },
        "population": {
            "samples": 10,
            "modules": 4,
            "records": 40,
            "input_elements": 696240000,
            "packed_bytes": 87030000,
            "active_elements": sum(row["active"] for row in decoded),
        },
        "records": decoded,
        "claim_boundary": {
            "exact_binary_input_payload": True,
            "cycles": False,
            "pgpr_tdr": False,
            "speedup": False,
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    staging = Path(tempfile.mkdtemp(
        prefix=output_dir.name + ".staging.", dir=str(output_dir.parent)))
    quarantine = output_dir.with_name(
        output_dir.name + ".quarantine.failed.{}.{}".format(
            os.getpid(), uuid.uuid4().hex))
    require(not quarantine.exists(), "verifier quarantine target exists")
    published = False
    try:
        (staging / "m511_payload_verify.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M511_INDEPENDENT_PAYLOAD_VERIFY\n", encoding="utf-8")
        write_seal(staging)
        staged_identity = verify_seal(staging)
        require(set(staged_identity["sealed_names"]) == {
            "m511_payload_verify.json", "RUN_COMPLETE.txt"
        } and (staging / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
                "PASS_M511_INDEPENDENT_PAYLOAD_VERIFY\n",
                "verifier staged output population/completion drift")
        require(not output_dir.exists(), "verifier output appeared")
        os.replace(staging, output_dir)
        published = True
        verify_seal(output_dir)
    except BaseException:
        if published:
            os.replace(output_dir, quarantine)
            failure_root = quarantine
        else:
            failure_root = staging
        if failure_root.exists():
            (failure_root / "FAILED.json").write_text(
                json.dumps({"status": "FAIL_CLOSED_NO_ADMISSION"}, indent=2)
                + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
