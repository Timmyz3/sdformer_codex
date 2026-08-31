#!/usr/bin/env python3
"""Capture H67 ep35 decoder-input sparsity on three frozen DSEC sequences.

This producer keeps the M511 checkpoint/model/evaluation identity and the
M686-r6 native deterministic cuDNN-TF32 controls.  It directly consumes 30
contract-pinned raw 10-bin event tensors (ten evenly spaced tensors from each
of three DSEC sequences), hooks D0--D3 ConvTranspose2d inputs, and classifies
the original FP32 bit patterns without thresholding, rounding, or coercion.
Exact {0,1} planes are bit-packed.  Exact {0,runtime-theta} planes are packed
as masks while retaining a hash of the original FP32 bytes.  Other planes
remain hash-only dense fallbacks.  The artifact contains identity, payload,
and density evidence only: no accuracy, cycle, speedup, system, RTL, EDA,
energy, PPA, or DATE-headline claim is emitted.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
M511_SHA256 = (
    "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a")
M686_SHA256 = (
    "1bcff2257e95983ddc77485a41cc4727e082c9297e7312ad534abbb28cf2c630")
EXPECTED_SEQUENCES = (
    "interlaken_01_a", "thun_01_b", "zurich_city_12_a")


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
        raise RuntimeError("M699 non-standard JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "M699 duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def checked_repo_path(relative, allow_missing=False):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "M699 unsafe repository-relative path")
    target = ROOT / relative
    cursor = Path(target.anchor)
    for index, part in enumerate(target.parts[1:], 1):
        cursor = cursor / part
        leaf = index == len(target.parts) - 1
        if os.path.lexists(str(cursor)):
            require(not cursor.is_symlink(),
                    "M699 rejects symlink component: " + str(cursor))
        else:
            require(allow_missing and leaf,
                    "M699 missing path component: " + str(cursor))
    return target.resolve(strict=not allow_missing)


def load_module(name, path, expected_sha):
    require(sha256(path) == expected_sha, "M699 helper identity drift: " + name)
    entrypoint = str(path.parent)
    if entrypoint not in sys.path:
        sys.path.insert(0, entrypoint)
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "M699 cannot import helper: " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha256(path) == expected_sha,
            "M699 helper mutated across import: " + name)
    return module


def seal_members(directory):
    excluded = {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    return [path.relative_to(directory) for path in sorted(directory.rglob("*"))
            if path.is_file() and
            path.relative_to(directory).as_posix() not in excluded]


def write_double_seal(directory):
    directory = Path(directory)
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / item), item.as_posix())
        for item in seal_members(directory)), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(seal)), encoding="utf-8")


def verify_double_seal(directory):
    directory = Path(directory)
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(seal.is_file() and not seal.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "M699 missing double seal")
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(seal) == expected,
            "M699 outer-seal mismatch")
    sealed = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sealed and ".." not in Path(name).parts and
                not Path(name).is_absolute(), "M699 unsafe sealed member")
        path = directory / name
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected, "M699 member-seal mismatch: " + name)
        sealed.add(name)
    actual = {path.relative_to(directory).as_posix()
              for path in directory.rglob("*") if path.is_file() and
              path.relative_to(directory).as_posix() not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == sealed, "M699 sealed-population mismatch")


def verify_core_inputs(contract, launcher):
    expected_names = {"launcher", "runner", "m511_producer", "m686_helper",
                      "m511_contract", "config", "checkpoint", "docs359"}
    require(set(contract["inputs"]) == expected_names,
            "M699 core-input population drift")
    observed = {}
    for name, entry in contract["inputs"].items():
        path = checked_repo_path(entry["path"])
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == int(entry["bytes"]) and
                sha256(path) == entry["sha256"],
                "M699 core-input identity drift: " + name)
        observed[name] = {"path": str(path), "bytes": path.stat().st_size,
                          "sha256": sha256(path)}
    require(observed["launcher"]["path"] == str(launcher) and
            observed["m511_producer"]["sha256"] == M511_SHA256 and
            observed["m686_helper"]["sha256"] == M686_SHA256 and
            observed["docs359"]["sha256"] == DOCS359_SHA256,
            "M699 critical frozen-root drift")
    return observed


def evenly_spaced_indices(population, samples=10):
    require(population >= samples and samples == 10,
            "M699 invalid selection population")
    return [round(index * (population - 1) / (samples - 1))
            for index in range(samples)]


def verify_selected_sources(contract):
    rows = contract["selected_sources"]
    require(len(rows) == 30 and
            [row["global_sample_id"] for row in rows] == list(range(30)),
            "M699 selected-source population/order drift")
    observed = []
    for sequence_position, sequence in enumerate(EXPECTED_SEQUENCES):
        cohort = rows[sequence_position * 10:(sequence_position + 1) * 10]
        require(all(row["sequence"] == sequence for row in cohort) and
                [row["sequence_sample_id"] for row in cohort] == list(range(10)),
                "M699 sequence cohort order drift: " + sequence)
        directory = checked_repo_path(contract["source_root"] + "/" + sequence)
        files = sorted(directory.glob("*.npy"))
        require(len(files) == int(cohort[0]["source_population"]) and
                not any(path.is_symlink() for path in files),
                "M699 source-directory population drift: " + sequence)
        indices = evenly_spaced_indices(len(files))
        require(indices == [int(row["source_index"]) for row in cohort],
                "M699 evenly-spaced selection drift: " + sequence)
        for row, index in zip(cohort, indices):
            path = checked_repo_path(row["path"])
            require(path == files[index].resolve() and path.is_file() and
                    path.stat().st_size == int(row["bytes"]) and
                    sha256(path) == row["sha256"],
                    "M699 selected NPY identity drift: " + row["path"])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            require(array.shape == (10, 480, 640) and
                    array.dtype == np.dtype("float32"),
                    "M699 selected NPY tensor identity drift: " + row["path"])
            observed.append({**row, "resolved_path": str(path),
                             "shape": [10, 480, 640], "dtype": "float32"})
    return observed


def runtime_identity(contract):
    allowed = set(contract["runtime"]["allowed_environment_names"])
    require(set(os.environ) <= allowed,
            "M699 runtime environment exceeds allowlist")
    expected_env = dict(contract["runtime"]["expected_environment"])
    expected_env["M699_EXPECTED_CONTRACT_SHA256"] = sha256(
        checked_repo_path(contract["contract_path"]))
    require(dict(os.environ) == expected_env,
            "M699 runtime environment values/population drift")
    python = Path(sys.executable).resolve()
    require(str(python) == contract["runtime"]["python"]["path"] and
            sha256(python) == contract["runtime"]["python"]["sha256"],
            "M699 Python identity drift")
    require([str(python)] + sys.argv == contract["runtime"]["exact_python_argv"],
            "M699 exact argv drift")
    hostname = subprocess.run(["/usr/bin/hostname"], check=True,
                              stdout=subprocess.PIPE, text=True).stdout.strip()
    gpu_row = subprocess.run([
        "/usr/bin/nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits"], check=True,
        stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()
    require(len(gpu_row) == 1, "M699 requires one visible physical GPU")
    fields = [item.strip() for item in gpu_row[0].split(",")]
    observed_gpu = {"index": int(fields[0]), "name": fields[1],
                    "uuid": fields[2], "driver_version": fields[3],
                    "memory_total_mib": int(fields[4])}
    require(hostname == contract["runtime"]["host_gpu"]["hostname"] and
            observed_gpu == contract["runtime"]["host_gpu"]["gpu"],
            "M699 host/GPU identity drift")
    require(torch.cuda.is_available(), "M699 CUDA unavailable")
    return {"hostname": hostname, "gpu": observed_gpu,
            "python": {"path": str(python), "sha256": sha256(python)},
            "argv": [str(python)] + sys.argv,
            "environment": dict(os.environ)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=int)
    parser.add_argument("--samples-per-sequence", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--chunk-elements", required=True, type=int)
    args = parser.parse_args()
    require(sys.byteorder == "little" and args.sequences == 3 and
            args.samples_per_sequence == 10 and args.num_workers == 0 and
            args.chunk_elements == 8388608,
            "M699 fixed S3x10 capture arguments drift")

    launcher = Path(__file__).resolve()
    contract_path = args.contract.resolve()
    contract_sha = sha256(contract_path)
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m699_h67_ep35_multisequence_decoder_payload_contract_v1" and
            contract.get("status") ==
            "STATIC_AUTHOR_HANDOFF__FRESH_HAMMER_REQUIRED_BEFORE_GPU",
            "M699 contract schema/status drift")
    require(contract_path == checked_repo_path(contract["contract_path"]),
            "M699 contract path drift")
    identities = verify_core_inputs(contract, launcher)
    sources = verify_selected_sources(contract)
    output = checked_repo_path(contract["output"]["canonical_directory"],
                               allow_missing=True)
    require(output == args.output_dir.resolve() and output.parent.is_dir() and
            not os.path.lexists(str(output)), "M699 canonical output not fresh")
    attempt = checked_repo_path(contract["one_shot"]["attempt_directory"])
    require(attempt.is_dir() and not attempt.is_symlink(),
            "M699 independently consumed one-shot receipt missing")
    verify_double_seal(attempt)

    m511 = load_module("m699_frozen_m511",
                       Path(identities["m511_producer"]["path"]), M511_SHA256)
    m686 = load_module("m699_frozen_m686",
                       Path(identities["m686_helper"]["path"]), M686_SHA256)
    m511_contract = strict_json(identities["m511_contract"]["path"])
    frozen_m511_inputs = m511.verify_inputs(m511_contract, M511_SHA256)
    determinism = m686.configure_deterministic_execution()
    m686.require_deterministic_execution(determinism)
    runtime = runtime_identity(contract)

    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.",
                                    dir=str(output.parent)))
    (staging / "calls").mkdir()
    handles = []
    records = []
    current = {"sample": None, "order": 0, "source": None}
    published = False
    quarantine = None
    try:
        config, device = m511.profile.load_config(identities["config"]["path"])
        m686.require_deterministic_execution(m686.observe_execution_controls())
        require(torch.device(device).type == "cuda" and
                config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
                config["model"]["use_upsample_conv"] is False and
                int(config["model"]["kernel_size"]) == 3 and
                config["loader"].get("crop") is None,
                "M699 H67 model/config identity drift")
        model = m511.profile.build_model(
            config, identities["checkpoint"]["path"], device)
        load_audit = m511.profile.validate_h9_load_audit(model, config)
        require(load_audit is not None and
                int(load_audit.get("missing_count", 0)) == 0 and
                int(load_audit.get("unexpected_count", 0)) == 0,
                "M699 checkpoint load is not exact")
        bn_policy = config.get("test", {}).get("bn_policy", "running")
        bn_changed = m511.profile.configure_batch_norm_evaluation(model,
                                                                   bn_policy)
        protocol = {
            "resolution": list(config["loader"]["resolution"]),
            "crop": config["loader"].get("crop"),
            "window_size": list(config["swin_transformer"]["window_size"]),
            "pretrained_window_size": config["swin_transformer"].get(
                "pretrained_window_size"),
            "tokens_per_window": int(np.prod(
                config["swin_transformer"]["window_size"])),
            "remap": config["loader"].get("remap"),
            "bn_policy": bn_policy, "bn_modules_changed": bn_changed,
            "eval_batch_size": 1, "num_workers": 0,
            "module_counts": m511.profile.h9_module_counts(model),
        }
        require(protocol == m511_contract["eval_protocol"],
                "M699 frozen evaluation protocol drift")
        module_ids = m511.module_identities(model, m511_contract["modules"])
        named = dict(model.named_modules())
        require([name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.ConvTranspose2d)] ==
                [row["name"] for row in m511_contract["modules"]],
                "M699 ConvTranspose topology drift")
        d1_theta, d1_identity = m686.decoder_threshold_identity(
            model, m511_contract["modules"][1])

        def hook_factory(expected):
            def hook(_module, inputs, output_tensor):
                sample = current["sample"]
                index = int(expected["module_index"])
                require(sample is not None and current["order"] == index and
                        isinstance(inputs, tuple) and len(inputs) == 1 and
                        torch.is_tensor(inputs[0]) and
                        torch.is_tensor(output_tensor),
                        "M699 decoder hook order/arity drift")
                value = inputs[0]
                require([int(item) for item in value.shape] ==
                        expected["input_shape"] and
                        [int(item) for item in output_tensor.shape] ==
                        expected["output_shape"] and
                        value.dtype == torch.float32,
                        "M699 decoder hook tensor identity drift")
                raw = m686.summarize_d1_fallback(value, args.chunk_elements)
                binary = (raw["nonfinite_count"] == 0 and
                          raw["nonbinary_finite_count"] == 0)
                relative = None
                if binary:
                    relative = "calls/s{:02d}_d{}.binary.le.bitpack".format(
                        sample, index)
                    packed = m686.stream_binary_input(
                        value, args.chunk_elements, staging / relative)
                    route = "EXACT_BINARY_BITPACK"
                    stats = packed
                else:
                    candidate = "calls/s{:02d}_d{}.theta.le.bitpack".format(
                        sample, index)
                    scaled = m686.stream_theta_binary_candidate(
                        value, d1_theta, args.chunk_elements,
                        staging / candidate)
                    if scaled["theta_gate_pass"]:
                        relative = candidate
                        route = "EXACT_SCALED_BINARY_BITPACK"
                    else:
                        require(not (staging / candidate).exists(),
                                "M699 invalid scaled candidate survived")
                        route = "COMMON_FP32_HASH_ONLY_FALLBACK"
                    stats = {"raw": raw, "scaled_binary_audit": scaled}
                source = current["source"]
                records.append({
                    "global_call_index": sample * 4 + index,
                    "global_sample_id": sample,
                    "sequence": source["sequence"],
                    "sequence_sample_id": source["sequence_sample_id"],
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                    "module_index": index, "name": expected["name"],
                    "input_shape": expected["input_shape"],
                    "input_dtype": str(value.dtype),
                    "input_stride": [int(item) for item in value.stride()],
                    "raw_fp32_content_sha256": raw["content_sha256"],
                    "route": route, "relative_path": relative,
                    "statistics": stats,
                    "thresholded": False, "rounded": False,
                    "coerced": False,
                })
                current["order"] += 1
            return hook

        for module in m511_contract["modules"]:
            handles.append(named[module["name"]].register_forward_hook(
                hook_factory(module)))

        sync = {"before_capture": 0, "per_sample_post_forward": 0,
                "final_pre_manifest": 0}
        torch.cuda.synchronize(device)
        sync["before_capture"] += 1
        with torch.no_grad():
            for source in sources:
                sample = int(source["global_sample_id"])
                m511.functional.reset_net(model)
                current.update({"sample": sample, "order": 0,
                                "source": source})
                array = np.load(source["resolved_path"], allow_pickle=False)
                chunk = torch.from_numpy(array.copy()).unsqueeze(0)
                label = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
                mask = torch.ones((1, 480, 640), dtype=torch.float32)
                x, _label, _mask = m511.profile.preprocess_chunk(
                    config, chunk, label, mask, None, device)
                model(x)
                torch.cuda.synchronize(device)
                sync["per_sample_post_forward"] += 1
                require(current["order"] == 4,
                        "M699 incomplete decoder hook population")
                current.update({"sample": None, "order": 0, "source": None})
                print("[M699] captured {}/30 {}".format(
                    sample + 1, source["path"]), flush=True)
        torch.cuda.synchronize(device)
        sync["final_pre_manifest"] += 1
        require(sync == {"before_capture": 1,
                         "per_sample_post_forward": 30,
                         "final_pre_manifest": 1} and len(records) == 120,
                "M699 final synchronization/record population drift")
        while handles:
            handles.pop().remove()

        # Close source/helper replacement windows before publication.
        verify_core_inputs(contract, launcher)
        require(m511.verify_inputs(m511_contract, M511_SHA256) ==
                frozen_m511_inputs,
                "M699 frozen M511 dependency identity changed during capture")
        sources_final = verify_selected_sources(contract)
        require([(row["path"], row["sha256"]) for row in sources_final] ==
                [(row["path"], row["sha256"]) for row in sources],
                "M699 source identity changed during capture")
        require(sha256(contract_path) == contract_sha and
                sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
                DOCS359_SHA256, "M699 frozen roots changed during capture")
        route_counts = {name: sum(row["route"] == name for row in records)
                        for name in ("EXACT_BINARY_BITPACK",
                                     "EXACT_SCALED_BINARY_BITPACK",
                                     "COMMON_FP32_HASH_ONLY_FALLBACK")}
        per_sequence = {}
        for sequence in EXPECTED_SEQUENCES:
            sequence_rows = [row for row in records
                             if row["sequence"] == sequence]
            per_sequence[sequence] = {
                "samples": 10, "hook_calls": len(sequence_rows),
                "routes": {name: sum(row["route"] == name
                                     for row in sequence_rows)
                           for name in route_counts},
            }
        manifest = {
            "schema": "m699_h67_ep35_multisequence_decoder_payload_v1",
            "status": "PASS_CAPTURE_ONLY__FRESH_RESULT_HAMMER_REQUIRED",
            "identity": {"contract": {"path": str(contract_path),
                                          "sha256": contract_sha},
                         "core_inputs": identities,
                         "frozen_m511_inputs": frozen_m511_inputs,
                         "checkpoint_load_audit": load_audit,
                         "runtime": runtime},
            "selection": {"algorithm": "round(i*(N-1)/9), i=0..9",
                          "sequences": list(EXPECTED_SEQUENCES),
                          "selected_sources": sources},
            "deterministic_execution": determinism,
            "eval_protocol": protocol,
            "module_identities": module_ids,
            "d1_runtime_threshold_identity": d1_identity,
            "cuda_synchronization": sync,
            "population": {"sequences": 3, "samples_per_sequence": 10,
                           "samples": 30, "modules": 4,
                           "hook_calls": 120, "route_counts": route_counts,
                           "per_sequence": per_sequence},
            "records": records,
            "claim_boundary": {
                "same_h67_ep35_checkpoint": True,
                "checkpoint_exact_load_missing_unexpected_zero": True,
                "original_fp32_content_hashed": True,
                "no_threshold_round_or_coercion": True,
                "payload_and_density_only": True,
                "accuracy": False, "cycles": False, "speedup": False,
                "system_speedup": False, "rtl": False, "vcs": False,
                "eda": False, "dc": False, "formality": False,
                "ptpx": False, "energy": False, "ppa": False,
                "date_headline": False,
                "fresh_result_hammer_required": True,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M699_CAPTURE_ONLY__FRESH_RESULT_HAMMER_REQUIRED\n",
            encoding="utf-8")
        write_double_seal(staging)
        verify_double_seal(staging)
        quarantine = output.with_name(output.name + ".quarantine.failed.{}.{}"
                                      .format(os.getpid(), uuid.uuid4().hex))
        require(not os.path.lexists(str(output)) and
                not os.path.lexists(str(quarantine)),
                "M699 publication target/quarantine collision")
        os.replace(staging, output)
        published = True
        verify_double_seal(output)
        print("PASS M699 {} {}".format(
            output / "manifest.json", sha256(output / "manifest.json")),
            flush=True)
    except BaseException as error:
        if published:
            os.replace(output, quarantine)
            failure_root = quarantine
            failure_name = "FAILED_POSTPUBLICATION.json"
        else:
            failure_root = staging
            failure_name = "FAILED.json"
        failure = failure_root / failure_name
        if not failure.exists():
            failure.write_text(json.dumps({
                "schema": "m699_multisequence_decoder_capture_failure_v1",
                "status": "FAIL_CLOSED_NO_CANONICAL_RESULT",
                "reason": "{}: {}".format(type(error).__name__, error),
                "completed_records": len(records),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        for handle in handles:
            try:
                handle.remove()
            except BaseException:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
