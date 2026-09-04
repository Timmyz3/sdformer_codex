#!/usr/bin/env python3
"""Read-only M692 fresh-result hammer for the frozen M686-r6 S10 payload."""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile

import numpy as np


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
OUTPUT = HW / "system_handoff/outgoing/m686r6_h67_ep35_layer_static_decoder_payload_s10_r1_20260828"
ATTEMPT = HW / "results/.m686r6_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed"
CONTRACT = HW / "contracts/m686r6_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json"
RUNNER = HW / "system_handoff/scripts/run_m686r6_h67_layer_static_decoder_payload_one_shot.sh"
EXPECTED_MANIFEST_SHA = "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33"
EXPECTED_OUTER_FILE_SHA = "e468b03a60a0531c95555908cef5aaffbc9b7e8887a14f37b985186642354592"
EXPECTED_DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

NAMES = {
    0: "sttmultires_unet.decoders.0.deconv.0",
    1: "sttmultires_unet.decoders.1.deconv.0",
    2: "sttmultires_unet.decoders.2.deconv.0",
    3: "sttmultires_unet.decoders.3.deconv.0",
}
SHAPES = {
    0: [10, 1, 1536, 15, 20],
    1: [10, 1, 770, 30, 40],
    2: [10, 1, 386, 60, 80],
    3: [10, 1, 194, 120, 160],
}
BYTES = {0: 576000, 1: 1155000, 2: 2316000, 3: 4656000}


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
    member = PurePosixPath(name)
    require(not member.is_absolute() and member.parts and
            ".." not in member.parts and member.parts[0] not in ("", "."),
            "unsafe member: " + str(name))
    return member


def tree_inventory(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "unsafe tree root")
    files, directories = set(), {"."}
    for base, dirnames, filenames in os.walk(str(directory), followlinks=False):
        base_path = Path(base)
        for name in dirnames:
            path = base_path / name
            observed = os.lstat(str(path))
            require(stat.S_ISDIR(observed.st_mode) and
                    not stat.S_ISLNK(observed.st_mode),
                    "symlink/non-directory in tree: " + str(path))
            directories.add(path.relative_to(directory).as_posix())
        for name in filenames:
            path = base_path / name
            observed = os.lstat(str(path))
            require(stat.S_ISREG(observed.st_mode) and
                    not stat.S_ISLNK(observed.st_mode),
                    "symlink/non-file in tree: " + str(path))
            files.add(path.relative_to(directory).as_posix())
    return files, directories


def verify_double_seal(directory, require_complete_population=True):
    directory = Path(directory)
    files, _directories = tree_inventory(directory)
    manifest_path = directory / "SHA256SUMS"
    outer_path = directory / "SHA256SUMS.seal.sha256"
    require(manifest_path.is_file() and outer_path.is_file(), "missing seals")
    outer = outer_path.read_text(encoding="utf-8").strip().split()
    require(outer == [sha256(manifest_path), "SHA256SUMS"],
            "outer seal mismatch")
    sealed = set()
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split(None, 1)
        require(len(fields) == 2, "malformed seal line")
        expected, raw_name = fields
        member = safe_member(raw_name.strip())
        name = member.as_posix()
        require(name not in sealed, "duplicate sealed member")
        path = directory / name
        observed = os.lstat(str(path))
        require(stat.S_ISREG(observed.st_mode) and
                not stat.S_ISLNK(observed.st_mode), "unsafe sealed member")
        require(sha256(path) == expected, "sealed member mismatch: " + name)
        sealed.add(name)
    if require_complete_population:
        require(sealed == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
                "sealed population mismatch")
    return {
        "members": len(sealed),
        "manifest_sha256": sha256(manifest_path),
        "outer_seal_file_sha256": sha256(outer_path),
    }


def product(values):
    value = 1
    for item in values:
        require(isinstance(item, int) and not isinstance(item, bool) and item > 0,
                "invalid shape dimension")
        value *= item
    return value


def c_stride(shape):
    result = []
    for index in range(len(shape)):
        result.append(product(shape[index + 1:]) if index + 1 < len(shape) else 1)
    return result


def bitpack_popcount(path):
    values = np.fromfile(str(path), dtype=np.uint8)
    return int(np.unpackbits(values, bitorder="little").sum(dtype=np.uint64))


def parse_key_values(path):
    result = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        key, value = raw.split("=", 1)
        require(key not in result, "duplicate key-value receipt field")
        result[key] = value
    return result


def reseal(directory):
    directory = Path(directory)
    files, _directories = tree_inventory(directory)
    members = sorted(files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (directory / "SHA256SUMS").write_text("".join(
        "{}  {}\n".format(sha256(directory / name), name)
        for name in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(directory / "SHA256SUMS")),
        encoding="utf-8")


def main():
    manifest_path = OUTPUT / "manifest.json"
    require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA,
            "external manifest root mismatch")
    require(sha256(OUTPUT / "SHA256SUMS.seal.sha256") ==
            EXPECTED_OUTER_FILE_SHA, "external outer-seal-file root mismatch")
    manifest = strict_json(manifest_path)
    contract = strict_json(CONTRACT)
    receipt = strict_json(OUTPUT / "runtime_receipt/runtime_receipt.json")

    expected_calls = set()
    for sample in range(10):
        expected_calls.update({
            "calls/s{:02d}_d0.activation.le.bitpack".format(sample),
            "calls/s{:02d}_d1.activation.theta.le.bitpack".format(sample),
            "calls/s{:02d}_d2.activation.le.bitpack".format(sample),
            "calls/s{:02d}_d3.activation.le.bitpack".format(sample),
        })
    expected_files = expected_calls | {
        "RUN_COMPLETE.txt", "manifest.json", "SHA256SUMS",
        "SHA256SUMS.seal.sha256", "runtime_receipt/runtime_receipt.json",
        "runtime_receipt/SHA256SUMS",
        "runtime_receipt/SHA256SUMS.seal.sha256",
        "weights/SHA256SUMS", "weights/SHA256SUMS.seal.sha256",
        "weights/d0.weight.f32le", "weights/d1.weight.f32le",
        "weights/d1.weight.folded_theta.f32le",
        "weights/d1.original_weight_output_scale.sidecar.json",
        "weights/d2.weight.f32le", "weights/d3.weight.f32le",
    }
    files, directories = tree_inventory(OUTPUT)
    require(files == expected_files, "canonical output file population drift")
    require(directories == {".", "calls", "runtime_receipt", "weights"},
            "canonical output directory population drift")
    seals = {
        "output": verify_double_seal(OUTPUT),
        "runtime_receipt": verify_double_seal(OUTPUT / "runtime_receipt"),
        "weights": verify_double_seal(OUTPUT / "weights"),
    }
    require((OUTPUT / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M660_S10_LAYER_STATIC_DECODER_PAYLOAD\n",
            "completion sentinel mismatch")

    attempt_files, attempt_dirs = tree_inventory(ATTEMPT)
    require(attempt_files == {
        "POSTCAPTURE_PASS.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256",
        "initial/ATTEMPT_CONSUMED.txt", "initial/identity.sha256",
        "initial/SHA256SUMS", "initial/SHA256SUMS.seal.sha256"},
        "attempt population drift")
    require(attempt_dirs == {".", "initial"}, "attempt directory drift")
    seals["attempt"] = verify_double_seal(ATTEMPT, False)
    seals["attempt_initial"] = verify_double_seal(ATTEMPT / "initial")
    attempt = parse_key_values(ATTEMPT / "initial/ATTEMPT_CONSUMED.txt")
    post = parse_key_values(ATTEMPT / "POSTCAPTURE_PASS.txt")
    require(attempt["status"] == "CONSUMED_IMMEDIATELY_BEFORE_M660R2_ONE_SHOT",
            "attempt was not consumed")
    require(post["status"] == "PASS_CAPTURE_AND_RUNNER_REHASH" and
            post["manifest_sha256"] == EXPECTED_MANIFEST_SHA and
            post["outer_seal_file_sha256"] == EXPECTED_OUTER_FILE_SHA,
            "post-capture runner receipt drift")
    runner_text = RUNNER.read_text(encoding="utf-8")
    require("set -euo pipefail" in runner_text and
            "m660r2_success=1\ntrap - EXIT\necho \"PASS M686-r6 one-shot capture" in runner_text,
            "runner exit-zero completion path drift")

    require(manifest["schema"] == "m660_h67_ep35_layer_static_decoder_payload_v1" and
            manifest["status"] ==
            "PASS_S10_ALL4_SCALED_BINARY__D1_FOLDED_WEIGHT_MITER_NONEXACT",
            "manifest schema/status drift")
    require(manifest["packing"] == {
        "values": [0, 1], "bit_order": "little", "order": "C_ORDER_FLAT",
        "whole_call_contiguous_copy_allowed": False}, "packing drift")
    require(manifest["population"] == {
        "binary_packed_bytes_by_module": {"0": 5760000, "1": 11550000,
                                             "2": 23160000, "3": 46560000},
        "binary_packed_bytes_total": 87030000,
        "binary_payload_records": 40,
        "d0_d2_d3_binary_payload_records": 30,
        "d1_metadata_records": 10,
        "d1_raw_payload_bytes": 0,
        "d1_raw_payload_files": 0,
        "d1_theta_binary_payload_records": 10,
        "hook_calls": 40, "samples": 10}, "population ledger drift")

    binary_rows = manifest["d0_d2_d3_binary_records"]
    d1_rows = manifest["d1_records"]
    require([(row["sample_id"], row["module_index"]) for row in binary_rows] ==
            [(sample, module) for sample in range(10) for module in (0, 2, 3)],
            "D0/D2/D3 lattice/order drift")
    require([(row["sample_id"], row["module_index"]) for row in d1_rows] ==
            [(sample, 1) for sample in range(10)], "D1 lattice/order drift")
    by_lattice = {(row["sample_id"], row["module_index"]): row
                  for row in binary_rows + d1_rows}
    require(len(by_lattice) == 40, "duplicate sample/module record")
    expected_order = []
    module_totals = {0: 0, 1: 0, 2: 0, 3: 0}
    popcounts = {}
    for sample in range(10):
        sample_key = None
        for module in range(4):
            row = by_lattice[(sample, module)]
            route = ("EXACT_SCALED_BINARY_BITPACK" if module == 1 else
                     "EXACT_BINARY_BITPACK")
            suffix = ("activation.theta.le.bitpack" if module == 1 else
                      "activation.le.bitpack")
            rel = "calls/s{:02d}_d{}.{}".format(sample, module, suffix)
            require(row["global_call_index"] == sample * 4 + module and
                    row["name"] == NAMES[module] and row["route"] == route and
                    row["relative_path"] == rel and
                    row["input_shape"] == SHAPES[module] and
                    row["input_stride"] == c_stride(SHAPES[module]),
                    "record identity/order/shape drift")
            if sample_key is None:
                sample_key = row["sample_key"]
            require(row["sample_key"] == sample_key and
                    row["sequence_key"] == "zurich_city_09_a",
                    "sample identity drift")
            identity = (row["theta_binary_candidate"] if module == 1 else
                        row["input"])
            path = OUTPUT / rel
            elements = product(SHAPES[module])
            require(elements % 8 == 0 and BYTES[module] == elements // 8 and
                    path.stat().st_size == BYTES[module] and
                    identity["elements"] == elements and
                    identity["packed_bytes"] == BYTES[module] and
                    identity["packed_sha256"] == sha256(path) and
                    identity["bit_order"] == "little" and
                    identity["packing_order"] == "C_ORDER_FLAT",
                    "bitpack range/count/hash drift")
            ones = bitpack_popcount(path)
            require(ones + (elements - ones) == elements,
                    "bit population arithmetic drift")
            if module == 1:
                require(identity["theta_gate_pass"] is True and
                        identity["theta_count"] == ones and
                        identity["zero_count"] == elements - ones and
                        identity["other_finite_count"] == 0 and
                        identity["nonfinite_count"] == 0 and
                        identity["raw_payload_saved"] is False and
                        identity["thresholded"] is False and
                        identity["rounded"] is False,
                        "D1 exact scaled-binary gate drift")
            else:
                require(identity["one_count"] == ones and
                        identity["zero_count"] == elements - ones and
                        identity["exact_binary_count"] == elements and
                        identity["nonbinary_finite_count"] == 0 and
                        identity["nonfinite_count"] == 0,
                        "binary count drift")
            popcounts["s{:02d}d{}".format(sample, module)] = ones
            module_totals[module] += BYTES[module]
            expected_order.append({"global_call_index": sample * 4 + module,
                "sample_id": sample, "module_index": module,
                "name": NAMES[module], "route": route})
    require(manifest["global_call_order"] == expected_order,
            "global call ledger drift")
    require(module_totals == {0: 5760000, 1: 11550000,
                              2: 23160000, 3: 46560000},
            "module byte total drift")
    require(popcounts["s00d0"] == 839586 and
            by_lattice[(0, 0)]["input"]["zero_count"] == 3768414 and
            by_lattice[(0, 0)]["input"]["packed_sha256"] ==
            "ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447",
            "S00D0 sentinel mismatch")

    # The mapper is deliberately not executed.  These are its exact static
    # input predicates: schema, packing, complete lattice, route, path, shape,
    # byte length, tail bits and digest.  All files are byte aligned, so each
    # admitted file occupies the exact bit interval [0, elements) with no tail.
    mapper_static_fields = {
        "schema": True, "packing": True, "complete_s10_d0_d3_lattice": True,
        "safe_regular_relative_paths": True, "shape_and_c_order": True,
        "byte_intervals_are_0_to_packed_bytes": True,
        "bit_intervals_are_0_to_elements_no_padding": True,
        "route_specific_hash_and_counts": True,
    }

    # Weight files are payload members, but only original D0-D3 weights are
    # identity material.  The D1 folded weight and sidecar remain unadmitted.
    for key, entry in manifest["weight_payloads"].items():
        path = OUTPUT / safe_member(entry["relative_path"]).as_posix()
        require(path.stat().st_size == entry["content_bytes"] and
                sha256(path) == entry["content_sha256"],
                "weight payload identity drift: " + key)
        if key in ("0", "1", "2", "3"):
            require(product(entry["shape"]) * 4 == entry["content_bytes"] and
                    manifest["module_identities"][entry["name"]]["weight"] ==
                    {field: entry[field] for field in
                     ("byte_order", "content_bytes", "content_sha256",
                      "dtype", "layout", "shape")},
                    "original weight identity drift")
    miter_rows = [row["folded_weight_miter"] for row in d1_rows]
    require(all(row["bit_exact"] is False and row["hashes_equal"] is False and
                row["bit_exact_mismatch_count"] > 0 and
                row["max_ulp_error"] > 0 for row in miter_rows),
            "D1 folded-miter result drift")
    require(manifest["d1_dual_result_decision"] == {
        "exact_zero_or_runtime_scalar_theta_s10": True,
        "fallback_selected": False,
        "folded_weight_convtranspose_miter_bit_exact_s10": False,
        "folded_weight_deployment_admitted": False,
        "folded_weight_payload_role": "DIAGNOSTIC_CANDIDATE_NOT_ADMITTED",
        "miter_nonexact_is_not_silently_admitted": True,
        "original_weight_output_scale_sidecar_role": "UNMITERED_CANDIDATE_NOT_ADMITTED",
        "scaled_binary_representation_admitted": True},
        "D1 decision boundary drift")
    boundary = manifest["claim_boundary"]
    require(boundary["d1_exact_scaled_binary_observed_s10"] is True and
            boundary["d1_folded_weight_miter_bit_exact"] is False and
            boundary["decoder_numeric_equivalence"] is False and
            all(boundary[key] is False for key in
                ("cycles", "speedup", "rtl", "vcs", "dc", "formality",
                 "ptpx", "eda", "energy", "ppa", "system_speedup",
                 "date_headline")), "claim boundary drift")

    load = manifest["identity"]["checkpoint_load_audit"]
    require(load["missing_count"] == 0 and load["unexpected_count"] == 0 and
            load["overlay_missing_count"] == 0 and
            load["overlay_unexpected_count"] == 0,
            "checkpoint/overlay exact-load drift")
    external_inputs = {}
    for container in (manifest["identity"]["inputs"],
                      manifest["identity"]["frozen_m511_inputs"]):
        for key, entry in container.items():
            path = Path(entry["path"])
            observed = sha256(path)
            require(path.is_file() and not path.is_symlink() and
                    observed == entry["sha256"], "external input drift: " + key)
            external_inputs[str(path)] = observed
    require(external_inputs[str(HW / "docs/359_DATE终局冻结_20260813.md")] ==
            EXPECTED_DOCS359_SHA, "docs359 drift")

    deterministic = {
        "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
    }
    require(manifest["deterministic_execution"] == deterministic and
            receipt["deterministic_execution"] == deterministic and
            manifest["cuda_synchronization"] == {
                "before_capture": 1, "per_sample_post_forward": 10,
                "final_pre_manifest": 1},
            "deterministic/synchronization receipt drift")
    producer_text = Path(manifest["identity"]["inputs"]["launcher"]["path"]).read_text(
        encoding="utf-8")
    require("for chunk, mask, label in take_exact(loader, args.samples):" in producer_text and
            producer_text.count("require_deterministic_execution(observe_execution_controls())") >= 4 and
            "sync_counts[\"per_sample_post_forward\"] += 1" in producer_text,
            "sealed per-sample determinism control-flow drift")
    require(receipt["command"]["argv"] ==
            contract["runtime_provenance"]["exact_python_argv"] and
            receipt["command"]["shell"] is False and
            receipt["environment"]["all_observed_names_allowlisted"] is True and
            receipt["environment"]["allowlist"] ==
            contract["runtime_provenance"]["allowed_environment_names"],
            "runtime argv/environment drift")
    expected_env = dict(contract["runtime_provenance"]["expected_environment"])
    expected_env["M660R2_EXPECTED_CONTRACT_SHA256"] = sha256(CONTRACT)
    require(receipt["environment"]["observed"] == expected_env,
            "runtime environment value drift")
    require(manifest["runtime_receipt"]["outer_seal_file_sha256"] ==
            sha256(OUTPUT / "runtime_receipt/SHA256SUMS.seal.sha256"),
            "runtime receipt outer root drift")

    attacks = {}
    with tempfile.TemporaryDirectory(prefix="m692_private_") as temp:
        private = Path(temp) / "payload"
        shutil.copytree(str(OUTPUT), str(private))
        attacked = private / "calls/s00_d0.activation.le.bitpack"
        with attacked.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 1]))
        try:
            verify_double_seal(private)
            attacks["unsealed_member_tamper_rejected"] = False
        except RuntimeError:
            attacks["unsealed_member_tamper_rejected"] = True
        shutil.copy2(str(OUTPUT / "calls/s00_d0.activation.le.bitpack"),
                     str(attacked))
        altered = strict_json(private / "manifest.json")
        altered["status"] = "ATTACK_CONSISTENTLY_RESEALED"
        (private / "manifest.json").write_text(
            json.dumps(altered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        reseal(private)
        verify_double_seal(private)
        attacks["consistent_reseal_internally_valid"] = True
        attacks["consistent_reseal_manifest_external_root_rejected"] = (
            sha256(private / "manifest.json") != EXPECTED_MANIFEST_SHA)
        attacks["consistent_reseal_outer_external_root_rejected"] = (
            sha256(private / "SHA256SUMS.seal.sha256") !=
            EXPECTED_OUTER_FILE_SHA)
    require(all(attacks.values()), "tamper attack escaped")

    result = {
        "schema": "m692_m686r6_s10_payload_independent_hammer_v1",
        "status": "PASS_EXACT_SCALED_BINARY_PAYLOAD_ONLY",
        "manifest_sha256": sha256(manifest_path),
        "outer_seal_file_sha256": sha256(OUTPUT / "SHA256SUMS.seal.sha256"),
        "seals": seals,
        "runner": {"attempt_consumed": True,
                   "postcapture_runner_rehash": True,
                   "exit_zero_completion_path": True},
        "canonical_population": {"files": len(files),
                                  "directories": sorted(directories),
                                  "symlinks": 0},
        "payload": {"samples": 10, "records": 40,
                    "module_packed_bytes": module_totals,
                    "packed_bytes_total": sum(module_totals.values()),
                    "s00d0_popcount": popcounts["s00d0"],
                    "s00d0_zero_count": 3768414,
                    "s00d0_sha256":
                    "ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447",
                    "all_bitpacks_rehashed_and_popcounted": 40,
                    "all_byte_aligned_no_tail_padding": True},
        "m672_static_input_fields": mapper_static_fields,
        "checkpoint_load": {"missing_count": 0, "unexpected_count": 0,
                            "overlay_missing_count": 0,
                            "overlay_unexpected_count": 0},
        "determinism": {"receipt_exact": True,
                        "per_sample_pre_post_checks_in_sealed_control_flow": True,
                        "post_forward_cuda_sync_count": 10},
        "d1": {"route": "EXACT_SCALED_BINARY_BITPACK",
               "theta_gate_pass_records": 10,
               "folded_miter_bit_exact_records": 0,
               "folded_weight_deployment_admitted": False,
               "sidecar_deployment_admitted": False,
               "decoder_numeric_equivalence_claimed": False},
        "external_input_identities_rehashed": len(external_inputs),
        "attacks": attacks,
        "execution_boundary": {"mapper_run": False, "cycle_simulator": False,
                               "performance": False, "gpu": False,
                               "rtl": False, "eda": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
