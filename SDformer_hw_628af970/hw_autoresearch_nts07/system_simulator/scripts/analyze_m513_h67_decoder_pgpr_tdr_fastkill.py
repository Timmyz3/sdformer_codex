#!/usr/bin/env python3
"""Exact S10 fast-kill for H67 decoder PGPR and temporal-delta RTL ideas.

The analysis uses the independently verified M511 bitpacks.  It reports exact
96-lane product-issue counts and psum traffic bounds, not measured RTL cycles,
energy, PPA, accuracy, multi-sequence behavior, or a headline system speedup.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import uuid

import numpy as np


LANES = 96
OLD_INCLUDED_SCOPE_CYCLES = 620302905
TDR_MIN_IDEAL_SPEEDUP = 1.30
EXPECTED_CONTRACT_SHA256 = \
    "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e"
EXPECTED_PAYLOAD_VERIFIER_SHA256 = \
    "222d0402a57789671c975bac4a59a34a5188279b6b6a02319ddd26ad37c9ed1b"
EXPECTED_RUNNER_SHA256 = \
    "788d674eb3df23f3af6cd8525b3a6471fd26596459e298ef8c9df7aa6369b7fa"
EXPECTED_MODULES = [
    (0, 1536, 384, [10, 1, 1536, 15, 20],
     [10, 1, 384, 30, 40]),
    (1, 770, 192, [10, 1, 770, 30, 40],
     [10, 1, 192, 60, 80]),
    (2, 386, 96, [10, 1, 386, 60, 80],
     [10, 1, 96, 120, 160]),
    (3, 194, 96, [10, 1, 194, 120, 160],
     [10, 1, 96, 240, 320]),
]


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
    require(not path.is_absolute() and path.parts and
            ".." not in path.parts,
            "unsafe sealed path: " + name)
    return path


def verify_directory(directory):
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            seal.is_file() and outer.is_file() and
            not seal.is_symlink() and not outer.is_symlink(),
            "missing/symlinked sealed directory")
    members = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        member = safe_member(name)
        path = directory.joinpath(*member.parts)
        require(name not in members and path.is_file() and
                not path.is_symlink() and sha256(path) == expected,
                "sealed member mismatch: " + name)
        members[name] = expected
    expected, name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(seal) == expected,
            "outer seal mismatch")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    require(actual == set(members), "sealed/actual member-set mismatch")
    return {
        "members": members,
        "sha256sums_sha256": sha256(seal),
        "seal_file_sha256": sha256(outer),
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


def unpack_record(path, shape, expected_sha):
    elements = int(np.prod(shape))
    require(elements % 8 == 0 and path.stat().st_size == elements // 8 and
            sha256(path) == expected_sha,
            "M513 bitpack size/hash drift")
    packed = np.fromfile(path, dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little", count=elements).reshape(
        shape).astype(np.bool_, copy=False)


def scatter_destination_contributors(site_counts):
    timesteps, height, width = site_counts.shape
    destination = np.zeros((timesteps, 2 * height, 2 * width),
                           dtype=np.int32)
    for ky in range(3):
        source_y = np.arange(1 if ky == 0 else 0, height, dtype=np.int64)
        output_y = 2 * source_y - 1 + ky
        for kx in range(3):
            source_x = np.arange(1 if kx == 0 else 0, width,
                                 dtype=np.int64)
            output_x = 2 * source_x - 1 + kx
            destination[:, output_y[:, None], output_x[None, :]] += \
                site_counts[:, source_y[:, None], source_x[None, :]]
    return destination


def percentile_from_histogram(histogram, percentile):
    total = sum(histogram.values())
    require(total > 0, "empty contributor histogram")
    target = int(np.ceil(percentile * total))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return value
    raise RuntimeError("histogram percentile overflow")


def analyze_record(bits, module):
    timesteps, batch, channels_in, height, width = bits.shape
    require(timesteps == 10 and batch == 1 and
            channels_in == module["in_channels"],
            "M513 record shape/channel drift")
    channels_out = module["out_channels"]
    require(channels_out % LANES == 0, "M513 Cout lane-tail drift")
    q = channels_out // LANES
    fanout_y = np.full(height, 3, dtype=np.int64)
    fanout_x = np.full(width, 3, dtype=np.int64)
    fanout_y[0] = 2
    fanout_x[0] = 2
    fanout = fanout_y[:, None] * fanout_x[None, :]
    current_sites = bits[:, 0].sum(axis=1, dtype=np.int32)
    current_vectors = int((current_sites * fanout).sum(dtype=np.int64))
    destination = scatter_destination_contributors(current_sites)
    require(int(destination.sum(dtype=np.int64)) == current_vectors,
            "M513 source/destination vector conservation failed")
    nonzero_destination = destination[destination > 0]
    histogram = defaultdict(int)
    for value, count in zip(*np.unique(nonzero_destination,
                                       return_counts=True)):
        histogram[int(value)] += int(count)

    previous = np.zeros_like(bits)
    previous[1:] = bits[:-1]
    rises = np.logical_and(bits, np.logical_not(previous))
    falls = np.logical_and(previous, np.logical_not(bits))
    delta = np.logical_xor(bits, previous)
    delta_sites = delta[:, 0].sum(axis=1, dtype=np.int32)
    rise_sites = rises[:, 0].sum(axis=1, dtype=np.int32)
    fall_sites = falls[:, 0].sum(axis=1, dtype=np.int32)
    delta_vectors = int((delta_sites * fanout).sum(dtype=np.int64))
    rise_vectors = int((rise_sites * fanout).sum(dtype=np.int64))
    fall_vectors = int((fall_sites * fanout).sum(dtype=np.int64))
    require(delta_vectors == rise_vectors + fall_vectors,
            "M513 signed-delta conservation failed")
    dense_vectors = int(timesteps * channels_in * fanout.sum(dtype=np.int64))
    return {
        "active_sources": int(bits.sum(dtype=np.int64)),
        "input_elements": int(bits.size),
        "dense_vectors": dense_vectors,
        "a1_source_tap_vectors": current_vectors,
        "a1_products": current_vectors * channels_out,
        "a1_product_issue_cycles": current_vectors * q,
        "nonempty_destinations": int(nonzero_destination.size),
        "destination_commits_96wide": int(nonzero_destination.size) * q,
        "source_driven_psum_updates_96wide": current_vectors * q,
        "destination_contributor_sum": int(
            nonzero_destination.sum(dtype=np.int64)),
        "destination_contributor_histogram": dict(histogram),
        "delta_sources": int(delta.sum(dtype=np.int64)),
        "rise_sources": int(rises.sum(dtype=np.int64)),
        "fall_sources": int(falls.sum(dtype=np.int64)),
        "delta_source_tap_vectors": delta_vectors,
        "rise_source_tap_vectors": rise_vectors,
        "fall_source_tap_vectors": fall_vectors,
        "tdr_products": delta_vectors * channels_out,
        "tdr_product_issue_cycles": delta_vectors * q,
    }


def merge_layer(rows, module):
    summed_keys = (
        "active_sources", "input_elements", "dense_vectors",
        "a1_source_tap_vectors", "a1_products", "a1_product_issue_cycles",
        "nonempty_destinations", "destination_commits_96wide",
        "source_driven_psum_updates_96wide", "destination_contributor_sum",
        "delta_sources", "rise_sources", "fall_sources",
        "delta_source_tap_vectors", "rise_source_tap_vectors",
        "fall_source_tap_vectors", "tdr_products",
        "tdr_product_issue_cycles")
    result = {key: sum(row[key] for row in rows) for key in summed_keys}
    histogram = defaultdict(int)
    for row in rows:
        for value, count in row["destination_contributor_histogram"].items():
            histogram[int(value)] += int(count)
    result.update({
        "module_index": module["module_index"],
        "name": module["name"],
        "channels_in": module["in_channels"],
        "channels_out": module["out_channels"],
        "activity_rate": result["active_sources"] /
                         float(result["input_elements"]),
        "dense_over_a1_opportunity": result["dense_vectors"] /
                                     float(result["a1_source_tap_vectors"]),
        "tdr_delta_over_a1": result["delta_source_tap_vectors"] /
                             float(result["a1_source_tap_vectors"]),
        "tdr_ideal_product_speedup": result["a1_source_tap_vectors"] /
                                     float(result["delta_source_tap_vectors"]),
        "source_rmw_over_ideal_commit":
            result["source_driven_psum_updates_96wide"] /
            float(result["destination_commits_96wide"]),
        "destination_contributors_p50": percentile_from_histogram(
            histogram, 0.50),
        "destination_contributors_p90": percentile_from_histogram(
            histogram, 0.90),
        "destination_contributors_p99": percentile_from_histogram(
            histogram, 0.99),
        "destination_contributors_max": max(histogram),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--payload-verify-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    script_start = sha256(Path(__file__).resolve())
    hw_root = Path(__file__).resolve().parents[2]
    canonical_contract = hw_root / "contracts" / \
        "m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
    canonical_capture = hw_root / "system_handoff/outgoing" / \
        "m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"
    canonical_verify = hw_root / "results" / \
        "m511_h67_ep35_convtranspose_payload_verify_r1_20260827"
    canonical_output = hw_root / "results" / \
        "m513_h67_decoder_pgpr_tdr_fastkill_r1_20260827"
    lexical = [Path(os.path.abspath(path)) for path in (
        args.contract, args.capture_dir, args.payload_verify_dir,
        args.output_dir)]
    require(lexical == [canonical_contract, canonical_capture,
                        canonical_verify, canonical_output] and
            all(not path.is_symlink() for path in lexical),
            "M513 noncanonical/symlinked input/output path")
    contract_path, capture_dir, verify_dir, output_dir = (
        path.resolve() for path in lexical)
    require(output_dir.parent.is_dir() and not output_dir.exists(),
            "M513 output must be a new child of an existing directory")
    contract_start = sha256(contract_path)
    require(contract_start == EXPECTED_CONTRACT_SHA256,
            "M513 unreviewed contract generation")
    payload_verifier_path = hw_root / "system_simulator/scripts" / \
        "verify_m511_h67_convtranspose_binary_input_payload.py"
    require(sha256(payload_verifier_path) ==
            EXPECTED_PAYLOAD_VERIFIER_SHA256,
            "M513 payload verifier source generation drift")
    capture_identity = verify_directory(capture_dir)
    verify_identity = verify_directory(verify_dir)
    contract = strict_json(contract_path)
    manifest = strict_json(capture_dir / "manifest.json")
    verified = strict_json(verify_dir / "m511_payload_verify.json")
    require(contract["schema"] ==
            "m511_h67_ep35_convtranspose_binary_input_capture_contract_v1" and
            contract["status"] ==
            "LOCKED_STATIC_PREFLIGHT_REQUIRED_BEFORE_REMOTE_S10_CAPTURE" and
            manifest["schema"] ==
            "m511_h67_ep35_convtranspose_binary_input_trace_v1" and
            manifest["status"] ==
            "PASS_EXACT_S10_FOUR_CONVTRANSPOSE_BINARY_INPUT_BITPACKS",
            "M513 contract/capture schema-status drift")
    require(set(verify_identity["members"]) == {
        "m511_payload_verify.json", "RUN_COMPLETE.txt"
    } and (verify_dir / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M511_INDEPENDENT_PAYLOAD_VERIFY\n",
            "M513 payload-verifier sealed output population drift")
    require(verified["schema"] ==
            "m511_h67_convtranspose_payload_independent_verify_v1" and
            verified["status"] ==
            "PASS_EXACT_REHASH_AND_FULL_BITPACK_POPCOUNT" and
            verified["identity"]["verifier_sha256"] ==
            EXPECTED_PAYLOAD_VERIFIER_SHA256 and
            verified["identity"]["contract_sha256"] == contract_start and
            verified["identity"]["capture"]["sha256sums_sha256"] ==
            capture_identity["sha256sums_sha256"] and
            verified["identity"]["runner_attempt"]["runner_sha256"] ==
            EXPECTED_RUNNER_SHA256 and
            verified["population"] == {
                "samples": 10,
                "modules": 4,
                "records": 40,
                "input_elements": 696240000,
                "packed_bytes": 87030000,
                "active_elements": manifest["population"]["active_elements"],
            } and verified["claim_boundary"] == {
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
            "M513 independent payload verification identity drift")
    runner_attempt_identity = verified["identity"]["runner_attempt"]
    runner_attempt_dir = hw_root / "results" / \
        ".m511_h67_ep35_convtranspose_binary_input_capture_r1_attempt_consumed"
    require(runner_attempt_identity["runner_sha256"] ==
            EXPECTED_RUNNER_SHA256 and
            sha256(runner_attempt_dir / "SHA256SUMS") ==
            runner_attempt_identity["final_sha256sums_sha256"] and
            sha256(runner_attempt_dir / "SHA256SUMS.seal.sha256") ==
            runner_attempt_identity["final_seal_file_sha256"],
            "M513 runner final-attempt receipt drift")
    require(len(contract["modules"]) == 4, "M513 module population drift")
    require(set(manifest["module_identities"]) == {
        module["name"] for module in contract["modules"]
    }, "M513 manifest module identity population drift")
    for module, expected in zip(contract["modules"], EXPECTED_MODULES):
        index, channels_in, channels_out, input_shape, output_shape = expected
        require(module["module_index"] == index and
                module["operator"] == "ConvTranspose2d" and
                module["in_channels"] == channels_in and
                module["out_channels"] == channels_out and
                module["input_shape"] == input_shape and
                module["output_shape"] == output_shape and
                module["kernel_size"] == [3, 3] and
                module["stride"] == [2, 2] and
                module["padding"] == [1, 1] and
                module["output_padding"] == [1, 1] and
                module["dilation"] == [1, 1] and module["groups"] == 1 and
                module["weight_shape"] ==
                [channels_in, channels_out, 3, 3] and
                manifest["module_identities"][module["name"]]["bias"] is None,
                "M513 exact decoder geometry/bias drift")
    require(len(manifest["records"]) == 40 and
            len(verified["records"]) == 40,
            "M513 raw record list population drift")
    capture_records = {
        (row["sample_id"], row["module_index"]): row
        for row in manifest["records"]
    }
    verify_records = {
        (row["sample_id"], row["module_index"]): row
        for row in verified["records"]
    }
    expected_record_keys = {
        (sample["sample_id"], module["module_index"])
        for sample in contract["samples"] for module in contract["modules"]
    }
    require(len(capture_records) == len(verify_records) == 40 and
            set(capture_records) == set(verify_records) ==
            expected_record_keys,
            "M513 record identity population drift")
    by_layer = defaultdict(list)
    for sample in contract["samples"]:
        for module in contract["modules"]:
            key = (sample["sample_id"], module["module_index"])
            capture = capture_records[key]
            checked = verify_records[key]
            require(capture["file_sha256"] == checked["file_sha256"] and
                    capture["active"] == checked["active"],
                    "M513 capture/verifier record disagreement")
            path = capture_dir.joinpath(
                *safe_member(capture["relative_path"]).parts)
            bits = unpack_record(
                path, module["input_shape"], capture["file_sha256"])
            row = analyze_record(bits, module)
            require(row["active_sources"] == capture["active"],
                    "M513 independently decoded activity drift")
            by_layer[module["module_index"]].append(row)
    layers = [merge_layer(by_layer[module["module_index"]], module)
              for module in contract["modules"]]
    aggregate_keys = (
        "dense_vectors", "a1_source_tap_vectors", "a1_products",
        "a1_product_issue_cycles", "nonempty_destinations",
        "destination_commits_96wide", "source_driven_psum_updates_96wide",
        "delta_source_tap_vectors", "tdr_products",
        "tdr_product_issue_cycles", "rise_source_tap_vectors",
        "fall_source_tap_vectors")
    aggregate = {key: sum(layer[key] for layer in layers)
                 for key in aggregate_keys}
    aggregate["samples"] = 10
    aggregate["a1_cycles_per_sample_mean"] = \
        aggregate["a1_product_issue_cycles"] / 10.0
    aggregate["tdr_cycles_per_sample_mean"] = \
        aggregate["tdr_product_issue_cycles"] / 10.0
    aggregate["dense_over_a1_opportunity"] = \
        aggregate["dense_vectors"] / float(aggregate["a1_source_tap_vectors"])
    aggregate["tdr_delta_over_a1"] = \
        aggregate["delta_source_tap_vectors"] / float(
            aggregate["a1_source_tap_vectors"])
    aggregate["tdr_ideal_product_speedup"] = \
        aggregate["a1_source_tap_vectors"] / float(
            aggregate["delta_source_tap_vectors"])
    aggregate["source_rmw_over_ideal_commit"] = \
        aggregate["source_driven_psum_updates_96wide"] / float(
            aggregate["destination_commits_96wide"])
    aggregate[
        "nonadmissible_s10_decoder_plus_s100_included_scope_sensitivity_cycles"
    ] = \
        OLD_INCLUDED_SCOPE_CYCLES + aggregate["a1_cycles_per_sample_mean"]
    aggregate[
        "nonadmissible_decoder_share_mixed_s10_s100_sensitivity"
    ] = \
        aggregate["a1_cycles_per_sample_mean"] / aggregate[
            "nonadmissible_s10_decoder_plus_s100_included_scope_sensitivity_cycles"]

    previous_input_bytes = sum(
        int(np.prod(module["input_shape"][1:])) // 8
        for module in contract["modules"])
    previous_output_elements = sum(
        int(np.prod(module["output_shape"][1:]))
        for module in contract["modules"])
    tdr_survives = aggregate["tdr_ideal_product_speedup"] >= \
        TDR_MIN_IDEAL_SPEEDUP
    decision = {
        "pgpr_speedup_verdict":
            "NO_GO__PRODUCT_ISSUE_EQUALS_STRONG_1R1W_OUTPUT_STATIONARY_A1",
        "pgpr_reason": (
            "Exact PGPR and A1 products are identical; a strong 96-wide "
            "1R1W output-stationary A1 sustains one vector per product cycle. "
            "The source-RMW/commit ratio is only an energy/dataflow opportunity, "
            "and the same commit lower bound belongs to the strong baseline."),
        "tdr_verdict": (
            "CONDITIONAL_GO__BUILD_STATE_CYCLE_MODEL_BEFORE_RTL"
            if tdr_survives else
            "NO_GO__IDEAL_PRODUCT_SPEEDUP_BELOW_1P30_BEFORE_STATE_TAX"),
        "tdr_min_ideal_speedup": TDR_MIN_IDEAL_SPEEDUP,
        "new_performance_rtl_authorized": False,
        "decoder_support_mode_authorized":
            "Only an exact C2 polyphase address-generation support mode; do "
            "not list it as an independent speedup contribution.",
    }
    result = {
        "schema": "m513_h67_decoder_pgpr_tdr_fastkill_v1",
        "status": "PASS_EXACT_S10_DECODER_FASTKILL_NO_RTL_ADMISSION",
        "identity": {
            "analyzer_sha256": script_start,
            "contract_sha256": contract_start,
            "payload_verifier_sha256": EXPECTED_PAYLOAD_VERIFIER_SHA256,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "runner_final_seal_file_sha256":
                runner_attempt_identity["final_seal_file_sha256"],
            "capture_sha256sums_sha256":
                capture_identity["sha256sums_sha256"],
            "payload_verify_sha256sums_sha256":
                verify_identity["sha256sums_sha256"],
        },
        "model": {
            "lanes": LANES,
            "decoder_cohort_samples": 10,
            "included_scope_cohort_samples": 100,
            "mixed_cohort_sensitivity_admitted": False,
            "strong_a1": "exact bit-sparse polyphase, 96 lanes, full Cout slices, 96-wide 1R1W psum, output-stationary admitted",
            "pgpr": "same exact products; source-driven RMW traffic versus ideal destination commits is not a fair speedup baseline",
            "tdr": "previous state starts at zero for each S10 sample; t>0 uses exact XOR and signed rise/fall replay",
        },
        "layers": layers,
        "aggregate": aggregate,
        "tdr_state_tax": {
            "previous_input_bitmap_bytes": previous_input_bytes,
            "previous_input_bitmap_mib": previous_input_bytes / (1024.0 ** 2),
            "previous_output_elements": previous_output_elements,
            "previous_output_int16_bytes": previous_output_elements * 2,
            "previous_output_int16_mib": previous_output_elements * 2 /
                                         (1024.0 ** 2),
            "previous_output_acc24_bytes": previous_output_elements * 3,
            "previous_output_acc24_mib": previous_output_elements * 3 /
                                         (1024.0 ** 2),
            "dynamic_no_running_bn_bridge_proven": False,
        },
        "decision": decision,
        "claim_boundary": {
            "exact_s10_coordinates": True,
            "exact_a0_a1_product_issue": True,
            "exact_tdr_transition_product_issue": True,
            "mixed_cohort_system_sensitivity_only": True,
            "multi_sequence": False,
            "cycle_simulator_with_sram": False,
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start and
            sha256(contract_path) == contract_start and
            sha256(payload_verifier_path) ==
            EXPECTED_PAYLOAD_VERIFIER_SHA256 and
            sha256(runner_attempt_dir / "SHA256SUMS") ==
            runner_attempt_identity["final_sha256sums_sha256"] and
            sha256(runner_attempt_dir / "SHA256SUMS.seal.sha256") ==
            runner_attempt_identity["final_seal_file_sha256"] and
            verify_directory(capture_dir) == capture_identity and
            verify_directory(verify_dir) == verify_identity,
            "M513 input/analyzer identity drift")
    staging = Path(tempfile.mkdtemp(
        prefix=output_dir.name + ".staging.", dir=str(output_dir.parent)))
    quarantine = output_dir.with_name(
        output_dir.name + ".quarantine.failed.{}.{}".format(
            os.getpid(), uuid.uuid4().hex))
    require(not quarantine.exists(), "M513 quarantine target exists")
    published = False
    try:
        (staging / "m513_decoder_pgpr_tdr_fastkill.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M513_EXACT_S10_DECODER_FASTKILL\n", encoding="utf-8")
        write_seal(staging)
        staged_identity = verify_directory(staging)
        require(set(staged_identity["members"]) == {
            "m513_decoder_pgpr_tdr_fastkill.json", "RUN_COMPLETE.txt"
        } and (staging / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
                "PASS_M513_EXACT_S10_DECODER_FASTKILL\n",
                "M513 staged output population/completion drift")
        require(not output_dir.exists(), "M513 output appeared")
        os.replace(staging, output_dir)
        published = True
        verify_directory(output_dir)
    except BaseException:
        if published:
            os.replace(output_dir, quarantine)
        raise


if __name__ == "__main__":
    main()
