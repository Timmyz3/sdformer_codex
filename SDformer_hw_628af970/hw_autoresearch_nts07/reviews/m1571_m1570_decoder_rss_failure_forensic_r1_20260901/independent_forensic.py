#!/usr/bin/env python3
"""Read-only M1570 failure forensic; never imports or runs the M1556 pilot."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / "results/m1570_ep34_decoder_d0_call0_nonproduct_one_shot_actual_r1_20260901"
RUNNER = HW / "system_simulator/scripts/run_m1560_ep34_decoder_d0_call0_nonproduct_one_shot.py"
SOURCE = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
KERNEL = HW / "system_simulator/scripts/build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
PAYLOAD_ROOT = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831"

EXPECTED = {
    str(RUNNER): "890a7cf66b8132b23df77d864d08d75766e0f967b194b0dd40c2f244e76c674f",
    str(SOURCE): "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa",
    str(KERNEL): "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4",
    str(RESULT / "WORK_STARTED.json"): "bc046ef00c5e5c9ad72b105a84aff6e8e70d39ab8d47315d9d72ece09bade42f",
    str(RESULT / "FAILED_OR_INCOMPLETE.json"): "bcced1f957320556f56a4a4ce5aa93f0240fb3724cfc61bdd5fd1ccb38f00184",
}


class ForensicError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ForensicError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ForensicError("nonfinite JSON " + token)))


def deep_size(value, seen=None):
    """Small synthetic-object estimator; values are diagnostic, not measured RSS."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(key, seen) + deep_size(item, seen)
                    for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(deep_size(item, seen) for item in value)
    return size


def load_kernel_only():
    spec = importlib.util.spec_from_file_location("m1571_frozen_m1539", str(KERNEL))
    require(spec is not None and spec.loader is not None, "cannot load frozen M1539")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for name, expected in EXPECTED.items():
        path = Path(name)
        require(path.exists() and path.is_file() and not path.is_symlink(),
                "missing or non-regular evidence: " + name)
        require(sha256(path) == expected, "evidence SHA drift: " + name)
    names = sorted(path.name for path in RESULT.iterdir() if path.is_file())
    require(names == ["FAILED_OR_INCOMPLETE.json", "WORK_STARTED.json"],
            "M1570 failure namespace changed")
    started = strict_json(RESULT / "WORK_STARTED.json")
    failed = strict_json(RESULT / "FAILED_OR_INCOMPLETE.json")
    require(started["attempt_consumed"] is True and
            started["automatic_retry"] is False and
            started["configurations"][0] == "DENSE_TYPED_K8",
            "start boundary drift")
    require(failed == {
        "attempt_consumed": True,
        "automatic_retry": False,
        "completed_configurations": 0,
        "exception": "streaming pilot exceeded the strict 8 GiB RSS limit",
        "exception_type": "M1543Error",
        "schema": "m1560_ep34_decoder_d0_call0_nonproduct_one_shot_source_r1_v1",
        "status": "FAILED_OR_INCOMPLETE",
    }, "failure receipt drift")

    runner_text = RUNNER.read_text(encoding="utf-8")
    source_text = SOURCE.read_text(encoding="utf-8")
    kernel_text = KERNEL.read_text(encoding="utf-8")
    require('row = module.stream_actual_call(config)' in runner_text and
            'results.append(row)' in runner_text and
            '"completed_configurations": len(results)' in runner_text,
            "M1560 result sequencing drift")
    require('resource.getrusage(resource.RUSAGE_SELF).ru_maxrss' in source_text and
            'require(value < PEAK_RSS_LIMIT_KIB' in source_text and
            'memory_gate()' in source_text,
            "M1556 RSS gate drift")
    require('stream.read(1 << 20)' in kernel_text and
            'verify_sealed_directory(' in kernel_text,
            "M1539 streaming SHA preflight drift")

    module = load_kernel_only()
    cin, cout, hin, win, _hout, _wout = module.GEOMETRY[0]
    contributors = module.contributors_for_destination(
        lambda _channel, _y, _x: 0, "DENSE_TYPED_K8",
        cin, hin, win, 0, 0)
    groups = module.bank_unique_groups(contributors, cin)
    cache = module.WeightTileCache()
    block_requests = []
    kinds = {}
    for output_block in range(int(math.ceil(float(cout) / 96))):
        count = 0
        for row in module.destination_transactions(
                "DENSE_TYPED_K8", 0, 0, 0, output_block, contributors,
                "DENSE_TYPED_K8:c0:control_done", cache):
            count += 1
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        count += 1  # M1556 appends one commit request per output block.
        kinds["commit"] = kinds.get("commit", 0) + 1
        block_requests.append(count)

    # Recreate the simultaneous queue/output containers used by
    # bank_unique_groups. This is a conservative object-graph proxy, not RSS.
    queues = [[] for _ in range(8)]
    for tap, channel in contributors:
        queues[channel % 8].append((tap, channel))
    grouped = []
    for ordinal in range(max(len(row) for row in queues)):
        grouped.append(tuple(row[ordinal] for row in queues
                             if ordinal < len(row)))
    produced = 3 + sum(value for key, value in kinds.items() if key != "commit")
    token_proxy = {"DENSE_TYPED_K8:c0:t0:d0:token_%d" % ordinal: ordinal
                   for ordinal in range(produced)}

    start_stat = (RESULT / "WORK_STARTED.json").stat()
    fail_stat = (RESULT / "FAILED_OR_INCOMPLETE.json").stat()
    payload_files = [path for path in PAYLOAD_ROOT.rglob("*") if path.is_file()]
    payload_sizes = [path.stat().st_size for path in payload_files]
    evidence = {
        "schema": "m1571_m1570_decoder_rss_failure_forensic_r1_v1",
        "status": "PASS_FORENSIC__M1570_FAILED_BEFORE_FIRST_CONFIGURATION__NO_CYCLES__NO_RETRY",
        "identity": {path: digest for path, digest in sorted(EXPECTED.items())},
        "namespace": {
            "files": names,
            "work_started_mtime_ns": start_stat.st_mtime_ns,
            "failure_mtime_ns": fail_stat.st_mtime_ns,
            "receipt_separation_seconds":
                (fail_stat.st_mtime_ns - start_stat.st_mtime_ns) / 1e9,
            "pid": started["pid"],
            "attempt_consumed": True,
            "automatic_retry": False,
            "completed_configurations": 0,
            "partial_results": 0,
            "cycle_results": 0,
            "traffic_results": 0,
        },
        "first_gate_path": {
            "first_configuration": "DENSE_TYPED_K8",
            "module": 0,
            "geometry_cin_cout_hin_win_hout_wout": list(module.GEOMETRY[0]),
            "immutable_payload_snapshot_bytes": (10 * cin * hin * win + 7) // 8,
            "first_destination_source_sites":
                len(module.destination_sources(0, 0, hin, win)),
            "first_destination_contributors": len(contributors),
            "first_destination_k8_groups": len(groups),
            "output_blocks": int(math.ceil(float(cout) / 96)),
            "requests_per_output_block_including_commit": block_requests,
            "requests_before_first_destination_retirement_including_three_initial":
                3 + sum(block_requests),
            "fallback_gate_period_requests": 65536,
            "therefore_first_reachable_gate": "retire_destination after destination 0",
            "gate_predicate": "ru_maxrss_kib >= 8388608",
            "observed_ru_maxrss_kib": None,
            "observed_ru_maxrss_lower_bound_kib": 8388608,
        },
        "bounded_python_object_proxies": {
            "method": "recursive sys.getsizeof over synthetic first-destination objects; not RSS",
            "contributors_deep_bytes": deep_size(contributors),
            "returned_groups_deep_bytes": deep_size(groups),
            "queue_plus_group_construction_peak_proxy_deep_bytes":
                deep_size({"queues": queues, "output": grouped}),
            "produced_tokens_before_first_retirement": produced,
            "synthetic_token_dictionary_deep_bytes": deep_size(token_proxy),
            "interpretation": "visible first-destination live Python containers are MiB-scale, not GiB-scale",
        },
        "same_process_preflight": {
            "runs_before_work_started": True,
            "sealed_directory_file_count": len(payload_files),
            "sealed_directory_total_file_bytes": sum(payload_sizes),
            "largest_member_bytes": max(payload_sizes),
            "hash_read_block_bytes": 1 << 20,
            "manifest_bytes": (PAYLOAD_ROOT / "manifest.json").stat().st_size,
            "sha_manifest_bytes": (PAYLOAD_ROOT / "SHA256SUMS").stat().st_size,
            "interpretation": "preflight is in the same interpreter and contaminates a process-lifetime high-water, although its visible verifier is streaming and has no GiB-scale container",
        },
        "cause_assessment": {
            "proven": [
                "the unique failing predicate observed ru_maxrss at or above 8 GiB",
                "the failure preceded completion of DENSE_TYPED_K8",
                "no configuration, cycle, traffic, result, seal, or retry exists",
                "ru_maxrss is a process-lifetime high-water metric rather than current RSS",
            ],
            "strong_inference": "the 8 GiB gate was contaminated by process history before the first destination RSS check, not created by the bounded first-destination replay containers",
            "not_recoverable_from_receipt": "the exact ru_maxrss and contemporaneous VmRSS values were not recorded",
        },
        "claim_boundary": {
            "pilot_rerun": False,
            "gpu": False,
            "eda": False,
            "rtl": False,
            "cycles": False,
            "traffic": False,
            "speedup": False,
            "paper_citable_performance": False,
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
