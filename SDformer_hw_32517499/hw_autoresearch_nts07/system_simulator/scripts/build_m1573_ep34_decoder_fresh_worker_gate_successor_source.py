#!/usr/bin/env python3
"""M1573 source-only fresh-worker gate for the ep34 decoder replay.

This module does not change one hardware request, dependency, address, cache
decision, port calendar, or cycle from the independently reviewed M1556
engine.  It only moves the actual replay behind a fresh-exec worker boundary
and records both Linux current RSS and process high-water RSS at the existing
destination/request memory gates.  M1570 remains consumed and is never
retried.  A distinct, independently reviewed one-shot runner is required
before an actual D0/call0 execution.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1556_SOURCE = HERE / "build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
M1556_SOURCE_SHA256 = "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa"
M1571 = HW / "reviews/m1571_m1570_decoder_rss_failure_forensic_r1_20260901"
M1571_REVIEW_SHA256 = "9039e6a4793fe237e615a8b31c1642e1a870266f4990081132f09e832246bfd4"
M1571_OUTER_SHA256 = "ceea708a62cdfeb946c893cd0d5bb1ade221a14effd7b75dcf28c31659711587"
M1572 = HW / "reviews/m1572_decoder_compact_cycle_simulator_design_review_r1_20260901"
M1572_REVIEW_SHA256 = "34e109794409ad0c1af56101862cd9ce2c21a3ae327a94e3044cf5cfc9b3f9d1"
M1572_OUTER_SHA256 = "a6f44cd77dbb278feee693e386f9a3587fb7f5906af82d7c47f80a33f89efdd6"

SCHEMA = "m1573_ep34_decoder_fresh_worker_gate_successor_source_r1_v1"
STATUS = "SOURCE_ONLY__FRESH_WORKER_AND_DUAL_RSS_TELEMETRY__NO_EXECUTION"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
RSS_LIMIT_KIB = 8 * 1024 * 1024


class M1573Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1573Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def verify_flat_seal(path, review_sha, outer_sha, expected_status):
    path = Path(path)
    regular_exact(path / "review.json", review_sha, "review")
    regular_exact(path / "SHA256SUMS.seal.sha256", outer_sha, "outer seal")
    require((path / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [sha256(path / "SHA256SUMS"), "SHA256SUMS"],
            "outer seal content drift")
    value = json.loads((path / "review.json").read_text(encoding="utf-8"))
    require(value.get("status") == expected_status, "review status drift")
    return {"review_sha256": review_sha,
            "outer_seal_file_sha256": outer_sha}


def load_m1556():
    regular_exact(M1556_SOURCE, M1556_SOURCE_SHA256, "M1556 source")
    spec = importlib.util.spec_from_file_location("m1573_bound_m1556",
                                                  str(M1556_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import M1556")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.M.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG,
            "M1556 configuration boundary drift")
    return module


U = load_m1556()


def current_rss_kib():
    for line in Path("/proc/self/status").read_text(
            encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            require(len(fields) == 3 and fields[2] == "kB",
                    "VmRSS format drift")
            return int(fields[1])
    raise M1573Error("VmRSS unavailable")


def peak_rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class DualRssGate(object):
    """Callable replacement for M1556's gate; timing state is untouched."""
    def __init__(self):
        self.calls = 0
        self.baseline_current_kib = current_rss_kib()
        self.baseline_peak_kib = peak_rss_kib()
        self.max_current_kib = self.baseline_current_kib
        self.max_peak_kib = self.baseline_peak_kib

    def __call__(self):
        current = current_rss_kib()
        peak = peak_rss_kib()
        self.calls += 1
        self.max_current_kib = max(self.max_current_kib, current)
        self.max_peak_kib = max(self.max_peak_kib, peak)
        require(current < RSS_LIMIT_KIB,
                "fresh worker current RSS exceeds strict 8 GiB limit")
        require(peak < RSS_LIMIT_KIB,
                "fresh worker high-water RSS exceeds strict 8 GiB limit")
        return peak

    def receipt(self):
        return {"gate_calls": self.calls,
                "baseline_current_rss_kib": self.baseline_current_kib,
                "baseline_peak_rss_kib": self.baseline_peak_kib,
                "max_current_rss_kib": self.max_current_kib,
                "max_peak_rss_kib": self.max_peak_kib,
                "absolute_limit_kib": RSS_LIMIT_KIB,
                "fresh_exec_required": True}


def validate_authorities(full_payload=False):
    m1571 = verify_flat_seal(
        M1571, M1571_REVIEW_SHA256, M1571_OUTER_SHA256,
        "PASS_M1571_FORENSIC__M1570_ATTEMPT_CONSUMED_AT_FIRST_DENSE_DESTINATION_RSS_GATE__ZERO_RESULTS__NO_RETRY")
    m1572 = verify_flat_seal(
        M1572, M1572_REVIEW_SHA256, M1572_OUTER_SHA256,
        "GO_COMPACT_SOURCE_ONLY_AFTER_EXACT_M1539_MITER_CONTRACT__NO_EXECUTION_AUTHORIZED")
    upstream = U.validate_authorities(bool(full_payload))
    return {"m1571": m1571, "m1572": m1572, "m1556": upstream,
            "m1570_retry": False, "actual_execution": False}


def _hardware_projection(row):
    return {key: row[key] for key in (
        "configuration", "resource_manifest_sha256", "total_cycles",
        "request_count", "kind_counts", "byte_counts",
        "transaction_address_sha256", "commit_sequence_sha256")}


def synthetic_self_test():
    original_gate = U.memory_gate
    before = U.synthetic_self_test()
    gate = DualRssGate()
    try:
        U.memory_gate = gate
        require(U.memory_gate is gate, "M1556 memory gate replacement failed")
        # The upstream synthetic kernel manually sets its destination count
        # and therefore does not call retire_destination().  Exercise the
        # exact callable seam once without pretending it is an actual replay.
        gate()
        after = U.synthetic_self_test()
    finally:
        U.memory_gate = original_gate
    require([_hardware_projection(row) for row in before["results"]] ==
            [_hardware_projection(row) for row in after["results"]],
            "dual-RSS gate changed the frozen schedule")
    require(gate.calls > 0, "dual-RSS gate was not exercised")
    return {"schema": SCHEMA,
            "status": "PASS_M1573_GATE_ONLY_SYNTHETIC_EXACT_MITER__NO_ACTUAL_EXECUTION",
            "hardware_projection_exact": True,
            "configurations": list(CONFIGS), "rss": gate.receipt(),
            "actual_execution": False, "production": False}


def fresh_worker_entry(config):
    """Internal callable for a future independently pinned worker runner.

    Calling this in the author CLI is impossible.  The future runner must
    start a new interpreter for each configuration and record this function's
    RSS receipt next to the unchanged M1556 hardware result.
    """
    require(config in CONFIGS and config != FORBIDDEN_CONFIG,
            "configuration is not admitted")
    gate = DualRssGate()
    original_gate = U.memory_gate
    try:
        U.memory_gate = gate
        result = U.stream_actual_call(config)
    finally:
        U.memory_gate = original_gate
    result = dict(result)
    result["m1573_rss"] = gate.receipt()
    result["fresh_exec_required"] = True
    return result


def production_release(_token=None):
    raise M1573Error("M1573 is source-only; production and M1570 retry forbidden")


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "configurations": list(CONFIGS),
            "forbidden_configuration": FORBIDDEN_CONFIG,
            "representation": {"hardware_schedule": "unchanged_M1556",
                "fresh_exec_per_configuration": True,
                "current_and_peak_rss": True,
                "compact_M1572_engine_implemented": False},
            "claim_boundary": {"source_only": True,
                "m1570_retry": False, "actual_execution": False,
                "paper_citable_performance": False,
                "cycles": False, "traffic": False, "energy": False,
                "system_speedup": False, "rtl": False, "ppa": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--verify-payload-members", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        require(not args.verify_payload_members, "describe accepts no payload flag")
        value = describe()
    elif args.preflight:
        value = {"schema": SCHEMA,
                 "status": "PASS_M1573_SOURCE_PREFLIGHT__NO_EXECUTION",
                 "authorities": validate_authorities(
                     args.verify_payload_members),
                 "actual_execution": False}
    else:
        require(not args.verify_payload_members,
                "synthetic test accepts no payload flag")
        value = synthetic_self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
