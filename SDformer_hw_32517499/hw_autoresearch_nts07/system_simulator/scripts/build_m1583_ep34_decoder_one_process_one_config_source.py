#!/usr/bin/env python3
"""Close the M1573 fresh-process and returned-result admission gaps.

This remains source-only.  The callable captures the exact M1573 worker at
clean import, consumes a private one-call token before opening the payload,
and validates every returned schedule field plus dual-RSS telemetry.  A future
runner must start a new interpreter for each admitted configuration.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1573_PATH = HERE / "build_m1573_ep34_decoder_fresh_worker_gate_successor_source.py"
M1573_SHA256 = "f26203424c4034230ee696ecf3b6d95685ed21647f41eb0c38b6961f0c83d02c"
M1577 = HW / "reviews/m1577_m1573_decoder_fresh_worker_gate_successor_independent_hammer_r1_20260901"
M1577_REVIEW_SHA256 = "cbc8dbd19d56584c09a2a54f017415e0409975a5bb2cfbe673227fddcff5a131"
M1577_OUTER_SHA256 = "730f86346be93ca9d390896b9e422e4956c3cfd9e96c93eeb8acb88f165166e5"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1583_ep34_decoder_one_process_one_config_source_r1_v1"
STATUS = "SOURCE_ONLY__ONE_PROCESS_ONE_CONFIG_AND_RESULT_VALIDATION__NO_ACTUAL"
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
RESOURCE_SHA256 = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
RSS_LIMIT_KIB = 8 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


class M1583Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1583Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
            label + " must be regular")
    require(sha256(path) == expected, label + " SHA drift")


def verify_m1577():
    regular_exact(M1577 / "review.json", M1577_REVIEW_SHA256, "M1577 review")
    regular_exact(M1577 / "SHA256SUMS.seal.sha256", M1577_OUTER_SHA256,
                  "M1577 outer seal")
    require((M1577 / "SHA256SUMS.seal.sha256").read_text(
        encoding="ascii").split() == [sha256(M1577 / "SHA256SUMS"), "SHA256SUMS"],
        "M1577 outer content drift")
    row = json.loads((M1577 / "review.json").read_text(encoding="utf-8"))
    require(row.get("status") ==
            "NO_GO_M1577_M1573_ONE_SHOT_RUNNER_AUTHORING__FRESHNESS_RSS_AND_RESULT_BINDING_NOT_ENFORCED" and
            row.get("authorization", {}).get("successor_source_authoring") is True and
            row.get("authorization", {}).get("actual_execution") is False,
            "M1577 decision drift")


def load_m1573():
    regular_exact(M1573_PATH, M1573_SHA256, "M1573 source")
    spec = importlib.util.spec_from_file_location("m1583_exact_m1573", str(M1573_PATH))
    require(spec is not None and spec.loader is not None, "cannot import M1573")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == FORBIDDEN_CONFIG and
            module.RSS_LIMIT_KIB == RSS_LIMIT_KIB,
            "M1573 boundary drift")
    return module


verify_m1577()
regular_exact(DOCS359, DOCS359_SHA256, "docs359")
U = load_m1573()


def _hex64(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in HEX for character in value), label + " is not hex64")


def validate_result(config, row):
    require(type(row) is dict, "worker result is not an object")
    required = {
        "configuration", "resource_manifest_sha256", "total_cycles",
        "request_count", "kind_counts", "byte_counts",
        "transaction_address_sha256", "commit_sequence_sha256",
        "streaming", "schema", "pilot_call_ordinal", "module_ordinal",
        "timesteps", "diagnostic_only", "paper_result", "product_capture",
        "production", "payload_fd_sha256", "payload_fd_size", "m1573_rss",
        "fresh_exec_required",
    }
    require(required.issubset(set(row)), "worker result fields missing")
    require(row["configuration"] == config and
            row["resource_manifest_sha256"] == RESOURCE_SHA256,
            "configuration/resource result drift")
    require(type(row["total_cycles"]) is int and row["total_cycles"] > 0 and
            type(row["request_count"]) is int and row["request_count"] > 0,
            "cycle/request result invalid")
    for name in ("kind_counts", "byte_counts"):
        require(type(row[name]) is dict and row[name] and
                all(type(key) is str and type(value) is int and value >= 0
                    for key, value in row[name].items()),
                name + " invalid")
    require(sum(row["kind_counts"].values()) == row["request_count"],
            "request/kind conservation mismatch")
    _hex64(row["transaction_address_sha256"], "address digest")
    _hex64(row["commit_sequence_sha256"], "commit digest")
    _hex64(row["payload_fd_sha256"], "payload digest")
    require(type(row["payload_fd_size"]) is int and row["payload_fd_size"] > 0,
            "payload extent invalid")
    require(row["pilot_call_ordinal"] == 0 and row["module_ordinal"] == 0 and
            row["timesteps"] == 10 and row["diagnostic_only"] is True and
            row["paper_result"] is False and row["product_capture"] is False and
            row["production"] is False and row["fresh_exec_required"] is True,
            "pilot scope result drift")
    streaming = row["streaming"]
    require(type(streaming) is dict and streaming.get("timesteps") == 10 and
            type(streaming.get("destinations")) is int and
            streaming["destinations"] > 0 and
            streaming.get("materialized_transaction_list") is False,
            "streaming result invalid")
    rss = row["m1573_rss"]
    require(type(rss) is dict and type(rss.get("gate_calls")) is int and
            rss["gate_calls"] > 0 and rss.get("absolute_limit_kib") == RSS_LIMIT_KIB and
            rss.get("fresh_exec_required") is True,
            "dual-RSS gate was not exercised")
    for key in ("baseline_current_rss_kib", "baseline_peak_rss_kib",
                "max_current_rss_kib", "max_peak_rss_kib"):
        require(type(rss.get(key)) is int and 0 <= rss[key] < RSS_LIMIT_KIB,
                "RSS result outside strict limit")
    require(rss["max_current_rss_kib"] >= rss["baseline_current_rss_kib"] and
            rss["max_peak_rss_kib"] >= rss["baseline_peak_rss_kib"],
            "RSS monotonicity drift")
    return dict(row)


def _build_one_shot(bound_entry):
    used = [False]

    def one_shot(config):
        require(type(config) is str and config in CONFIGS and
                config != FORBIDDEN_CONFIG, "configuration not admitted")
        require(not used[0], "this process already consumed its one configuration")
        used[0] = True
        return validate_result(config, bound_entry(config))
    return one_shot


one_shot_worker_entry = _build_one_shot(U.fresh_worker_entry)


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "configurations": list(CONFIGS),
            "fresh_interpreter_per_configuration": True,
            "one_call_token_consumed_before_payload": True,
            "bound_m1573_sha256": M1573_SHA256,
            "result_validation": ["configuration", "resource", "cycles",
                "requests", "kind_counts", "byte_counts", "address_digest",
                "commit_digest", "payload_digest", "scope", "streaming",
                "dual_rss_gate_calls"],
            "claim_boundary": {"source_only": True, "actual_execution": False,
                "m1570_retry": False, "production": False, "cycles": False,
                "traffic": False, "speedup": False, "system_speedup": False,
                "energy": False, "rtl": False, "eda": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        U.validate_authorities(False)
    print(json.dumps(describe(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
