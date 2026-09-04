#!/usr/bin/env python3
"""Read-only independent hammer for the sealed M1595 diagnostic result.

The authoritative result and attempt marker are never modified.  Mutation
tests operate only on temporary copies of the small JSON receipt tree.  No
decoder payload is opened and no child, GPU or EDA process is launched.
"""
from __future__ import print_function

import ast
import copy
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / "results/m1595_ep34_decoder_one_process_per_config_r1_20260901"
ATTEMPT = HW / "results/.m1595_ep34_decoder_one_process_per_config_attempt_consumed"
WORK = HW / "results/.m1595_ep34_decoder_one_process_per_config_work"
FAILURE = HW / (
    "results/m1595_ep34_decoder_one_process_per_config_r1_20260901."
    "failed_or_incomplete")
SOURCE = HW / "system_simulator/scripts/run_m1595_ep34_decoder_one_process_per_config_runner_source.py"
TEST = HW / "system_simulator/tests/test_m1595_ep34_decoder_one_process_per_config_runner_source.py"
CONTRACT = HW / "contracts/m1595_m1592_decoder_one_process_per_config_runner_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1595_m1592_decoder_one_process_per_config_runner_source_author_receipt_r1_20260901"
M1583 = HW / "system_simulator/scripts/build_m1583_ep34_decoder_one_process_one_config_source.py"
M1573 = HW / "system_simulator/scripts/build_m1573_ep34_decoder_fresh_worker_gate_successor_source.py"
M1556 = HW / "system_simulator/scripts/build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
M1539 = HW / "system_simulator/scripts/build_m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
KINDS = ("external_read", "external_write", "weight_read", "weight_write",
         "psum_read", "psum_write", "compute", "commit")
HEX = frozenset("0123456789abcdef")
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"

EXPECTED = {
    "source": "c5797aefebed319a59558fa8aadd344c5074272f1c16011e4dd1d183244b8136",
    "test": "8b8b8d85c37c98a381c58058b329b44cffe94a026fb039803ea0f4377875e261",
    "contract": "ba5c9fe3dae7ede92f2b34f0c90f46b45ddd6dafed324fc7ad0e3a8003983978",
    "contract_manifest_file": "1d334164a0062fe2a08934411e0e549da91877683d181f18932e8c5ec3de3f70",
    "contract_outer_file": "e66aa4a4b563a6154f32e5deca2570da2305d77f8d51f1bbf4d25fdc6a3b449c",
    "author_review": "a01bc3107021b7bf5db4b95e9ca8bce20ab4a646291f1c188172d438ff84ebf4",
    "author_manifest": "d8a6fe095c8be43230fce87ba17d8fdb3c6720e106db09e3df0397c1310f2604",
    "author_outer_file": "89959635d95187826af01ee5b82a3f7a82a1463110bc490bc6d6efdbc45b043a",
    "m1583": "f92c91f0a6f3a3d79e53ec232fee339ead72edcf14d22a2d51e6f9e86e3f48c4",
    "m1573": "f26203424c4034230ee696ecf3b6d95685ed21647f41eb0c38b6961f0c83d02c",
    "m1556": "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa",
    "m1539": "9acc4d316061b1791f0ad49793d2f2a7a79eb24fdf0d0c5867cde6648a64b4b4",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "attempt": "d82765f45b01fc8d192ffd2712e074bcda4766046e62aa1ff8f2f90c051c4042",
    "result_manifest": "4e58b301db449bd5e84d2d6ff312c7a71bb5eeb8566add8baddfa07c6b877cd1",
    "result_outer_file": "5aeabb2a030c6cd7c59505cec296d15566840997cdc794fd0e0534b028c8ab9b",
}

EXPECTED_FILES = {
    "RUN_COMPLETE.json": "76d3c1f67f4b29f8a6ecbc1749cb400cb497ef8621e29a1480d9a7281fc7c907",
    "child_0_DENSE_TYPED_K8.json": "faa7e5c7873a088910db1fd55527d67cdabe0112f752c7a32755271b17468ffe",
    "child_1_BIT_EQUAL_SERVICE_K1X8.json": "af8b6605cf5e424e9c02eb286d6a7eddd90900faffcf15ae375bba60fd3113f7",
    "child_2_BIT_TYPED_K8.json": "b4434c31f8371f8b2cb3182f2635c42be762b8bfe69284c50130b7b63537f538",
    "result.json": "aab01f2e7481d377ba0a3a05585c9802fb29576f8d32caf6ca9cebe49fa0f4c5",
}

EXPECTED_ROWS = {
    "DENSE_TYPED_K8": {
        "cycles": 411885307, "requests": 119671692,
        "bytes": {"commit": 18432000, "compute": 5741936640,
            "external_read": 16269396624, "external_write": 144,
            "psum_read": 5741936640, "psum_write": 5741936640,
            "weight_read": 15311831040, "weight_write": 15311831040},
        "kinds": {"commit": 48000, "compute": 19937280,
            "external_read": 29905931, "external_write": 1,
            "psum_read": 19937280, "psum_write": 19937280,
            "weight_read": 19937280, "weight_write": 9968640},
        "address": "aa0044e0a173810adfe62ac9a8d066e188011c065237a2bba43522982f304291",
        "pid": 3690416,
        "ticket": "b38c6127d7478f7a761728d8040fe267dad5261d1cf21f8c35ba7475264dfd34",
    },
    "BIT_EQUAL_SERVICE_K1X8": {
        "cycles": 245418899, "requests": 82947668,
        "bytes": {"commit": 18432000, "compute": 975207168,
            "external_read": 22899133968, "external_write": 144,
            "psum_read": 975207168, "psum_write": 975207168,
            "weight_read": 2082108672, "weight_write": 22551539712},
        "kinds": {"commit": 48000, "compute": 3386136,
            "external_read": 36370635, "external_write": 1,
            "psum_read": 3386136, "psum_write": 3386136,
            "weight_read": 21688632, "weight_write": 14681992},
        "address": "3e2fea214447b97d1845b13d811b340fc98446a538b834c54519462d89698112",
        "pid": 3993068,
        "ticket": "6918a83f726429b617e3e6998c4450d3b38d7d7eb746a20215fc29d6a03cf70f",
    },
    "BIT_TYPED_K8": {
        "cycles": 227080791, "requests": 46342676,
        "bytes": {"commit": 18432000, "compute": 975207168,
            "external_read": 22693048560, "external_write": 144,
            "psum_read": 975207168, "psum_write": 975207168,
            "weight_read": 2082108672, "weight_write": 22551539712},
        "kinds": {"commit": 48000, "compute": 3386136,
            "external_read": 18068139, "external_write": 1,
            "psum_read": 3386136, "psum_write": 3386136,
            "weight_read": 3386136, "weight_write": 14681992},
        "address": "ed6c95d1d595663439152bfb63206df11d8d90866b7c1dfafe67a4b429d52394",
        "pid": 4145605,
        "ticket": "5433fd3d6cee123a6b7b83f308f0eea846e846b12adc96b74a11df3d2b22778d",
    },
}

PARENT_PID = 3690376
RESOURCE = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
PAYLOAD_SHA = "37208563da5f5b218f3aff5b292f05e10a5db16b078672762b2cb9ed60678a1c"
CHECKPOINT_SHA = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
COMMIT_SHA = "3880a2353bfc8b210795136c63fc6937c97158406302a3c8761c6ece20965649"
RESULT_STATUS = "PASS_M1595_D0_CALL0_THREE_PROCESS_DIAGNOSTIC__INDEPENDENT_RESULT_HAMMER_REQUIRED"


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


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
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HammerError("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys),
            label + " key topology drift")


def hex64(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in HEX for character in value),
            label + " is not lowercase hex64")


def canonical_sha(value):
    raw = (json.dumps(value, indent=2, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_manifest(directory):
    manifest = directory / MANIFEST
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "manifest row shape")
        digest, name = fields[0], fields[1].strip()
        rel = Path(name)
        hex64(digest, "manifest digest")
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts and
                len(rel.parts) == 1,
                "duplicate/noncanonical/nested manifest member")
        expected[name] = digest
    return expected


def verify_topology(directory, pin_original):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "result root invalid")
    members = {}
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "result symlink member")
        require(stat.S_ISREG(mode), "nested directory or special member")
        members[relative] = member
    require(set(members) == set(EXPECTED_FILES) | {MANIFEST, OUTER},
            "extra/missing/nested result topology")
    manifest = parse_manifest(directory)
    require(set(manifest) == set(EXPECTED_FILES), "manifest set drift")
    for name, digest in manifest.items():
        regular_exact(directory / name, digest, "result member " + name)
    outer_fields = (directory / OUTER).read_text(encoding="ascii").split()
    require(outer_fields == [sha256(directory / MANIFEST), MANIFEST],
            "outer seal content drift")
    if pin_original:
        require(manifest == EXPECTED_FILES, "original member identity drift")
        regular_exact(directory / MANIFEST, EXPECTED["result_manifest"],
                      "result manifest")
        regular_exact(directory / OUTER, EXPECTED["result_outer_file"],
                      "result outer file")


def validate_rss(row):
    exact_keys(row, ("absolute_limit_kib", "baseline_current_rss_kib",
                     "baseline_peak_rss_kib", "fresh_exec_required",
                     "gate_calls", "max_current_rss_kib",
                     "max_peak_rss_kib"), "RSS")
    require(row["absolute_limit_kib"] == 8388608 and
            row["fresh_exec_required"] is True and
            type(row["gate_calls"]) is int and row["gate_calls"] > 0,
            "RSS gate identity/count drift")
    for field in ("baseline_current_rss_kib", "baseline_peak_rss_kib",
                  "max_current_rss_kib", "max_peak_rss_kib"):
        require(type(row[field]) is int and 0 <= row[field] < 8388608,
                "RSS strict bound drift")
    require(row["max_current_rss_kib"] >= row["baseline_current_rss_kib"] and
            row["max_peak_rss_kib"] >= row["baseline_peak_rss_kib"],
            "RSS monotonicity drift")


def validate_worker(row, configuration):
    exact_keys(row, ("byte_counts", "commit_sequence_sha256",
        "configuration", "diagnostic_only", "fresh_exec_required",
        "kind_counts", "m1573_rss", "module_ordinal", "paper_result",
        "payload_fd_sha256", "payload_fd_size", "pilot_call_ordinal",
        "product_capture", "production", "request_count",
        "resource_manifest_sha256", "schema", "streaming", "timesteps",
        "total_cycles", "transaction_address_sha256"), "worker result")
    expected = EXPECTED_ROWS[configuration]
    require(row["configuration"] == configuration and
            row["resource_manifest_sha256"] == RESOURCE and
            row["total_cycles"] == expected["cycles"] and
            row["request_count"] == expected["requests"] and
            row["kind_counts"] == expected["kinds"] and
            row["byte_counts"] == expected["bytes"] and
            row["transaction_address_sha256"] == expected["address"] and
            row["commit_sequence_sha256"] == COMMIT_SHA and
            row["payload_fd_sha256"] == PAYLOAD_SHA and
            row["payload_fd_size"] == 576000,
            "worker exact identity/cycle/request/byte/digest drift")
    require(set(row["kind_counts"]) == set(KINDS) and
            set(row["byte_counts"]) == set(KINDS) and
            sum(row["kind_counts"].values()) == row["request_count"],
            "worker kind/byte ledger topology/conservation drift")
    require(row["kind_counts"]["commit"] == 48000 and
            row["byte_counts"]["commit"] == 18432000,
            "dense commit population drift")
    require(row["schema"] ==
            "m1556_ep34_decoder_nonproduct_streaming_single_call_pilot_immutable_snapshot_source_r4_v1" and
            row["pilot_call_ordinal"] == 0 and row["module_ordinal"] == 0 and
            row["timesteps"] == 10 and row["diagnostic_only"] is True and
            row["fresh_exec_required"] is True and
            row["paper_result"] is False and
            row["product_capture"] is False and row["production"] is False,
            "worker scope/claim drift")
    streaming = row["streaming"]
    exact_keys(streaming, ("destinations", "materialized_transaction_list",
        "max_live_dependency_tokens", "max_live_outstanding_entries",
        "peak_rss_kib", "peak_rss_limit_kib", "timesteps"), "streaming")
    require(streaming["destinations"] == 12000 and
            streaming["timesteps"] == 10 and
            streaming["materialized_transaction_list"] is False and
            type(streaming["max_live_dependency_tokens"]) is int and
            streaming["max_live_dependency_tokens"] > 0 and
            type(streaming["max_live_outstanding_entries"]) is int and
            streaming["max_live_outstanding_entries"] > 0 and
            streaming["peak_rss_limit_kib"] == 8388608 and
            0 <= streaming["peak_rss_kib"] < 8388608,
            "streaming/RSS result drift")
    validate_rss(row["m1573_rss"])


def validate_semantics(directory, attempt_path=ATTEMPT):
    run = strict_json(Path(directory) / "RUN_COMPLETE.json")
    exact_keys(run, ("status",), "RUN_COMPLETE")
    require(run["status"] == RESULT_STATUS, "completion status drift")
    root = strict_json(Path(directory) / "result.json")
    exact_keys(root, ("attempt_consumed", "automatic_retry", "child_pids",
        "claim_boundary", "identity", "population", "results", "schema",
        "status"), "root result")
    require(root["schema"] ==
            "m1595_ep34_decoder_one_process_per_config_result_r1_v1" and
            root["status"] == RESULT_STATUS and
            root["attempt_consumed"] is True and
            root["automatic_retry"] is False,
            "root status/attempt drift")
    exact_keys(root["identity"], ("docs359_sha256", "m1583_source_sha256",
        "m1592_review_sha256", "python_sha256"), "root identity")
    require(root["identity"] == {
        "docs359_sha256": EXPECTED["docs359"],
        "m1583_source_sha256": EXPECTED["m1583"],
        "m1592_review_sha256":
            "e2a46df1db6b13ed7dff801427cecb77cd00b0331e6120a976706db32a57fe80",
        "python_sha256":
            "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"},
        "root identity drift")
    exact_keys(root["population"], ("call_ordinal", "configurations",
        "decoder_stage", "fresh_child_processes", "module_ordinal",
        "timesteps"), "population")
    require(root["population"] == {"call_ordinal": 0,
        "configurations": list(CONFIGS), "decoder_stage": "D0",
        "fresh_child_processes": 3, "module_ordinal": 0, "timesteps": 10},
        "population/configuration order drift")
    exact_keys(root["claim_boundary"], ("diagnostic_only", "eda", "energy",
        "paper_citable_performance", "production", "rtl",
        "system_speedup"), "claim boundary")
    require(root["claim_boundary"] == {"diagnostic_only": True,
        "eda": False, "energy": False, "paper_citable_performance": False,
        "production": False, "rtl": False, "system_speedup": False},
        "diagnostic claim boundary drift")
    require(type(root["child_pids"]) is list and
            root["child_pids"] == [EXPECTED_ROWS[name]["pid"]
                                   for name in CONFIGS] and
            len(set(root["child_pids"])) == 3 and
            PARENT_PID not in root["child_pids"],
            "three distinct child PID evidence drift")
    require(type(root["results"]) is list and len(root["results"]) == 3,
            "root result population drift")
    tickets = []
    children = []
    for ordinal, configuration in enumerate(CONFIGS):
        child_path = Path(directory) / (
            "child_%d_%s.json" % (ordinal, configuration))
        child = strict_json(child_path)
        exact_keys(child, ("child_pid", "configuration", "m1583_source_sha256",
            "parent_pid", "result", "result_sha256", "schema",
            "ticket_sha256"), "child envelope")
        expected = EXPECTED_ROWS[configuration]
        require(child["schema"] == "m1595_ep34_decoder_child_result_r1_v1" and
                child["configuration"] == configuration and
                child["parent_pid"] == PARENT_PID and
                child["child_pid"] == expected["pid"] and
                child["m1583_source_sha256"] == EXPECTED["m1583"] and
                child["ticket_sha256"] == expected["ticket"],
                "child process/ticket/config identity drift")
        hex64(child["ticket_sha256"], "child ticket")
        validate_worker(child["result"], configuration)
        require(canonical_sha(child["result"]) == child["result_sha256"] and
                child["result"] == root["results"][ordinal],
                "child passthrough/canonical digest drift")
        tickets.append(child["ticket_sha256"])
        children.append(child)
    require(len(set(tickets)) == 3, "child process tickets are not distinct")
    require(FORBIDDEN_CONFIG not in root["population"]["configurations"] and
            all(child["configuration"] != FORBIDDEN_CONFIG
                for child in children), "product configuration present")
    require(len(set(child["result"]["resource_manifest_sha256"]
                    for child in children)) == 1 and
            len(set(child["result"]["payload_fd_sha256"]
                    for child in children)) == 1 and
            len(set(child["result"]["commit_sequence_sha256"]
                    for child in children)) == 1 and
            len(set(child["result"]["kind_counts"]["commit"]
                    for child in children)) == 1 and
            len(set(child["result"]["byte_counts"]["commit"]
                    for child in children)) == 1,
            "cross-configuration resource/payload/commit drift")
    regular_exact(attempt_path, EXPECTED["attempt"], "attempt marker")
    attempt = strict_json(attempt_path)
    exact_keys(attempt, ("attempt_consumed", "automatic_retry",
        "configurations", "parent_pid", "schema", "started_unix", "status"),
        "attempt marker")
    require(attempt["attempt_consumed"] is True and
            attempt["automatic_retry"] is False and
            attempt["configurations"] == list(CONFIGS) and
            attempt["parent_pid"] == PARENT_PID and
            attempt["schema"] ==
                "m1595_ep34_decoder_one_process_per_config_runner_source_r1_v1" and
            attempt["status"] == "ATTEMPT_CONSUMED_BEFORE_CHILD" and
            type(attempt["started_unix"]) is float and
            math.isfinite(attempt["started_unix"]),
            "attempt-before-child/no-retry marker drift")
    return root


def source_chain_and_control_proof():
    for path, key in ((SOURCE, "source"), (TEST, "test"),
                      (CONTRACT, "contract"), (M1583, "m1583"),
                      (M1573, "m1573"), (M1556, "m1556"),
                      (M1539, "m1539"), (DOCS359, "docs359")):
        regular_exact(path, EXPECTED[key], key)
    regular_exact(Path(str(CONTRACT) + ".sha256"),
                  EXPECTED["contract_manifest_file"], "contract manifest")
    regular_exact(Path(str(CONTRACT) + ".sha256.seal.sha256"),
                  EXPECTED["contract_outer_file"], "contract outer")
    regular_exact(AUTHOR / "review.json", EXPECTED["author_review"],
                  "author review")
    regular_exact(AUTHOR / MANIFEST, EXPECTED["author_manifest"],
                  "author manifest")
    regular_exact(AUTHOR / OUTER, EXPECTED["author_outer_file"],
                  "author outer")
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, ast.FunctionDef))
    execute = functions["execute_controlled"]
    child_main = functions["child_main"]
    launcher_calls = [node for node in ast.walk(execute)
                      if isinstance(node, ast.Call) and
                      isinstance(node.func, ast.Name) and
                      node.func.id == "launcher"]
    worker_calls = [node for node in ast.walk(child_main)
                    if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr == "one_shot_worker_entry"]
    require(len(launcher_calls) == 1 and len(worker_calls) == 1 and
            source_text.index("write_new(layout.attempt") <
                source_text.index("for ordinal, config in enumerate(CONFIGS)") <
                source_text.index("envelope = launcher(config"),
            "attempt/one-call control-flow proof drift")
    require("automatic_retry\": False" in source_text and
            "failure_permanently_consumes_attempt" in
                CONTRACT.read_text(encoding="utf-8"),
            "no-retry source/contract proof drift")
    m1539_text = M1539.read_text(encoding="utf-8")
    require(('CHECKPOINT_SHA256 = "' + CHECKPOINT_SHA + '"') in m1539_text and
            ('M1539_SOURCE_SHA256 = "' + EXPECTED["m1539"] + '"') in
                M1556.read_text(encoding="utf-8"),
            "ep34 checkpoint source-chain binding drift")
    return {"attempt_write_precedes_config_loop": True,
            "launcher_call_sites_in_config_loop": 1,
            "worker_call_sites_in_child": 1,
            "checkpoint_sha256_via_exact_source_chain": CHECKPOINT_SHA}


def metrics(root):
    getcontext().prec = 40
    values = dict((row["configuration"], row) for row in root["results"])
    totals = dict((name, sum(values[name]["byte_counts"].values()))
                  for name in CONFIGS)

    def ratio(label, baseline, candidate):
        base_cycles = values[baseline]["total_cycles"]
        candidate_cycles = values[candidate]["total_cycles"]
        base_bytes = totals[baseline]
        candidate_bytes = totals[candidate]
        return {"label": label, "baseline": baseline,
                "candidate": candidate,
                "cycle_ratio_of_sums": str(
                    Decimal(base_cycles) / Decimal(candidate_cycles)),
                "time_reduction": str(
                    Decimal(base_cycles - candidate_cycles) /
                    Decimal(base_cycles)),
                "byte_ratio_of_sums": str(
                    Decimal(base_bytes) / Decimal(candidate_bytes)),
                "byte_reduction": str(
                    Decimal(base_bytes - candidate_bytes) /
                    Decimal(base_bytes))}
    return {"cycles": dict((name, values[name]["total_cycles"])
                           for name in CONFIGS),
            "modeled_transaction_bytes": totals,
            "all_three_modeled_transaction_bytes": sum(totals.values()),
            "comparisons": [
                ratio("dense_vs_bit_equal", CONFIGS[0], CONFIGS[1]),
                ratio("dense_vs_bit_k8", CONFIGS[0], CONFIGS[2]),
                ratio("bit_equal_vs_bit_k8", CONFIGS[1], CONFIGS[2])],
            "metric_boundary": "single D0/call0 diagnostic; ratio-of-sums; not system or paper performance"}


def reseal(directory):
    members = sorted(path for path in Path(directory).iterdir()
                     if path.name not in (MANIFEST, OUTER))
    text = "".join("{}  {}\n".format(sha256(path), path.name)
                   for path in members)
    (Path(directory) / MANIFEST).write_text(text, encoding="ascii")
    (Path(directory) / OUTER).write_text(
        "{}  {}\n".format(sha256(Path(directory) / MANIFEST), MANIFEST),
        encoding="ascii")


def expect_reject(action, label):
    try:
        action()
    except (HammerError, OSError, ValueError, UnicodeError):
        return label
    raise HammerError("mutation survived: " + label)


def mutate_copy(mutator, semantic=True):
    with tempfile.TemporaryDirectory(prefix="m1637_m1595_hammer.") as root:
        candidate = Path(root) / "result"
        shutil.copytree(str(RESULT), str(candidate))
        mutator(candidate)
        if semantic:
            reseal(candidate)
        verify_topology(candidate, False)
        validate_semantics(candidate)


def mutation_hammer():
    rejected = []
    rejected.append(expect_reject(
        lambda: mutate_copy(lambda root: (root / "EXTRA.json").write_text(
            "{}\n", encoding="utf-8"), semantic=False), "extra_member"))
    rejected.append(expect_reject(
        lambda: mutate_copy(lambda root: ((root / "nested").mkdir(),
            (root / "nested/x.json").write_text("{}\n", encoding="utf-8")),
            semantic=False), "nested_member"))

    def symlink(root):
        target = root / "child_0_DENSE_TYPED_K8.json"
        target.unlink()
        target.symlink_to("result.json")
    rejected.append(expect_reject(
        lambda: mutate_copy(symlink, semantic=False), "symlink_member"))

    def manifest_duplicate(root):
        manifest = root / MANIFEST
        first = manifest.read_text(encoding="ascii").splitlines()[0]
        manifest.write_text(manifest.read_text(encoding="ascii") + first + "\n",
                            encoding="ascii")
        (root / OUTER).write_text(
            "{}  {}\n".format(sha256(manifest), MANIFEST), encoding="ascii")
    rejected.append(expect_reject(
        lambda: mutate_copy(manifest_duplicate, semantic=False),
        "duplicate_manifest_row"))

    def duplicate_json(root):
        path = root / "RUN_COMPLETE.json"
        path.write_text('{"status":"%s","status":"%s"}\n' %
                        (RESULT_STATUS, RESULT_STATUS), encoding="utf-8")
    rejected.append(expect_reject(
        lambda: mutate_copy(duplicate_json), "duplicate_json_key"))

    def nonfinite(root):
        path = root / "result.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"attempt_consumed": true',
                                     '"alias": NaN,\n  "attempt_consumed": true', 1),
                        encoding="utf-8")
    rejected.append(expect_reject(
        lambda: mutate_copy(nonfinite), "nonfinite_json"))

    def mutate_child_and_root(root, configuration, mutator):
        ordinal = CONFIGS.index(configuration)
        path = root / ("child_%d_%s.json" % (ordinal, configuration))
        child = strict_json(path)
        mutator(child)
        child["result_sha256"] = canonical_sha(child["result"])
        path.write_text(json.dumps(child, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
        result_path = root / "result.json"
        result = strict_json(result_path)
        result["results"][ordinal] = copy.deepcopy(child["result"])
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n",
                               encoding="utf-8")

    rejected.append(expect_reject(lambda: mutate_copy(
        lambda root: mutate_child_and_root(root, CONFIGS[0],
            lambda child: child.update({"child_pid": PARENT_PID}))),
        "child_identity"))
    rejected.append(expect_reject(lambda: mutate_copy(
        lambda root: mutate_child_and_root(root, CONFIGS[0],
            lambda child: child["result"].update(
                {"total_cycles": child["result"]["total_cycles"] + 1}))),
        "cycle_ledger"))
    rejected.append(expect_reject(lambda: mutate_copy(
        lambda root: mutate_child_and_root(root, CONFIGS[1],
            lambda child: child["result"]["byte_counts"].update(
                {"external_read": child["result"]["byte_counts"][
                    "external_read"] + 1}))), "byte_ledger"))
    rejected.append(expect_reject(lambda: mutate_copy(
        lambda root: mutate_child_and_root(root, CONFIGS[2],
            lambda child: child["result"].update({"paper_result": True}))),
        "worker_claim"))

    def root_claim(root):
        path = root / "result.json"
        value = strict_json(path)
        value["claim_boundary"]["system_speedup"] = True
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    rejected.append(expect_reject(
        lambda: mutate_copy(root_claim), "root_claim"))

    def nested_alias(root):
        path = root / "result.json"
        value = strict_json(path)
        value["results"][0]["claim_boundary"] = {
            "paper_result": True, "system_speedup": True}
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    rejected.append(expect_reject(
        lambda: mutate_copy(nested_alias), "nested_alias_claim"))

    def product_alias(root):
        path = root / "result.json"
        value = strict_json(path)
        value["population"]["configurations"][2] = FORBIDDEN_CONFIG
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    rejected.append(expect_reject(
        lambda: mutate_copy(product_alias), "forbidden_product_config"))
    require(len(rejected) == 13, "mutation count drift")
    return rejected


def main():
    require(not WORK.exists() and not WORK.is_symlink() and
            not FAILURE.exists() and not FAILURE.is_symlink(),
            "successful result coexists with work/failure namespace")
    verify_topology(RESULT, True)
    control = source_chain_and_control_proof()
    root = validate_semantics(RESULT)
    calculated = metrics(root)
    rejected = mutation_hammer()
    output = {"schema":
        "m1637_m1595_decoder_d0_call0_three_process_diagnostic_result_independent_hammer_r1_v1",
        "status": "PASS_M1637_M1595_SEALED_DIAGNOSTIC_RESULT__DIAGNOSTIC_ONLY_NO_PAPER_RESULT",
        "python": sys.version, "control_proof": control,
        "metrics": calculated, "mutations_rejected": rejected,
        "sealed_topology": {"manifest_sha256": EXPECTED["result_manifest"],
            "outer_seal_file_sha256": EXPECTED["result_outer_file"],
            "members": sorted(EXPECTED_FILES), "child_processes": 3,
            "distinct_child_pids": True, "distinct_tickets": True,
            "one_call_each": True},
        "shared_identity": {"checkpoint_sha256": CHECKPOINT_SHA,
            "checkpoint_binding": "inferred through exact M1583-M1573-M1556-M1539 source chain",
            "payload_fd_sha256": PAYLOAD_SHA,
            "resource_manifest_sha256": RESOURCE,
            "commit_sequence_sha256": COMMIT_SHA,
            "dense_commit_count": 48000,
            "dense_commit_bytes": 18432000},
        "claim_boundary": {"diagnostic_only": True,
            "paper_result": False, "system_speedup": False,
            "energy": False, "rtl": False, "eda": False},
        "actual_payload_opened": False, "rerun": False,
        "gpu": False, "eda": False}
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
