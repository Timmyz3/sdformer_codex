#!/usr/bin/env python3
"""Different-author, payload-free hammer for the frozen M1595 runner source."""

from __future__ import print_function

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/run_m1595_ep34_decoder_one_process_per_config_runner_source.py"
TEST = HW / "system_simulator/tests/test_m1595_ep34_decoder_one_process_per_config_runner_source.py"
ENGINE = HW / "system_simulator/scripts/build_m1583_ep34_decoder_one_process_one_config_source.py"
CONTRACT = HW / "contracts/m1595_m1592_decoder_one_process_per_config_runner_source_contract_r1_20260901.json"
CONTRACT_INNER = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1592 = HW / "reviews/m1592_m1583_decoder_one_process_one_config_engineering_qa_r1_20260901"
AUTHOR = HW / "reviews/m1595_m1592_decoder_one_process_per_config_runner_source_author_receipt_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON310 = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    SOURCE: "c5797aefebed319a59558fa8aadd344c5074272f1c16011e4dd1d183244b8136",
    TEST: "8b8b8d85c37c98a381c58058b329b44cffe94a026fb039803ea0f4377875e261",
    ENGINE: "f92c91f0a6f3a3d79e53ec232fee339ead72edcf14d22a2d51e6f9e86e3f48c4",
    CONTRACT: "ba5c9fe3dae7ede92f2b34f0c90f46b45ddd6dafed324fc7ad0e3a8003983978",
    CONTRACT_INNER: "1d334164a0062fe2a08934411e0e549da91877683d181f18932e8c5ec3de3f70",
    CONTRACT_OUTER: "e66aa4a4b563a6154f32e5deca2570da2305d77f8d51f1bbf4d25fdc6a3b449c",
    M1592 / "review.json": "e2a46df1db6b13ed7dff801427cecb77cd00b0331e6120a976706db32a57fe80",
    M1592 / "SHA256SUMS": "ba4192f11aa531c19401da7bbb6a75f82d2cb53577fd2cadbefb5c45295d883a",
    M1592 / "SHA256SUMS.seal.sha256": "f3b0805ccc50c391d541e934a5615f753bf6e589d0651d44c66e39547cc02ef8",
    AUTHOR / "review.json": "a01bc3107021b7bf5db4b95e9ca8bce20ab4a646291f1c188172d438ff84ebf4",
    AUTHOR / "SHA256SUMS": "d8a6fe095c8be43230fce87ba17d8fdb3c6720e106db09e3df0397c1310f2604",
    AUTHOR / "SHA256SUMS.seal.sha256": "89959635d95187826af01ee5b82a3f7a82a1463110bc490bc6d6efdbc45b043a",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    PYTHON310: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
}

CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
PRODUCT = "PRODUCT_CAPTURE_TYPED_K8"
RESOURCE = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"
RSS_LIMIT = 8388608


class HammerError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise HammerError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerError("nonfinite JSON: " + token)))


def verify_sealed_tree(directory, expected_status):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "outer seal content drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "unsafe manifest row")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed tree symlink")
        if stat.S_ISREG(mode):
            actual.add(rel)
        else:
            require(stat.S_ISDIR(mode), "sealed tree special member")
    require(actual == set(expected), "sealed tree member-set drift")
    for name, digest in expected.items():
        require(sha256(directory / name) == digest, "sealed member drift: " + name)
    review = strict_json_text((directory / "review.json").read_text(encoding="utf-8"))
    require(review.get("status") == expected_status, "sealed review status drift")


def import_source():
    spec = importlib.util.spec_from_file_location("m1600_frozen_m1595", str(SOURCE))
    require(spec is not None and spec.loader is not None, "cannot load M1595")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_row(config):
    return {
        "configuration": config,
        "resource_manifest_sha256": RESOURCE,
        "total_cycles": 101,
        "request_count": 3,
        "kind_counts": {"compute": 2, "commit": 1},
        "byte_counts": {"compute": 288, "commit": 3},
        "transaction_address_sha256": "a" * 64,
        "commit_sequence_sha256": "b" * 64,
        "streaming": {"materialized_transaction_list": False,
                      "destinations": 7, "timesteps": 10},
        "schema": "synthetic_upstream",
        "pilot_call_ordinal": 0,
        "module_ordinal": 0,
        "timesteps": 10,
        "diagnostic_only": True,
        "paper_result": False,
        "product_capture": False,
        "production": False,
        "payload_fd_sha256": "c" * 64,
        "payload_fd_size": 4096,
        "m1573_rss": {"gate_calls": 2,
                      "baseline_current_rss_kib": 100,
                      "baseline_peak_rss_kib": 120,
                      "max_current_rss_kib": 130,
                      "max_peak_rss_kib": 140,
                      "absolute_limit_kib": RSS_LIMIT,
                      "fresh_exec_required": True},
        "fresh_exec_required": True,
    }


def make_layout(M, root, name):
    root = Path(root)
    return M.Layout(root / (name + ".result"), root / (name + ".attempt"),
                    root / (name + ".work"), root / (name + ".failure"),
                    root / (name + ".lock"))


def envelope(M, config, target, parent_pid, nonce, child_pid, row=None):
    row = clean_row(config) if row is None else row
    ticket = M.child_ticket(nonce, config, target, parent_pid)
    return {"schema": "m1595_ep34_decoder_child_result_r1_v1",
            "configuration": config, "parent_pid": parent_pid,
            "child_pid": child_pid, "ticket_sha256": ticket,
            "m1583_source_sha256": M.ENGINE_SHA256,
            "result_sha256": M.canonical_sha(row), "result": row}


def rejected(function):
    try:
        function()
    except Exception:
        return True
    return False


def verify_contract(contract):
    require(contract["schema"] ==
            "m1595_m1592_decoder_one_process_per_config_runner_source_contract_r1_v1",
            "contract schema drift")
    require(contract["population"] == {
        "checkpoint": "motion_ep34_live93", "decoder_stage": "D0",
        "call_ordinal": 0, "module_ordinal": 0, "timesteps": 10,
        "configurations_in_order": list(CONFIGS),
        "forbidden_configuration": PRODUCT, "production": False},
        "pilot population drift")
    require(contract["authorization"] == {
        "different_author_runner_source_hammer": True,
        "runner_execution": False, "actual_execution": False,
        "payload": False, "retry": False, "product": False,
        "production": False, "gpu": False, "rtl": False, "eda": False},
        "author contract exceeded source-only authority")
    require(contract["author_evidence"]["actual_child_processes_launched"] == 0 and
            contract["author_evidence"]["actual_m1583_worker_calls"] == 0 and
            contract["author_evidence"]["payload_opened"] is False,
            "author evidence execution drift")


def source_structure(text):
    require('CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")' in text,
            "ordered configurations drift")
    require('FORBIDDEN_CONFIG = "' + PRODUCT + '"' in text,
            "product prohibition drift")
    require('PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")' in text,
            "fixed child interpreter drift")
    require("for ordinal, config in enumerate(CONFIGS):" in text,
            "sequential three-config loop missing")
    require("write_new(layout.attempt" in text and
            text.index("write_new(layout.attempt") < text.index("for ordinal, config"),
            "attempt is not consumed before first child")
    require("environment = dict(CHILD_ENV_BASE)" in text and
            "subprocess.run(command, env=environment" in text,
            "clean child environment not bound to subprocess")
    require('command = [str(PYTHON), str(SOURCE_FILE), "--child-config", config,' in text,
            "fresh fixed-Python child command drift")
    child_block = text[text.index("def child_main("):text.index("def launch_real_child(")]
    require(child_block.count("M.one_shot_worker_entry(config)") == 1,
            "child does not contain exactly one M1583 call")
    require(child_block.index("config in CONFIGS and config != FORBIDDEN_CONFIG") <
            child_block.index("M.one_shot_worker_entry(config)"),
            "product/config gate occurs after actual entry")
    require("len(set(child_pids)) == len(CONFIGS)" in text and
            "len(set(ticket_hashes)) == len(CONFIGS)" in text,
            "fresh-child PID/ticket conservation missing")
    require("rows[0][\"resource_manifest_sha256\"] == RESOURCE_SHA256" in text and
            "len(set(row[\"commit_sequence_sha256\"] for row in rows)) == 1" in text,
            "cross-config resource/commit conservation missing")
    require("ATTEMPT_CONSUMED_BEFORE_CHILD" in text and
            "automatic_retry\": False" in text and
            "FAILED_OR_INCOMPLETE" in text,
            "permanent attempt/failure receipt missing")
    require("renameat2" in text and "rename_noreplace(layout.work, layout.result)" in text and
            "rename_noreplace(layout.work, layout.failure)" in text,
            "no-replace publication missing")
    lower = text.lower()
    for pattern in (r"\bimport\s+torch\b", r"\bfrom\s+torch\b", r"\bcuda\.",
                    r"nvidia-smi", r"\bvcs\s+-", r"\bdc_shell\b",
                    r"\bpt_shell\b", r"\bfm_shell\b"):
        require(re.search(pattern, lower) is None,
                "unexpected GPU/EDA entry: " + pattern)
    require("range(120)" not in text and '"call_ordinal": 119' not in text and
            "all_120" not in lower and "full_120" not in lower,
            "120-call population leaked into pilot runner")
    return {"ordered_configs": list(CONFIGS), "fresh_child_command": True,
            "one_m1583_call_per_child": True, "product_gate_before_entry": True,
            "attempt_before_child": True, "full_120_call_path": False}


def hammer(M):
    counters = {"synthetic_launcher_calls": 0, "actual_worker_calls": 0,
                "mutations_rejected": 0}

    def forbidden_actual(_config):
        counters["actual_worker_calls"] += 1
        raise HammerError("actual M1583 entry invoked by static hammer")

    M.M.one_shot_worker_entry = forbidden_actual

    with tempfile.TemporaryDirectory(prefix="m1600_success.") as root:
        layout = make_layout(M, root, "success")
        pids = []

        def launcher(config, target, parent_pid, nonce):
            ordinal = len(pids)
            item = envelope(M, config, target, parent_pid, nonce,
                            parent_pid + 1000 + ordinal)
            M.write_new(target, item)
            pids.append(item["child_pid"])
            counters["synthetic_launcher_calls"] += 1
            return item

        result = M.execute_controlled(layout, launcher)
        require([row["configuration"] for row in result["results"]] == list(CONFIGS),
                "success configuration order drift")
        require(len(set(pids)) == 3 and result["population"]["fresh_child_processes"] == 3,
                "synthetic child process accounting drift")
        require(layout.attempt.is_file() and layout.result.is_dir() and
                not layout.work.exists() and not layout.failure.exists(),
                "success publication layout drift")
        before = counters["synthetic_launcher_calls"]
        require(rejected(lambda: M.execute_controlled(layout, launcher)),
                "success namespace allowed retry")
        require(counters["synthetic_launcher_calls"] == before,
                "retry reached launcher")

    with tempfile.TemporaryDirectory(prefix="m1600_failure.") as root:
        layout = make_layout(M, root, "failure")
        calls = []

        def fail_second(config, target, parent_pid, nonce):
            calls.append(config)
            if len(calls) == 2:
                raise HammerError("injected child-2 failure")
            item = envelope(M, config, target, parent_pid, nonce, parent_pid + 2000)
            M.write_new(target, item)
            return item

        require(rejected(lambda: M.execute_controlled(layout, fail_second)),
                "injected failure accepted")
        require(calls == list(CONFIGS[:2]) and layout.attempt.is_file() and
                layout.failure.is_dir() and not layout.result.exists() and
                not layout.work.exists(), "failure publication/consumption drift")
        require(rejected(lambda: M.execute_controlled(layout, fail_second)),
                "failed attempt allowed retry")
        require(calls == list(CONFIGS[:2]), "failed retry reached launcher")

    with tempfile.TemporaryDirectory(prefix="m1600_envelope.") as root:
        target = Path(root) / "child.json"
        parent_pid, nonce = 2345, "1" * 64
        good = envelope(M, CONFIGS[0], target, parent_pid, nonce, 6789)
        ticket = M.child_ticket(nonce, CONFIGS[0], target, parent_pid)
        require(M.verify_child_envelope(CONFIGS[0], good, parent_pid, ticket) == good["result"],
                "clean child envelope rejected")

        mutations = []
        for key, value in (("schema", "bad"), ("configuration", CONFIGS[1]),
                           ("parent_pid", parent_pid + 1), ("child_pid", parent_pid),
                           ("child_pid", 1), ("ticket_sha256", "0" * 64),
                           ("m1583_source_sha256", "0" * 64),
                           ("result_sha256", "0" * 64)):
            bad = copy.deepcopy(good); bad[key] = value
            mutations.append(("envelope_" + key,
                              lambda bad=bad: M.verify_child_envelope(
                                  CONFIGS[0], bad, parent_pid, ticket)))

        required = list(good["result"].keys())
        for key in required:
            bad = copy.deepcopy(good)
            del bad["result"][key]
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            mutations.append(("remove_" + key,
                              lambda bad=bad: M.verify_child_envelope(
                                  CONFIGS[0], bad, parent_pid, ticket)))

        row_mutations = (
            ("configuration", CONFIGS[1]),
            ("resource_manifest_sha256", "0" * 64),
            ("total_cycles", 0), ("total_cycles", True),
            ("request_count", 0), ("request_count", True),
            ("kind_counts", {"compute": 2}),
            ("kind_counts", {"compute": -1, "commit": 4}),
            ("byte_counts", {"compute": -1}),
            ("transaction_address_sha256", "g" * 64),
            ("commit_sequence_sha256", "b" * 63),
            ("payload_fd_sha256", "C" * 64),
            ("payload_fd_size", 0),
            ("pilot_call_ordinal", 1), ("module_ordinal", 1),
            ("timesteps", 9), ("diagnostic_only", False),
            ("paper_result", True), ("product_capture", True),
            ("production", True), ("fresh_exec_required", False),
        )
        for ordinal, (key, value) in enumerate(row_mutations):
            bad = copy.deepcopy(good); bad["result"][key] = value
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            mutations.append(("row_%02d_%s" % (ordinal, key),
                              lambda bad=bad: M.verify_child_envelope(
                                  CONFIGS[0], bad, parent_pid, ticket)))

        streaming_mutations = (("materialized_transaction_list", True),
                               ("destinations", 0), ("timesteps", 9))
        for key, value in streaming_mutations:
            bad = copy.deepcopy(good); bad["result"]["streaming"][key] = value
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            mutations.append(("streaming_" + key,
                              lambda bad=bad: M.verify_child_envelope(
                                  CONFIGS[0], bad, parent_pid, ticket)))

        rss_mutations = (("gate_calls", 0), ("absolute_limit_kib", RSS_LIMIT - 1),
                         ("fresh_exec_required", False),
                         ("baseline_current_rss_kib", RSS_LIMIT),
                         ("baseline_peak_rss_kib", RSS_LIMIT),
                         ("max_current_rss_kib", RSS_LIMIT),
                         ("max_peak_rss_kib", RSS_LIMIT),
                         ("max_current_rss_kib", 99),
                         ("max_peak_rss_kib", 119))
        for key, value in rss_mutations:
            bad = copy.deepcopy(good); bad["result"]["m1573_rss"][key] = value
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            mutations.append(("rss_" + key + "_" + str(value),
                              lambda bad=bad: M.verify_child_envelope(
                                  CONFIGS[0], bad, parent_pid, ticket)))

        for name, function in mutations:
            require(rejected(function), "mutation accepted: " + name)
            counters["mutations_rejected"] += 1

    require(counters["actual_worker_calls"] == 0,
            "hammer reached actual M1583 worker")
    require(counters["mutations_rejected"] >= 60,
            "insufficient mutation population")
    return counters


def main(output):
    for path, expected in EXPECTED.items():
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) and not path.is_symlink(),
                "nonregular frozen identity: " + str(path))
        require(sha256(path) == expected, "frozen SHA drift: " + str(path))
    require(CONTRACT_INNER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT], CONTRACT.name], "contract inner seal drift")
    require(CONTRACT_OUTER.read_text(encoding="ascii").split() ==
            [EXPECTED[CONTRACT_INNER], CONTRACT_INNER.name],
            "contract outer seal drift")
    verify_sealed_tree(M1592,
        "PASS_M1592_M1583_SOURCE_ENGINEERING_QA__INDEPENDENT_PROCESS_RUNNER_SOURCE_AUTHORING_AUTHORIZED__ACTUAL_NOT_AUTHORIZED")
    verify_sealed_tree(AUTHOR,
        "PASS_AUTHOR_DUAL_RUNTIME_CONTROL_PLANE__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_ACTUAL")
    contract = strict_json_text(CONTRACT.read_text(encoding="utf-8"))
    verify_contract(contract)
    structure = source_structure(SOURCE.read_text(encoding="utf-8"))

    M = import_source()
    require(tuple(M.CONFIGS) == CONFIGS and M.FORBIDDEN_CONFIG == PRODUCT and
            M.RESOURCE_SHA256 == RESOURCE and M.RSS_LIMIT_KIB == RSS_LIMIT,
            "imported runner boundary drift")
    require(sha256(M.ENGINE) == EXPECTED[ENGINE] and
            M.M1592_REVIEW_SHA256 == EXPECTED[M1592 / "review.json"] and
            M.M1592_MANIFEST_SHA256 == EXPECTED[M1592 / "SHA256SUMS"] and
            M.M1592_OUTER_SHA256 == EXPECTED[M1592 / "SHA256SUMS.seal.sha256"],
            "M1592/M1583 imported identity drift")
    description = M.describe()
    require(description["execution"] == {"attempt_consumed": False,
            "child_processes": 0, "actual": False, "payload": False,
            "gpu": False, "eda": False}, "describe execution boundary drift")
    collisions = M.layout_collisions(M.PRODUCTION_LAYOUT)
    require(collisions == (), "production namespace already consumed")
    preflight = M.preflight(M.PRODUCTION_LAYOUT)
    require(preflight["attempt_consumed"] is False and
            preflight["child_processes"] == 0 and
            preflight["actual_execution"] is False,
            "read-only production preflight drift")
    require(M.layout_collisions(M.PRODUCTION_LAYOUT) == (),
            "preflight modified production namespace")

    counters = hammer(M)
    output_value = {
        "schema": "m1600_m1595_decoder_runner_independent_hammer_r1_v1",
        "status": "PASS_M1600_M1595_DIFFERENT_AUTHOR_HAMMER__ONE_D0_CALL0_THREE_CONFIG_PILOT_ACTUAL_AUTHORIZED__NOT_EXECUTED",
        "identity": {str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path): digest
                     for path, digest in EXPECTED.items()},
        "source_structure": structure,
        "synthetic_hammer": counters,
        "production_preflight": {"namespace_fresh": True,
                                 "attempt_consumed": False,
                                 "actual_execution": False,
                                 "payload_opened": False},
        "execution_by_m1600": {"actual": 0, "payload": 0, "gpu": 0,
                               "eda": 0, "fresh_real_children": 0,
                               "m1583_worker_calls": 0},
        "authorization": {
            "runner_execution": True,
            "actual_attempts": 1,
            "scope": {"decoder_stage": "D0", "call_ordinal": 0,
                      "module_ordinal": 0, "timesteps": 10,
                      "configurations_in_order": list(CONFIGS),
                      "fresh_child_processes": 3},
            "full_120_call": False, "product": False, "retry": False,
            "production": False, "gpu": False, "rtl": False, "eda": False,
            "execute_from_hammer": False,
            "independent_result_hammer_required": True,
        },
        "claim_boundary": {"source_hammer_only": True, "cycles": False,
                           "traffic": False, "speedup": False,
                           "system_speedup": False, "energy": False,
                           "paper_result": False},
    }
    Path(output).write_text(json.dumps(output_value, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.output)
