#!/usr/bin/env python3
"""Independent local-only hammer of M1573.

The hammer never opens the canonical decoder payload and never executes the
actual pilot.  Actual-entry attacks replace the frozen upstream call with a
small in-memory witness before invoking the M1573 host wrapper.
"""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1573_ep34_decoder_fresh_worker_gate_successor_source.py"
TEST = HW / "system_simulator/tests/test_m1573_ep34_decoder_fresh_worker_gate_successor_source.py"
CONTRACT = HW / "contracts/m1573_ep34_decoder_fresh_worker_gate_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1573_ep34_decoder_fresh_worker_gate_successor_author_r1_20260901"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
AUTHOR_COMMIT = "46209401b72ae52f0b1f46742aa8d37d5e84b9c6"

PINNED = {
    "source": "f26203424c4034230ee696ecf3b6d95685ed21647f41eb0c38b6961f0c83d02c",
    "test": "ad8f0f60f26dcb6ac3cf98d73193667fb290399c9440d3d0b76936c0e2211d6c",
    "contract": "6ab5397d50de8a3bc036856af87a40be78ce017829549c8eee7459f8ae152c41",
    "author_review": "fae205894bd8e76262e62875ad9af5572b7a0ddf1505aad3c3313cfd13a3c1a5",
    "author_manifest": "84175b5da1c170b64711e22114410413a5cecf154db23c7655d27cb656eb590a",
    "author_outer": "b0f06728321ec39fab714b6aa0543c7f9ba7799cbf6d7ae49c19b96f00ac457a",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


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
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(function, exceptions, label):
    try:
        function()
    except exceptions:
        return
    raise RuntimeError(label + " did not fail closed")


def run_command(arguments):
    process = subprocess.Popen(arguments, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def verify_author_seal():
    regular_exact(AUTHOR / "review.json", PINNED["author_review"],
                  "author review")
    regular_exact(AUTHOR / "SHA256SUMS", PINNED["author_manifest"],
                  "author manifest")
    regular_exact(AUTHOR / "SHA256SUMS.seal.sha256", PINNED["author_outer"],
                  "author outer seal")
    require((AUTHOR / "SHA256SUMS.seal.sha256").read_text(
                encoding="ascii").split() ==
            [PINNED["author_manifest"], "SHA256SUMS"],
            "author outer seal content drift")
    expected = {}
    for line in (AUTHOR / "SHA256SUMS").read_text(
            encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in expected and
                "/" not in fields[1] and ".." not in fields[1],
                "author manifest malformed")
        expected[fields[1]] = fields[0]
    for name, digest in expected.items():
        regular_exact(AUTHOR / name, digest, "author member " + name)
    actual = set(path.name for path in AUTHOR.iterdir()
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(actual == set(expected), "author seal coverage drift")
    return len(expected)


def mutate_projection_hammer(module, baseline):
    mutations = [
        ("configuration", "MUTATED_CONFIG"),
        ("resource_manifest_sha256", "0" * 64),
        ("total_cycles", int(baseline["results"][0]["total_cycles"]) + 1),
        ("request_count", int(baseline["results"][0]["request_count"]) + 1),
        ("transaction_address_sha256", "1" * 64),
        ("commit_sequence_sha256", "2" * 64),
    ]
    rejected = []
    original = module.U.synthetic_self_test
    original_current = module.current_rss_kib
    original_peak = module.peak_rss_kib
    try:
        module.current_rss_kib = lambda: 1024
        module.peak_rss_kib = lambda: 1024
        for key, replacement in mutations:
            before = copy.deepcopy(baseline)
            after = copy.deepcopy(baseline)
            after["results"][0][key] = replacement
            rows = [before, after]

            def sequenced(rows=rows):
                require(rows, "synthetic sequence exhausted")
                return rows.pop(0)

            module.U.synthetic_self_test = sequenced
            expect_reject(module.synthetic_self_test, module.M1573Error,
                          "projection mutation " + key)
            rejected.append(key)

        before = copy.deepcopy(baseline)
        after = copy.deepcopy(baseline)
        after["results"][0]["kind_counts"] = dict(
            after["results"][0]["kind_counts"])
        after["results"][0]["kind_counts"]["compute"] += 1
        rows = [before, after]
        module.U.synthetic_self_test = lambda: rows.pop(0)
        expect_reject(module.synthetic_self_test, module.M1573Error,
                      "kind-count mutation")
        rejected.append("kind_counts")

        before = copy.deepcopy(baseline)
        after = copy.deepcopy(baseline)
        after["results"][0]["byte_counts"] = dict(
            after["results"][0]["byte_counts"])
        after["results"][0]["byte_counts"]["external_read"] += 1
        rows = [before, after]
        module.U.synthetic_self_test = lambda: rows.pop(0)
        expect_reject(module.synthetic_self_test, module.M1573Error,
                      "byte-count mutation")
        rejected.append("byte_counts")

        before = copy.deepcopy(baseline)
        after = copy.deepcopy(baseline)
        after["results"] = after["results"][:-1]
        rows = [before, after]
        module.U.synthetic_self_test = lambda: rows.pop(0)
        expect_reject(module.synthetic_self_test, module.M1573Error,
                      "configuration population mutation")
        rejected.append("result_population")
    finally:
        module.U.synthetic_self_test = original
        module.current_rss_kib = original_current
        module.peak_rss_kib = original_peak
    return rejected


def wrapper_boundary_hammer(module):
    """Use a fake upstream result; never open or replay the actual payload."""
    original_stream = module.U.stream_actual_call
    original_current = module.current_rss_kib
    original_peak = module.peak_rss_kib
    calls = []
    try:
        module.current_rss_kib = lambda: 1024
        module.peak_rss_kib = lambda: 1024

        def fake_with_gate(config):
            calls.append(config)
            module.U.memory_gate()
            return {"configuration": config, "synthetic_witness": True,
                    "actual_payload_opened": False}

        module.U.stream_actual_call = fake_with_gate
        first = module.fresh_worker_entry(module.CONFIGS[0])
        second = module.fresh_worker_entry(module.CONFIGS[1])
        nonfresh_accepted = (
            calls == [module.CONFIGS[0], module.CONFIGS[1]] and
            first["m1573_rss"]["gate_calls"] == 1 and
            second["m1573_rss"]["gate_calls"] == 1 and
            first["fresh_exec_required"] is True and
            second["fresh_exec_required"] is True)

        def fake_without_gate(config):
            return {"configuration": config, "synthetic_witness": True,
                    "actual_payload_opened": False}

        module.U.stream_actual_call = fake_without_gate
        ungated = module.fresh_worker_entry(module.CONFIGS[2])
        zero_gate_accepted = (ungated["m1573_rss"]["gate_calls"] == 0 and
                              ungated["fresh_exec_required"] is True)

        def fake_mutated(_config):
            return {"configuration": module.FORBIDDEN_CONFIG,
                    "resource_manifest_sha256": "f" * 64,
                    "total_cycles": 1, "request_count": 1,
                    "kind_counts": {}, "byte_counts": {},
                    "transaction_address_sha256": "e" * 64,
                    "commit_sequence_sha256": "d" * 64,
                    "synthetic_witness": True,
                    "actual_payload_opened": False}

        module.U.stream_actual_call = fake_mutated
        mutated = module.fresh_worker_entry(module.CONFIGS[0])
        mutated_result_accepted = (
            mutated["configuration"] == module.FORBIDDEN_CONFIG and
            mutated["resource_manifest_sha256"] == "f" * 64)
    finally:
        module.U.stream_actual_call = original_stream
        module.current_rss_kib = original_current
        module.peak_rss_kib = original_peak
    return {"two_configurations_same_process_accepted": nonfresh_accepted,
            "zero_rss_gate_calls_accepted": zero_gate_accepted,
            "mutated_configuration_and_resource_result_accepted":
                mutated_result_accepted,
            "actual_payload_opened": False}


def main():
    regular_exact(SOURCE, PINNED["source"], "M1573 source")
    regular_exact(TEST, PINNED["test"], "M1573 test")
    regular_exact(CONTRACT, PINNED["contract"], "M1573 contract")
    regular_exact(DOC359, PINNED["docs359"], "docs/359")
    author_members = verify_author_seal()
    review = strict_json(AUTHOR / "review.json")
    contract = strict_json(CONTRACT)
    require(review.get("status") ==
            "PASS_SOURCE_AUTHORING__INDEPENDENT_HAMMER_REQUIRED__NO_ACTUAL_EXECUTION" and
            contract.get("status") ==
            "SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_ACTUAL_EXECUTION",
            "M1573 author/contract status drift")

    git_root = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]).decode(
            "ascii").strip()
    resolved = subprocess.check_output(
        ["git", "-C", git_root, "rev-parse", AUTHOR_COMMIT]).decode(
            "ascii").strip()
    require(resolved == AUTHOR_COMMIT, "M1573 author commit drift")
    relative_source = str(SOURCE.relative_to(Path(git_root)))
    committed_source = subprocess.check_output(
        ["git", "-C", git_root, "show", AUTHOR_COMMIT + ":" + relative_source])
    require(hashlib.sha256(committed_source).hexdigest() == PINNED["source"],
            "M1573 committed source byte drift")

    module = load(SOURCE, "m1577_bound_m1573")
    require(tuple(module.CONFIGS) == (
        "DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8") and
        module.FORBIDDEN_CONFIG == "PRODUCT_CAPTURE_TYPED_K8",
        "configuration boundary drift")

    author_return, author_stdout, author_stderr = run_command(
        [sys.executable, str(TEST)])
    require(author_return == 0 and
            "PASS M1573 tests=9 actual_execution=false" in author_stdout and
            author_stderr == "", "author test failed under hammer runtime")

    cli_attacks = []
    for arguments in (("--actual",), ("--pilot",), ("--retry-m1570",),
                      ("--config", module.FORBIDDEN_CONFIG),
                      ("--synthetic-self-test", "--verify-payload-members")):
        code, stdout, _stderr = run_command(
            [sys.executable, str(SOURCE)] + list(arguments))
        require(code != 0 and "total_cycles" not in stdout,
                "forbidden CLI accepted: " + " ".join(arguments))
        cli_attacks.append(" ".join(arguments))

    expect_reject(module.production_release, module.M1573Error,
                  "production release")
    original_stream = module.U.stream_actual_call
    module.U.stream_actual_call = lambda _config: (_ for _ in ()).throw(
        RuntimeError("forbidden product reached upstream"))
    try:
        expect_reject(lambda: module.fresh_worker_entry(module.FORBIDDEN_CONFIG),
                      module.M1573Error, "product configuration")
    finally:
        module.U.stream_actual_call = original_stream

    original_current = module.current_rss_kib
    original_peak = module.peak_rss_kib
    try:
        module.current_rss_kib = lambda: module.RSS_LIMIT_KIB
        module.peak_rss_kib = lambda: 1024
        gate = module.DualRssGate()
        expect_reject(gate, module.M1573Error, "current RSS equality")
        module.current_rss_kib = lambda: 1024
        module.peak_rss_kib = lambda: module.RSS_LIMIT_KIB
        gate = module.DualRssGate()
        expect_reject(gate, module.M1573Error, "peak RSS equality")
    finally:
        module.current_rss_kib = original_current
        module.peak_rss_kib = original_peak

    baseline = module.U.synthetic_self_test()
    require(len(baseline["results"]) == 3, "upstream synthetic population drift")
    clean = module.synthetic_self_test()
    require(clean["hardware_projection_exact"] is True and
            clean["actual_execution"] is False and
            clean["rss"]["gate_calls"] > 0,
            "clean synthetic exact miter failed")
    projection_rejections = mutate_projection_hammer(module, baseline)
    wrapper = wrapper_boundary_hammer(module)
    require(all(wrapper[key] is True for key in
                ("two_configurations_same_process_accepted",
                 "zero_rss_gate_calls_accepted",
                 "mutated_configuration_and_resource_result_accepted")),
            "P0 wrapper bypass proof drift")

    inputs = {"source": sha256(SOURCE), "test": sha256(TEST),
              "contract": sha256(CONTRACT),
              "author_review": sha256(AUTHOR / "review.json"),
              "author_manifest": sha256(AUTHOR / "SHA256SUMS"),
              "author_outer": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
              "docs359": sha256(DOC359)}
    result = {
        "schema": "m1577_m1573_decoder_fresh_worker_gate_independent_hammer_r1_v1",
        "status": "NO_GO_M1577_M1573_ONE_SHOT_RUNNER_AUTHORING__FRESHNESS_RSS_AND_RESULT_BINDING_NOT_ENFORCED",
        "runtime": {"executable": sys.executable,
                    "version": sys.version.split()[0]},
        "author_commit": AUTHOR_COMMIT,
        "pinned_inputs": inputs,
        "passed": {
            "author_seal_members": author_members,
            "author_test": True,
            "clean_synthetic_hardware_projection_exact": True,
            "projection_mutations_rejected": projection_rejections,
            "current_rss_equal_limit_rejected": True,
            "peak_rss_equal_limit_rejected": True,
            "forbidden_actual_cli_attacks": cli_attacks,
            "product_configuration_rejected_before_upstream": True,
            "production_release_rejected": True,
            "m1570_retry_cli_absent": True},
        "p0_findings": wrapper,
        "required_successor_fix": [
            "Bind the exact upstream actual-call function and expected hardware projection inside a clean-import closure instead of consulting mutable module attributes.",
            "Consume a private per-process one-shot token before the first worker entry so a second configuration in the same interpreter fails before upstream execution.",
            "Require at least one dual-RSS gate call and reject a result whose requested configuration, resource digest, or frozen hardware projection drifts.",
            "Repeat a fresh different-author hammer before any exactly-once runner or actual pilot authorization."],
        "authorization": {
            "successor_source_authoring": True,
            "one_shot_runner_authoring": False,
            "actual_execution": False,
            "m1570_retry": False,
            "production": False,
            "gpu": False, "eda": False, "rtl": False},
        "claim_boundary": {
            "synthetic_only": True, "actual_payload_opened": False,
            "actual_pilot_executed": False, "production_executed": False,
            "cycles": False, "traffic": False, "speedup": False,
            "system_speedup": False, "energy": False, "rtl": False,
            "eda": False, "ppa": False, "paper_result": False}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
