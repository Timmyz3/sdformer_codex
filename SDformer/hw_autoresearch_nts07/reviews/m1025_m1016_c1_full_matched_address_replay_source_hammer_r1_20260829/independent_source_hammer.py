#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1025 hammer of the M1016 full matched-address replay source.

No production replay, attempt runner, EDA, or remote tool is invoked.  The only
subprocesses are the frozen ten small unit tests, engine self-test, source
checker, bash syntax checker, and a deliberately invalid CLI invocation.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tracemalloc

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
RUNNER = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay_one_shot.sh"
CHECKER = HW / "system_simulator/scripts/check_m1016_c1_full_matched_address_replay_source.py"
TESTS = HW / "system_simulator/tests/test_m1016_c1_full_matched_address_replay_source.py"
CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
M1010 = HW / "reviews/m1010_m1007_c1_matched_common_charge_address_replay_source_hammer_r1_20260829"
RECEIPT = HW / "reviews/m1016_m1010_c1_full_matched_address_replay_source_receipt_r1_20260829"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m1016_m1010_c1_full_matched_address_replay_r1_20260829"
ATTEMPT = HW / "results/.m1016_m1010_c1_full_matched_address_replay_attempt_consumed"

EXPECTED = {
    "engine": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "runner": "e11de2a48e87700aeb927a837c3fb50605bda3fa4020d24c58d702c6c622a54e",
    "checker": "f08f45ca10f41524dd8f4b7f679af11b0621ea3097dbe0d9c642d9f8259f06a3",
    "tests": "f2d92f41eda1bf5f74bc63fbdba3d6315e172ce8175e04542b238cc729c5759c",
    "contract": "b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90",
    "m1010_review": "c74812b03ca17b698ec5f80d086427937aea312668fd8d34df35544a930d669e",
    "m1010_manifest": "5bc8ea19bfb658cf737e227d632461a21096d5035efad8e88a20fc5cdb704e27",
    "m1010_outer": "4885bee6283a09551fa5f95088a01683ce2b561e9305a33365ad807bfeb618f7",
    "receipt_review": "4f81a5d765353a172e6aef0bcfcdcef4406cdb569123dba58ac61e76f290db7e",
    "receipt_manifest": "f5c1a7469bfd6653e8ceaf089ccb00b1f9412b76ec6258d6331a390e56874df2",
    "receipt_outer": "c1b365a72be1e21bb4f1354d345f4d6b0d0028ad88f51e55aedd1c1776edc148",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def verify_flat(directory: Path, expected: tuple[str, str, str]) -> dict:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) == expected,
            "sealed identity drift: " + directory.name)
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "sealed member drift: " + str(member))
        listed[rel] = digest
    require(outer.read_text().split() == [expected[1], "SHA256SUMS"],
            "outer content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
              "__pycache__" not in path.parts}
    require(set(listed) == actual, "sealed exact-set drift: " + directory.name)
    return load_json(review)


def load_engine():
    spec = importlib.util.spec_from_file_location("m1025_independent_m1016", ENGINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_complete(module):
    """Construct a proof state at the gate, without replaying the production ledger."""
    coverage = module.DerivedCoverage()
    coverage.seen_tiles = bytearray([1]) * module.TASKS
    coverage.phase_rows = np.full(module.PHASES, module.ROWS_PER_PHASE, dtype=np.int32)
    coverage.raw_rows = module.RAW_ROWS
    coverage.unique_tiles = module.TASKS
    coverage.ledger_sha256 = module.ROWS_SHA
    for name, state in coverage.designs.items():
        state.block_tasks = module.BLOCK_TASKS
        state.services = Counter(module.EXPECTED_SERVICE_COUNTS)
        state.service_digest.update(b"same frozen ordered service stream")
        state.service_merge_finished = True
        state.parent = Counter(module.EXPECTED_PARENT[name])
    require(coverage.proof()["raw_full_replay_complete"], "synthetic positive gate does not close")
    return coverage


def assert_incomplete(coverage, label: str) -> dict:
    try:
        proof = coverage.proof()
    except (KeyError, RuntimeError, ValueError):
        return {"fault": label, "rejected": True, "mode": "exception"}
    require(not proof["raw_full_replay_complete"], label + " derived complete")
    return {"fault": label, "rejected": True, "mode": "proof_false"}


def main() -> dict:
    for path, key in ((ENGINE, "engine"), (RUNNER, "runner"), (CHECKER, "checker"),
                      (TESTS, "tests"), (CONTRACT, "contract"), (ROWS, "rows"),
                      (DOC359, "docs359")):
        require(sha(path) == EXPECTED[key], key + " identity drift")
    m1010 = verify_flat(M1010, (EXPECTED["m1010_review"], EXPECTED["m1010_manifest"],
                               EXPECTED["m1010_outer"]))
    receipt = verify_flat(RECEIPT, (EXPECTED["receipt_review"], EXPECTED["receipt_manifest"],
                                   EXPECTED["receipt_outer"]))
    require(m1010["status"] == "PASS_M1010_M1007_C1_MATCHED_COMMON_CHARGE_SOURCE_HAMMER" and
            m1010["p0_count"] == 0 and m1010["p1_count"] == 1,
            "M1010 authority drift")
    require(receipt["status"] == "PASS_M1016_FULL_REPLAY_SOURCE_PACKAGE__NO_EXECUTION",
            "M1016 receipt status drift")
    contract = load_json(CONTRACT)
    identities = contract["source_identity"]
    require(identities["engine"]["sha256"] == EXPECTED["engine"] and
            identities["runner"]["sha256"] == EXPECTED["runner"] and
            identities["checker"]["sha256"] == EXPECTED["checker"] and
            identities["tests"]["sha256"] == EXPECTED["tests"],
            "contract source pins drift")
    source_tsv = (RECEIPT / "source_sha256.tsv").read_text()
    for key in ("engine", "runner", "checker", "tests", "contract", "rows", "docs359"):
        require(EXPECTED[key] in source_tsv, "receipt does not transitively seal " + key)

    subprocess.run(["bash", "-n", str(RUNNER)], check=True, timeout=10)
    tests_proc = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest",
                                 "-v", str(TESTS)], text=True, capture_output=True,
                                check=True, timeout=30)
    require("Ran 10 tests" in tests_proc.stderr + tests_proc.stdout and
            "OK" in tests_proc.stderr + tests_proc.stdout, "frozen 10/10 did not pass")
    selftest = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(ENGINE),
                               "--self-test"], text=True, capture_output=True,
                              check=True, timeout=30)
    require("PASS_M1016_SMALL_ORACLE__NO_FULL_REPLAY" in selftest.stdout,
            "engine self-test failed")
    checker = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(CHECKER),
                              "--contract", str(CONTRACT)], text=True, capture_output=True,
                             check=True, timeout=30)
    require("PASS_M1016_FULL_REPLAY_SOURCE_CHECK__NO_EXECUTION" in checker.stdout,
            "source checker failed")

    module = load_engine()
    faults: list[dict] = []
    empty = module.DerivedCoverage()
    faults.append(assert_incomplete(empty, "empty"))
    tiny = module.DerivedCoverage()
    index = tiny.observe_tile(0, 0, 0, 0, 64)
    receipt0 = module.common_receipt(index, 64)
    for design in module.DESIGNS:
        _, parent = module.parent_for_design(design, [1, 3, 5])
        tiny.observe_design(design, receipt0, parent)
    faults.append(assert_incomplete(tiny, "tiny"))

    duplicate = module.DerivedCoverage()
    duplicate.observe_tile(0, 0, 0, 0, 64)
    try:
        duplicate.observe_tile(0, 0, 0, 0, 64)
        raise RuntimeError("duplicate tile accepted")
    except RuntimeError as exc:
        require("duplicate frozen tile" in str(exc), "duplicate wrong rejection")
    faults.append({"fault": "duplicate_tile", "rejected": True, "mode": "exception"})

    truncated = synthetic_complete(module)
    truncated.raw_rows -= 1
    faults.append(assert_incomplete(truncated, "truncated_row"))
    missing_phase = synthetic_complete(module)
    missing_phase.phase_rows[0] -= 1
    faults.append(assert_incomplete(missing_phase, "missing_phase_row"))
    reordered_phase = synthetic_complete(module)
    reordered_phase.ledger_sha256 = "0" * 64
    faults.append(assert_incomplete(reordered_phase, "reordered_phase_changes_frozen_ledger_sha"))
    missing_block = synthetic_complete(module)
    missing_block.designs["candidate"].block_tasks -= module.BLOCKS
    faults.append(assert_incomplete(missing_block, "missing_block"))
    missing_design = synthetic_complete(module)
    del missing_design.designs["same_coordinate_bit"]
    faults.append(assert_incomplete(missing_design, "missing_design"))
    service_count = synthetic_complete(module)
    service_count.designs["candidate"].services["weight"] -= 1
    faults.append(assert_incomplete(service_count, "service_count_mismatch"))
    service_digest = synthetic_complete(module)
    service_digest.designs["candidate"].service_digest.update(b"reordered phase receipt")
    faults.append(assert_incomplete(service_digest, "service_digest_or_phase_order_mismatch"))
    parent = synthetic_complete(module)
    parent.designs["candidate"].parent["reads"] -= 1
    faults.append(assert_incomplete(parent, "parent_conservation_mismatch"))
    unfinished = synthetic_complete(module)
    unfinished.designs["candidate"].service_merge_finished = False
    faults.append(assert_incomplete(unfinished, "unfinished_service_merge"))

    # CLI cannot supply coverage; argparse must reject it without creating output.
    with tempfile.TemporaryDirectory(prefix="m1025_cli_") as td:
        out = Path(td) / "must_not_exist"
        cli = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(ENGINE),
                              "--self-test", "--coverage-complete", "--out", str(out)],
                             text=True, capture_output=True, check=False, timeout=20)
        require(cli.returncode != 0 and not out.exists(), "coverage CLI reached output")
        faults.append({"fault": "coverage_cli", "rejected": True, "mode": "argparse"})

    # Environment injection is ignored by the internally derived proof.
    env = os.environ.copy()
    env["COVERAGE_COMPLETE"] = "1"
    env_run = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(ENGINE),
                              "--self-test"], env=env, text=True, capture_output=True,
                             check=True, timeout=20)
    require("PASS_M1016_SMALL_ORACLE__NO_FULL_REPLAY" in env_run.stdout,
            "coverage environment changed self-test")
    faults.append({"fault": "coverage_environment", "rejected": True, "mode": "ignored"})

    # An unknown JSON flag can be parsed but has no data path into DerivedCoverage.
    with tempfile.TemporaryDirectory(prefix="m1025_json_") as td:
        injected = copy.deepcopy(contract)
        injected["coverage_complete"] = True
        path = Path(td) / "injected.json"
        path.write_text(json.dumps(injected))
        preflight = module.validate_source_only(path)
        require(preflight["full_replay_executed"] is False and
                not module.DerivedCoverage().proof()["raw_full_replay_complete"],
                "JSON coverage derived complete")
        duplicate_json = Path(td) / "duplicate.json"
        duplicate_json.write_text('{"status":"a","status":"b"}')
        try:
            module.strict_json(duplicate_json)
            raise RuntimeError("duplicate JSON accepted")
        except RuntimeError as exc:
            require("duplicate JSON key" in str(exc), "duplicate JSON wrong rejection")
    faults.append({"fault": "coverage_json_unknown_or_duplicate", "rejected": True,
                   "mode": "ignored_or_strict_reject"})

    # Production file-size and SHA guards precede coverage completion.  A tiny
    # replacement is rejected before any task loop or output directory creation.
    with tempfile.TemporaryDirectory(prefix="m1025_truncated_") as td:
        old_rows = module.ROWS
        tiny_rows = Path(td) / "truncated.memh"
        tiny_rows.write_text("00000000\n")
        out = Path(td) / "out"
        module.ROWS = tiny_rows
        try:
            try:
                module.run_full(CONTRACT, out)
                raise RuntimeError("truncated production ledger accepted")
            except RuntimeError as exc:
                require("M410 size drift" in str(exc), "truncated ledger wrong rejection")
            require(not out.exists(), "truncated ledger created output")
        finally:
            module.ROWS = old_rows
    faults.append({"fault": "truncated_m410_file", "rejected": True, "mode": "size_preflight"})

    run_source = inspect.getsource(module.run_full)
    require("os.pread(fd, count * BYTES_PER_LINE, offset)" in run_source and
            "read_bytes" not in run_source, "raw ledger is not pread-streamed")
    require(module.ROW_TILE * module.BYTES_PER_LINE == 576,
            "raw tile pread upper bound drift")
    require(inspect.isgeneratorfunction(module.iter_parent_address_events),
            "parent address event API is not generator")
    tracemalloc.start()
    event_count = sum(1 for _ in module.iter_parent_address_events([0xFFFF] * 64, 7, 0))
    _, parent_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    require(event_count > 0 and parent_peak < 2_000_000,
            "single-tile parent stream memory bound failed")

    service = contract["matched_common_service"]
    require("M528 aggregate traffic" in service["semantics"] and
            "not a measured physical SRAM/DRAM trace" in service["semantics"] and
            service["physical_memory_energy"] is False,
            "common service model boundary drift")
    require(contract["packing_audit"]["capacity_only_hypothesis_bytes"] == 214_912 and
            contract["admission_gates"]["capacity_only_214912B_admission_before_result_hammer"] is False and
            contract["admission_gates"]["speedup_admission_before_result_hammer"] is False,
            "capacity/speedup boundary drift")
    require(not RESULT.exists() and not ATTEMPT.exists(), "production namespace consumed")

    return {
        "schema": "m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_v1",
        "date": "2026-08-29",
        "milestone": "M1025",
        "status": "PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_EXECUTION_RELEASE_AND_EXACT_RUNNER_ONLY",
        "score_out_of_100": 98,
        "p0_count": 0,
        "p1_count": 1,
        "p2_count": 0,
        "identity": {
            "engine_sha256": sha(ENGINE), "runner_sha256": sha(RUNNER),
            "checker_sha256": sha(CHECKER), "tests_sha256": sha(TESTS),
            "contract_sha256": sha(CONTRACT),
            "m1010_outer_seal_file_sha256": sha(M1010 / "SHA256SUMS.seal.sha256"),
            "m1016_receipt_outer_seal_file_sha256": sha(RECEIPT / "SHA256SUMS.seal.sha256"),
            "m410_rows_sha256": sha(ROWS), "docs359_sha256": sha(DOC359),
        },
        "mechanical": {
            "frozen_unittests": "PASS_10_OF_10", "engine_selftest": "PASS",
            "source_checker": "PASS", "bash_n": "PASS",
            "fault_injections": faults,
            "fault_count": len(faults),
        },
        "streaming_memory_bound": {
            "raw_ledger_access": "os.pread per tile",
            "maximum_pread_bytes": 576,
            "parent_address_api_generator": True,
            "one_worst_pattern_tile_events": event_count,
            "one_worst_pattern_tile_peak_python_bytes": parent_peak,
            "full_ledger_materialized": False,
        },
        "model_boundary": {
            "common_service_model": "M528 aggregate-anchored same-coordinate cycle model",
            "physical_sram_dram_trace": False,
            "physical_memory_energy": False,
            "capacity_only_hypothesis_bytes": 214_912,
            "capacity_admitted": False,
            "speedup_admitted": False,
        },
        "p1": [{
            "id": "P1_SPECIALIZE_FUTURE_RELEASE_AUTHORITY",
            "finding": "The generic future one-shot accepts caller-selected release and hammer paths plus caller-supplied expected hashes. It is safe while unlaunched, but the execution successor must hardcode one additive release/hammer namespace and cross-bind their identities before consuming an attempt.",
            "required_action": "Author a new exact execution release and additive runner; independently hammer that chain. Do not execute the generic source runner directly.",
        }],
        "authorization": {
            "author_execution_release_and_exact_runner": True,
            "execute_51840000_replay": False,
            "execute_eda": False,
            "admit_capacity_214912B": False,
            "admit_cycles": False,
            "admit_speedup": False,
        },
        "scope": {
            "source_only_hammer": True, "full_51840000_replayed": False,
            "production_attempt_consumed": False, "eda_runs": 0,
            "docs359_modified": False,
        },
        "claim_boundary": {
            "source_ready": True, "full_result": False,
            "capacity_only_214912B_admitted": False,
            "matched_cycles": False, "speedup": False,
            "physical_memory_trace": False, "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    result = main()
    (HERE / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
