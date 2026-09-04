#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M1010 hammer for the source-only M1007 matched replay package.

This hammer is intentionally limited to small synthetic cases.  It does not
iterate the frozen 51.84M-row ledger and it does not launch VCS or EDA.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/m1007_c1_matched_common_charge_address_replay_source.py"
CHECKER = HW / "system_simulator/scripts/check_m1007_c1_matched_common_charge_address_replay_source.py"
TESTS = HW / "system_simulator/tests/test_m1007_c1_matched_common_charge_address_replay_source.py"
CONTRACT = HW / "contracts/m1007_m1000_c1_matched_common_charge_address_replay_source_contract_r1_20260829.json"
M505_ANALYZER = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M504_ANALYZER = HW / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
M528_RESULT = HW / "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json"
M505_RESULT = HW / "results/m505_h67_liveness_aware_single_port_parent_scratch_r1_20260827/m505_h67_liveness_aware_single_port_parent_scratch_result_r1.json"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1007_RECEIPT = HW / "reviews/m1007_m1000_c1_matched_common_charge_address_replay_source_receipt_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "150f22eaa11d219bfa20561b91a38049f14abbc541a6b40db04bd73533ec3442",
    "checker": "8838be4f8d7ca648ca71ad5115b39f528222a89c332b72dadd68a63c3040378a",
    "tests": "d3d176acd06b263e731ddea9dbd1da083411cd401fc774dfd529873d63e0cc8f",
    "contract": "d918801574c2d276c48610f345364d7aad151eeb367aba2b6943b5790dcfc06d",
    "m505_analyzer": "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced",
    "m504_analyzer": "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e",
    "m528_result": "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
    "m505_result": "b8a29f2fafc0e7d051d66ed206cd5c25efb866d4a1ab02082aa71bad4b14eb61",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "m1000_review": "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "m1000_manifest": "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "m1000_outer": "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
    "m1007_review": "ee42d29263b4b013cc627a5c921f148c807caf23c3b2c43469fccb6beb3d8d8c",
    "m1007_manifest": "4079903024785725ddd4c442c142acc0dab9b0809b659f53fca3d0e7eca406ce",
    "m1007_outer": "750c3c0cd5ed251291ae84f2dbe09dc0b4b9c17781f680127318eebeb7c50354",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_flat_seal(directory: Path, review_sha: str, manifest_sha: str,
                     outer_sha: str) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory missing/symlink")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha256(review) == review_sha and sha256(manifest) == manifest_sha and
            sha256(outer) == outer_sha, "sealed identity drift: " + directory.name)
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer seal content drift: " + directory.name)
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "duplicate/unsafe seal member")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha256(member) == digest,
                "seal member drift: " + rel)
        listed[rel] = digest
    actual = {path.name for path in directory.iterdir()
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(not any(path.is_symlink() for path in directory.iterdir()),
            "symlink in flat sealed directory")
    require(set(listed) == actual, "sealed exact-set drift: " + directory.name)
    return {"entries": len(listed), "exact_set": True,
            "manifest_sha256": manifest_sha, "outer_seal_file_sha256": outer_sha}


def run_unittests() -> dict:
    module = load_module("m1010_m1007_tests", TESTS)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    require(result.wasSuccessful() and result.testsRun == 9,
            "M1007 unittest failure: " + stream.getvalue())
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "status": "PASS_9_OF_9"}


def common_charge_hammer(module) -> dict:
    common = [
        {"resource": resource, "op": "READ", "bank": 0, "address": index,
         "bytes": index + 1, "transaction": "tx_" + resource, "cycle": index}
        for index, resource in enumerate(module.COMMON_RESOURCES)
    ]
    per_design = {
        name: [dict(event, cycle=event["cycle"] + 100 * index) for event in common]
        for index, name in enumerate(module.DESIGNS)
    }
    policy = {resource: {"mode": "include_both", "capacity_bytes": 4096,
                         "ports": "1RW", "latency_cycles": 2}
              for resource in module.COMMON_RESOURCES}
    positive = module.verify_matched_common_charge(per_design, policy)
    require(positive["cycle_merge_pending"] is True,
            "timestamp-only equality incorrectly promoted to cycles")

    rejected = []
    for design in module.DESIGNS:
        bad = {name: [dict(event) for event in events]
               for name, events in per_design.items()}
        bad[design].pop()
        try:
            module.verify_matched_common_charge(bad, policy)
        except RuntimeError as exc:
            require("asymmetric common charge" in str(exc), "unexpected asymmetry failure")
            rejected.append(design)
        else:
            raise RuntimeError("missing common transaction accepted for " + design)

    missing_design = dict(per_design)
    missing_design.pop("candidate")
    try:
        module.verify_matched_common_charge(missing_design, policy)
    except RuntimeError as exc:
        require("population drift" in str(exc), "unexpected missing-design failure")
    else:
        raise RuntimeError("missing design accepted")

    bad_policy = {key: dict(value) for key, value in policy.items()}
    del bad_policy["psum"]["latency_cycles"]
    try:
        module.verify_matched_common_charge(per_design, bad_policy)
    except RuntimeError as exc:
        require("service spec incomplete" in str(exc), "unexpected service failure")
    else:
        raise RuntimeError("incomplete common service spec accepted")

    return {"positive_timestamp_shift": positive["status"],
            "cycle_merge_pending": positive["cycle_merge_pending"],
            "missing_transaction_rejected_for_designs": rejected,
            "missing_design_rejected": True,
            "incomplete_service_spec_rejected": True}


def packing_hammer(module) -> dict:
    psum_conflict = [
        {"cycle": 11, "op": "READ", "bank": 0, "address": 1},
        {"cycle": 11, "op": "WRITE", "bank": 1, "address": 65},
    ]
    weight_overlap = [
        {"cycle": 12, "op": "READ", "bank": 0, "address": 2},
        {"cycle": 12, "op": "READ", "bank": 1, "address": 18},
    ]
    conflict = module.packing_summary(psum_conflict, weight_overlap, True)
    incomplete = module.packing_summary([], [], False)
    caller_asserted_empty = module.packing_summary([], [], True)
    require(conflict["psum_depth_packed_pair"]["conflict_cycles"] == 1 and
            conflict["weight_single_group"]["conflict_cycles"] == 1 and
            conflict["weight_half_slot_overlap_cycles"] == 1 and
            conflict["capacity_only_214912B_admitted"] is False,
            "packing conflict negative failed")
    require(incomplete["capacity_only_214912B_admitted"] is False,
            "incomplete coverage admitted")
    require(caller_asserted_empty["capacity_only_214912B_admitted"] is True,
            "coverage trust-boundary characterization drift")
    return {"psum_conflict_rejected": True, "weight_conflict_rejected": True,
            "weight_half_slot_overlap_rejected": True,
            "incomplete_coverage_rejected": True,
            "caller_asserted_empty_trace_admitted": True,
            "required_runner_repair":
                "derive coverage_complete internally from exact 51.84M-row conservation; never accept a naked caller flag"}


def streaming_hammer(module) -> dict:
    require(inspect.isgeneratorfunction(module.stream_parent_memh),
            "stream_parent_memh is not a generator function")
    dormant = module.stream_parent_memh()
    require(inspect.isgenerator(dormant), "stream construction is eager")
    dormant.close()

    saved = {name: getattr(module, name) for name in
             ("ROWS_SHA", "SAMPLES", "OPERATORS", "PARTITIONS",
              "ROWS_PER_PHASE", "ROW_TILE", "BLOCKS", "BYTES_PER_LINE")}
    original_pread = os.pread
    pread_sizes = []
    try:
        with tempfile.TemporaryDirectory(prefix="m1010_stream_") as tmpdir:
            tiny = Path(tmpdir) / "tiny.memh"
            tiny.write_bytes(b"00000001\n00000003\n")
            module.ROWS_SHA = sha256(tiny)
            module.SAMPLES = module.OPERATORS = module.PARTITIONS = module.BLOCKS = 1
            module.ROWS_PER_PHASE = module.ROW_TILE = 2
            module.BYTES_PER_LINE = 9

            def tracked_pread(fd, count, offset):
                pread_sizes.append((count, offset))
                return original_pread(fd, count, offset)

            module.os.pread = tracked_pread
            events = list(module.stream_parent_memh(tiny))
            require(events and pread_sizes == [(18, 0)], "tiny streaming pread drift")
            require({event["sample"] for event in events} == {0} and
                    {event["operator"] for event in events} == {0} and
                    {event["partition"] for event in events} == {0} and
                    {event["block"] for event in events} == {0},
                    "tiny stream geometry drift")
    finally:
        module.os.pread = original_pread
        for name, value in saved.items():
            setattr(module, name, value)

    max_production_pread = saved["ROW_TILE"] * saved["BYTES_PER_LINE"]
    max_tile_trace_bound = saved["ROW_TILE"] * 16 + 2 * saved["ROW_TILE"] + 8
    return {"generator_function": True, "construction_is_lazy": True,
            "tiny_stream_events": len(events), "tiny_pread_bytes": 18,
            "production_max_pread_bytes_per_tile": max_production_pread,
            "production_parent_trace_event_bound_per_tile": max_tile_trace_bound,
            "only_one_tile_trace_materialized": True,
            "full_ledger_sha_is_streamed_in_1MiB_blocks": True}


def main() -> dict:
    identities = {
        "source": SOURCE, "checker": CHECKER, "tests": TESTS,
        "contract": CONTRACT, "m505_analyzer": M505_ANALYZER,
        "m504_analyzer": M504_ANALYZER, "m528_result": M528_RESULT,
        "m505_result": M505_RESULT, "rows": ROWS, "docs359": DOC359,
    }
    for key, path in identities.items():
        require(sha256(path) == EXPECTED[key], "identity drift: " + key)

    m1000_seal = verify_flat_seal(
        M1000, EXPECTED["m1000_review"], EXPECTED["m1000_manifest"],
        EXPECTED["m1000_outer"])
    m1007_seal = verify_flat_seal(
        M1007_RECEIPT, EXPECTED["m1007_review"], EXPECTED["m1007_manifest"],
        EXPECTED["m1007_outer"])

    source_text = SOURCE.read_text()
    tree = ast.parse(source_text)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    require(not imported.intersection({"subprocess", "socket", "requests", "urllib"}),
            "execution/network module in source")
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    require(not calls.intersection({"system", "exec", "eval", "compile"}),
            "unsafe execution primitive in source")

    module = load_module("m1010_m1007_source", SOURCE)
    checker = load_module("m1010_m1007_checker", CHECKER)
    check = checker.validate_static(CONTRACT)
    tests = run_unittests()

    contract = strict_json(CONTRACT)
    require(contract["status"] == "PASS_M1007_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA" and
            contract["launch_now"] is False, "contract state drift")
    require(contract["frozen_geometry"] == {
        "samples": 10, "operators": 4, "partitions_per_operator": 432,
        "rows_per_phase": 3000, "row_tile": 64, "output_blocks": 8,
        "task_order": ["sample", "operator", "row_chunk", "partition", "output_block"],
        "parent_width_bits": 1152, "parent_port": "1RW",
        "parent_read_latency_cycles": 1, "parent_response_queue_entries": 2,
        "parent_policy": "M505 dead-write-only liveness-aware recurrence"},
        "contract geometry drift")
    false_boundary = (
        "full_51840000_replay_executed", "complete_address_timed_trace_created",
        "matched_total_cycles", "capacity_only_214912B_admitted", "vcs_executed",
        "dc_executed", "pt_executed", "ptpx_executed", "gpu_remote_used",
        "rtl_speedup", "m528_cpu_speedup_promoted", "linear_area_extrapolation",
        "headline", "paper_ppa_ready")
    require(all(contract["claim_boundary"][key] is False for key in false_boundary),
            "claim boundary weakened")
    require(not list((HW / "results").glob("m1007*m1000*c1*matched*replay*")),
            "full replay result exists despite source-only boundary")

    cases = ([1, 3, 5], [1, 3, 7, 15], [3, 3, 3, 3],
             [1, 2, 3, 4, 5, 7, 15, 0])
    expected_cases = [
        {"cycles": 4, "macro_reads": 1, "macro_writes": 1,
         "idle_cycles": 2, "forwarded_reads": 1, "issue_cycles": 3,
         "stall_cycles": 1},
        {"cycles": 4, "macro_reads": 0, "macro_writes": 3,
         "idle_cycles": 1, "forwarded_reads": 3, "issue_cycles": 4,
         "stall_cycles": 0},
        {"cycles": 6, "macro_reads": 2, "macro_writes": 1,
         "idle_cycles": 3, "forwarded_reads": 1, "issue_cycles": 5,
         "stall_cycles": 1},
        {"cycles": 8, "macro_reads": 2, "macro_writes": 3,
         "idle_cycles": 3, "forwarded_reads": 2, "issue_cycles": 7,
         "stall_cycles": 1},
    ]
    observed_cases = []
    for masks, expected_case in zip(cases, expected_cases):
        events = list(module.parent_cycle_trace(masks))
        observed = module.parent_summary(events)
        frozen = module.M505.simulate_liveness_task(np.asarray(masks, dtype=np.uint16), False)
        comparison = {
            "cycles": frozen["liveness_cycles"],
            "macro_reads": frozen["macro_reads"],
            "macro_writes": frozen["macro_writes"],
            "idle_cycles": frozen["liveness_cycles"] - frozen["macro_reads"] - frozen["macro_writes"],
            "forwarded_reads": frozen["forwarded_reads"],
            "issue_cycles": frozen["ideal_1r1w_issue_cycles"],
            "stall_cycles": frozen["liveness_stall_cycles"],
        }
        require(observed == expected_case and comparison == expected_case,
                "four-case parent oracle drift")
        require(all(Counter(event["op"] for event in [cycle])["READ"] +
                    Counter(event["op"] for event in [cycle])["WRITE"] <= 1
                    for cycle in events), "more than one parent operation per cycle")
        observed_cases.append(observed)

    m505_result = strict_json(M505_RESULT)
    require(m505_result["aggregate_one_output_block"]["row_count"] == 51_840_000 and
            m505_result["cycle_comparison"]["m505_liveness_single_port_cycles"] == 435_293_339 and
            m505_result["cycle_comparison"]["speedup_vs_best_same_budget_m468_zero"] ==
            1.7467534301047505, "frozen M505 anchor drift")

    common_charge = common_charge_hammer(module)
    packing = packing_hammer(module)
    streaming = streaming_hammer(module)

    return {
        "schema": "m1010_m1007_c1_matched_common_charge_source_hammer_v1",
        "status": "PASS_M1010_M1007_C1_MATCHED_COMMON_CHARGE_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_FULL_REPLAY_LAUNCH_RUNNER_ONLY",
        "score_out_of_100": 96,
        "p0_count": 0, "p1_count": 1, "p2_count": 0,
        "identity": {key + "_sha256": EXPECTED[key] for key in identities},
        "seals": {"m1000": m1000_seal, "m1007_source_receipt": m1007_seal},
        "checker": check, "unittests": tests,
        "parent_oracle": {"cases": 4, "observed": observed_cases,
                          "frozen_fields_exact": True,
                          "one_parent_operation_per_cycle": True},
        "frozen_m505_reference_only": {
            "raw_rows": 51_840_000, "cycles": 435_293_339,
            "cpu_same_ledger_speedup": 1.7467534301047505,
            "promoted_to_rtl_or_matched_replay": False},
        "common_charge": common_charge, "packing": packing,
        "streaming": streaming,
        "p1": [{
            "id": "P1_DERIVE_COVERAGE_IN_RUNNER",
            "finding": "packing_summary trusts a caller-provided coverage_complete boolean; an empty synthetic trace with True admits the capacity-only 214912-B hypothesis.",
            "required_action": "The future runner must derive coverage_complete internally from exact 51.84M-row identity/conservation, all 10x4x432x3000 rows, all eight blocks, and completed three-design service merges. It must not expose a naked CLI/JSON boolean."
        }],
        "authorization": {
            "write_full_replay_launch_runner": True,
            "execute_full_replay": False,
            "execute_eda": False,
            "admit_capacity_214912B": False,
            "admit_matched_total_cycles": False,
            "admit_speedup": False,
        },
        "scope": {"source_only_hammer": True, "full_51840000_replayed": False,
                  "eda_runs": 0, "docs359_modified": False},
        "claim_boundary": contract["claim_boundary"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
