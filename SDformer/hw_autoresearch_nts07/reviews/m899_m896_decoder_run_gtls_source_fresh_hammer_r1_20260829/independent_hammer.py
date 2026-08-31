#!/usr/bin/env python3
"""Fresh independent bounded hammer for the M896 RUN-GTLS source.

The executable scope is deliberately limited to synthetic and sealed real
D0/A1/t0 prefixes of at most 100,000 expanded requests.  This hammer cannot
launch a full row, a production population, EDA, GPU, or remote work.
"""

from __future__ import annotations

from array import array
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import py_compile
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
SOURCE = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
TESTS = HW / "system_simulator/tests/test_m896_decoder_run_gtls_source_candidate.py"
CONTRACT = HW / "contracts/m896_decoder_run_gtls_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m896_decoder_run_gtls_source_candidate_r1_20260829.json"
M890_SOURCE = HW / "system_simulator/scripts/analyze_m890_decoder_gtls_source_candidate.py"
HANDOFF_DIR = HW / "reviews/m896_decoder_run_gtls_source_author_handoff_r1_20260829"
REQUEST_DIR = HW / "reviews/m897_m896_decoder_run_gtls_source_fresh_hammer_REQUEST_r1_20260829"
M893_DIR = HW / "reviews/m893_m890_decoder_gtls_source_fresh_hammer_r1_20260829"
M883_DIR = HW / "reviews/m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_FILES = {
    SOURCE: "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    TESTS: "12c1e092253ff078b52f7b5f7fcce9e17d4cb721e0f0d5aad2d75e86ca4d90eb",
    CONTRACT: "0f3f9faa31f2ab9b2221eec5b65030a37279c290266ae84f325ba4ea60e1780d",
    CANDIDATE: "8f70dcdc2445f31d4a90d65626978cdef03301de379d9cda8b541249ba7922fe",
    M890_SOURCE: "cacc118ea33616ae4284403ad69656bbeacaa7bc83d227c0d9b5a86c2ead459e",
    HANDOFF_DIR / "handoff.json": "073d02564e52c5f7193d59b5e7548128a5e2fcc0481735922e6521bcff283ff6",
    REQUEST_DIR / "request.json": "64959a2e565d4bc6afcf82e09343ab250ecafab9cfdfaf791b439b71a0f82268",
}

EXPECTED_SEALED_DIRS = {
    HANDOFF_DIR: (
        "073d02564e52c5f7193d59b5e7548128a5e2fcc0481735922e6521bcff283ff6",
        "handoff.json",
        "35352a5e15f8dad5454a5557123ae05564dadfe30fdcfcab8cd493b20bae2ee6",
        "1d301737bd1adadbe39518f0939f5665836514d4715c123aa76ab574084ad585"),
    REQUEST_DIR: (
        "64959a2e565d4bc6afcf82e09343ab250ecafab9cfdfaf791b439b71a0f82268",
        "request.json",
        "11f2291a58acbfdbd1473265527ba5666615875ad1c19b34b33c5363c7d415c6",
        "f2f287814fb7f48c432122bb1a671aa6456a14fa9c1c73ff124020c6bb208981"),
    M893_DIR: (
        "f883f68ca27aca654a558e2cb27ee3d9a56b490c4cba0e481523781ae4e7d102",
        "review.json",
        "8642b26197cfbdf7f71e47d22c2ad92e3586f1555d975dd3dcb938f13709ced9",
        "a21108afcea9b0ed2e85314c20878338835370151b41923019e990827addaf3b"),
    M883_DIR: (
        "ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88",
        "review.json",
        "3cdd7be9cde8177e4cce6dfd16fc42dda5a84ba729757c92638eb242fe6fed0d",
        "4ddece71698ee0b83c18d039eb34205a0f2c93b4e5b95fd349f011686ab8d5a1"),
}

DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
FULL_REQUESTS = 38672612
STATE_GATE_BYTES = 512 * 1024 * 1024


class HammerFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def pairs(rows: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in rows:
            if key in output:
                raise HammerFailure("duplicate JSON key: " + key)
            output[key] = value
        return output

    def constant(value: str) -> None:
        raise HammerFailure("nonfinite JSON constant: " + value)

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=constant)


def strict_json_attacks() -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for name, payload in (("duplicate", '{"a":1,"a":2}'),
                          ("nan", '{"a":NaN}'),
                          ("positive_infinity", '{"a":Infinity}'),
                          ("negative_infinity", '{"a":-Infinity}')):
        rejected = False
        try:
            json.loads(payload,
                       object_pairs_hook=lambda rows: _reject_duplicate(rows),
                       parse_constant=lambda value: _reject_nonfinite(value))
        except HammerFailure:
            rejected = True
        require(rejected, name + " JSON attack accepted")
        results[name + "_rejected"] = True
    return results


def _reject_duplicate(rows: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in rows:
        if key in output:
            raise HammerFailure("duplicate")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> None:
    raise HammerFailure("nonfinite: " + value)


def verify_sidecar(path: Path) -> Dict[str, str]:
    require(path.is_file() and not path.is_symlink(), "missing/symlink input: " + str(path))
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(), "missing sidecar: " + str(sidecar))
    require(outer.is_file() and not outer.is_symlink(), "missing outer sidecar: " + str(outer))
    row = sidecar.read_text(encoding="utf-8").strip().split()
    require(row == [sha256(path), path.name], "sidecar mismatch: " + path.name)
    outer_row = outer.read_text(encoding="utf-8").strip().split()
    require(outer_row == [sha256(sidecar), sidecar.name], "outer sidecar mismatch: " + path.name)
    return {"payload_sha256": row[0], "sidecar_sha256": outer_row[0]}


def verify_sealed_directory(directory: Path,
                            expected: Tuple[str, str, str, str]) -> Dict[str, Any]:
    primary_sha, primary_name, manifest_sha, outer_sha = expected
    require(directory.is_dir() and not directory.is_symlink(), "missing/symlink dir: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest missing/symlink")
    require(outer.is_file() and not outer.is_symlink(), "outer seal missing/symlink")
    require(sha256(manifest) == manifest_sha, "manifest identity drift: " + directory.name)
    require(sha256(outer) == outer_sha, "outer seal-file identity drift: " + directory.name)
    require(outer.read_text(encoding="utf-8").strip().split() ==
            [manifest_sha, "SHA256SUMS"], "outer seal content drift: " + directory.name)
    listed: Dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        require(len(fields) == 2, "malformed manifest row")
        digest, name = fields
        require(name not in listed and Path(name).name == name and
                "/" not in name and "\\" not in name, "unsafe/duplicate manifest name")
        payload = directory / name
        require(payload.is_file() and not payload.is_symlink(), "manifest payload missing/symlink")
        require(sha256(payload) == digest, "manifest payload drift: " + name)
        if payload.suffix == ".json":
            strict_load(payload)
        listed[name] = digest
    actual = {row.name for row in directory.iterdir() if row.is_file()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(actual == set(listed), "unsealed/stale artifact: " + directory.name)
    require(listed.get(primary_name) == primary_sha, "primary identity drift")
    return {"primary_sha256": primary_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha, "sealed_files": len(listed)}


def import_m896():
    spec = importlib.util.spec_from_file_location("m899_frozen_m896", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M896")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounded_miters(m) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    cases = (
        ("synthetic_1k", m.M890.synthetic_transactions(1000), True),
        ("synthetic_10k", m.M890.synthetic_transactions(10000), True),
        ("real_1k", m.M890.real_prefix_transactions(1000), True),
        ("real_10k", m.M890.real_prefix_transactions(10000), True),
        ("real_100k", m.M890.real_prefix_transactions(100000), False),
    )
    expected_fields = [
        "total_cycles", "expanded_request_count", "compressed_transaction_count",
        "scheduled_requests", "compressed_schedule", "transaction_address_sha256",
        "commit_sequence_sha256", "population_ids", "configs", "cycle_classes",
        "same_cycle_response_slot_reuse", "terminal_readiness",
        "terminal_readiness_sha256", "port_calendars"]
    for name, rows, include_old in cases:
        started = time.monotonic()
        result = m.exact_miter(rows, include_old=include_old)
        expected = ("PASS_EXACT_M768_M861_M890_RUN_GTLS_MITER" if include_old
                    else "PASS_EXACT_M861_M890_RUN_GTLS_MITER")
        require(result["status"] == expected, name + " status drift")
        require(result["fields"] == expected_fields, name + " exact-field coverage drift")
        require(result["expanded_requests"] in (1000, 10000, 100000), name + " count drift")
        outputs[name] = dict(result)
        outputs[name]["hammer_elapsed_seconds"] = round(time.monotonic() - started, 6)
    require(outputs["real_100k"]["terminal_readiness_sha256"] ==
            "a55d8cfa67f47863bc561323d01c674f1dd8d35555f3a972ab78d72bf44891ee",
            "sealed real 100K readiness digest drift")
    return outputs


def run_and_priority_attacks(m) -> Dict[str, Any]:
    runs = m.MaximalHalfOpenRuns()
    runs.add(10, 12)
    runs.add(1, 3)
    runs.add(3, 10)
    runs.add(15, 17)
    runs.add(16, 19)
    runs.add(2, 4)
    require(list(runs.iter_runs()) == [(1, 12), (15, 19)],
            "touching/non-touching maximal merge drift")
    require(runs.covered == 15 and runs.count == 2, "run coverage/count drift")
    empty = m.MaximalHalfOpenRuns()
    empty.add(5, 5)
    require(empty.count == 0, "empty run retained")
    invalid = False
    try:
        empty.add(7, 6)
    except m.Failure:
        invalid = True
    require(invalid, "invalid half-open run accepted")

    ap = m.MaximalHalfOpenRuns()
    ap.add_counted_progression(2, 1, 7)
    ap.add_counted_progression(20, 3, 4)
    require(list(ap.iter_runs()) == [(2, 9), (20, 21), (23, 24), (26, 27), (29, 30)],
            "counted AP expansion/run union drift")
    for start, step, count in ((0, 1, 1), (4, 1, 100), (3, 2, 19), (7, 11, 31)):
        got = m.MaximalHalfOpenRuns()
        got.add_counted_progression(start, step, count)
        expected = m.MaximalHalfOpenRuns()
        for index in range(count):
            point = start + index * step
            expected.add(point, point + 1)
        require(list(got.iter_runs()) == list(expected.iter_runs()), "counted AP exactness drift")

    # Independently rebuild the frozen recurrence, rather than comparing the
    # candidate closed form to a second spelling of its own expression.
    recurrence_cases = 0
    for count in range(1, 33):
        for base in (0, 3, 19):
            for service in range(1, 6):
                for distance in range(0, 11):
                    for outstanding in range(1, 5):
                        next_port = 0
                        returns: List[int] = []
                        expected_issues: List[int] = []
                        for _ in range(count):
                            initial = max(base, next_port)
                            occupied = [value for value in returns if value > initial]
                            bound = initial
                            if len(occupied) >= outstanding:
                                bound = max(bound, sorted(occupied)[len(occupied) - outstanding])
                            issue = max(initial, bound)
                            expected_issues.append(issue)
                            next_port = issue + service
                            returns = [value for value in returns if value > issue] + [issue + distance]
                        got_issues = [m.RUNGTLSScheduler._counted_ap_issue(
                            index, base, service, distance, outstanding)
                            for index in range(count)]
                        require(got_issues == expected_issues,
                                "counted AP issue recurrence drift")
                        recurrence_cases += 1

    rows = (
        (0, 0, 0, 3, "none"),
        (0, 3, 3, 5, "dependency_completion"),
        (0, 0, 7, 9, "weight_bank"),
        (2, 2, 8, 11, "psum_bank"),
        (1, 1, 12, 15, "memory"),
        (4, 4, 4, 4, "none"),
        (15, 15, 15, 16, "none"),
    )
    old = m.M890.PackedPriorityEvents()
    new = m.OnlinePriorityRuns()
    for row in rows:
        old.observe(*row)
        new.observe(*row)
    require(old.finalize(20) == new.finalize(20), "same-cycle priority reduction drift")
    require(sum(new.finalize(20).values()) == 20, "six-class conservation drift")
    return {
        "touching_and_nested_merge_exact": True,
        "non_touching_preserved": True,
        "invalid_run_rejected": invalid,
        "counted_ap_exact": True,
        "counted_ap_issue_recurrence_cases": recurrence_cases,
        "same_cycle_priority_exact": True,
        "cycle_classes": new.finalize(20),
        "maximal_runs": list(runs.iter_runs()),
    }


def liveness_shard_hash_attacks(m) -> Dict[str, Any]:
    base = m.liveness_attack_self_test()
    require(base["premature_retirement_rejected"] and base["post_retirement_rejected"],
            "liveness attack drift")
    producer = m.M890.synthetic_transactions(64)[0]
    bad = m.CompressedTransaction(
        transaction_id="bad_consumer", population_id="M890_SYNTHETIC",
        config="TYPED_SIGNED_K8", kind="compute", base_address=1 << 60,
        address_stride_bytes=0, count=1, bank_pattern=(0,), width_bytes=288,
        dependency_tokens=(m.M890.token_for(producer, 0),),
        produces_token_prefix="bad_consumer:done")
    nonterminal = False
    try:
        m.RunGroupIR([producer, bad], ("M890_SYNTHETIC", "TYPED_SIGNED_K8", 0, 0, 0))
    except m.Failure:
        nonterminal = True
    require(nonterminal, "nonterminal dependency accepted")

    rows = m.M890.synthetic_transactions(1000)
    shards: Dict[str, int] = {}
    for count in (1, 2, 7, 13, 257):
        ir = m.RunGroupIR(rows, ("M899", "TYPED_SIGNED_K8", 11, 3, 9))
        first = ir.deterministic_shard(count)
        second = ir.deterministic_shard(count)
        require(first == second and 0 <= first < count, "deterministic shard drift")
        shards[str(count)] = first
    rejected = False
    try:
        m.RunGroupIR(rows, ("M899", "TYPED_SIGNED_K8", 0, 0, 0)).deterministic_shard(0)
    except m.Failure:
        rejected = True
    require(rejected, "invalid shard accepted")

    ir = m.RunGroupIR(rows, ("M899", "TYPED_SIGNED_K8", 0, 0, 0))
    result = m.RUNGTLSScheduler(m.M861._synthetic_resource()).schedule(
        ir, retain_details=False, retain_expanded_address_sha=True)
    require(result["compressed_group_ir_sha256"] == ir.compressed_group_ir_sha256,
            "compressed IR hash drift")
    require(result["expanded_address_sha256"] == result["transaction_address_sha256"],
            "expanded address alias drift")
    require(result["compressed_group_ir_sha256"] != result["expanded_address_sha256"],
            "compressed/expanded hash domain collision")
    require(result["detail_retained"] is False and "scheduled_requests" not in result,
            "non-detail schedule retained expanded rows")
    second_schedule_rejected = False
    try:
        m.RUNGTLSScheduler(m.M861._synthetic_resource()).schedule(
            ir, retain_details=False, retain_expanded_address_sha=False)
    except m.Failure:
        second_schedule_rejected = True
    require(second_schedule_rejected, "one-shot liveness ledger was reused")
    return {
        "premature_retirement_rejected": True,
        "post_retirement_rejected": True,
        "nonterminal_dependency_rejected": nonterminal,
        "one_shot_reuse_rejected": second_schedule_rejected,
        "shards": shards,
        "invalid_shard_rejected": rejected,
        "compressed_ir_sha256": result["compressed_group_ir_sha256"],
        "expanded_address_sha256": result["expanded_address_sha256"],
        "hash_domains_distinct": True,
    }


def measured_state_probe() -> Dict[str, Any]:
    child = r'''
import importlib.util, json, time
p = "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
s = importlib.util.spec_from_file_location("m899_state", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
t = time.monotonic(); r = m.measure_real_100k_state(); elapsed = time.monotonic()-t
r["child_elapsed_seconds"] = elapsed
print(json.dumps(r, sort_keys=True, allow_nan=False))
'''
    with tempfile.TemporaryDirectory(prefix="m899_state_") as temporary:
        timing = Path(temporary) / "time.txt"
        completed = subprocess.run(
            ["/usr/bin/time", "-f", "%e %M", "-o", str(timing),
             str(PYTHON), "-c", child], cwd=str(HW), text=True,
            capture_output=True, check=False, timeout=120)
        require(completed.returncode == 0,
                "100K live-state child failed: " + completed.stdout + completed.stderr)
        measured = json.loads(completed.stdout)
        timing_fields = timing.read_text(encoding="utf-8").strip().split()
    require(len(timing_fields) == 2, "time receipt malformed")
    require(measured["status"] == "PASS_RUN_GTLS_100K_COMBINED_STATE_PROJECTION_GATE",
            "state gate status drift")
    state = int(measured["combined_live_event_state_bytes"])
    independent_projection = (state * FULL_REQUESTS + 100000 - 1) // 100000
    require(independent_projection == measured["conservative_projection_bytes"],
            "ceil projection mismatch")
    require(independent_projection <= STATE_GATE_BYTES, "512 MiB state gate failed")
    require(measured["serialized_or_compressed_file_size_used"] is False and
            measured["input_transaction_objects_excluded"] is True,
            "state metric provenance drift")
    return {
        "scope": "REAL_D0_A1_T0_100K_LIVE_IN_PROCESS_STATE__NOT_FULL_ROW",
        "measurement": measured,
        "independent_ceil_projection_bytes": independent_projection,
        "projection_margin_bytes": STATE_GATE_BYTES - independent_projection,
        "projection_margin_mib": (STATE_GATE_BYTES - independent_projection) / float(1 << 20),
        "state_gate_bytes": STATE_GATE_BYTES,
        "state_gate_passed": True,
        "child_process_elapsed_seconds": float(timing_fields[0]),
        "child_process_peak_rss_kib": int(timing_fields[1]),
        "rss_is_not_state_gate": True,
        "full_row_runtime_100x_gate_passed": False,
    }


def fail_closed(contract: Mapping[str, Any], candidate: Mapping[str, Any],
                handoff: Mapping[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("launch_now", "full_first_row", "full_population", "production",
                "decoder_complete", "cycles_or_speedup_citable", "paper_citable",
                "vcs_eda_gpu_remote"):
        require(contract.get(key) is False, "contract authority drift: " + key)
    for key in ("launch_now", "full_first_row", "full_row_100x_gate_passed",
                "full_population", "decoder_complete", "paper_citable", "table_a"):
        require(candidate["claims"].get(key) is False, "candidate claim drift: " + key)
    for key in ("production_cycles", "production_speedup", "accelerator_speedup"):
        require(candidate["claims"].get(key) is None, "candidate numeric claim drift: " + key)
    require(request["future_runtime_gate"]["execution_authorized_by_this_request"] is False,
            "request authorized future runtime gate")
    require(handoff["claim_boundary"]["full_first_row"] is False and
            handoff["claim_boundary"]["full_population"] is False,
            "handoff authorized full work")

    refused: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m899_refuse_") as temporary:
        target = Path(temporary) / "forbidden.json"
        commands = {
            "full_first_row": [str(PYTHON), str(SOURCE), "--run-full-first-row"],
            "production": [str(PYTHON), str(SOURCE), "--run-production"],
            "publication": [str(PYTHON), str(SOURCE), "--output", str(target)],
        }
        for name, command in commands.items():
            completed = subprocess.run(command, cwd=str(HW), text=True,
                                       capture_output=True, check=False, timeout=30)
            require(completed.returncode != 0, name + " forbidden mode accepted")
            refused[name] = True
        require(not target.exists(), "forbidden publication target created")
    forbidden = []
    for base in (HW / "results", HW / "dc_handoff/runs"):
        if base.is_dir():
            forbidden.extend(str(path.relative_to(HW)) for path in base.glob("*m896*"))
    require(not forbidden, "M896 full-row/EDA artifacts exist: " + repr(forbidden))
    text = SOURCE.read_text(encoding="utf-8")
    require(all(word not in text for word in ("subprocess", "paramiko", "torch", "socket")),
            "M896 source contains prohibited execution dependency")
    return {
        "forbidden_modes_refused": refused,
        "m896_result_or_eda_artifacts": forbidden,
        "full_first_row": False,
        "full_population": False,
        "production": False,
        "decoder_complete": False,
        "cycles_or_speedup_citable": False,
        "paper_citable": False,
        "vcs_eda_gpu_remote": False,
    }


def main() -> int:
    started = time.monotonic()
    checks: List[str] = []
    output: Dict[str, Any] = {
        "schema": "m899_m896_decoder_run_gtls_source_fresh_hammer_output_v1",
        "date": "2026-08-29",
        "status": "FAIL_CLOSED_PENDING",
    }
    try:
        require(sys.version_info[:2] == (3, 10), "hammer must run under Python 3.10")
        require(sha256(DOCS359) == DOCS359_SHA, "docs/359 drift before hammer")
        checks.append("docs359_pre")

        identities: Dict[str, Any] = {}
        for path, expected in EXPECTED_FILES.items():
            require(path.is_file() and not path.is_symlink(), "missing/symlink input: " + str(path))
            require(sha256(path) == expected, "required identity drift: " + str(path))
            if path in (SOURCE, TESTS, CONTRACT, CANDIDATE):
                identities[str(path.relative_to(HW))] = verify_sidecar(path)
            if path.suffix == ".json":
                strict_load(path)
        output["source_identities"] = identities
        checks.extend(["source_identity", "source_sidecars", "strict_json_inputs", "m890_source_identity"])

        seals: Dict[str, Any] = {}
        for directory, expected in EXPECTED_SEALED_DIRS.items():
            seals[str(directory.relative_to(HW))] = verify_sealed_directory(directory, expected)
        output["sealed_authorities"] = seals
        checks.extend(["m896_handoff_double_seal", "m897_request_double_seal",
                       "m893_authority_double_seal", "m883_authority_double_seal"])
        m893 = strict_load(M893_DIR / "review.json")
        # The tuple in M893 is [review, manifest, outer], not the local
        # verify_sealed_directory argument ordering.
        require(m893["upstream_authorities_recomputed"]["m883"] == [
            EXPECTED_SEALED_DIRS[M883_DIR][0], EXPECTED_SEALED_DIRS[M883_DIR][2],
            EXPECTED_SEALED_DIRS[M883_DIR][3]], "M893 upstream M883 binding drift")
        checks.append("m893_to_m883_binding")

        output["strict_json_attacks"] = strict_json_attacks()
        checks.append("duplicate_nonfinite_json_attacks")
        contract = strict_load(CONTRACT)
        candidate = strict_load(CANDIDATE)
        handoff = strict_load(HANDOFF_DIR / "handoff.json")
        request = strict_load(REQUEST_DIR / "request.json")

        py_compile.compile(str(SOURCE), doraise=True)
        py_compile.compile(str(TESTS), doraise=True)
        completed = subprocess.run(
            [str(PYTHON), "-m", "pytest", "-q", str(TESTS)], cwd=str(HW),
            text=True, capture_output=True, check=False, timeout=180)
        require(completed.returncode == 0 and "11 passed" in completed.stdout,
                "directed pytest failed: " + completed.stdout + completed.stderr)
        output["pytest"] = completed.stdout.strip()
        checks.extend(["py_compile", "pytest_11_of_11"])

        m = import_m896()
        validation = m.validate_source_candidate(CONTRACT)
        require(validation["status"] == "PASS_M896_SOURCE_IDENTITY_ONLY__NO_FULL_ROW",
                "source validator drift")
        output["source_validation"] = validation
        checks.append("source_validator")

        output["bounded_miters"] = bounded_miters(m)
        checks.extend(["synthetic_1k_exact", "synthetic_10k_exact", "real_1k_exact",
                       "real_10k_exact", "real_100k_exact", "every_endpoint_exact",
                       "expanded_address_hash_exact", "commit_hash_exact",
                       "terminal_readiness_exact", "port_calendars_exact",
                       "same_cycle_response_slot_exact", "six_cycle_classes_exact"])
        output["run_priority_attacks"] = run_and_priority_attacks(m)
        checks.extend(["maximal_run_merge", "touching_non_touching", "counted_ap",
                       "same_cycle_priority", "priority_conservation"])
        output["liveness_shard_hash_attacks"] = liveness_shard_hash_attacks(m)
        checks.extend(["premature_retirement", "post_retirement", "nonterminal_dependency",
                       "one_shot_liveness", "deterministic_shard", "invalid_shard",
                       "compressed_expanded_hash_domains", "no_detail_retention"])
        output["measured_state_probe"] = measured_state_probe()
        checks.extend(["real_100k_live_state", "independent_ceil_projection",
                       "state_projection_below_512mib", "state_rss_separated",
                       "runtime_100x_unmeasured"])
        output["fail_closed"] = fail_closed(contract, candidate, handoff, request)
        checks.extend(["fail_closed_contract", "refuse_full_row", "refuse_production",
                       "refuse_publication", "absence_result_eda", "absence_gpu_remote",
                       "full_population_false", "decoder_complete_false", "paper_citable_false"])

        require(sha256(DOCS359) == DOCS359_SHA, "docs/359 drift after hammer")
        checks.append("docs359_post")
        require(len(checks) == len(set(checks)), "duplicate check labels")
        output.update({
            "status": "PASS100_M896_RUN_GTLS_BOUNDED_EXACT__STATE_GATE_PASS__ONLY_FRESH_INERT_FULLROW_RELEASE_AUTHOR_AUTHORIZED",
            "score": 100,
            "checks_passed": len(checks),
            "checks": checks,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "hammer_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "authority": {
                "may_authorize": "A different author may create one inert full-first-row runtime-gate release request.",
                "full_row_execution": False,
                "full_population": False,
                "runtime_100x_gate_passed": False,
                "decoder_complete": False,
                "cycles_or_speedup_citable": False,
                "paper_citable": False,
                "vcs_eda_gpu_remote": False,
            },
            "docs359_sha256": sha256(DOCS359),
        })
    except Exception as error:
        output.update({
            "status": "FAIL_CLOSED_M896_RUN_GTLS_SOURCE_HAMMER",
            "score": 0,
            "checks_passed": len(checks),
            "checks": checks,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "authority": {"may_authorize": None, "full_row_execution": False,
                          "full_population": False, "production": False,
                          "paper_citable": False},
        })
    path = HERE / "independent_hammer_output.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0 if output["score"] == 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
