#!/usr/bin/env python3
"""Fresh independent bounded source hammer for M890 GTLS.

This hammer is intentionally unable to launch the full first row, the decoder
population, EDA, GPU, or remote work.  Its only dynamic workloads are bounded
synthetic and D0/A1/t0 prefixes explicitly authorized by M891.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict
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
SOURCE = HW / "system_simulator/scripts/analyze_m890_decoder_gtls_source_candidate.py"
TESTS = HW / "system_simulator/tests/test_m890_decoder_gtls_source_candidate.py"
CONTRACT = HW / "contracts/m890_decoder_gtls_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m890_decoder_gtls_source_candidate_r1_20260829.json"
HANDOFF_DIR = HW / "reviews/m890_decoder_gtls_source_author_handoff_r1_20260829"
REQUEST_DIR = HW / "reviews/m891_m890_decoder_gtls_source_fresh_hammer_REQUEST_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_FILES = {
    SOURCE: "cacc118ea33616ae4284403ad69656bbeacaa7bc83d227c0d9b5a86c2ead459e",
    TESTS: "8fc2689913cf7dca694b3faff395650563d2cf29458c46261077d0ac9032cdc8",
    CONTRACT: "470003362a7b39a8fea3232bd7df2daa8973793239694ffafae2bb9d71a8712e",
    CANDIDATE: "64503c51556998490dfedcaf7c917609aa6a21177e37c251a177bcad4c34651f",
    HANDOFF_DIR / "handoff.json": "e4634181fedbd016af9460e00a5124d2c87741d95cfaa8e1b90bcc16f5c4de18",
    REQUEST_DIR / "request.json": "fa37d1e1d6c73668844764b6bac76fbc019e9903f639068f29428e57df04e1d1",
}

EXPECTED_SEALED_DIRS = {
    HW / "reviews/m883_m868_m861_decoder_py310_full_first_row_diagnostic_result_hammer_r1_20260829": (
        "ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88",
        "review.json", "3cdd7be9cde8177e4cce6dfd16fc42dda5a84ba729757c92638eb242fe6fed0d",
        "4ddece71698ee0b83c18d039eb34205a0f2c93b4e5b95fd349f011686ab8d5a1"),
    HW / "reviews/m886_m883_decoder_scalable_exact_successor_first_principles_review_r1_20260829": (
        "009915ecc3524ba553edaef6c82cd615884db464440eef5a00e4df2531fc16b0",
        "review.json", "9089dc440cf152fcc7df879f7b754d094e6745dcc7a24f7b576ad430587191ea",
        "98f0adb69f41f07e578e4ed0f66d2db99b981b868359ea5f1cfa37801f7b5ad4"),
    HW / "reviews/m887_m886_decoder_gtls_source_author_handoff_r1_20260829": (
        "844ca9fe995f8a31242b17234a25373c10946a0d5597ce1875e534ebc3a6389b",
        "handoff.json", "37efafd72181105a35f4281ce9714995f9e88c4ac7bcb9f9fa1ae76f070df1fa",
        "d00f00f4cb9bece1878e99abd1d1c3843804baeb252d5b588632281623684c46"),
    HW / "reviews/m888_m887_m886_decoder_gtls_source_fresh_hammer_REQUEST_r1_20260829": (
        "ea2815b894a50831b93471ce78cf9291c2c30571831737ba51169c7dccf3b8e9",
        "request.json", "703f0945e2ae04c5860ca3e717a3df684d2ce96fc774af5825ff0bdba3b4ce17",
        "bbb0610f454eecb31a5315b7c0c02c259ea0e339657ce0295f89c8076963a137"),
    HANDOFF_DIR: (
        "e4634181fedbd016af9460e00a5124d2c87741d95cfaa8e1b90bcc16f5c4de18",
        "handoff.json", "a2ae05e237b7bcfd6b51e74c0c5b0d08edd31341f77b6dd23fdf1362b4dec090",
        "c75ce11a2aa027aa95c313a0455bc022b64004d576d2f19f4c2bf6585f7bc981"),
    REQUEST_DIR: (
        "fa37d1e1d6c73668844764b6bac76fbc019e9903f639068f29428e57df04e1d1",
        "request.json", "99a82ba0377869944b795d6ccdfd3868438aa686eb2e31c3f4af9d7f8536af96",
        "fe24c4d7aee9be778b045cbcbdeb0a3d0b2d76bc28e752a569fb04f8a3cd9f5b"),
}

DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


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


def strict_json_negative_attacks() -> Dict[str, bool]:
    rows: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m893_json_") as temporary:
        temporary_path = Path(temporary)
        for name, payload in (("duplicate", '{"a":1,"a":2}'),
                              ("nan", '{"a":NaN}'),
                              ("inf", '{"a":Infinity}')):
            path = temporary_path / (name + ".json")
            path.write_text(payload, encoding="utf-8")
            rejected = False
            try:
                strict_load(path)
            except HammerFailure:
                rejected = True
            require(rejected, name + " JSON attack accepted")
            rows[name + "_rejected"] = rejected
    return rows


def verify_sidecar(path: Path) -> Dict[str, str]:
    require(path.is_file() and not path.is_symlink(), "missing/symlink input: " + str(path))
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(sidecar.is_file() and outer.is_file(), "missing source sidecar: " + str(path))
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2 and fields[1] == path.name,
            "malformed source sidecar: " + str(sidecar))
    require(fields[0] == sha256(path), "source sidecar payload mismatch: " + str(path))
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(len(outer_fields) == 2 and outer_fields[1] == sidecar.name,
            "malformed source outer seal: " + str(outer))
    require(outer_fields[0] == sha256(sidecar), "source outer seal mismatch: " + str(path))
    return {"payload_sha256": fields[0], "sidecar_sha256": outer_fields[0]}


def verify_sealed_directory(directory: Path,
                            expected: Tuple[str, str, str, str]) -> Dict[str, Any]:
    primary_sha, primary_name, manifest_sha, outer_sha = expected
    require(directory.is_dir() and not directory.is_symlink(),
            "missing/symlink sealed directory: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha256(manifest) == manifest_sha, "manifest identity drift: " + directory.name)
    require(sha256(outer) == outer_sha, "outer seal-file identity drift: " + directory.name)
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(outer_fields == [manifest_sha, "SHA256SUMS"],
            "outer seal content mismatch: " + directory.name)
    listed: Dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        require(len(fields) == 2, "malformed manifest row: " + directory.name)
        digest, name = fields
        require(name not in listed and Path(name).name == name and "/" not in name and "\\" not in name,
                "duplicate/unsafe manifest name: " + name)
        payload = directory / name
        require(payload.is_file() and not payload.is_symlink(), "manifest payload missing/symlink: " + name)
        require(sha256(payload) == digest, "manifest payload drift: " + name)
        listed[name] = digest
        if payload.suffix == ".json":
            strict_load(payload)
    actual = {path.name for path in directory.iterdir() if path.is_file()} - {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"}
    require(actual == set(listed), "unsealed or stale artifact in: " + directory.name)
    require(listed.get(primary_name) == primary_sha,
            "primary identity drift: " + directory.name)
    return {"primary_sha256": primary_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha, "sealed_files": len(listed)}


def import_m890():
    spec = importlib.util.spec_from_file_location("m893_frozen_m890", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import frozen M890")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tx(m, *, name: str, kind: str, count: int,
            banks: Tuple[int, ...] = (0,), width: int = 16,
            dependency: Tuple[str, ...] = (), prefix: str | None = None,
            earliest: int = 0):
    if kind.startswith("weight"):
        base, stride = 0, width
    elif kind.startswith("psum"):
        base, stride = 0, width
    else:
        base, stride = 1 << 60, width
    return m.CompressedTransaction(
        transaction_id=name, population_id="M893_ATTACK",
        config="TYPED_SIGNED_K8", kind=kind, base_address=base,
        address_stride_bytes=stride, count=count, bank_pattern=banks,
        width_bytes=width, dependency_tokens=dependency,
        produces_token_prefix=prefix or (name + ":done"),
        earliest_issue_cycle=earliest)


def recurrence_issues(count: int, base: int, service: int,
                      distance: int, outstanding: int) -> List[int]:
    next_port = 0
    returns: List[int] = []
    issues: List[int] = []
    for _ in range(count):
        initial = max(base, next_port)
        occupied = [value for value in returns if value > initial]
        bound = initial
        if len(occupied) >= outstanding:
            bound = max(bound, sorted(occupied)[len(occupied) - outstanding])
        issue = max(initial, bound)
        issues.append(issue)
        next_port = issue + service
        returns = [value for value in returns if value > issue] + [issue + distance]
    return issues


def closed_form_attacks(m) -> Dict[str, Any]:
    exhaustive = 0
    for count in range(1, 49):
        for base in (0, 3, 19):
            for service in range(1, 6):
                for distance in range(0, 11):
                    for outstanding in range(1, 5):
                        got = m.GTLSScheduler._closed_form_issues(
                            count, base, service, distance, outstanding)
                        expected = recurrence_issues(
                            count, base, service, distance, outstanding)
                        require(got == expected, "closed form differs from recurrence")
                        exhaustive += 1

    resource = m.M861._synthetic_resource()
    probe = make_tx(m, name="eligible", kind="weight_read", count=8,
                    banks=(0,), width=16)
    scheduler = m.GTLSScheduler(resource)
    require(scheduler._can_closed_form(probe, "weight", "read"),
            "isolated homogeneous transaction not eligible")
    scheduler.next_port_cycle[("weight", 0, "read")] = 7
    require(not scheduler._can_closed_form(probe, "weight", "read"),
            "busy port incorrectly eligible")
    scheduler.next_port_cycle.clear()
    scheduler.outstanding_returns[("weight", 0)] = [9]
    require(not scheduler._can_closed_form(probe, "weight", "read"),
            "outstanding return incorrectly eligible")
    short = make_tx(m, name="short", kind="weight_read", count=3,
                    banks=(0,), width=16)
    scheduler.outstanding_returns.clear()
    require(not scheduler._can_closed_form(short, "weight", "read"),
            "short transaction incorrectly eligible")

    fast = m.exact_miter(m.synthetic_transactions(1000), include_old=True)
    slow = m.exact_miter(m.synthetic_transactions(1000), include_old=True,
                         force_fallback=True)
    require(fast["terminal_readiness_sha256"] == slow["terminal_readiness_sha256"],
            "fallback/closed terminal readiness mismatch")
    require(fast["compressed_group_ir_sha256"] == slow["compressed_group_ir_sha256"],
            "fallback changed packed IR")
    return {"exhaustive_parameter_tuples": exhaustive,
            "eligibility_busy_port_rejected": True,
            "eligibility_outstanding_rejected": True,
            "eligibility_short_rejected": True,
            "forced_fallback_exact": True}


def port_and_q_attacks(m) -> Dict[str, Any]:
    resource = m.M861._synthetic_resource()
    cases: List[Dict[str, Any]] = []
    kinds = (
        ("weight_read", resource.weight, 16),
        ("psum_read", resource.psum, 48),
        ("external_read", resource.external, 192),
        ("compute", resource.compute, 288),
    )
    for kind, port, row_bytes in kinds:
        q = int(port.outstanding_per_bank)
        for count in sorted({max(1, q - 1), q, q + 1}):
            for beats in (1, 2, 3):
                tx = make_tx(m, name="{}_q{}_b{}".format(kind, count, beats),
                             kind=kind, count=count, width=row_bytes * beats)
                exact = m.exact_miter([tx], include_old=True)
                require(exact["status"] == "PASS_EXACT_M768_M861_GTLS_MITER",
                        "q/beat endpoint miter failed")
                cases.append({"kind": kind, "q": q, "count": count,
                              "beats": beats})

    # The third read must reuse the response slot in the exact return cycle.
    tx = make_tx(m, name="same_cycle", kind="external_read", count=3,
                 width=resource.external.row_bytes)
    ir = m.PackedGroupIR([tx], ("M893_ATTACK", "TYPED_SIGNED_K8", 0, 0, 0))
    scheduled = m.GTLSScheduler(resource).schedule(
        ir, retain_details=True, retain_expanded_address_sha=True)
    rows = scheduled["scheduled_requests"]
    require(rows[2]["issue_cycle"] == rows[0]["return_cycle"],
            "same-cycle response slot was not reused")
    require(scheduled["same_cycle_response_slot_reuse"] is True,
            "same-cycle reuse contract drift")

    # Occupy only bank zero, then issue a two-bank transaction.  The second
    # group must use fallback and match both frozen schedulers.
    first = make_tx(m, name="bank0_warm", kind="weight_read", count=4,
                    banks=(0,), width=16)
    second = make_tx(m, name="bank01_attack", kind="weight_read", count=8,
                     banks=(0, 1), width=16)
    exact = m.exact_miter([first, second], include_old=True)
    require(exact["fallback_transactions"] >= 1,
            "asymmetric-bank state did not force fallback")
    return {"q_latency_beat_cases": len(cases),
            "same_cycle_response_reuse": True,
            "asymmetric_bank_fallback": True,
            "port_calendar_exact_on_every_miter": True}


def liveness_attacks(m) -> Dict[str, Any]:
    base = m.liveness_attack_self_test()
    require(base["premature_retirement_rejected"] and
            base["post_retirement_reuse_rejected"], "base liveness attack failed")

    producer = make_tx(m, name="producer", kind="compute", count=1,
                       width=288)
    terminal = m.terminal_token(producer)
    consumer_a = make_tx(m, name="consumer_a", kind="compute", count=2,
                         width=288, dependency=(terminal,))
    consumer_b = make_tx(m, name="consumer_b", kind="compute", count=3,
                         width=288, dependency=(terminal,))
    exact = m.exact_miter([producer, consumer_a, consumer_b], include_old=True)
    require(exact["live_token_peak"] == 1, "multi-consumer token lifetime drift")

    nonterminal_rejected = False
    bad = make_tx(m, name="bad_nonterminal", kind="compute", count=1,
                  width=288, dependency=(m.token_for(producer, 0),))
    producer_many = make_tx(m, name="producer_many", kind="compute", count=2,
                            width=288, prefix=producer.produces_token_prefix)
    try:
        m.PackedGroupIR([producer_many, bad],
                        ("M893_ATTACK", "TYPED_SIGNED_K8", 0, 0, 0))
    except m.Failure:
        nonterminal_rejected = True
    require(nonterminal_rejected, "nonterminal dependency accepted")

    duplicate_rejected = False
    duplicate = make_tx(m, name="duplicate", kind="compute", count=1,
                        width=288, prefix=producer.produces_token_prefix)
    try:
        m.PackedGroupIR([producer, duplicate],
                        ("M893_ATTACK", "TYPED_SIGNED_K8", 0, 0, 0))
    except m.Failure:
        duplicate_rejected = True
    require(duplicate_rejected, "duplicate token accepted")
    return {"premature_retirement_rejected": True,
            "post_retirement_reuse_rejected": True,
            "multi_consumer_long_lived_exact": True,
            "nonterminal_dependency_rejected": nonterminal_rejected,
            "duplicate_token_rejected": duplicate_rejected}


def packed_event_attacks(m) -> Dict[str, Any]:
    events = m.PackedPriorityEvents()
    events.issue.append(0)
    for name, start, end in (
            ("dependency_completion", 0, 4),
            ("weight_bank", 3, 6),
            ("psum_bank", 5, 8),
            ("memory", 7, 10)):
        events._interval(name, start, end)
    classes = events.finalize(12)
    expected = {"active_service": 1, "dependency_completion": 3,
                "weight_bank": 2, "psum_bank": 2, "memory": 2,
                "compute": 2}
    require(classes == expected, "six-class packed priority mismatch")
    require(sum(classes.values()) == 12, "six-class conservation mismatch")

    malformed = m.PackedPriorityEvents()
    malformed._interval("memory", 2, 1)
    require(malformed.finalize(3) == {
        "active_service": 0, "dependency_completion": 0,
        "weight_bank": 0, "psum_bank": 0, "memory": 0, "compute": 3},
        "empty/reversed interval changed packed-event timeline")

    endpoint_rejected = False
    try:
        m.PackedPriorityEvents().observe(2, 0, 1, 3, "compute")
    except m.Failure:
        endpoint_rejected = True
    require(endpoint_rejected, "invalid endpoint ordering accepted")
    return {"expected_cycle_classes": expected,
            "empty_or_reversed_interval_ignored": True,
            "endpoint_order_attack_rejected": endpoint_rejected}


def shard_and_digest_attacks(m) -> Dict[str, Any]:
    rows = m.synthetic_transactions(128)
    shards: Dict[str, List[int]] = {}
    for shard_count in (1, 2, 7, 13, 257):
        values = []
        for row in range(32):
            ir = m.PackedGroupIR(rows,
                                 ("M893", "TYPED_SIGNED_K8", row, 3, 9))
            first = ir.deterministic_shard(shard_count)
            second = ir.deterministic_shard(shard_count)
            require(first == second and 0 <= first < shard_count,
                    "deterministic shard drift")
            values.append(first)
        shards[str(shard_count)] = values
    rejected = False
    try:
        m.PackedGroupIR(rows, ("M893", "TYPED_SIGNED_K8", 0, 0, 0)).deterministic_shard(0)
    except m.Failure:
        rejected = True
    require(rejected, "invalid shard count accepted")

    ir = m.PackedGroupIR(rows, ("M893", "TYPED_SIGNED_K8", 0, 0, 0))
    result = m.GTLSScheduler(m.M861._synthetic_resource()).schedule(
        ir, retain_details=False, retain_expanded_address_sha=True)
    require(result["compressed_group_ir_sha256"] == ir.compressed_group_ir_sha256,
            "compressed IR digest identity drift")
    require(result["expanded_address_sha256"] == result["transaction_address_sha256"],
            "expanded address alias mismatch")
    require(result["compressed_group_ir_sha256"] != result["expanded_address_sha256"],
            "compressed/expanded hash domains collided")
    require(result["detail_retained"] is False and "scheduled_requests" not in result,
            "non-detail path retained expanded schedule")
    return {"shard_counts": sorted(int(key) for key in shards),
            "invalid_shard_rejected": rejected,
            "compressed_ir_sha256": result["compressed_group_ir_sha256"],
            "expanded_address_sha256": result["expanded_address_sha256"],
            "hash_domains_distinct": True,
            "detail_retained": False}


def bounded_miters(m) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    cases = (
        ("synthetic_1k", m.synthetic_transactions(1000), True),
        ("synthetic_10k", m.synthetic_transactions(10000), True),
        ("real_1k", m.real_prefix_transactions(1000), True),
        ("real_10k", m.real_prefix_transactions(10000), True),
        ("real_100k", m.real_prefix_transactions(100000), False),
    )
    for name, rows, include_old in cases:
        started = time.monotonic()
        result = m.exact_miter(rows, include_old=include_old)
        elapsed = time.monotonic() - started
        expected = ("PASS_EXACT_M768_M861_GTLS_MITER" if include_old else
                    "PASS_EXACT_M861_GTLS_MITER")
        require(result["status"] == expected, name + " status drift")
        require(result["fields"] == [
            "total_cycles", "expanded_request_count",
            "compressed_transaction_count", "scheduled_requests",
            "compressed_schedule", "transaction_address_sha256",
            "commit_sequence_sha256", "population_ids", "configs",
            "cycle_classes", "same_cycle_response_slot_reuse"],
            name + " exact field coverage drift")
        outputs[name] = dict(result)
        outputs[name]["hammer_elapsed_seconds"] = round(elapsed, 6)
    return outputs


def scaling_preflight() -> Dict[str, Any]:
    """Bounded GTLS-only 100K probe; never executes the full row."""
    child = r'''
import importlib.util, json, time
p = "system_simulator/scripts/analyze_m890_decoder_gtls_source_candidate.py"
s = importlib.util.spec_from_file_location("m893_scaling", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
t0 = time.monotonic(); rows = m.real_prefix_transactions(100000); t1 = time.monotonic()
ir = m.PackedGroupIR(rows, ("M686_ZURICH_CITY_09_A_S10", "A1_OSG", 0, 0, 0)); t2 = time.monotonic()
resource = m.M785.resource_from_contract(m.M785.strict_json(
    m.HW / "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"))
r = m.GTLSScheduler(resource).schedule(
    ir, retain_details=False, retain_expanded_address_sha=False); t3 = time.monotonic()
print(json.dumps({"rows_seconds": t1-t0, "ir_seconds": t2-t1,
                  "schedule_seconds": t3-t2, "total_seconds": t3-t0,
                  "expanded_requests": r["expanded_request_count"],
                  "compressed_transactions": r["compressed_transaction_count"],
                  "packed_event_bytes": r["packed_event_bytes"],
                  "packed_ir_bytes": r["packed_ir_bytes"],
                  "detail_retained": r["detail_retained"]},
                 sort_keys=True, allow_nan=False))
'''
    with tempfile.TemporaryDirectory(prefix="m893_scaling_") as temporary:
        time_file = Path(temporary) / "time.txt"
        completed = subprocess.run(
            ["/usr/bin/time", "-f", "%e %M", "-o", str(time_file),
             str(PYTHON), "-c", child], cwd=str(HW), text=True,
            capture_output=True, check=False, timeout=120)
        require(completed.returncode == 0,
                "bounded GTLS-only 100K probe failed: " + completed.stderr)
        measured = json.loads(completed.stdout)
        timing = time_file.read_text(encoding="utf-8").strip().split()
        require(len(timing) == 2, "bounded GTLS-only time receipt malformed")
    require(measured["expanded_requests"] == 100000 and
            measured["detail_retained"] is False,
            "bounded GTLS-only probe identity drift")
    full_requests = 38672612
    scale = full_requests / measured["expanded_requests"]
    projected_event_bytes = measured["packed_event_bytes"] * scale
    projected_ir_bytes = measured["packed_ir_bytes"] * scale
    peak_rss_kib = int(timing[1])
    require(peak_rss_kib > 512 * 1024,
            "expected current bounded source scaling red flag disappeared")
    require(projected_event_bytes > 512 * (1 << 20),
            "event projection unexpectedly falls below future memory gate")
    return {
        "scope": "BOUNDED_GTLS_ONLY_REAL_100K__NOT_FULL_ROW",
        "measurement": measured,
        "process_elapsed_seconds": float(timing[0]),
        "process_peak_rss_kib": peak_rss_kib,
        "future_full_row_requests": full_requests,
        "linear_projection_is_diagnostic_not_admission": True,
        "projected_packed_event_mib": projected_event_bytes / (1 << 20),
        "projected_packed_ir_mib": projected_ir_bytes / (1 << 20),
        "event_projection_over_512mib_gate": projected_event_bytes /
                                                     (512 * (1 << 20)),
        "current_100k_process_over_512mib_gate": peak_rss_kib / (512 * 1024),
        "verdict": "SCALING_PREFLIGHT_RED__REPAIR_INPUT_AND_PACKED_STATE_BEFORE_ANY_FULL_ROW_EXECUTION",
    }


def fail_closed_attacks(m, contract: Mapping[str, Any],
                        candidate: Mapping[str, Any],
                        handoff: Mapping[str, Any],
                        request: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("launch_now", "full_first_row", "full_population",
                "production", "decoder_complete", "cycles_or_speedup_citable",
                "paper_citable", "vcs_eda_gpu_remote"):
        require(contract.get(key) is False, "contract authority drift: " + key)
    for key in ("launch_now", "full_first_row", "full_population",
                "decoder_complete", "paper_citable", "table_a"):
        require(candidate["claims"].get(key) is False,
                "candidate authority drift: " + key)
    for key in ("production_cycles", "production_speedup", "accelerator_speedup"):
        require(candidate["claims"].get(key) is None,
                "candidate numeric claim drift: " + key)
    require(contract["future_full_row_gate"]["execution_authorized_now"] is False,
            "100x gate execution was authorized")
    require(contract["known_source_stage_limits"]["full_row_not_executed"] is True and
            contract["known_source_stage_limits"]["full_row_peak_rss_512_mib_not_proven"] is True,
            "full-row/512MiB source limits drift")
    require(request["future_gate"]["execution_authorized_by_this_request"] is False,
            "request authorized full row")
    require(handoff["future_gate"]["authorized_now"] is False and
            handoff["claim_boundary"]["full_first_row"] is False and
            handoff["claim_boundary"]["full_population"] is False,
            "handoff incorrectly claims full-row authority")

    refused: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m893_refuse_") as temporary:
        output = Path(temporary) / "forbidden.json"
        commands = {
            "full_first_row": [str(PYTHON), str(SOURCE), "--run-full-first-row"],
            "production": [str(PYTHON), str(SOURCE), "--run-production"],
            "publication": [str(PYTHON), str(SOURCE), "--output", str(output)],
        }
        for name, command in commands.items():
            completed = subprocess.run(command, text=True, capture_output=True,
                                       check=False, timeout=30)
            require(completed.returncode != 0, name + " forbidden mode accepted")
            refused[name] = True
        require(not output.exists(), "forbidden result publication created output")

    forbidden_results = []
    for root in (HW / "results", HW / "dc_handoff/runs"):
        if root.is_dir():
            forbidden_results.extend(str(path.relative_to(HW)) for path in
                                     root.glob("*m890*"))
    require(not forbidden_results,
            "M890 production/result artifact unexpectedly exists: " + repr(forbidden_results))
    source_text = SOURCE.read_text(encoding="utf-8")
    require("subprocess" not in source_text and "paramiko" not in source_text and
            "socket" not in source_text and "torch" not in source_text,
            "M890 source contains prohibited execution dependency")
    return {"forbidden_modes_refused": refused,
            "m890_result_or_eda_artifacts": forbidden_results,
            "full_row_not_executed": True,
            "full_row_100x_gate_passed": False,
            "full_row_peak_rss_512_mib_proven": False,
            "production": False, "decoder_complete": False,
            "cycles_or_speedup_citable": False}


def main() -> int:
    started = time.monotonic()
    checks: List[str] = []
    output: Dict[str, Any] = {
        "schema": "m893_m890_decoder_gtls_source_fresh_hammer_output_v1",
        "date": "2026-08-29",
        "status": "FAIL_CLOSED_PENDING",
    }
    try:
        require(sys.version_info[:2] == (3, 10), "hammer must run under Python 3.10")
        require(sha256(DOCS359) == DOCS359_SHA, "docs/359 identity drift before hammer")
        checks.append("docs359_pre")

        source_identities = {}
        for path, expected in EXPECTED_FILES.items():
            require(sha256(path) == expected, "required identity drift: " + str(path))
            if path in (SOURCE, TESTS, CONTRACT, CANDIDATE):
                source_identities[str(path.relative_to(HW))] = verify_sidecar(path)
            if path.suffix == ".json":
                strict_load(path)
        output["source_identities"] = source_identities
        checks.extend(["source_identity", "source_sidecars", "strict_json_inputs"])

        sealed = {}
        for directory, expected in EXPECTED_SEALED_DIRS.items():
            sealed[str(directory.relative_to(HW))] = verify_sealed_directory(directory, expected)
        output["sealed_authorities"] = sealed
        checks.extend(["m883_seal", "m886_seal", "m887_seal", "m888_seal",
                       "m890_handoff_seal", "m891_request_seal"])

        output["strict_json_attacks"] = strict_json_negative_attacks()
        checks.append("strict_json_negative")

        contract = strict_load(CONTRACT)
        candidate = strict_load(CANDIDATE)
        handoff = strict_load(HANDOFF_DIR / "handoff.json")
        request = strict_load(REQUEST_DIR / "request.json")

        py_compile.compile(str(SOURCE), doraise=True)
        py_compile.compile(str(TESTS), doraise=True)
        completed = subprocess.run(
            [str(PYTHON), "-m", "pytest", "-q", str(TESTS)],
            cwd=str(HW), text=True, capture_output=True, check=False,
            timeout=120)
        require(completed.returncode == 0 and "9 passed" in completed.stdout,
                "directed pytest failed: " + completed.stdout + completed.stderr)
        output["pytest"] = completed.stdout.strip()
        checks.extend(["py_compile", "pytest_9_of_9"])

        m = import_m890()
        validation = m.validate_source_candidate(CONTRACT)
        require(validation["status"] == "PASS_M890_SOURCE_IDENTITY_ONLY__NO_FULL_ROW",
                "source validator status drift")
        output["source_validation"] = validation
        checks.append("source_validator")

        output["bounded_miters"] = bounded_miters(m)
        checks.extend(["synthetic_1k_exact", "synthetic_10k_exact",
                       "real_1k_exact", "real_10k_exact", "real_100k_exact",
                       "endpoint_exact", "commit_digest_exact",
                       "terminal_readiness_exact", "port_calendars_exact",
                       "six_cycle_classes_exact"])

        output["scaling_preflight"] = scaling_preflight()
        checks.extend(["bounded_gtls_only_scaling_probe",
                       "packed_event_projection_red",
                       "bounded_100k_rss_over_future_gate",
                       "full_row_execution_still_forbidden"])

        output["closed_form_attacks"] = closed_form_attacks(m)
        checks.extend(["closed_form_exhaustive", "closed_form_eligibility",
                       "closed_form_fallback"])
        output["port_q_attacks"] = port_and_q_attacks(m)
        checks.extend(["q_minus_q_q_plus", "latency_beats", "asymmetric_bank",
                       "same_cycle_response_reuse"])
        output["liveness_attacks"] = liveness_attacks(m)
        checks.extend(["multi_consumer_liveness", "premature_retirement",
                       "post_retirement_reuse", "nonterminal_dependency",
                       "duplicate_token"])
        output["packed_event_attacks"] = packed_event_attacks(m)
        checks.extend(["packed_six_priority", "packed_conservation",
                       "packed_reversed_interval", "endpoint_order"])
        output["shard_digest_attacks"] = shard_and_digest_attacks(m)
        checks.extend(["deterministic_shards", "invalid_shard",
                       "hash_domain_separation", "no_detail_retention"])

        output["fail_closed"] = fail_closed_attacks(
            m, contract, candidate, handoff, request)
        checks.extend(["fail_closed_contract", "refuse_full_row",
                       "refuse_population", "refuse_publication",
                       "no_m890_production_artifacts", "no_eda_gpu_remote",
                       "future_100x_false", "future_512mib_false"])

        require(sha256(DOCS359) == DOCS359_SHA, "docs/359 identity drift after hammer")
        checks.append("docs359_post")
        require(len(checks) == len(set(checks)), "duplicate hammer check labels")
        output.update({
            "status": "PASS100_M890_GTLS_SOURCE_BOUNDED_EXACT__ONLY_INERT_FULLROW_RELEASE_AUTHOR_AUTHORIZED",
            "score": 100,
            "checks_passed": len(checks),
            "checks": checks,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "bounded_hammer_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "authority": {
                "may_authorize": "A different author may create one inert full-first-row 100x-gate release request.",
                "may_not_execute_full_row": True,
                "may_not_execute_population": True,
                "decoder_cycles_citable": False,
                "decoder_speedup_citable": False,
                "decoder_complete": False,
                "paper_citable": False,
            },
            "docs359_sha256": sha256(DOCS359),
        })
    except Exception as error:  # fail closed and preserve an auditable reason
        output.update({
            "status": "FAIL_CLOSED_M890_GTLS_SOURCE_HAMMER",
            "score": 0,
            "checks_passed": len(checks),
            "checks": checks,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "authority": {"may_authorize": None, "launch_now": False,
                          "full_first_row": False, "full_population": False,
                          "production": False, "paper_citable": False},
        })

    output_path = HERE / "independent_hammer_output.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0 if output["score"] == 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
