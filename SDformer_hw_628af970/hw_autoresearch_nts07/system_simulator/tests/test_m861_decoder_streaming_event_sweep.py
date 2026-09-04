#!/usr/bin/env python3
"""Bounded adversarial tests for the M861 streaming event sweep."""

import importlib.util
import itertools
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m861_decoder_streaming_event_sweep.py")
SPEC = importlib.util.spec_from_file_location("m861_event_sweep", SCRIPT)
M861 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M861)


def test_manual_e_d_i_r_priority_exposes_all_classes():
    result = M861.manual_endpoint_priority_miter()
    assert result["status"] == "PASS_MANUAL_E_D_I_R_PRIORITY_MITER"
    assert set(result["cycle_classes"]) == set(M861.CYCLE_CLASS_ORDER)
    assert all(value > 0 for value in result["cycle_classes"].values())


def test_interval_union_handles_touch_overlap_and_out_of_order_exactly():
    union = M861.IntervalUnion()
    for start, end in ((10, 12), (3, 5), (5, 9), (15, 16),
                       (8, 11), (20, 21), (16, 20)):
        union.add(start, end)
    assert union.intervals == ((3, 12), (15, 21))
    assert union.cardinality == 15
    assert union.out_of_order_insertions > 0


@pytest.mark.parametrize("seed,count", [(861, 1), (862, 64),
                                         (863, 512), (864, 2048)])
def test_random_dag_exact_old_new_all_frozen_fields(seed, count):
    result = M861.exact_old_new_miter(
        M861.deterministic_random_dag(count, seed))
    assert result["status"] == "PASS_EXACT_OLD_NEW_MITER"
    assert result["requests"] == count
    assert set(result["fields"]) == set(M861.M768_RESULT_FIELDS)


def test_1rw_1r1w_outstanding_and_same_cycle_return_slot_reuse():
    rows = [
        # Weight 1R1W: read/write are allowed to issue together.
        M861._request(0, "weight_read", earliest=0),
        M861._request(1, "weight_write", earliest=0),
        # Psum 1RW: read/write share the port and serialize.
        M861._request(2, "psum_read", earliest=0),
        M861._request(8, "psum_write", earliest=0),
        # External outstanding limit=2 and fixed latency exercise return-slot
        # retirement while later requests are considered.
        M861._request(4, "external_read", earliest=0),
        M861._request(5, "external_read", earliest=0),
        M861._request(6, "external_read", earliest=0),
    ]
    old_scheduler = M861.M785.AddressTimedScheduler(M861._synthetic_resource())
    new_scheduler = M861.StreamingAddressTimedScheduler(
        M861._synthetic_resource())
    old = old_scheduler.schedule(rows)
    new = new_scheduler.schedule(rows, retain_details=True)
    for field in M861.M768_RESULT_FIELDS:
        assert old[field] == new[field]
    by_id = {row["request_id"]: row for row in new["scheduled_requests"]}
    assert by_id["r0"]["issue_cycle"] == by_id["r1"]["issue_cycle"]
    assert by_id["r8"]["issue_cycle"] > by_id["r2"]["issue_cycle"]


def test_outstanding_one_reuses_slot_at_exact_return_cycle():
    resource = M861._synthetic_resource()
    resource = M861.M785.CommonResource(
        lanes=resource.lanes,
        accumulator_bits=resource.accumulator_bits,
        clock_ns=resource.clock_ns,
        external_bytes_per_cycle=resource.external_bytes_per_cycle,
        onchip_budget_bytes_macro_rounded=
            resource.onchip_budget_bytes_macro_rounded,
        macro_round_bytes=resource.macro_round_bytes,
        weight_bytes_logical=resource.weight_bytes_logical,
        psum_bytes_logical=resource.psum_bytes_logical,
        descriptor_control_bytes_logical=
            resource.descriptor_control_bytes_logical,
        reserved_unallocated_bytes=resource.reserved_unallocated_bytes,
        weight=resource.weight,
        psum=resource.psum,
        external=M861.M785.PortSpec(1, "1RW", 192, 2, 2, 1, 1),
        compute=resource.compute)
    rows = [M861._request(index, "external_read", earliest=0)
            for index in range(3)]
    old_scheduler = M861.M785.AddressTimedScheduler(resource)
    new_scheduler = M861.StreamingAddressTimedScheduler(resource)
    old = old_scheduler.schedule(rows)
    new = new_scheduler.schedule(rows, retain_details=True)
    assert old["scheduled_requests"] == new["scheduled_requests"]
    returned = old["scheduled_requests"][0]["return_cycle"]
    assert old["scheduled_requests"][1]["issue_cycle"] == returned
    assert new["same_cycle_response_slot_reuse"] is True


def test_streaming_summary_retains_no_expanded_or_compressed_rows():
    scheduler = M861.StreamingAddressTimedScheduler(
        M861._synthetic_resource())
    result = scheduler.schedule(M861.synthetic_prefix_requests(10000))
    assert result["expanded_request_count"] == 10000
    assert result["detail_retained"] is False
    assert "scheduled_requests" not in result
    assert "compressed_schedule" not in result
    assert sum(result["cycle_classes"].values()) == result["total_cycles"]


def test_compressed_count_matches_reference_even_when_endpoints_change():
    rows = [M861._request(index, "compute", earliest=index,
                         transaction="same_transaction")
            for index in range(128)]
    result = M861.exact_old_new_miter(rows)
    assert result["status"] == "PASS_EXACT_OLD_NEW_MITER"
    streamed = M861.StreamingAddressTimedScheduler(
        M861._synthetic_resource()).schedule(rows)
    assert streamed["compressed_transaction_count"] == 1


def test_bounded_real_prefix_exact_miter_and_streaming_summary():
    bounded = list(itertools.islice(M861.real_prefix_requests(512), 512))
    assert len(bounded) == 512
    assert M861.exact_old_new_miter(bounded)["status"] == \
        "PASS_EXACT_OLD_NEW_MITER"
    streamed = M861.StreamingAddressTimedScheduler(
        M861.M785.resource_from_contract(M861.M785.strict_json(
            ROOT / "contracts/"
            "m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json"
        ))).schedule(iter(bounded))
    assert streamed["expanded_request_count"] == 512
    assert streamed["detail_retained"] is False


def test_production_flag_is_fail_closed():
    completed = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SCRIPT),
         "--run-production"], text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "refuses production replay" in completed.stderr


def test_full_first_row_and_full_population_are_not_called(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unbounded replay called")

    monkeypatch.setattr(M861, "real_prefix_requests", forbidden)
    rows = M861.run_scale_prefixes((1000, 10000))
    assert [row["prefix_requests"] for row in rows] == [1000, 10000]


def test_docs359_and_m857_failure_authority_remain_pinned():
    assert M861._sha256(
        ROOT / "docs/359_DATE终局冻结_20260813.md") == \
        M861.DOCS359_SHA256
    identity = M861.M785.verify_sealed_directory(M861.M857_DIR)
    assert M861._sha256(M861.M857_DIR / "review.json") == \
        M861.M857_REVIEW_SHA256
    assert identity["manifest_sha256"] == M861.M857_MANIFEST_SHA256
    assert identity["outer_seal_file_sha256"] == \
        M861.M857_OUTER_SEAL_FILE_SHA256
