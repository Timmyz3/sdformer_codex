#!/usr/bin/env python3
"""Independent M2032 C1 calibration source hammer; no EDA or GPU work."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
TB = HW / "tb_m528_dw1rw/tb_m2031_ep34_c1_first64_model_rtl_calibration.sv"
FIXTURE = HW / "tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh"
SOURCE_AUDIT = HW / "system_simulator/scripts/check_m2031_ep34_c1_first64_model_rtl_calibration_source.py"
DUT = HW / "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
SCRATCH = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M1590 = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901"
LEDGER = M1590 / "ep34_c1_support16_rows.memh"
M1590_RESULT = M1590 / "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json"
M1597 = HW / "reviews/m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901"
M1597_REVIEW = M1597 / "review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
MACRO_V = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")

EXPECTED = {
    TB: "8cac9b384ce6812336d6961bc9ae50ca5a46e636ee8e74d2d49de40c0b4d74f1",
    FIXTURE: "4601182ca0dbba23d444de7d65cd2d7969159aa8564fd54a516a1934bf8112b3",
    SOURCE_AUDIT: "c3937a5d069f56cee3bd641eda0b78777acda8c15aae54e8650360e1105c485a",
    DUT: "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    SCRATCH: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    LEDGER: "daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835",
    M1590_RESULT: "facfecaf3b25a4c79299517de31283ed3815af26a5dd87c91a6985f6fc68516f",
    M1590 / "SHA256SUMS": "50881cd508bec486e6527ec483e451a1f03b7aba1fea7a047d54f1c1f5f08707",
    M1590 / "SHA256SUMS.seal.sha256": "9e7de8638deb0875ba7e2bd27c20859905fdbf441e8cce9759b32bb06b8b3127",
    M1597_REVIEW: "bfa3414ebb69d4a3022182ef7a4989d738c8370a855dff3ce5232c320623c33f",
    M1597 / "SHA256SUMS": "36dc79f7ca76bb98dfe1126aa05c7158dfc460d33215ee39d6fee4edd98e016c",
    M1597 / "SHA256SUMS.seal.sha256": "8f53a7fa74a2d0245448e822bc35b040df31b3e7d40d46d8ea739e6856d4df8b",
    MACRO_V: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_COUNTERS = {
    "rows": 64,
    "active_rows": 64,
    "input_nnz": 565,
    "residual_nnz": 192,
    "exact_parent_rows": 4,
    "issue_accepts": 196,
    "parent_edges": 58,
    "dead_write_elisions": 31,
    "macro_reads": 54,
    "macro_writes": 33,
    "forwards": 4,
    "deadline_holds": 6,
    "issue_stalls": 14,
    "liveness_cycles": 210,
    "psum_commits": 64,
    "row_completions": 64,
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def popcount(value):
    return bin(value & 0xffff).count("1")


def verify_seal(directory, rows_expected):
    assert directory.is_dir() and not directory.is_symlink()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    outer_digest, outer_name = outer.read_text().strip().split(maxsplit=1)
    assert outer_name.lstrip(" *") == manifest.name
    assert outer_digest == sha(manifest)
    rows = manifest.read_text().splitlines()
    assert len(rows) == rows_expected
    listed = set()
    for row in rows:
        digest, relative = row.split(maxsplit=1)
        relative = relative.lstrip(" *")
        assert relative not in listed
        listed.add(relative)
        target = directory / relative
        assert target.is_file() and not target.is_symlink()
        assert sha(target) == digest


def load_exact_prefix():
    fixture_lines = FIXTURE.read_bytes().splitlines(keepends=True)
    assert len(fixture_lines) == 64
    assert all(re.match(br"^0000[0-9a-f]{4}\n$", row) for row in fixture_lines)
    with LEDGER.open("rb") as stream:
        ledger_prefix = [stream.readline() for _ in range(64)]
    assert fixture_lines == ledger_prefix
    return [int(row.strip(), 16) & 0xffff for row in fixture_lines]


def independent_match(masks):
    residual = list(masks)
    parent = [-1] * len(masks)
    for row, current in enumerate(masks):
        if popcount(current) < 2:
            continue
        candidates = []
        for candidate, value in enumerate(masks):
            is_subset = (value & current) == value
            equal_at_current_or_later = value == current and candidate >= row
            if is_subset and not equal_at_current_or_later and popcount(value) >= 1:
                candidates.append(candidate)
        if candidates:
            # Maximum-popcount parent; earliest row breaks equal-popcount ties.
            chosen = max(candidates, key=lambda item: (popcount(masks[item]), -item))
            parent[row] = chosen
            residual[row] = current ^ masks[chosen]
    return residual, parent


def independent_dead_write_schedule(masks, residual, parent):
    active_order = sorted(
        [row for row, value in enumerate(masks) if value != 0],
        key=lambda row: (popcount(masks[row]), row))
    requirements = [parent[row] for row in active_order if parent[row] >= 0]
    consumers = [cursor for cursor, row in enumerate(active_order)
                 if parent[row] >= 0]
    use_count = [0] * len(masks)
    for required in requirements:
        use_count[required] += 1

    queue = []
    pending = None
    next_requirement = 0
    row_cursor = 0
    beat = 0
    issue_accepts = 0
    issue_stalls = 0
    macro_reads = 0
    macro_writes = 0
    forwards = 0
    deadline_holds = 0
    dead_write_elisions = 0
    cycles = 0
    written = [False] * len(masks)

    while row_cursor < len(active_order):
        row = active_order[row_cursor]
        parent_id = parent[row]
        work = popcount(residual[row])
        if parent_id >= 0 and work == 0:
            work = 1
        assert work > 0

        parent_ready = parent_id < 0 or bool(queue and queue[0] == parent_id)
        final_if_issued = bool(parent_ready and beat + 1 == work)
        reserved = len(queue) + int(pending is not None)
        assert reserved <= 2
        request_exists = next_requirement < len(requirements)
        requested_parent = requirements[next_requirement] if request_exists else -1
        requested_consumer = consumers[next_requirement] if request_exists else -1
        has_capacity = reserved < 2
        producer_ready = bool(request_exists and written[requested_parent])

        predicted_write = bool(final_if_issued and use_count[row] > 0)
        deadline_hold = bool(
            predicted_write and request_exists and has_capacity
            and producer_ready and requested_parent != row
            and requested_consumer == row_cursor + 1)
        issue = bool(parent_ready and not deadline_hold)
        last = bool(issue and beat + 1 == work)
        forward = bool(last and request_exists and has_capacity
                       and requested_parent == row)
        write = bool(last and use_count[row] > 0)
        read = bool(not write and not forward and request_exists
                    and has_capacity and written[requested_parent])
        assert not (read and write)

        deadline_holds += int(deadline_hold)
        issue_accepts += int(issue)
        issue_stalls += int(not issue)
        if last and parent_id >= 0:
            assert queue and queue[0] == parent_id
            queue.pop(0)
        if pending is not None:
            assert len(queue) < 2
            queue.append(pending)
        if forward:
            assert len(queue) < 2
            queue.append(requested_parent)
            next_requirement += 1
            forwards += 1
        if read:
            assert written[requested_parent]
            next_pending = requested_parent
            next_requirement += 1
            macro_reads += 1
        else:
            next_pending = None
        pending = next_pending

        if last:
            if write:
                written[row] = True
                macro_writes += 1
            elif use_count[row] == 0:
                dead_write_elisions += 1
            else:
                raise AssertionError("dead-write-only schedule elided live row")
        if issue:
            if last:
                row_cursor += 1
                beat = 0
            else:
                beat += 1
        cycles += 1
        assert cycles <= 1000

    assert next_requirement == len(requirements)
    assert pending is None and not queue
    return {
        "rows": len(masks),
        "active_rows": len(active_order),
        "input_nnz": sum(popcount(value) for value in masks),
        "residual_nnz": sum(popcount(value) for value in residual),
        "exact_parent_rows": sum(
            int(parent[row] >= 0 and residual[row] == 0 and masks[row] != 0)
            for row in range(len(masks))),
        "issue_accepts": issue_accepts,
        "parent_edges": len(requirements),
        "dead_write_elisions": dead_write_elisions,
        "macro_reads": macro_reads,
        "macro_writes": macro_writes,
        "forwards": forwards,
        "deadline_holds": deadline_holds,
        "issue_stalls": issue_stalls,
        "liveness_cycles": cycles,
        "psum_commits": len(active_order),
        "row_completions": len(active_order),
    }


def module_ports(text):
    header = text[text.index("module "):text.index(");", text.index("module "))]
    return re.findall(
        r"^\s*(?:input|output)\s+logic(?:\s+\[[^\]]+\])?\s+(\w+)\s*,?\s*$",
        header, re.M)


def verify_tb_static(tb, dut, scratch, macro):
    ports = module_ports(dut)
    assert len(ports) >= 50 and len(ports) == len(set(ports))
    declaration_prefix = tb[:tb.index(
        "m528_dead_write_only_1rw_product_capture_island_r2 dut (.*);")]
    for port in ports:
        assert re.search(
            r"\blogic(?:\s+\[[^\]]+\])?\s+[^;]*\b" +
            re.escape(port) + r"\b[^;]*;", declaration_prefix), port
    assert tb.count("m528_dead_write_only_1rw_product_capture_island_r2 dut (.*);") == 1
    assert not re.search(r"\b(force|release)\b", tb)
    assert tb.count("issue_data_valid = issue_request_valid;") == 1
    assert tb.count("issue_psum_prior = '0;") == 1
    assert "if (issue_request_source_valid)" in tb
    assert "while (!prep_ready) @(negedge clk_core);" in tb
    assert "prep_task_start = (row == 0);" in tb
    assert "prep_task_last = (row == 63);" in tb
    assert "prep_epoch = 16'd34;" in tb
    assert "if (psum_write_valid && psum_write_ready) begin" in tb
    assert "!row_complete_valid || !row_complete_ready" in tb
    assert "$signed(psum_write_data[lane*19 +: 19])" in tb
    assert "expected_lane(fixture_mask[psum_write_address], lane)" in tb
    assert "while (!(task_done_valid && task_done_epoch == 16'd34))" in tb
    final_wait = tb.index("while (!(task_done_valid && task_done_epoch == 16'd34))")
    sample = tb.index("if (count_issue_accepts != 64'd196")
    assert final_wait < tb.index("@(negedge clk_core);", final_wait) < sample
    for token in (
            "count_parent_edges != 64'd58",
            "count_dead_write_elisions != 64'd31",
            "count_macro_reads != 64'd54",
            "count_macro_writes != 64'd33",
            "count_forwards != 64'd4",
            "count_deadline_holds != 64'd6",
            "count_issue_stalls != 64'd14",
            "count_psum_commits != 64'd64",
            "count_row_completions != 64'd64",
            "commit_checks != 64",
            "rtl_cycle_speedup=false", "full_network=false",
            "system_speedup=false", "global watchdog expired"):
        assert token in tb
    assert tb.count("$finish;") == 1

    assert "m528_dw1rw_parent_scratch_9x128_macro u_parent_scratch" in dut
    assert "for (genvar slice = 0; slice < 9;" in scratch
    assert "TS1N28HPCPHVTB128X128M4S u_parent_sram" in scratch
    assert "module TS1N28HPCPHVTB128X128M4S (" in macro
    assert "`ifdef UNIT_DELAY" in macro
    assert "parameter SRAM_DELAY = 0.0100;" in macro
    assert "always #1.5 clk_core = ~clk_core;" in tb
    assert "$sdf_annotate" not in tb


def main():
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == digest, (path, sha(path), digest)
    verify_seal(M1590, 5)
    verify_seal(M1597, 5)
    m1597 = json.loads(M1597_REVIEW.read_text())
    assert m1597["status"] == "PASS_M1597_M1590_EP34_C1_RESULT_HAMMER_WITH_CAPACITY_SUPERSESSION"
    assert m1597["admission"]["rtl_cycle"] is False
    assert m1597["admission"]["system_speedup"] is False

    masks = load_exact_prefix()
    residual, parent = independent_match(masks)
    counters = independent_dead_write_schedule(masks, residual, parent)
    assert counters == EXPECTED_COUNTERS, counters
    verify_tb_static(TB.read_text(), DUT.read_text(), SCRATCH.read_text(),
                     MACRO_V.read_text(errors="replace"))

    audit = SOURCE_AUDIT.read_text()
    assert audit.startswith("#!/usr/bin/env python3\n")
    assert "from __future__ import annotations" in audit
    assert "import numpy as np" in audit
    assert "fixture_lines == ledger_prefix" in audit
    assert '"rtl_cycle_speedup": False' in audit
    assert '"system_speedup": False' in audit
    print("PASS_M2032_INDEPENDENT_SOURCE_HAMMER rows=64 prefix=1 issue=196 edges=58 dead=31 reads=54 writes=33 forwards=4 holds=6 stalls=14 cycles=210 p0=0 p1=1 p2=1")


if __name__ == "__main__":
    main()
