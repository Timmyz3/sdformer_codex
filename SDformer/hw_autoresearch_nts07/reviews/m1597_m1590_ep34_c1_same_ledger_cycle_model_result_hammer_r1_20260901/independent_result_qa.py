#!/usr/bin/env python3
"""Independent read-only result QA for the sealed M1590/M1579 production.

This checker never imports or runs the production replay.  It authenticates
the result/capture/release chain, scans the already-published 51.84M-row ledger
once for extent/format/SHA/popcount, and independently recomputes every CSV
ratio and distribution.  Adversarial copies exist only in memory.
"""

from decimal import Decimal, getcontext
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
RESULT_DIR = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901"
RESULT = RESULT_DIR / "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json"
LEDGER = RESULT_DIR / "ep34_c1_support16_rows.memh"
SAMPLE_CSV = RESULT_DIR / "sample_major_cycles.csv"
OPERATOR_CSV = RESULT_DIR / "operator_isolated_cycles.csv"
ATTEMPT = HW / "results/.m1590_ep34_c1_same_ledger_cycle_model_attempt.json"
RELEASE = HW / "contracts/m1590_ep34_c1_same_ledger_cycle_model_release_r1_20260901.json"
SOURCE = HW / "system_simulator/scripts/run_m1579_ep34_c1_same_ledger_cycle_model.py"
TEST = HW / "system_simulator/tests/test_m1579_ep34_c1_same_ledger_cycle_model.py"
M1524 = HW / "system_simulator/scripts/build_m1524_ep34_c1_same_ledger_rebind_source.py"
M528 = HW / "system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute.py"
M505 = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M504 = HW / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
CAPTURE = HW / "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831"
CAPTURE_MANIFEST = CAPTURE / "manifest.json"
ORDERED = CAPTURE / "unified_ordered_records.jsonl"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M1589 = HW / "reviews/m1589_m1579_regular_file_gate_final_incremental_qa_r1_20260901"
M1591 = HW / "results/m1591_c1_full_storage_macro_area_model_r1_20260901/m1591_c1_full_storage_macro_area_model_result_r1.json"
M1596 = HW / "reviews/m1596_m1591_c1_full_storage_macro_area_model_independent_review_r1_20260901/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    RESULT: "facfecaf3b25a4c79299517de31283ed3815af26a5dd87c91a6985f6fc68516f",
    LEDGER: "daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835",
    SAMPLE_CSV: "f49541f12761df9b0530807781df25ca9e3e9f3ac906bc9f2ac81f196e6f426b",
    OPERATOR_CSV: "7a5ec8c98fbde2f78e500b5912f2da1a43a24827492d7cf1357d2f4be1ed92c2",
    RESULT_DIR / "RUN_COMPLETE.txt": "bc341d4869f71210c3263bc9989d4d2119caed1c6b50f5fb363b446aa1ab20ef",
    RESULT_DIR / "SHA256SUMS": "50881cd508bec486e6527ec483e451a1f03b7aba1fea7a047d54f1c1f5f08707",
    RESULT_DIR / "SHA256SUMS.seal.sha256": "9e7de8638deb0875ba7e2bd27c20859905fdbf441e8cce9759b32bb06b8b3127",
    ATTEMPT: "27d71813dcd736e3dfe29deda02d86983cd6ecddd4fbfcf09ecf854d35f57e30",
    RELEASE: "569cff0a8da02d7b9f7056dd7db00391bf9c129662025f6147ee640c3ddcad64",
    SOURCE: "e0f09bd218af6733c17b50781ab9c3a4f13117821e24e14ea0eaa2864c1535b5",
    TEST: "c2cc102be14496b79bde2cee57a892bf76383c3180dba74563f78162ae4dec89",
    M1524: "a089650bad2e6acb338cb19a6ffea52bf4a823d6e32b6fb70ef3b101ed96e416",
    M528: "c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a",
    M505: "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced",
    M504: "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e",
    CAPTURE_MANIFEST: "3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d",
    ORDERED: "5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c",
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    CAPTURE / "SHA256SUMS": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    CAPTURE / "SHA256SUMS.seal.sha256": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    M1589 / "review.json": "bd2afbe1e926c1c011f568c1c762ee141db76e2cdc8e7aed838813c03d1b506c",
    M1589 / "SHA256SUMS": "e4fe97db06b8cc4a693b0bf0659d00992449ac4246aee86d7e418d5861a54eca",
    M1589 / "SHA256SUMS.seal.sha256": "e97965c747adc1de3c7b2262c4924b19d3b73ad244323bd08fa3b08805c1582f",
    M1591: "9b10348228780ca46b950cd4d603971d4d67bf9ba67412937a88c8eb4c8b3a2b",
    M1596: "d95f00333ecc8227eede483e875eb15ca2eaedf368a0edd9593cc71dba1ffe0e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

MODULES = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
CYCLE_KEYS = (
    "m468_strong_zero_cycles",
    "m473_same_coordinate_bit_cycles",
    "m473_fused_concurrent_1r1w_ceiling_cycles",
    "m504_all_write_1rw_cycles",
    "m505_dead_write_only_1rw_cycles",
    "m505_combined_pvrf_1rw_cycles",
)
RATIO_KEYS = (
    "speedup_vs_m468_strong_zero",
    "speedup_vs_m473_same_coordinate_bit",
    "port_tax_vs_m473_ceiling",
    "m504_to_dead_write_speedup",
    "dead_to_combined_speedup",
)
SAMPLE_HEADER = ("sample", "aggregation_semantics") + CYCLE_KEYS + RATIO_KEYS
OPERATOR_HEADER = ("sample", "operator", "module", "aggregation_semantics") + CYCLE_KEYS + RATIO_KEYS
PUBLICATION_LABEL = "[cycle model]"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regular(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)),
    )
    require(type(value) is dict, "JSON root is not object")
    return value


def strict_json(path):
    return strict_json_text(path.read_text(encoding="utf-8"))


def verify_flat_seal(directory, expected_members):
    require(directory.is_dir() and not directory.is_symlink(), "sealed dir drift")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), "seal files not regular")
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n", "outer seal content drift")
    listed = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "manifest row malformed")
        digest, name = fields
        relative = Path(name)
        require(name not in listed and not relative.is_absolute() and
                ".." not in relative.parts, "unsafe/duplicate manifest member")
        member = directory / relative
        require(regular(member) and sha256(member) == digest,
                "manifest member drift: " + name)
        listed[name] = digest
    require(set(listed) == set(expected_members), "manifest member set drift")
    actual = set(item.name for item in directory.iterdir() if item.is_file())
    require(actual == set(expected_members) |
            {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "result exact member set drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer),
            "members": sorted(listed)}


def verify_capture_bindings():
    manifest = CAPTURE / "SHA256SUMS"
    outer = CAPTURE / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n", "capture outer seal drift")
    entries = {}
    prefix = CAPTURE.relative_to(ROOT).as_posix() + "/"
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith(prefix):
            name = name[len(prefix):]
        require(name not in entries and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "capture manifest unsafe/duplicate")
        entries[name] = digest
    require(entries.get("manifest.json") == PINS[CAPTURE_MANIFEST] and
            entries.get("unified_ordered_records.jsonl") == PINS[ORDERED],
            "capture seal lacks required identity")
    capture = strict_json(CAPTURE_MANIFEST)
    require(capture["schema"] ==
            "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1" and
            capture["status"] ==
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "capture schema/status drift")
    selected = capture["identity"]["selection"]["selected"]
    require(selected["candidate_id"] == "resume_ep34" and
            selected["epoch"] == 34 and
            selected["checkpoint"]["sha256"] == PINS[CHECKPOINT],
            "capture checkpoint selection drift")
    require(capture["module_inventory"]["c1_conv3x3"] == list(MODULES),
            "capture C1 module inventory drift")

    retained = []
    with ORDERED.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("cohort") == "c1" and row.get("category") == "c1_conv3x3":
                retained.append(row)
    require(len(retained) == 40, "ordered C1 population drift")
    counts = [0, 0, 0, 0]
    for sample in range(10):
        group = retained[sample * 4:(sample + 1) * 4]
        require([row["name"] for row in group] == list(MODULES),
                "ordered operator sequence drift")
        for operator, row in enumerate(group):
            require(row["global_sample_id"] == sample and
                    row["payload"]["retained"] is True and
                    row["input"]["negative"] == 0 and
                    row["input"]["nonfinite"] == 0 and
                    row["input"]["active"] == row["input"]["positive"],
                    "ordered C1 record semantics drift")
            counts[operator] += row["input"]["active"]
    require(counts == [2908684, 1322881, 3299475, 1744577] and
            sum(counts) == 9275617, "ordered captured activity drift")
    return {"retained_c1_records": len(retained),
            "active_values_by_operator": counts,
            "active_values_total": sum(counts),
            "capture_manifest_members": len(entries)}


def scan_ledger():
    require(LEDGER.stat().st_size == 466560000, "ledger byte extent drift")
    pattern = re.compile(b"(?:0000[0-9a-f]{4}\\n)*\\Z")
    pop = [0] * 256
    for value, char in enumerate(b"0123456789abcdef"):
        pop[char] = bin(value).count("1")
    pop_table = bytes(pop)
    digest = hashlib.sha256()
    rows = 0
    active = 0
    with LEDGER.open("rb") as stream:
        while True:
            block = stream.read(9 * 262144)
            if not block:
                break
            require(len(block) % 9 == 0 and pattern.fullmatch(block) is not None,
                    "ledger fixed-row lowercase format drift")
            digest.update(block)
            rows += len(block) // 9
            active += sum(block[4::9].translate(pop_table))
            active += sum(block[5::9].translate(pop_table))
            active += sum(block[6::9].translate(pop_table))
            active += sum(block[7::9].translate(pop_table))
    require(rows == 51840000 and active == 78668732 and
            digest.hexdigest() == PINS[LEDGER], "ledger scan mismatch")
    return {"rows": rows, "bytes": LEDGER.stat().st_size,
            "sha256": digest.hexdigest(), "support16_popcount": active,
            "all_rows_exact_0000_lowerhex_newline": True}


def ratio_fields(row):
    zero = Decimal(str(row["m468_strong_zero_cycles"]))
    bit = Decimal(str(row["m473_same_coordinate_bit_cycles"]))
    fused = Decimal(str(row["m473_fused_concurrent_1r1w_ceiling_cycles"]))
    all_write = Decimal(str(row["m504_all_write_1rw_cycles"]))
    dead = Decimal(str(row["m505_dead_write_only_1rw_cycles"]))
    combined = Decimal(str(row["m505_combined_pvrf_1rw_cycles"]))
    return {
        "speedup_vs_m468_strong_zero": zero / dead,
        "speedup_vs_m473_same_coordinate_bit": bit / dead,
        "port_tax_vs_m473_ceiling": dead / fused - Decimal(1),
        "m504_to_dead_write_speedup": all_write / dead,
        "dead_to_combined_speedup": dead / combined,
    }


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def read_cycle_csv(path, expected_header, count):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == expected_header,
                "CSV header drift: " + path.name)
        rows = list(reader)
    require(len(rows) == count, "CSV population drift: " + path.name)
    parsed = []
    for row in rows:
        output = dict(row)
        for key in CYCLE_KEYS:
            require(row[key].isdigit() and int(row[key]) > 0, "invalid cycle field")
            output[key] = int(row[key])
        for key in RATIO_KEYS:
            value = float(row[key])
            require(math.isfinite(value) and value > 0, "invalid ratio field")
            output[key] = value
        expected = ratio_fields(output)
        require(all(close(output[key], expected[key]) for key in RATIO_KEYS),
                "row ratio arithmetic drift")
        parsed.append(output)
    return parsed


def describe(values):
    values = [float(item) for item in values]
    mean = statistics.fmean(values) if hasattr(statistics, "fmean") else sum(values) / len(values)
    return {
        "count": len(values),
        "arithmetic_mean": mean,
        "geometric_mean": math.exp(math.fsum(math.log(value) for value in values) /
                                   len(values)),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation_population": statistics.pstdev(values) / mean,
    }


def validate_describe(actual, expected):
    require(set(actual) == set(expected), "distribution schema drift")
    require(actual["count"] == expected["count"], "distribution count drift")
    for key in set(expected) - {"count"}:
        require(close(actual[key], expected[key]), "distribution statistic drift: " + key)


def validate_result(value, ledger_scan, capture_scan, samples, operators):
    require(set(value) == {
        "aggregate_cycles", "capacity", "claim_boundary", "conservation",
        "distribution", "identity", "ledger", "ratio_semantics", "schema",
        "scope", "status", "traffic",
    }, "result root schema drift")
    require(value["schema"] == "m1579_ep34_c1_same_ledger_cycle_model_r1_v1" and
            value["status"] == "PASS_M1579_EP34_C1_SAME_LEDGER_CYCLE_MODEL",
            "result schema/status drift")
    require(value["identity"] == {
        "checkpoint_sha256": PINS[CHECKPOINT],
        "capture_manifest_sha256": PINS[CAPTURE_MANIFEST],
        "ordered_records_sha256": PINS[ORDERED],
        "source_sha256": PINS[SOURCE],
        "release_sha256": PINS[RELEASE],
        "frozen_m1524_sha256": PINS[M1524],
        "frozen_m528_sha256": PINS[M528],
        "frozen_m505_sha256": PINS[M505],
        "frozen_m504_sha256": PINS[M504],
    }, "result identity map drift")
    require(value["scope"] == {
        "checkpoint": "Motion C12 ep34 live93",
        "samples": 10,
        "operators": list(MODULES),
        "operator_class": "four bottleneck Conv3x3 only",
        "sequence": "zurich_city_09_a",
        "cycle_model": True,
        "same_ledger_all_baselines": True,
    }, "scope drift")
    require(value["ratio_semantics"] ==
            "ratio_of_sums_over_ten_ep34_samples", "ratio semantics drift")
    require(value["ledger"] == {
        "path": LEDGER.name,
        "sha256": ledger_scan["sha256"],
        "bytes": 466560000,
        "rows": 51840000,
        "line_format": "0000<support16_lowercase_hex>\\n",
        "phase_order": "sample,operator,partition",
        "row_order": "timestep,output_y,output_x",
        "captured_input_active_values": capture_scan["active_values_total"],
        "captured_input_active_values_by_operator":
            capture_scan["active_values_by_operator"],
    }, "ledger identity/metadata drift")

    for index, row in enumerate(samples):
        require(int(row["sample"]) == index and row["aggregation_semantics"] ==
                "sample_major_four_operator_continuous_pipeline_plus_commit",
                "sample CSV order/semantics drift")
    for index, row in enumerate(operators):
        sample, operator = divmod(index, 4)
        require(int(row["sample"]) == sample and int(row["operator"]) == operator and
                row["module"] == MODULES[operator] and
                row["aggregation_semantics"] ==
                "operator_isolated_pipeline_no_commit_not_summable",
                "operator CSV order/semantics drift")

    totals = {key: sum(row[key] for row in samples) for key in CYCLE_KEYS}
    ratios = ratio_fields(totals)
    aggregate = value["aggregate_cycles"]
    require(all(aggregate[key] == totals[key] for key in CYCLE_KEYS),
            "aggregate differs from sample ratio-of-sums")
    require(all(close(aggregate[key], ratios[key]) for key in RATIO_KEYS),
            "aggregate ratio-of-sums drift")
    require(totals == {
        "m468_strong_zero_cycles": 648741051,
        "m473_same_coordinate_bit_cycles": 646619098,
        "m473_fused_concurrent_1r1w_ceiling_cycles": 341057992,
        "m504_all_write_1rw_cycles": 402449385,
        "m505_dead_write_only_1rw_cycles": 382848700,
        "m505_combined_pvrf_1rw_cycles": 382848700,
    }, "cycle anchor drift")

    for scope, rows in (("sample_major", samples),
                        ("operator_isolated", operators)):
        published = value["distribution"][scope]
        require(set(published) == {"cycles", "ratios"}, "distribution scope drift")
        for key in CYCLE_KEYS:
            validate_describe(published["cycles"][key],
                              describe([row[key] for row in rows]))
        for key in RATIO_KEYS:
            validate_describe(published["ratios"][key],
                              describe([row[key] for row in rows]))

    conservation = value["conservation"]
    require(conservation["source_rows"] == ledger_scan["rows"] and
            conservation["input_nonzero_bit_issues_per_output_block"] ==
            ledger_scan["support16_popcount"] and
            conservation["product_arithmetic_issues_per_output_block"] ==
            conservation["residual_nonzero_bit_issues_per_output_block"] +
            conservation["exact_parent_only_issues_per_output_block"] and
            conservation["product_arithmetic_issues_all_eight_output_blocks"] ==
            8 * conservation["product_arithmetic_issues_per_output_block"] and
            conservation["dead_reads_plus_forwards"] ==
            conservation["parent_edges_per_output_block"] and
            conservation["dead_writes_plus_elisions"] ==
            conservation["active_rows_per_output_block"] and
            conservation["all_equalities_pass"] is True,
            "conservation arithmetic drift")
    require(conservation == {
        "source_rows": 51840000,
        "input_nonzero_bit_issues_per_output_block": 78668732,
        "residual_nonzero_bit_issues_per_output_block": 36931884,
        "exact_parent_only_issues_per_output_block": 2386770,
        "product_arithmetic_issues_per_output_block": 39318654,
        "product_arithmetic_issues_all_eight_output_blocks": 314549232,
        "parent_edges_per_output_block": 16189026,
        "dead_reads_plus_forwards": 16189026,
        "active_rows_per_output_block": 25304213,
        "dead_writes_plus_elisions": 25304213,
        "all_equalities_pass": True,
    }, "conservation anchors drift")

    traffic = value["traffic"]
    scale = 8 * 144
    require(traffic["dead_write_only_parent_read_bytes_all_eight_blocks"] % scale == 0 and
            traffic["dead_write_only_parent_write_bytes_all_eight_blocks"] % scale == 0,
            "traffic vector scaling drift")
    dead_reads = traffic["dead_write_only_parent_read_bytes_all_eight_blocks"] // scale
    dead_writes = traffic["dead_write_only_parent_write_bytes_all_eight_blocks"] // scale
    require(dead_reads == 14506449 and dead_writes == 9070756 and
            dead_reads <= conservation["parent_edges_per_output_block"] and
            dead_writes <= conservation["active_rows_per_output_block"] and
            traffic["dead_write_only_parent_total_bytes_all_eight_blocks"] ==
            traffic["dead_write_only_parent_read_bytes_all_eight_blocks"] +
            traffic["dead_write_only_parent_write_bytes_all_eight_blocks"] and
            traffic["traffic_scope"] ==
            "parent scratch only; not total SRAM or DRAM traffic",
            "parent traffic arithmetic/scope drift")

    # The source result retains the old M528 213376-B capacity coordinate.
    # It is authenticated but superseded for publication by M1591/M1596.
    require(value["capacity"] == {
        "macro_rounded_bytes": 213376,
        "budget_bytes": 245760,
        "margin_bytes": 32384,
        "fits": True,
    }, "historical capacity field drift")
    current_capacity = strict_json(M1591)
    current_review = strict_json(M1596)
    require(current_capacity["logical_storage"]["total_bytes"] == 214912 and
            current_capacity["conservative_macro_rounding"]["represented_bytes"] == 215040 and
            current_capacity["logical_storage"]["budget_bytes"] == 245760 and
            current_review["publication_admission"]["required_exact_label"] ==
            "[macro area model]", "current capacity authority drift")

    require(value["claim_boundary"] == {
        "paper_citable_after_independent_result_hammer": False,
        "cycle_model": True,
        "cpu_replay": True,
        "rtl_cycle": False,
        "wall_clock": False,
        "full_network": False,
        "system_speedup": False,
        "energy": False,
        "power": False,
        "ppa": False,
        "multi_sequence": False,
        "external_official_simulator": False,
    }, "claim boundary drift")
    return totals, ratios, dead_reads, dead_writes


def validate_release_attempt():
    release = strict_json(RELEASE)
    require(release == {
        "schema": "m1579_ep34_c1_same_ledger_cycle_model_release_r1_v1",
        "status": "RELEASED_EXACTLY_ONE_CPU_CYCLE_MODEL__NO_EDA_NO_GPU",
        "source_sha256": PINS[SOURCE],
        "output": str(RESULT_DIR.resolve()),
        "ledger": str(LEDGER.resolve()),
        "attempt_marker": str(ATTEMPT.resolve()),
        "cpu_runs": 1, "gpu_runs": 0, "eda_runs": 0,
        "maximum_workers": 3,
        "frozen_inputs": {
            "m1524": PINS[M1524], "m528": PINS[M528],
            "m505": PINS[M505], "m504": PINS[M504],
            "docs359": PINS[DOCS359],
        },
        "claim_boundary": {
            "cycle_model": True, "four_bottleneck_conv_only": True,
            "final_ep34_checkpoint_bound": True,
            "same_ledger_zero_bit_product": True,
            "rtl_cycle": False, "wall_clock": False, "full_network": False,
            "system_speedup": False, "energy": False, "power": False,
            "ppa": False,
        },
    }, "release value drift")
    attempt = strict_json(ATTEMPT)
    require(attempt == {
        "schema": "m1579_ep34_c1_same_ledger_attempt_r1_v1",
        "status": "ATTEMPT_CONSUMED_BEFORE_LEDGER_MATERIALIZATION",
        "release_sha256": PINS[RELEASE],
        "source_sha256": PINS[SOURCE],
        "output": str(RESULT_DIR.resolve()),
    }, "attempt marker drift")
    require(stat.S_IMODE(ATTEMPT.lstat().st_mode) & 0o222 == 0,
            "attempt marker unexpectedly writable")
    attempts = []
    outputs = []
    for item in (HW / "results").iterdir():
        if item.name.startswith(".m1590_ep34_c1_same_ledger_cycle_model_attempt"):
            attempts.append(item.name)
        if item.name.startswith("m1590_ep34_c1_same_ledger_cycle_model_r1_20260901"):
            outputs.append(item.name)
    require(attempts == [ATTEMPT.name] and outputs == [RESULT_DIR.name],
            "attempt/result namespace not unique")
    m1589 = strict_json(M1589 / "review.json")
    require(m1589["authorization"]["exactly_one_cpu_production_execution"] is True and
            m1589["authorization"]["maximum_workers"] == 3 and
            m1589["passed"]["failure_keeps_attempt_consumed"] is True and
            m1589["passed"]["retry_after_failure_rejected"] == "FileExistsError" and
            m1589["passed"]["workers_above_three_rejected"] is True,
            "M1589 one-shot/worker authority drift")
    return {
        "attempt_namespaces": len(attempts),
        "result_namespaces": len(outputs),
        "attempt_consumed_before_materialization": True,
        "release_cpu_runs": 1,
        "release_maximum_workers": 3,
        "workers_above_three_rejected_by_frozen_source": True,
        "retry_after_attempt_rejected_by_frozen_source": True,
        "automatic_retry_artifact_observed": False,
        "actual_worker_count_recorded_in_result_or_attempt": False,
    }


def require_publication_label(label):
    require(label == PUBLICATION_LABEL, "missing/wrong cycle-model label")


def expect_result_reject(value, name, ledger_scan, capture_scan, samples, operators):
    try:
        validate_result(value, ledger_scan, capture_scan, samples, operators)
    except Exception:
        return name
    raise RuntimeError("mutation accepted: " + name)


def mutate(value, path, replacement):
    output = copy.deepcopy(value)
    cursor = output
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return output


def run():
    getcontext().prec = 40
    for path, digest in PINS.items():
        require(regular(path) and sha256(path) == digest, "frozen pin drift: " + str(path))
    result_seal = verify_flat_seal(RESULT_DIR, {
        LEDGER.name, RESULT.name, SAMPLE_CSV.name, OPERATOR_CSV.name,
        "RUN_COMPLETE.txt",
    })
    require((RESULT_DIR / "RUN_COMPLETE.txt").read_text(encoding="ascii") ==
            "PASS_M1579_EP34_C1_SAME_LEDGER_CYCLE_MODEL\n",
            "RUN_COMPLETE token drift")
    # M1589 historical flat seal binds the authority files listed by it.
    require((M1589 / "SHA256SUMS.seal.sha256").read_text(encoding="ascii") ==
            sha256(M1589 / "SHA256SUMS") + "  SHA256SUMS\n",
            "M1589 outer seal drift")

    capture_scan = verify_capture_bindings()
    ledger_scan = scan_ledger()
    samples = read_cycle_csv(SAMPLE_CSV, SAMPLE_HEADER, 10)
    operators = read_cycle_csv(OPERATOR_CSV, OPERATOR_HEADER, 40)
    value = strict_json(RESULT)
    totals, ratios, dead_reads, dead_writes = validate_result(
        value, ledger_scan, capture_scan, samples, operators)
    attempt = validate_release_attempt()

    attacks = []
    specs = [
        (("aggregate_cycles", "m505_dead_write_only_1rw_cycles"), 382848699,
         "candidate_cycle_mutation"),
        (("aggregate_cycles", "speedup_vs_m468_strong_zero"), 2.0,
         "headline_ratio_forgery"),
        (("ledger", "rows"), 51839999, "ledger_row_forgery"),
        (("scope", "sequence"), "multi", "sequence_scope_forgery"),
        (("claim_boundary", "rtl_cycle"), True, "rtl_cycle_forgery"),
        (("claim_boundary", "full_network"), True, "full_network_forgery"),
        (("claim_boundary", "system_speedup"), True, "system_speedup_forgery"),
        (("claim_boundary", "energy"), True, "energy_forgery"),
        (("claim_boundary", "multi_sequence"), True, "multisequence_forgery"),
    ]
    for path, replacement, name in specs:
        attacks.append(expect_result_reject(
            mutate(value, path, replacement), name,
            ledger_scan, capture_scan, samples, operators))
    for label, name in ((None, "missing_cycle_model_label"),
                        ("cycle model", "unbracketed_cycle_model_label"),
                        ("[RTL]", "wrong_cycle_model_label")):
        try:
            require_publication_label(label)
        except Exception:
            attacks.append(name)
        else:
            raise RuntimeError("publication label mutation accepted")
    require_publication_label(PUBLICATION_LABEL)
    for payload, name in (("{\"a\":1,\"a\":2}", "duplicate_json_key"),
                          ("{\"a\":NaN}", "nonfinite_json")):
        try:
            strict_json_text(payload)
        except Exception:
            attacks.append(name)
        else:
            raise RuntimeError("strict JSON mutation accepted")

    sample_zero = value["distribution"]["sample_major"]["ratios"][
        "speedup_vs_m468_strong_zero"]
    sample_bit = value["distribution"]["sample_major"]["ratios"][
        "speedup_vs_m473_same_coordinate_bit"]
    operator_zero = value["distribution"]["operator_isolated"]["ratios"][
        "speedup_vs_m468_strong_zero"]
    operator_bit = value["distribution"]["operator_isolated"]["ratios"][
        "speedup_vs_m473_same_coordinate_bit"]
    return {
        "schema": "m1597_m1590_ep34_c1_result_hammer_mechanical_checks_r1_v1",
        "status": "PASS_M1597_M1590_EP34_C1_CYCLE_MODEL_RESULT_HAMMER",
        "identity": {path.relative_to(ROOT).as_posix(): digest
                     for path, digest in PINS.items()},
        "result_seal": result_seal,
        "capture": capture_scan,
        "ledger_scan": ledger_scan,
        "release_and_attempt": attempt,
        "ratio_of_sums": {
            "candidate_cycles": totals["m505_dead_write_only_1rw_cycles"],
            "m468_strong_zero_cycles": totals["m468_strong_zero_cycles"],
            "m473_same_coordinate_bit_cycles":
                totals["m473_same_coordinate_bit_cycles"],
            "speedup_vs_m468_strong_zero": str(ratios[
                "speedup_vs_m468_strong_zero"]),
            "speedup_vs_m473_same_coordinate_bit": str(ratios[
                "speedup_vs_m473_same_coordinate_bit"]),
        },
        "sample_distribution": {
            "count": 10,
            "speedup_vs_zero_min": sample_zero["minimum"],
            "speedup_vs_zero_max": sample_zero["maximum"],
            "speedup_vs_zero_geomean": sample_zero["geometric_mean"],
            "speedup_vs_bit_min": sample_bit["minimum"],
            "speedup_vs_bit_max": sample_bit["maximum"],
            "speedup_vs_bit_geomean": sample_bit["geometric_mean"],
        },
        "operator_distribution": {
            "count": 40,
            "not_summable": True,
            "speedup_vs_zero_min": operator_zero["minimum"],
            "speedup_vs_zero_max": operator_zero["maximum"],
            "speedup_vs_zero_geomean": operator_zero["geometric_mean"],
            "speedup_vs_bit_min": operator_bit["minimum"],
            "speedup_vs_bit_max": operator_bit["maximum"],
            "speedup_vs_bit_geomean": operator_bit["geometric_mean"],
        },
        "conservation_and_traffic": {
            "source_rows": 51840000,
            "support16_popcount": 78668732,
            "product_issues_per_output_block": 39318654,
            "product_issues_all_eight_blocks": 314549232,
            "dead_reads": dead_reads,
            "dead_writes": dead_writes,
            "dead_forwards": 16189026 - dead_reads,
            "dead_elisions": 25304213 - dead_writes,
            "parent_read_bytes_all_eight_blocks": 16711429248,
            "parent_write_bytes_all_eight_blocks": 10449510912,
            "parent_total_bytes_all_eight_blocks": 27160940160,
            "traffic_is_parent_scratch_only": True,
        },
        "capacity_resolution": {
            "m1590_historical_macro_rounded_bytes_not_for_publication": 213376,
            "current_logical_bytes_m1591": 214912,
            "current_mapped_bytes_m1591": 215040,
            "budget_bytes": 245760,
            "current_margin_bytes": 30720,
            "cycle_result_affected_by_capacity_correction": False,
            "same_resource_still_within_240kib": True,
        },
        "publication_boundary": {
            "paper_citable_after_this_review": True,
            "required_exact_label": PUBLICATION_LABEL,
            "four_bottleneck_conv_only": True,
            "samples": 10,
            "sequences": 1,
            "sequence": "zurich_city_09_a",
            "rtl_cycle": False,
            "wall_clock": False,
            "full_network": False,
            "system_speedup": False,
            "energy": False,
            "power": False,
            "ppa": False,
        },
        "attacks": {"attempted": len(attacks), "rejected": len(attacks),
                    "names": attacks},
    }


def main():
    value = run()
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(sys.argv) == 1:
        sys.stdout.write(payload)
        return 0
    require(len(sys.argv) == 3 and sys.argv[1] == "--out",
            "usage: independent_result_qa.py [--out FILE]")
    output = Path(sys.argv[2])
    require(not output.exists(), "refuse overwrite")
    output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
