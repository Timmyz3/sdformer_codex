#!/opt/anaconda3/bin/python3.12
"""Parse the additive M2067 R10 one-simv, 960-workload VCS transcript.

R10 changes process granularity, not workload semantics.  It imports the
frozen R8 fixture decoder and applies the same per-workload identity, exact
integer-oracle, address-observation, continuation, and claim-boundary checks
to every PASS line in one log.  A separate batch marker and 960 explicit reset
markers are mandatory.  This source never launches VCS or any other EDA tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
R8_PARSER_PATH = HW / (
    "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_r8_20260904.py")
SPEC = importlib.util.spec_from_file_location("m2067_r8_frozen", R8_PARSER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("frozen R8 parser unavailable")
R8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R8)

TB = HW / (
    "tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960_codex_batch_"
    "r10_20260904.sv")
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_batch_r10_20260904.f")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
R8_TB = HW / (
    "tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960_codex_"
    "zeroaware_r8_20260904.sv")
R8_PARSER_SHA256 = (
    "18394260d4056151d9b013516e8b3c0b02e2e4a18501e4d3c51d07c14ed5139e")
R8_TB_SHA256 = (
    "c2e3c1e2c61e3387e7c70d19e6787bdbd7f0b42e18530151d7392e16f9faa1d8")
DOC359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
PASS_PREFIX = R8.PASS_PREFIX
ROW_PREFIX = R8.ROW_PREFIX
RESET_PREFIX = "M2067_WORKLOAD_RESET_COMPLETE "
BATCH_PREFIX = "PASS_M2067_EP34_FC2_EXACT_CONTINUATION_BATCH "


class Failure(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    need(path.is_file() and not path.is_symlink(), "missing/symlink " + str(path))
    need(sha256(path) == digest, "identity drift " + str(path))


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    try:
        return R8.parse_fields(line, prefix)
    except Exception as exc:
        raise Failure(str(exc)) from exc


def validate_source() -> dict:
    exact(R8_PARSER_PATH, R8_PARSER_SHA256)
    exact(R8_TB, R8_TB_SHA256)
    exact(DOC359, DOC359_SHA256)
    fixture = R8.validate_fixture()
    need(TB.is_file() and not TB.is_symlink(), "R10 TB absent/symlink")
    need(FILELIST.is_file() and not FILELIST.is_symlink(),
         "R10 filelist absent/symlink")
    lines = [line.strip() for line in FILELIST.read_text().splitlines()
             if line.strip()]
    expected = [
        HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
        HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv",
        HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
        HW / "rtl_m2067/m2067_fc2_exact_continuation_wrapper.sv",
        HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv",
        TB,
    ]
    need(lines == [str(path.resolve()) for path in expected],
         "R10 filelist exact order")
    need(all(Path(line).is_absolute() and Path(line).is_file()
             and not Path(line).is_symlink() for line in lines),
         "R10 filelist source resolution")
    text = TB.read_text()
    r8_text = R8_TB.read_text()
    required_once = (
        "module tb_m2067_ep34_fc2_exact_continuation_s960_batch_r10;",
        "task automatic prepare_workload_boundary;",
        "for (workload_slot=0; workload_slot<WORKLOADS;",
        "prepare_workload_boundary();",
        "PASS_M2067_EP34_FC2_EXACT_CONTINUATION workload_slot=",
        "PASS_M2067_EP34_FC2_EXACT_CONTINUATION_BATCH workloads=960",
        "M2067_WORKLOAD_RESET_COMPLETE workload_slot=",
    )
    for token in required_once:
        need(text.count(token) == 1, "R10 TB structural token " + token)
    need("$value$plusargs(\"WORKLOAD_SLOT=" not in text,
         "R10 must not select one slot by plusarg")
    need("single_simv_batch=true" in text,
         "R10 batch boundary absent")
    need("dut_counters_zero=true tb_counters_zero=true "
         "scoreboards_zero=true" in text, "R10 reset marker boundary")
    need(text.count("reset_both();") == 1,
         "R10 alias reset task invocation source cardinality drift")
    # Every functional task inherited from R8 must remain byte-identical.  The
    # only allowed behavioral delta is the additive workload boundary plus the
    # outer exact-order loop and batch ledger.
    def task_body(source: str, name: str) -> str:
        marker = "task automatic " + name
        need(source.count(marker) == 1, "task cardinality " + name)
        start = source.index(marker)
        end = source.index("endtask", start) + len("endtask")
        return source[start:end]
    frozen_tasks = (
        "initialize_drives;", "reset_both;", "send_header_both(",
        "send_invalid_alias_header_both(", "load_descriptor_both;",
        "load_chunk(", "compute_expected_tile;", "run_alias_attack(",
        "run_output_tile(",
    )
    for name in frozen_tasks:
        need(task_body(text, name) == task_body(r8_text, name),
             "R8 functional task drift " + name)
    r8_function = r8_text[r8_text.index("function automatic integer directed_weight"):
                          r8_text.index("endfunction",
                                        r8_text.index("function automatic integer directed_weight"))
                          + len("endfunction")]
    r10_function = text[text.index("function automatic integer directed_weight"):
                        text.index("endfunction",
                                   text.index("function automatic integer directed_weight"))
                        + len("endfunction")]
    need(r10_function == r8_function, "R8 directed-weight oracle drift")
    r8_scoreboard = r8_text[r8_text.rindex("    always_comb begin"):
                            r8_text.index("    initial begin\n",
                                          r8_text.rindex("    always_comb begin"))]
    r10_scoreboard = text[text.rindex("    always_comb begin"):
                          text.index("    initial begin\n",
                                     text.rindex("    always_comb begin"))]
    need(r10_scoreboard == r8_scoreboard,
         "R8 ready/commit/address scoreboard drift")
    return {
        "status": "PASS_M2101_M2067_R10_BATCH_STATIC_SOURCE_AND_FIXTURE",
        "workloads": R8.WORKLOADS,
        "integer_checks_per_axis": fixture["integer_checks"],
        "metadata_row_chunk_descriptors": fixture["row_chunk_records"],
        "r8_parser_sha256": sha256(R8_PARSER_PATH),
        "r8_tb_sha256": sha256(R8_TB),
        "r10_tb_sha256": sha256(TB),
        "r10_filelist_sha256": sha256(FILELIST),
        "docs359_sha256": sha256(DOC359),
        "vcs_executed": False,
    }


def _parse_workload(pass_line: str, row_lines: list[str], row: dict,
                    expected_slot: int) -> dict:
    fields = parse_fields(pass_line, PASS_PREFIX)
    required_exact = {
        "physical_groups": "48", "oracle_mismatches": "0",
        "overflow": "0", "alias_attacks": "2",
        "alias_attacks_g96": "1", "alias_attacks_g192": "1",
        "alias_rejects_base": "2", "alias_rejects_tsbg": "2",
        "ordinary_tsbg_same_fixed_fees": "true",
        "real_ep34_sources": "true", "directed_weights": "true",
        "rtl_speedup_claimed": "false", "system_speedup": "false",
        "paper_admitted": "false",
    }
    for key, value in required_exact.items():
        need(fields.get(key) == value,
             f"slot {expected_slot} PASS field {key}")
    integer_keys = (
        "workload_slot", "sample_id", "layer_id", "token_start",
        "token_role_id", "sequence_id", "source_groups", "output_tiles",
        "chunks", "commits", "integer_checks", "row_chunk_records",
        "address_checks_base", "address_checks_tsbg", "base_cycles",
        "tsbg_cycles")
    try:
        values = {key: int(fields[key]) for key in integer_keys}
    except (KeyError, ValueError) as exc:
        raise Failure(f"slot {expected_slot} integer PASS schema") from exc
    need(values["workload_slot"] == expected_slot,
         f"PASS exact ascending order {expected_slot}")
    for key in ("sample_id", "layer_id", "token_start", "token_role_id",
                "sequence_id", "source_groups", "output_tiles", "chunks"):
        need(values[key] == int(row[key]),
             f"slot {expected_slot} metadata identity {key}")
    need(values["commits"] == int(row["expected_commits"])
         and values["integer_checks"] == int(row["integer_checks"]),
         f"slot {expected_slot} commit/check cardinality")
    expected_bases = row["global_group_bases"]
    need(fields.get("global_group_bases") ==
         ",".join(str(value) for value in expected_bases),
         f"slot {expected_slot} global bases")
    expected_cartesian = {
        (tile, chunk)
        for tile in range(int(row["output_tiles"]))
        for chunk in range(int(row["chunks"]))
    }
    seen: set[tuple[int, int]] = set()
    for line in row_lines:
        item = parse_fields(line, ROW_PREFIX)
        need(set(item) == {
            "workload_slot", "sample_id", "layer_id", "token_start",
            "output_tile", "source_groups", "chunk_index", "chunk_count",
            "global_group_base", "first", "intermediate", "final",
        }, f"slot {expected_slot} row field schema")
        key = (int(item["output_tile"]), int(item["chunk_index"]))
        need(key in expected_cartesian and key not in seen,
             f"slot {expected_slot} row Cartesian uniqueness")
        seen.add(key)
        chunk = key[1]
        need(int(item["workload_slot"]) == expected_slot
             and int(item["sample_id"]) == int(row["sample_id"])
             and int(item["layer_id"]) == int(row["layer_id"])
             and int(item["token_start"]) == int(row["token_start"])
             and int(item["source_groups"]) == int(row["source_groups"])
             and int(item["chunk_count"]) == int(row["chunks"])
             and int(item["global_group_base"]) == expected_bases[chunk]
             and int(item["first"]) == int(chunk == 0)
             and int(item["intermediate"]) ==
                 int(0 < chunk < int(row["chunks"]) - 1)
             and int(item["final"]) ==
                 int(chunk == int(row["chunks"]) - 1),
             f"slot {expected_slot} row transcript identity")
    need(seen == expected_cartesian,
         f"slot {expected_slot} complete row Cartesian set")
    need(values["row_chunk_records"] == len(row_lines),
         f"slot {expected_slot} row count")
    if int(row["nonzero_codes"]) == 0:
        need(values["address_checks_base"] == 0
             and values["address_checks_tsbg"] == 0,
             f"slot {expected_slot} zero workload addresses")
    else:
        need(values["address_checks_base"] > 0
             and values["address_checks_tsbg"] > 0,
             f"slot {expected_slot} nonzero workload addresses")
    need(values["base_cycles"] > 0 and values["tsbg_cycles"] > 0,
         f"slot {expected_slot} positive cycles")
    return {**values,
            "expected_nonzero_codes": int(row["nonzero_codes"]),
            "global_group_bases": expected_bases,
            "rtl_cycle_ratio_observed":
                values["base_cycles"] / values["tsbg_cycles"]}


def parse_log(path: Path, validate_source_identity: bool = True) -> dict:
    source = validate_source() if validate_source_identity else {
        "status": "PREVALIDATED_BY_EXACT_PINNED_R10_RUNNER"}
    text = path.read_text(errors="strict")
    need(re.search(r"(?im)(?:\$fatal|Assertion failed|Error-\[|Fatal:)",
                   text) is None, "VCS fatal/assertion/error")
    lines = text.splitlines()
    pass_lines = [line for line in lines if line.startswith(PASS_PREFIX)]
    need(len(pass_lines) == R8.WORKLOADS,
         "individual PASS cardinality must be 960")
    reset_lines = [line for line in lines if line.startswith(RESET_PREFIX)]
    need(len(reset_lines) == R8.WORKLOADS,
         "reset marker cardinality must be 960")
    for slot, line in enumerate(reset_lines):
        fields = parse_fields(line, RESET_PREFIX)
        need(fields.get("workload_slot") == str(slot)
             and fields.get("dut_counters_zero") == "true"
             and fields.get("tb_counters_zero") == "true"
             and fields.get("scoreboards_zero") == "true",
             f"reset exact order/boundary slot {slot}")
        need(int(fields.get("batch_cycle", "-1")) >= 0,
             f"reset batch cycle slot {slot}")
    row_lines_by_slot: list[list[str]] = [[] for _ in range(R8.WORKLOADS)]
    row_indices_by_slot: list[list[int]] = [[] for _ in range(R8.WORKLOADS)]
    for line_index, line in enumerate(lines):
        if line.startswith(ROW_PREFIX):
            fields = parse_fields(line, ROW_PREFIX)
            try:
                slot = int(fields["workload_slot"])
            except (KeyError, ValueError) as exc:
                raise Failure("row workload slot") from exc
            need(0 <= slot < R8.WORKLOADS, "row slot bounds")
            row_lines_by_slot[slot].append(line)
            row_indices_by_slot[slot].append(line_index)
    metadata = R8.strict_json(R8.META)
    rows = [_parse_workload(pass_lines[slot], row_lines_by_slot[slot],
                            metadata["rows"][slot], slot)
            for slot in range(R8.WORKLOADS)]
    batch_lines = [line for line in lines if line.startswith(BATCH_PREFIX)]
    need(len(batch_lines) == 1, "batch PASS cardinality")
    reset_indices = [index for index, line in enumerate(lines)
                     if line.startswith(RESET_PREFIX)]
    pass_indices = [index for index, line in enumerate(lines)
                    if line.startswith(PASS_PREFIX)]
    batch_index = next(index for index, line in enumerate(lines)
                       if line.startswith(BATCH_PREFIX))
    for slot in range(R8.WORKLOADS):
        need(reset_indices[slot] < pass_indices[slot],
             f"slot {slot} reset precedes PASS")
        need(all(reset_indices[slot] < index < pass_indices[slot]
                 for index in row_indices_by_slot[slot]),
             f"slot {slot} rows bounded by reset/PASS")
        if slot:
            need(pass_indices[slot - 1] < reset_indices[slot],
                 f"slot {slot} follows prior PASS")
    need(pass_indices[-1] < batch_index, "batch PASS must be last PASS")
    batch = parse_fields(batch_lines[0], BATCH_PREFIX)
    exact_batch = {
        "workloads": "960", "first_slot": "0", "last_slot": "959",
        "workload_passes": "960", "commits_per_axis": "115200",
        "integer_checks_per_axis": "1843200",
        "row_chunk_records": "13440", "alias_attacks": "1920",
        "single_simv_batch": "true", "real_ep34_sources": "true",
        "directed_weights": "true", "rtl_speedup_claimed": "false",
        "system_speedup": "false", "paper_admitted": "false",
    }
    for key, value in exact_batch.items():
        need(batch.get(key) == value, "batch PASS field " + key)
    base_cycles = sum(row["base_cycles"] for row in rows)
    tsbg_cycles = sum(row["tsbg_cycles"] for row in rows)
    need(int(batch.get("base_cycles", "-1")) == base_cycles
         and int(batch.get("tsbg_cycles", "-1")) == tsbg_cycles,
         "batch cycle sums")
    need(sum(row["integer_checks"] for row in rows) == 1843200,
         "batch integer check sum")
    need(sum(row["commits"] for row in rows) == 115200,
         "batch commit sum")
    need(sum(row["row_chunk_records"] for row in rows) == 13440,
         "batch row transcript sum")
    return {
        "schema": "m2067_ep34_fc2_exact_continuation_batch_log_r10_v1",
        "status": "PASS_M2101_M2067_R10_BATCH_LOG_PENDING_RESULT_HAMMER",
        "log_sha256": sha256(path), "workloads": len(rows),
        "ordinary_cycles_observed": base_cycles,
        "tsbg_cycles_observed": tsbg_cycles,
        "rtl_cycle_ratio_observed": base_cycles / tsbg_cycles,
        "integer_checks_per_axis": 1843200,
        "row_chunk_records": 13440, "reset_markers": 960,
        "rows": rows, "source_identity": source,
        "claim_boundary": {
            "directed_weights": True,
            "real_ep34_activity_and_sign_descriptors": True,
            "single_simv_process": True, "full_fc_wall_time": False,
            "system_speedup": False, "same_area": False,
            "power": False, "energy": False, "paper_admitted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static", action="store_true")
    group.add_argument("--log", type=Path)
    args = parser.parse_args()
    result = validate_source() if args.static else parse_log(args.log)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
