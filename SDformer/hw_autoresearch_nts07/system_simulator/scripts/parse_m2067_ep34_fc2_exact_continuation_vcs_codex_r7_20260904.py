#!/opt/anaconda3/bin/python
"""CODEX R7 NEW FILE -- context-last production parser / 2026-09-04.

Fail-closed M2067 r7 context-last source/fixture and per-slot VCS log parser.
It follows the sealed dual-geometry pilot and never cites quarantined R3/R5.

The source stage has no VCS or RTL-cycle result.  A future independently
authorized campaign may feed one log at a time through ``--log``; only those
observed cycles may be called RTL cycles.  The M2064 CPU ratios are never
promoted or preloaded here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / (
    "contracts/m2067_ep34_fc2_exact_continuation_vcs_source_contract_r7_codex_contextlast_20260904.json"
)
FIXTURE = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh"
STATS = HW / (
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960_stats.memh"
)
META = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json"
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_codex_contextlast_r7_20260904.f"
)
M2051_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
M2068_REVIEW = HW / (
    "reviews/m2068_m2067_ep34_fc2_exact_continuation_vcs_source_hammer_"
    "r1_20260903"
)
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2051_META_SHA256 = (
    "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5"
)
M2068_REVIEW_SHA256 = (
    "664c1f5188fe08afcbbd6332bc5237ec2161a6505b18b3b8d5f9067b8be971ca"
)
M2068_MANIFEST_SHA256 = (
    "78f5d23f73de14fab50a946373f0df3ded05bcda5bb13b253346db25c8d1fc14"
)
M2068_OUTER_FILE_SHA256 = (
    "72c4b9f444b1689ab392160a7782a38afaf2841c01988517a0129b64892d56d3"
)
DOC359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
WORKLOADS = 960
CONTEXTS = 4
MAX_GROUPS = 192
STAT_FIELDS = (
    "sample_id", "layer_id", "token_start", "source_groups",
    "output_tiles", "chunks", "token_role_id", "sequence_id",
    "expected_commits", "integer_checks", "nonzero_codes",
    "negative_codes",
)
PASS_PREFIX = "PASS_M2067_EP34_FC2_EXACT_CONTINUATION "
ROW_PREFIX = "M2067_ROW_CHUNK "

# Filled once after all M2067 source files stop moving.  Parser and contract
# are reviewed as separate identities to avoid a self-hash cycle.
SOURCE_SHA256 = {
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_codex_contextlast_r7_20260904.f": "73e67c47b8b1c85ac939ba0cd5290b7b9b691157c16a334d6afc06ffc39f390a",
    "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot_codex_r7_20260904.py": "873a7f714f6a7f83d749c82fd01673f9fb810ab318c3376ffbb4f7a71fa6d2dd",
    "system_simulator/scripts/build_m2067_ep34_fc2_exact_continuation_fixture.py": "4cf6f1ed16fbf43f19fc7fa9c1d196b67c6a6563750c447e13812695f0733f08",
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh": "c617c6311ce44f15fb820f5dba5460ebd127235a13acd56724b56ccbb10cd594",
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960_stats.memh": "4e2271ca56947ceb8d6abb8b753729576d4b4ed4ca2de297efa05f1b2e9bb80d",
    "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json": "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    "rtl_m2067/m2067_fc2_exact_continuation_wrapper.sv": "755027453b9fc91264f44918cc16e31b278cf70e1b13821666ca2be602022c92",
    "tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960_codex_contextlast_r7_20260904.sv": "dbbbe4b19812f937310ee1946f97289a08b8f3b05ed7d57038770dbbf645bf33",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv": "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv": "dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c",
    "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv": "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
    "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv": "64805bdedb7c80d5c6141bc36e59ef61234507b40942e69ccbf4a30ac2383436",
    "system_simulator/scripts/build_m2051_ep34_tsbg_full40_fixture.py": "3a8642914ccad60df89dfdad1b78c375c6d4e4609435c5731357f294d9acf8cf",
    "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json": "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/SHA256SUMS": "be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f",
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/SHA256SUMS.seal.sha256": "8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/result.json": "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/SHA256SUMS": "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/SHA256SUMS.seal.sha256": "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2",
    "reviews/m2065_m2064_ep34_fc2_exact_continuation_quick_gate_result_hammer_r1_20260903/review.json": "01152ad8d0c7539c4cd885cba8c434e5b98201d7b89c07a3f22c8b2cde1703b6",
    "reviews/m2065_m2064_ep34_fc2_exact_continuation_quick_gate_result_hammer_r1_20260903/SHA256SUMS": "aacd2a34a409ba2b38887f7cc0922b1ba1b24d8ca845c51ac5c08980f4dc8ebf",
    "reviews/m2065_m2064_ep34_fc2_exact_continuation_quick_gate_result_hammer_r1_20260903/SHA256SUMS.seal.sha256": "bb1165db65abb818c07a33f5174a61ac455a1d2afeb8cd4024e477b98747eeb8",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


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
    need(path.is_file() and not path.is_symlink(), f"missing/symlink {path}")
    need(sha256(path) == digest, f"identity drift {path}")


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, f"duplicate JSON key {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    need(type(value) is dict, f"JSON root {path}")
    return value


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    need(line.startswith(prefix), "line prefix")
    fields: dict[str, str] = {}
    for token in line[len(prefix):].split():
        need(token.count("=") == 1, "field syntax " + token)
        key, value = token.split("=", 1)
        need(key not in fields, "duplicate field " + key)
        fields[key] = value
    return fields


def decode_stats(path: Path) -> list[dict[str, int]]:
    lines = path.read_text().splitlines()
    need(len(lines) == WORKLOADS, "stats cardinality")
    rows = []
    for index, line in enumerate(lines):
        need(re.fullmatch(r"[0-9a-f]{96}", line) is not None,
             f"stats syntax {index}")
        value = int(line, 16)
        row = {field: (value >> (32 * offset)) & 0xffffffff
               for offset, field in enumerate(STAT_FIELDS)}
        rows.append(row)
    return rows


def validate_fixture() -> dict:
    exact(M2051_META, M2051_META_SHA256)
    exact(DOC359, DOC359_SHA256)
    metadata = strict_json(META)
    need(metadata.get("schema") ==
         "m2067_ep34_fc2_exact_continuation_fixture_r1_v1",
         "metadata schema")
    geometry = metadata.get("geometry", {})
    need(geometry == {
        "workloads": 960, "samples": 40, "sequences": 4,
        "layers": 8, "g96_layers": 6, "g192_layers": 2,
        "quartets_per_layer_sample": 3, "contexts": 4,
        "physical_source_groups": 48,
        "max_logical_source_groups": 192,
        "sources_per_group": 16, "integer_checks": 1843200,
    }, "metadata geometry")
    need(metadata.get("fixture_sha256") == sha256(FIXTURE),
         "fixture metadata hash")
    need(metadata.get("stats_sha256") == sha256(STATS),
         "stats metadata hash")
    identity = metadata.get("input_identity", {})
    m2051_key = "hw_autoresearch_nts07/tb_m2018/fixtures/" \
                "m2051_ep34_tsbg_full40_s1920.json"
    need(identity.get(m2051_key) == M2051_META_SHA256,
         "direct M2051 metadata pin")
    rows = metadata.get("rows")
    need(type(rows) is list and len(rows) == WORKLOADS, "metadata rows")
    stats = decode_stats(STATS)
    fixtures = FIXTURE.read_text().splitlines()
    need(len(fixtures) == WORKLOADS * CONTEXTS * MAX_GROUPS,
         "fixture word cardinality")
    nonzero_total = 0
    negative_total = 0
    for index, word in enumerate(fixtures):
        need(re.fullmatch(r"[0-9a-f]{8}", word) is not None,
             f"fixture syntax {index}")
        packed = int(word, 16)
        active = packed & 0xffff
        sign = packed >> 16
        need(sign & ~active == 0, f"sign without activity {index}")
        nonzero_total += active.bit_count()
        negative_total += sign.bit_count()
    seen = set()
    row_chunk_records = 0
    for slot, (row, stat) in enumerate(zip(rows, stats)):
        need(row.get("slot") == slot, f"slot order {slot}")
        for field in STAT_FIELDS:
            need(int(row[field]) == stat[field], f"stat mismatch {slot} {field}")
        groups = int(row["source_groups"])
        chunks = int(row["chunks"])
        need((groups, chunks) in {(96, 2), (192, 4)},
             f"logical geometry {slot}")
        expected_bases = list(range(0, groups, 48))
        need(row.get("global_group_bases") == expected_bases,
             f"row bases {slot}")
        chunk_rows = row.get("chunk_rows")
        need(type(chunk_rows) is list and len(chunk_rows) == chunks,
             f"chunk rows {slot}")
        for chunk_index, chunk in enumerate(chunk_rows):
            need(chunk.get("global_group_base") == expected_bases[chunk_index],
                 f"serialized chunk base {slot}/{chunk_index}")
            need(chunk.get("first") is (chunk_index == 0)
                 and chunk.get("intermediate") is
                 (0 < chunk_index < chunks - 1)
                 and chunk.get("final") is (chunk_index == chunks - 1),
                 f"serialized chunk flags {slot}/{chunk_index}")
            row_chunk_records += 1
        key = (int(row["sample_id"]), int(row["layer_id"]),
               int(row["token_role_id"]))
        need(key not in seen, f"duplicate workload identity {slot}")
        seen.add(key)
    need(sum(stat["integer_checks"] for stat in stats) == 1843200,
         "integer checks total")
    need(sum(stat["nonzero_codes"] for stat in stats) == nonzero_total,
         "nonzero descriptor total")
    need(sum(stat["negative_codes"] for stat in stats) == negative_total,
         "negative descriptor total")
    return {
        "status": "PASS_M2067_FIXTURE_STATIC",
        "workloads": WORKLOADS,
        "row_chunk_records": row_chunk_records,
        "fixture_sha256": sha256(FIXTURE),
        "stats_sha256": sha256(STATS),
        "metadata_sha256": sha256(META),
        "m2051_metadata_sha256": sha256(M2051_META),
        "integer_checks": 1843200,
    }


def validate_source() -> dict:
    need(SOURCE_SHA256, "M2067_DRAFT_NO_FROZEN_SOURCE_INVENTORY")
    for relative, digest in SOURCE_SHA256.items():
        exact(HW / relative, digest)
    contract = strict_json(CONTRACT)
    exact(M2068_REVIEW / "review.json", M2068_REVIEW_SHA256)
    exact(M2068_REVIEW / "SHA256SUMS", M2068_MANIFEST_SHA256)
    exact(M2068_REVIEW / "SHA256SUMS.seal.sha256",
          M2068_OUTER_FILE_SHA256)
    need(contract.get("schema") ==
         "m2067_ep34_fc2_exact_continuation_vcs_source_contract_r7_codex_contextlast_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_R7_ONLY__M2079_REVIEW_REQUIRED_BEFORE_EXECUTION__NO_VCS_NO_EDA__CONTEXT_LAST_FIXED",
         "contract status")
    rows = contract.get("m2067_frozen_sources")
    need(type(rows) is list and len(rows) == len(SOURCE_SHA256),
         "contract source inventory cardinality")
    need(all(type(row) is dict and set(row) == {"path", "sha256"}
             for row in rows), "contract source row schema")
    inventory = {row["path"]: row["sha256"] for row in rows}
    need(len(inventory) == len(rows) and inventory == SOURCE_SHA256,
         "contract source inventory")
    need(contract.get("direct_predecessor_pins", {}).get(
        "m2051_fixture_metadata_sha256") == M2051_META_SHA256,
        "contract direct M2051 metadata pin")
    predecessor = contract.get("direct_predecessor_pins", {})
    need(predecessor.get("m2068_rejection_review_json_sha256") ==
         M2068_REVIEW_SHA256
         and predecessor.get("m2068_rejection_manifest_sha256") ==
         M2068_MANIFEST_SHA256
         and predecessor.get("m2068_rejection_outer_file_sha256") ==
         M2068_OUTER_FILE_SHA256,
         "contract M2068 rejection lineage")
    need(contract.get("claim_boundary") == {
        "source_only": True, "vcs_executed": False,
        "rtl_cycles_observed": False, "rtl_speedup_claimed": False,
        "cpu_ratio_promoted_to_rtl": False, "eda": False,
        "energy": False, "full_fc_wall_time": False,
        "system_speedup": False, "paper_admitted": False,
    }, "contract claim boundary")
    filelist_lines = [line.strip() for line in FILELIST.read_text().splitlines()
                      if line.strip()]
    expected_filelist = [
        HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
        HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv",
        HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
        HW / "rtl_m2067/m2067_fc2_exact_continuation_wrapper.sv",
        HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv",
        HW / "tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960_codex_contextlast_r7_20260904.sv",
    ]
    need(filelist_lines == [str(path.resolve()) for path in expected_filelist],
         "filelist absolute/cwd-independent identity")
    need(all(Path(line).is_absolute() and Path(line).is_file()
             and not Path(line).is_symlink() for line in filelist_lines),
         "filelist absolute regular-source resolution")
    fixture = validate_fixture()
    return {**fixture,
            "status": "PASS_M2067_STATIC_SOURCE_AND_FIXTURE",
            "frozen_sources": len(SOURCE_SHA256)}


def parse_log(path: Path, validate_source_identity: bool = True,
              metadata: dict | None = None) -> dict:
    source = validate_source() if validate_source_identity else {
        "status": "PREVALIDATED_BY_EXACT_PINNED_ONE_SHOT_RUNNER"
    }
    text = path.read_text(errors="strict")
    need(re.search(r"(?im)(?:\$fatal|Assertion failed|Error-\[|Fatal:)",
                   text) is None, "VCS fatal/assertion/error")
    pass_lines = [line for line in text.splitlines()
                  if line.startswith(PASS_PREFIX)]
    need(len(pass_lines) == 1, "PASS cardinality")
    fields = parse_fields(pass_lines[0], PASS_PREFIX)
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
        need(fields.get(key) == value, "PASS field " + key)
    integer_fields = {
        key: int(fields[key]) for key in (
            "workload_slot", "sample_id", "layer_id", "token_start",
            "token_role_id", "sequence_id", "source_groups",
            "output_tiles", "chunks", "commits", "integer_checks",
            "row_chunk_records", "address_checks_base",
            "address_checks_tsbg", "base_cycles", "tsbg_cycles")
    }
    slot = integer_fields["workload_slot"]
    need(0 <= slot < WORKLOADS, "workload slot")
    if metadata is None:
        metadata = strict_json(META)
    row = metadata["rows"][slot]
    for key in ("sample_id", "layer_id", "token_start", "token_role_id",
                "sequence_id", "source_groups", "output_tiles", "chunks"):
        need(integer_fields[key] == int(row[key]), "row identity " + key)
    need(integer_fields["commits"] == int(row["expected_commits"])
         and integer_fields["integer_checks"] == int(row["integer_checks"]),
         "exact commit/check cardinality")
    expected_bases = row["global_group_bases"]
    need(fields.get("global_group_bases") ==
         ",".join(str(value) for value in expected_bases),
         "PASS global group bases")
    row_lines = [line for line in text.splitlines()
                 if line.startswith(ROW_PREFIX)]
    need(len(row_lines) == int(row["output_tiles"]) * int(row["chunks"]),
         "row/chunk line cardinality")
    transcript = []
    seen = set()
    expected_cartesian = {
        (tile, chunk)
        for tile in range(int(row["output_tiles"]))
        for chunk in range(int(row["chunks"]))
    }
    for line in row_lines:
        item = parse_fields(line, ROW_PREFIX)
        need(set(item) == {
            "workload_slot", "sample_id", "layer_id", "token_start",
            "output_tile", "source_groups", "chunk_index", "chunk_count",
            "global_group_base", "first", "intermediate", "final",
        }, "row/chunk field schema")
        key = (int(item["output_tile"]), int(item["chunk_index"]))
        need(key in expected_cartesian, "row/chunk Cartesian bounds")
        need(key not in seen, "duplicate row/chunk record")
        seen.add(key)
        chunk_index = key[1]
        need(int(item["workload_slot"]) == slot
             and int(item["sample_id"]) == int(row["sample_id"])
             and int(item["layer_id"]) == int(row["layer_id"])
             and int(item["token_start"]) == int(row["token_start"])
             and int(item["source_groups"]) == int(row["source_groups"])
             and int(item["chunk_count"]) == int(row["chunks"])
             and int(item["global_group_base"]) == expected_bases[chunk_index]
             and int(item["first"]) == int(chunk_index == 0)
             and int(item["intermediate"]) ==
                 int(0 < chunk_index < int(row["chunks"]) - 1)
             and int(item["final"]) ==
                 int(chunk_index == int(row["chunks"]) - 1),
             "row/chunk transcript identity")
        transcript.append({key: int(item[key]) for key in (
            "output_tile", "chunk_index", "chunk_count",
            "global_group_base", "first", "intermediate", "final")})
    need(seen == expected_cartesian, "row/chunk exact Cartesian set")
    need(integer_fields["row_chunk_records"] == len(transcript),
         "PASS row/chunk cardinality")
    need(integer_fields["address_checks_base"] > 0
         and integer_fields["address_checks_tsbg"] > 0
         and integer_fields["base_cycles"] > 0
         and integer_fields["tsbg_cycles"] > 0,
         "positive address/cycle observations")
    return {
        "status": "PASS_M2067_SINGLE_WORKLOAD_VCS_LOG",
        "log_sha256": sha256(path), **integer_fields,
        "global_group_bases": expected_bases,
        "row_chunk_transcript": transcript,
        "rtl_cycle_ratio_observed":
            integer_fields["base_cycles"] / integer_fields["tsbg_cycles"],
        "claim_boundary": {
            "one_workload_only": True, "directed_weights": True,
            "system_speedup": False, "paper_admitted": False,
        },
        "source_identity": source,
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
