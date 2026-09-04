#!/opt/anaconda3/bin/python3.12
"""Independent M2085 checker for the M2067 R9 full-960 VCS result.

This is a read-only consumer.  It does not run VCS or any other EDA tool, does
not seal a review, and does not authorize an EDA launch.  Production checking
requires explicit result, attempt, failure, and output paths.  ``--static``
does not inspect any R9 runtime namespace.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CHECKER = Path(__file__).resolve()
PARSER_PATH = HW / (
    "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_"
    "vcs_codex_r8_20260904.py")
RUNNER = HW / (
    "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_"
    "one_shot_codex_r9_ownerfix_20260904.py")
CONTRACT = HW / (
    "contracts/m2067_ep34_fc2_exact_continuation_vcs_source_contract_"
    "r9_codex_ownerfix_20260904.json")
M2084_REVIEW = HW / (
    "reviews/m2084_m2067_ep34_fc2_exact_continuation_vcs_source_r9_"
    "hammer_r1_20260904/review.json")
M2082_DIR = HW / (
    "reviews/m2082_m2067_ep34_fc2_exact_continuation_vcs_source_r8_"
    "hammer_r1_20260904")
M2083_DIR = HW / (
    "reviews/m2083_m2067_r8_external_interrupt_failure_hammer_"
    "r1_20260904")
M2084_DIR = M2084_REVIEW.parent

PARSER_SHA256 = (
    "18394260d4056151d9b013516e8b3c0b02e2e4a18501e4d3c51d07c14ed5139e")
RUNNER_SHA256 = (
    "3423f358fd1b91f92058c1ab5aac2f15add72787650962af94f28e4841e9d2c4")
CONTRACT_SHA256 = (
    "e92806d5f447f03d83e61217769afe34b7a4172a09c8d0def79a41a015296374")
M2082_REVIEW_SHA256 = (
    "a7f12388f8458364bc9b7bdc736cd97dc33b5b303a3acb06ebdf8e650f42c82b")
M2082_MANIFEST_SHA256 = (
    "61d963edf5c7d2d828192d80a20d30af3267b9301a2b2f1b12cf2ffd675cd37d")
M2082_OUTER_SHA256 = (
    "d0b730ec66ead887d45e460db21447d0f96d736ed3958b912c368b695ddd60ee")
M2083_REVIEW_SHA256 = (
    "3a2d7e7cf6b9865c0a6fc5e66076e0dc495a6566c7cc7fab512fdce2f96f9ce0")
M2083_MANIFEST_SHA256 = (
    "4405edc96508ff940ed6582bb12d957017f770eb81f914d696f5e53a41b38fbd")
M2083_OUTER_SHA256 = (
    "0b2402fa2d4cadcb01b0e9acfc279fe6da1c3489d1adac3223b64ca21b8cacb5")
M2084_REVIEW_SHA256 = (
    "08acb31a49bc693c6998ae1f8620b55c9252126bc4ddebfc47c49362be16b8b9")
M2084_MANIFEST_SHA256 = (
    "64365660ce69b877d86d82005d783d32df5caae760b963e08a6c3090933b78ed")
M2084_OUTER_SHA256 = (
    "a4d204aebe91a3013e8b10a5c7cf196950a579670c54f0d3cc9060dc98e0d12c")
DOC359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

WORKLOADS = 960
EXPECTED_SEQUENCE_COUNTS = {0: 240, 1: 240, 2: 240, 3: 240}
EXPECTED_LAYER_COUNTS = {
    17: 120, 19: 120, 21: 120, 23: 120,
    25: 120, 27: 120, 29: 120, 31: 120,
}
EXPECTED_GROUP_COUNTS = {96: 720, 192: 240}
EXPECTED_ROLE_COUNTS = {0: 320, 1: 320, 2: 320}
EXPECTED_NONZERO_WORKLOADS = 764
EXPECTED_ZERO_WORKLOADS = 196
EXPECTED_COMMITS_PER_AXIS = 115_200
EXPECTED_INTEGER_CHECKS_PER_AXIS = 1_843_200
EXPECTED_METADATA_DESCRIPTORS = 2_400
EXPECTED_VCS_TRANSCRIPT_RECORDS = 13_440
EXPECTED_ALIAS_ATTACKS = 1_920
EXPECTED_RESULT_NAME = (
    "m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_20260904")
EXPECTED_ATTEMPT_NAME = (
    ".m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "attempt_consumed")
EXPECTED_FAILURE_NAME = EXPECTED_RESULT_NAME + ".failed_or_incomplete.quarantine"

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


def load_frozen_parser():
    # Verify before importing: a drifted Python file must never execute merely
    # because the hammer is trying to reject it.
    exact(PARSER_PATH, PARSER_SHA256)
    spec = importlib.util.spec_from_file_location(
        "m2067_frozen_r8_parser", PARSER_PATH)
    need(spec is not None and spec.loader is not None,
         "frozen R8 parser unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value

    value = json.loads(
        path.read_text(errors="strict"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root " + str(path))
    return value


def all_entries_no_links(root: Path) -> set[str]:
    need(root.is_dir() and not root.is_symlink(), "sealed root missing/symlink")
    files: set[str] = set()
    for current, dirs, names in os.walk(root, followlinks=False):
        base = Path(current)
        for name in dirs:
            need(not (base / name).is_symlink(), "directory symlink in seal")
        for name in names:
            path = base / name
            need(path.is_file() and not path.is_symlink(), "non-regular seal member")
            files.add(path.relative_to(root).as_posix())
    return files


def sealed_directory(root: Path) -> dict[str, str]:
    """Validate an exhaustive SHA256SUMS plus its outer SHA seal."""
    files = all_entries_no_links(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need("SHA256SUMS" in files and "SHA256SUMS.seal.sha256" in files,
         "double seal absent")
    outer_fields = outer.read_text(errors="strict").split()
    need(outer_fields == [sha256(manifest), "SHA256SUMS"], "outer seal")
    mapping: dict[str, str] = {}
    for line in manifest.read_text(errors="strict").splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0])
             is not None, "manifest syntax")
        relative = Path(fields[1].lstrip("*"))
        name = relative.as_posix()
        need(not relative.is_absolute() and ".." not in relative.parts,
             "unsafe manifest member")
        need(name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
             "recursive seal member")
        need(name not in mapping, "duplicate manifest member")
        exact(root / relative, fields[0])
        mapping[name] = fields[0]
    need(set(mapping) == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
         "non-exhaustive sealed directory")
    return mapping


def validate_static() -> dict:
    exact(RUNNER, RUNNER_SHA256)
    exact(CONTRACT, CONTRACT_SHA256)
    exact(HW / "docs/359_DATE终局冻结_20260813.md", DOC359_SHA256)
    for root, review_sha, manifest_sha, outer_sha in (
        (M2082_DIR, M2082_REVIEW_SHA256, M2082_MANIFEST_SHA256,
         M2082_OUTER_SHA256),
        (M2083_DIR, M2083_REVIEW_SHA256, M2083_MANIFEST_SHA256,
         M2083_OUTER_SHA256),
        (M2084_DIR, M2084_REVIEW_SHA256, M2084_MANIFEST_SHA256,
         M2084_OUTER_SHA256),
    ):
        mapping = sealed_directory(root)
        exact(root / "review.json", review_sha)
        exact(root / "SHA256SUMS", manifest_sha)
        exact(root / "SHA256SUMS.seal.sha256", outer_sha)
        need(mapping.get("review.json") == review_sha,
             "review absent from authority seal " + str(root))
    frozen_parser = load_frozen_parser()
    parser_static = frozen_parser.validate_source()
    need(parser_static.get("status") == "PASS_M2067_STATIC_SOURCE_AND_FIXTURE",
         "frozen parser static status")
    need(parser_static.get("workloads") == WORKLOADS, "static workloads")
    need(parser_static.get("row_chunk_records") ==
         EXPECTED_METADATA_DESCRIPTORS, "metadata descriptor count")
    need(parser_static.get("integer_checks") ==
         EXPECTED_INTEGER_CHECKS_PER_AXIS, "static integer checks")
    m2084 = strict_json(M2084_REVIEW)
    need(m2084.get("status", "").startswith("PASS_M2084_"), "M2084 status")
    identity = m2084.get("reviewed_source_identity", {})
    need(identity.get("runner_sha256") == RUNNER_SHA256
         and identity.get("parser_sha256") == PARSER_SHA256
         and identity.get("contract_sha256") == CONTRACT_SHA256,
         "M2084 source identity")
    return {
        "status": "PASS_M2085_CHECKER_STATIC_ONLY__NO_R9_NAMESPACE_READ",
        "checker_sha256": sha256(CHECKER),
        "parser_sha256": PARSER_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "m2084_review_sha256": M2084_REVIEW_SHA256,
        "workloads_expected": WORKLOADS,
        "metadata_descriptors_expected": EXPECTED_METADATA_DESCRIPTORS,
        "vcs_transcript_records_expected": EXPECTED_VCS_TRANSCRIPT_RECORDS,
        "eda_executed": False,
        "r9_runtime_namespace_read": False,
    }


def validate_attempt(attempt_dir: Path, result: dict) -> dict:
    mapping = sealed_directory(attempt_dir)
    need(set(mapping) == {"attempt.json", "owner.json"},
         "attempt member inventory")
    attempt = strict_json(attempt_dir / "attempt.json")
    owner = strict_json(attempt_dir / "owner.json")
    need(attempt.get("schema") ==
         "m2067_ep34_fc2_exact_continuation_attempt_r9_v1",
         "attempt schema")
    need(attempt.get("runner_sha256") == RUNNER_SHA256
         and attempt.get("parser_sha256") == PARSER_SHA256
         and attempt.get("contract_sha256") == CONTRACT_SHA256,
         "attempt source identity")
    need(attempt.get("vcs_compiles_budget") == 1
         and attempt.get("simv_runs_budget") == WORKLOADS
         and attempt.get("inherited_logs") == 0
         and attempt.get("automatic_retry") is False, "attempt budget")
    need(owner.get("schema") == "m2067_r9_attempt_owner_v1",
         "owner schema")
    need(type(owner.get("pid")) is int and owner["pid"] > 1, "owner pid")
    need(re.fullmatch(r"[0-9a-f]{32}", str(owner.get("nonce", "")))
         is not None, "owner nonce")
    need(owner.get("runner_sha256") == RUNNER_SHA256, "owner runner")
    result_attempt = result.get("attempt_identity", {})
    need(result_attempt == {
        "owner_nonce": owner["nonce"],
        "attempt_json_sha256": sha256(attempt_dir / "attempt.json"),
        "owner_json_sha256": sha256(attempt_dir / "owner.json"),
        "manifest_sha256": sha256(attempt_dir / "SHA256SUMS"),
        "outer_file_sha256": sha256(attempt_dir / "SHA256SUMS.seal.sha256"),
    }, "result/attempt identity")
    return {
        "owner_pid": owner["pid"], "owner_nonce": owner["nonce"],
        "attempt_json_sha256": sha256(attempt_dir / "attempt.json"),
        "owner_json_sha256": sha256(attempt_dir / "owner.json"),
        "manifest_sha256": sha256(attempt_dir / "SHA256SUMS"),
        "outer_file_sha256": sha256(attempt_dir / "SHA256SUMS.seal.sha256"),
        "vcs_compiles_budget": 1, "simv_runs_budget": WORKLOADS,
        "inherited_logs": 0, "automatic_retry": False,
    }


def exact_counter(actual: Counter, expected: dict[int, int], label: str) -> None:
    need(dict(actual) == expected, label + " coverage")


def production_check(result_dir: Path, attempt_dir: Path, failure_path: Path,
                     output: Path) -> dict:
    need(result_dir.name == EXPECTED_RESULT_NAME
         and attempt_dir.name == EXPECTED_ATTEMPT_NAME
         and failure_path.name == EXPECTED_FAILURE_NAME,
         "R9 namespace basename identity")
    need(result_dir.parent.resolve() == attempt_dir.parent.resolve()
         and result_dir.parent.resolve() == failure_path.parent.resolve(),
         "R9 namespaces must share one result parent")
    need(not output.exists() and not output.is_symlink(), "output already exists")
    result_abs = result_dir.resolve(strict=False)
    attempt_abs = attempt_dir.resolve(strict=False)
    output_abs = output.resolve(strict=False)
    need(output_abs != result_abs and output_abs != attempt_abs
         and result_abs not in output_abs.parents
         and attempt_abs not in output_abs.parents,
         "output must not modify a sealed input namespace")
    need(output.parent.is_dir() and not output.parent.is_symlink(),
         "output parent must preexist as a real directory")
    need(not os.path.lexists(failure_path), "failure namespace exists")
    static = validate_static()
    frozen_parser = load_frozen_parser()
    manifest = sealed_directory(result_dir)
    expected_logs = {f"logs/slot_{slot:04d}.log" for slot in range(WORKLOADS)}
    need(set(manifest) == expected_logs | {
        "result.json", "vcs_compile.log", "lmstat.log"},
         "result member inventory")
    need((result_dir / "vcs_compile.log").stat().st_size > 0,
         "empty compile log")
    lmstat = (result_dir / "lmstat.log").read_text(errors="strict")
    need("Users of VCSCompiler_Net" in lmstat, "license preflight evidence")

    result = strict_json(result_dir / "result.json")
    need(result.get("schema") ==
         "m2067_ep34_fc2_exact_continuation_vcs_result_r9_v1",
         "result schema")
    need(result.get("status") ==
         "PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE", "result status")
    need(result.get("workloads") == WORKLOADS
         and result.get("inherited_logs") == 0, "result workload identity")
    source_identity = result.get("source_and_authority_identity", {})
    need(source_identity == {
        "runner_sha256": RUNNER_SHA256,
        "parser_sha256": PARSER_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "m2082_review_sha256": M2082_REVIEW_SHA256,
        "m2083_review_sha256": M2083_REVIEW_SHA256,
        "m2084_review_sha256": M2084_REVIEW_SHA256,
    }, "result source/authority identity")
    claim_boundary = {
        "directed_weights": True, "component_workloads": True,
        "full_fc_wall_time": False, "system_speedup": False,
        "energy": False, "paper_admitted": False,
    }
    need(result.get("claim_boundary") == claim_boundary,
         "producer claim boundary")
    attempt = validate_attempt(attempt_dir, result)

    producer_rows = result.get("rows")
    need(type(producer_rows) is list and len(producer_rows) == WORKLOADS,
         "producer row cardinality")
    metadata = frozen_parser.strict_json(frozen_parser.META)
    reparsed_rows = []
    sequence_counts: Counter = Counter()
    layer_counts: Counter = Counter()
    group_counts: Counter = Counter()
    role_counts: Counter = Counter()
    zero_nonzero_counts: Counter = Counter()
    sequence_cycles = defaultdict(lambda: [0, 0])
    layer_cycles = defaultdict(lambda: [0, 0])
    commits = 0
    integer_checks = 0
    metadata_descriptors = 0
    vcs_transcript_records = 0
    alias_attacks = 0
    alias_attacks_g96 = 0
    alias_attacks_g192 = 0
    alias_rejects_base = 0
    alias_rejects_tsbg = 0
    oracle_mismatches = 0
    acc24_overflows = 0
    zero_address_rows = 0
    positive_address_rows = 0

    for slot in range(WORKLOADS):
        log = result_dir / "logs" / f"slot_{slot:04d}.log"
        parsed = frozen_parser.parse_log(
            log, validate_source_identity=False, metadata=metadata)
        parsed.pop("source_identity", None)
        need(parsed == producer_rows[slot], f"producer/reparse row {slot}")
        need(parsed["workload_slot"] == slot, f"slot order {slot}")
        need(parsed["log_sha256"] == manifest[f"logs/slot_{slot:04d}.log"],
             f"sealed log hash {slot}")
        reparsed_rows.append(parsed)
        sequence_counts[parsed["sequence_id"]] += 1
        layer_counts[parsed["layer_id"]] += 1
        group_counts[parsed["source_groups"]] += 1
        role_counts[parsed["token_role_id"]] += 1
        zero_nonzero_counts[parsed["expected_nonzero_codes"] > 0] += 1
        sequence_cycles[parsed["sequence_id"]][0] += parsed["base_cycles"]
        sequence_cycles[parsed["sequence_id"]][1] += parsed["tsbg_cycles"]
        layer_cycles[parsed["layer_id"]][0] += parsed["base_cycles"]
        layer_cycles[parsed["layer_id"]][1] += parsed["tsbg_cycles"]
        commits += parsed["commits"]
        integer_checks += parsed["integer_checks"]
        metadata_descriptors += parsed["chunks"]
        vcs_transcript_records += parsed["row_chunk_records"]
        if parsed["expected_nonzero_codes"] == 0:
            need(parsed["address_checks_base"] == 0
                 and parsed["address_checks_tsbg"] == 0,
                 f"zero address evidence {slot}")
            zero_address_rows += 1
        else:
            need(parsed["address_checks_base"] > 0
                 and parsed["address_checks_tsbg"] > 0,
                 f"positive address evidence {slot}")
            positive_address_rows += 1

        text = log.read_text(errors="strict")
        pass_lines = [line for line in text.splitlines()
                      if line.startswith(frozen_parser.PASS_PREFIX)]
        need(len(pass_lines) == 1, f"raw PASS cardinality {slot}")
        fields = frozen_parser.parse_fields(
            pass_lines[0], frozen_parser.PASS_PREFIX)
        alias_attacks += int(fields["alias_attacks"])
        alias_attacks_g96 += int(fields["alias_attacks_g96"])
        alias_attacks_g192 += int(fields["alias_attacks_g192"])
        alias_rejects_base += int(fields["alias_rejects_base"])
        alias_rejects_tsbg += int(fields["alias_rejects_tsbg"])
        oracle_mismatches += int(fields["oracle_mismatches"])
        acc24_overflows += int(fields["overflow"])

    exact_counter(sequence_counts, EXPECTED_SEQUENCE_COUNTS, "sequence")
    exact_counter(layer_counts, EXPECTED_LAYER_COUNTS, "layer")
    exact_counter(group_counts, EXPECTED_GROUP_COUNTS, "source group")
    exact_counter(role_counts, EXPECTED_ROLE_COUNTS, "token role")
    need(zero_nonzero_counts == Counter({True: EXPECTED_NONZERO_WORKLOADS,
                                        False: EXPECTED_ZERO_WORKLOADS}),
         "nonzero/zero workload coverage")
    need(commits == EXPECTED_COMMITS_PER_AXIS, "commit count per axis")
    need(integer_checks == EXPECTED_INTEGER_CHECKS_PER_AXIS,
         "integer checks per axis")
    need(metadata_descriptors == EXPECTED_METADATA_DESCRIPTORS,
         "metadata descriptor count")
    need(vcs_transcript_records == EXPECTED_VCS_TRANSCRIPT_RECORDS,
         "VCS transcript record count")
    need(alias_attacks == EXPECTED_ALIAS_ATTACKS
         and alias_attacks_g96 == WORKLOADS
         and alias_attacks_g192 == WORKLOADS
         and alias_rejects_base == EXPECTED_ALIAS_ATTACKS
         and alias_rejects_tsbg == EXPECTED_ALIAS_ATTACKS,
         "alias attack/rejection totals")
    need(oracle_mismatches == 0 and acc24_overflows == 0,
         "functional mismatch/overflow")
    need(zero_address_rows == EXPECTED_ZERO_WORKLOADS
         and positive_address_rows == EXPECTED_NONZERO_WORKLOADS,
         "address evidence partition")

    ordinary_cycles = sum(row["base_cycles"] for row in reparsed_rows)
    tsbg_cycles = sum(row["tsbg_cycles"] for row in reparsed_rows)
    ratio_of_sums = ordinary_cycles / tsbg_cycles
    need(result.get("ordinary_cycles_observed") == ordinary_cycles
         and result.get("tsbg_cycles_observed") == tsbg_cycles,
         "producer cycle sums")
    need(math.isclose(result.get("rtl_cycle_ratio_observed", -1),
                      ratio_of_sums, rel_tol=0.0, abs_tol=1e-15),
         "producer ratio-of-sums")
    need(result.get("integer_checks_per_axis") == integer_checks,
         "producer integer checks")

    namespace_parent = result_dir.parent
    residue_patterns = (
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_work.*",
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_stage.*",
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_failure.*",
    )
    residues = sorted(str(path) for pattern in residue_patterns
                      for path in namespace_parent.glob(pattern))
    need(not residues, "private work/stage/failure residue " + repr(residues))

    by_sequence = {
        str(key): {
            "workloads": sequence_counts[key],
            "ordinary_cycles": value[0], "tsbg_cycles": value[1],
            "rtl_cycle_ratio_of_sums": value[0] / value[1],
        } for key, value in sorted(sequence_cycles.items())
    }
    by_layer = {
        str(key): {
            "workloads": layer_counts[key],
            "ordinary_cycles": value[0], "tsbg_cycles": value[1],
            "rtl_cycle_ratio_of_sums": value[0] / value[1],
        } for key, value in sorted(layer_cycles.items())
    }
    output_value = {
        "schema": "m2085_m2067_ep34_fc2_exact_continuation_vcs_r9_"
                  "independent_result_hammer_mechanical_v1",
        "status": "PASS_M2085_R9_RESULT_MECHANICALLY_REPARSED__REVIEW_"
                  "AND_PAPER_ADMISSION_STILL_SEPARATE",
        "input_identity": {
            "result_json_sha256": sha256(result_dir / "result.json"),
            "result_manifest_sha256": sha256(result_dir / "SHA256SUMS"),
            "result_outer_file_sha256":
                sha256(result_dir / "SHA256SUMS.seal.sha256"),
            "parser_sha256": PARSER_SHA256, "runner_sha256": RUNNER_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "m2084_review_sha256": M2084_REVIEW_SHA256,
            "docs359_sha256": DOC359_SHA256,
            "attempt": attempt,
        },
        "coverage": {
            "workloads": WORKLOADS, "inherited_logs": 0,
            "sequence_counts": {str(k): v for k, v in sorted(sequence_counts.items())},
            "layer_counts": {str(k): v for k, v in sorted(layer_counts.items())},
            "source_group_counts": {str(k): v for k, v in sorted(group_counts.items())},
            "token_role_counts": {str(k): v for k, v in sorted(role_counts.items())},
            "nonzero_workloads": EXPECTED_NONZERO_WORKLOADS,
            "zero_workloads": EXPECTED_ZERO_WORKLOADS,
        },
        "functional_evidence": {
            "oracle_mismatches": 0, "acc24_overflows": 0,
            "commits_per_axis": commits,
            "integer_checks_per_axis": integer_checks,
            "metadata_descriptors": metadata_descriptors,
            "vcs_output_tile_chunk_transcript_records": vcs_transcript_records,
            "descriptor_and_transcript_counts_are_distinct": True,
            "alias_attacks": alias_attacks,
            "alias_attacks_g96": alias_attacks_g96,
            "alias_attacks_g192": alias_attacks_g192,
            "alias_rejects_base": alias_rejects_base,
            "alias_rejects_tsbg": alias_rejects_tsbg,
            "zero_workloads_with_zero_address_observations": zero_address_rows,
            "nonzero_workloads_with_positive_address_observations":
                positive_address_rows,
            "address_observations_are_not_weight_request_cardinality": True,
        },
        "cycle_evidence": {
            "ordinary_cycles": ordinary_cycles, "tsbg_cycles": tsbg_cycles,
            "rtl_cycle_ratio_of_sums": ratio_of_sums,
            "aggregation": "ratio_of_sums_not_mean_of_per_workload_ratios",
            "by_sequence": by_sequence, "by_layer": by_layer,
        },
        "namespace_evidence": {
            "result_double_sealed_exhaustively": True,
            "attempt_double_sealed_exhaustively": True,
            "failure_namespace_absent": True,
            "private_work_stage_failure_residue_absent": True,
            "symlinks_absent": True,
        },
        "m2088_authorization_inputs": {
            "m2085_mechanical_result_gate_pass": True,
            "fresh_960_vcs_workloads": True,
            "same_io_schedule_axes": ["ordinary", "tsbg_b4"],
            "eligible_for_separate_m2088_source_authoring": True,
            "dc_runs_authorized_by_this_checker": 0,
            "pt_runs_authorized_by_this_checker": 0,
            "ptpx_runs_authorized_by_this_checker": 0,
            "vcs_runs_authorized_by_this_checker": 0,
            "automatic_retry_authorized": False,
        },
        "claim_boundary": {
            **claim_boundary,
            "real_ep34_activity_and_sign_descriptors": True,
            "real_checkpoint_weights": False,
            "continuation_cohort_only": True,
            "direct_g48_cohort_included": False,
            "full_fc": False, "same_area": False, "power": False,
            "throughput_per_area": False,
            "independent_human_review_completed": False,
        },
        "static_identity": static,
        "eda_executed_by_checker": False,
    }
    with output.open("x") as stream:
        json.dump(output_value, stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
    return output_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--failure-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    supplied = [args.result_dir, args.attempt_dir, args.failure_path, args.output]
    if args.static:
        need(not any(supplied), "--static accepts no production paths")
        value = validate_static()
    else:
        need(all(item is not None for item in supplied),
             "production mode requires all four explicit paths")
        value = production_check(args.result_dir, args.attempt_dir,
                                 args.failure_path, args.output)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(json.dumps({"status": "FAIL_M2085", "error": str(exc)},
                         sort_keys=True, allow_nan=False))
        raise SystemExit(1)
