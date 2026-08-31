#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author mutation hammer for M1345/R16 source admission only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKER = HW / "verif_m1345r16_c1_real_m935_runtime_witness/check_m1345r16_source.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("m1352_bound_m1345_checker", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("checker import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_checker()
SOURCE = M.WITNESS.read_text()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"non-unique mutation anchor ({text.count(old)}): {old!r}")
    return text.replace(old, new, 1)


def replace_in_stage(stage: str, next_stage: str, old: str, new: str) -> str:
    begin = SOURCE.index(stage + ": begin")
    end = SOURCE.index(next_stage + ": begin", begin + 1)
    body = SOURCE[begin:end]
    if body.count(old) != 1:
        raise AssertionError(f"stage anchor drift {stage}: {old!r}")
    return SOURCE[:begin] + body.replace(old, new, 1) + SOURCE[end:]


def control_mutant(control: str) -> str:
    begin = SOURCE.index("control_unknown = $isunknown({")
    end = SOURCE.index("});", begin) + 3
    block = SOURCE[begin:end]
    if block.count(control) != 1:
        raise AssertionError(f"control anchor drift: {control}")
    return SOURCE[:begin] + block.replace(control, "1'b0", 1) + SOURCE[end:]


def main() -> int:
    attacks: dict[str, str] = {}

    stages = (
        ("W_FIRST_REQUEST", "W_FIRST_ACCEPT", "(weight_request_fire === 1'b0)",
         "responses_q <= 4'd1;", "stage_q <= W_FIRST_ACCEPT;"),
        ("W_SECOND_REQUEST", "W_SECOND_ACCEPT", "(psum_commit_fire === 1'b0)",
         "core_accepts_q <= 4'd2;", "stage_q <= W_SECOND_ACCEPT;"),
        ("W_SECOND_ACCEPT", "W_PSUM_COMMIT", "(row_complete_fire === 1'b0)",
         "psum_commits_q <= 4'd1;", "stage_q <= W_PSUM_COMMIT;"),
        ("W_PSUM_COMMIT", "W_ROW_DONE", "(task_done_fire === 1'b0)",
         "row_completions_q <= 4'd1;", "stage_q <= W_ROW_DONE;"),
    )
    for stage, next_stage, guard, update, transition in stages:
        stem = stage.lower()
        attacks[f"{stem}_guard_delete"] = replace_in_stage(
            stage, next_stage, guard, "1'b1")
        attacks[f"{stem}_update_delete"] = replace_in_stage(
            stage, next_stage, update, "")
        attacks[f"{stem}_transition_redirect"] = replace_in_stage(
            stage, next_stage, transition, "stage_q <= W_TASK_DONE;")

    controls = (
        "weight_request_fire", "psum_request_fire", "response_accept", "core_accept",
        "psum_commit_fire", "row_complete_fire", "task_done_fire",
        "request_hold_attack_mode", "weight_service_attack_mode",
        "psum_service_attack_mode", "protocol_error", "boundary_fault", "core_fault",
        "m935_fault", "weight_service_fault", "psum_service_fault",
    )
    for control in controls:
        attacks[f"control_unknown_drop_{control}"] = control_mutant(control)

    for signal, expected in (
        ("design_issue_accepts", "64'd2"),
        ("design_psum_commits", "64'd1"),
        ("design_row_completions", "64'd1"),
    ):
        term = f"            && ({signal} === {expected})\n"
        attacks[f"final_drop_{signal}"] = replace_once(SOURCE, term, "")

    # A semantic deletion must not be rescued by leaving its spelling in comments.
    guard_line = "                                    && (weight_request_fire === 1'b0)\n"
    deleted = replace_in_stage(
        "W_FIRST_REQUEST", "W_FIRST_ACCEPT", guard_line.strip(), "")
    attacks["line_comment_residue_bypass"] = deleted + (
        "\n// residue only: (weight_request_fire === 1'b0)\n")
    attacks["block_comment_residue_bypass"] = deleted + (
        "\n/* residue only: (weight_request_fire === 1'b0) */\n")

    # Reordering the complete set keeps cardinality but changes the intended canonical oracle.
    control_pair = "weight_request_fire,\n                psum_request_fire"
    attacks["control_reorder_bypass"] = replace_once(
        SOURCE, control_pair, "psum_request_fire,\n                weight_request_fire")

    # Normalization deliberately permits whitespace only, but never a semantic operator change.
    attacks["normalized_operator_bypass"] = replace_once(
        SOURCE, "(design_issue_accepts === 64'd2)",
        "( design_issue_accepts >= 64'd2 )")

    pass_line = (
        "            $display(\"PASS_M1337R15_REAL_M935_RUNTIME_WITNESS "
        "wrapper_functional_candidate=true strict_registered_stages=true "
        "unknown_fail_closed=true structural_bind=true ledger_bytes=214912 "
        "functional_vcs=false timing_verified=false cycles_measured=false "
        "speedup=false ppa=false energy=false headline=false\");\n"
    )
    attacks["pass_token_unconditional"] = replace_once(
        SOURCE, "        if (pass === 1'b1) begin\n" + pass_line,
        pass_line + "        if (pass === 1'b1) begin\n")
    attacks["pass_branch_inverted"] = replace_once(
        SOURCE, "if (pass === 1'b1) begin", "if (pass !== 1'b1) begin")
    attacks["fatal_removed"] = replace_once(
        SOURCE, "$fatal(1, \"M1337R15 runtime witness incomplete, unknown, or attacked\");",
        "$display(\"failure ignored\");")
    attacks["early_finish"] = replace_once(
        SOURCE, "final begin : witness_final_oracle",
        "initial begin $finish; end\n\n    final begin : witness_final_oracle")

    outcomes = []
    false_negatives = []
    for name, mutant in attacks.items():
        if mutant == SOURCE:
            raise AssertionError("no-op mutation: " + name)
        rejected = False
        message = ""
        try:
            M.check_witness_text(mutant)
        except AssertionError as exc:
            rejected = True
            message = str(exc)
        outcomes.append({"attack": name, "rejected": rejected, "message": message})
        if not rejected:
            false_negatives.append(name)

    payload = {
        "schema": "m1352_m1345_c1_r16_runtime_witness_source_blind_hammer_r1_v1",
        "status": "PASS" if not false_negatives else "FAIL_DO_NOT_CITE",
        "attack_count": len(attacks),
        "rejected_count": len(attacks) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "categories": {
            "four_stage_guard_update_transition": 12,
            "complete_control_unknown": 16,
            "final_design_count_conjunct": 3,
            "normalized_comment_reorder": 4,
            "pass_token_branch": 4,
        },
        "outcomes": outcomes,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not false_negatives else 1


if __name__ == "__main__":
    raise SystemExit(main())
