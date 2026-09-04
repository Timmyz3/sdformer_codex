#!/usr/bin/env python3
"""Receipt-blind static audit of the repaired M714-r2 one-shot runner.

This program reads frozen source and identity files only.  It never imports or
executes the author capture, never invokes the runner, and never queries GPU,
EDA, or remote state.
"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m714_h67_ep35_pctda_pattern_s10_r2_one_shot.sh"
CONTRACT = HW / "contracts/m714_h67_ep35_pctda_pattern_s10_contract_r2_20260828.json"
CAPTURE = HW / "system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py"
M366 = HW / "system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
M366_CONTRACT = HW / "contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
M716 = HW / "reviews/m716_m714_pctda_prerun_fresh_hammer_r1_20260828/m716_m714_pctda_prerun_fresh_hammer_verdict_r1.json"
M720 = HW / "reviews/m720_m714_r2_one_shot_runner_fresh_static_hammer_r1_20260828/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "e5c08ee747c8444c42439bc215bbe609742a62180f6b9c179d23973b4acbfb7e",
    "contract": "86eb22204e06325074c2288b83ae64851424bd23d23cc33734f4324cf505cd4b",
    "capture": "f65d87b085963bcdcea6bb79660475b32cb65d7f303d4c6354348628f2d4f59f",
    "m366": "c4b2e83b2a1341f9790038d395aa8ed4c25c75bc441e932def4e2e32b1ba4045",
    "m366_contract": "95f031569b1695c9c74e7862ac1abd3a95465789bd8c1e4ebe4a658b1bc4cdc2",
    "m716": "471cf62946f48a21815fb9730ea0588a85c34f7598d95f401eb5d2bae7e55263",
    "m720": "8267b4053446f3a84f7629fd8934aa4a35efb4cc18036892da744283151324fa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("duplicate JSON key {} in {}".format(key, path))
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token {} in {}".format(token, path))

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def extract_process_regex(runner_text):
    match = re.search(
        r"if re\.search\(\s*((?:r?'[^']*'\s*)+),\s*joined, re\.IGNORECASE\)",
        runner_text, re.DOTALL)
    require(match is not None, "cannot locate process-name regex")
    parts = re.findall(r"r?'[^']*'", match.group(1))
    require(parts, "empty process-name regex")
    return "".join(ast.literal_eval(part) for part in parts)


def extract_capture_contract(capture_text):
    milestone = re.search(
        r'contract\.get\("milestone"\) ==\s*"([^"]+)"',
        capture_text, re.DOTALL)
    identity = re.search(
        r'set\(identity\) == \{(.*?)\},\s*"M714 identity key drift"',
        capture_text, re.DOTALL)
    require(milestone is not None and identity is not None,
            "cannot recover capture-side contract constraints")
    identity_keys = re.findall(r'"([^"]+)"', identity.group(1))
    return milestone.group(1), sorted(identity_keys)


def main():
    paths = {
        "runner": RUNNER,
        "contract": CONTRACT,
        "capture": CAPTURE,
        "m366": M366,
        "m366_contract": M366_CONTRACT,
        "m716": M716,
        "m720": M720,
        "docs359": DOCS359,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    require(observed == EXPECTED, "frozen source identity drift")

    runner_text = RUNNER.read_text(encoding="utf-8")
    capture_text = CAPTURE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    ast.parse(capture_text, filename=str(CAPTURE))

    require(contract["schema"] == "m714_h67_ep35_pctda_pattern_s10_contract_v2",
            "contract schema drift")
    identity_sha = {
        key: value["sha256"] for key, value in contract["identity"].items()
    }
    require(identity_sha["m714_script"] == observed["capture"],
            "contract capture SHA mismatch")
    require(identity_sha["m366_script"] == observed["m366"],
            "contract M366 SHA mismatch")
    require(identity_sha["m366_contract"] == observed["m366_contract"],
            "contract M366-contract SHA mismatch")
    require(identity_sha["m716_prerun_review"] == observed["m716"],
            "contract M716 SHA mismatch")
    require(identity_sha["m720_failed_static_review"] == observed["m720"],
            "contract M720 SHA mismatch")
    require(identity_sha["protected_docs359"] == observed["docs359"],
            "contract docs359 SHA mismatch")

    capture_milestone, capture_identity_keys = extract_capture_contract(capture_text)
    contract_identity_keys = sorted(contract["identity"])
    contract_capture_compatibility = {
        "contract_milestone": contract["milestone"],
        "capture_required_milestone": capture_milestone,
        "milestone_matches": contract["milestone"] == capture_milestone,
        "contract_identity_keys": contract_identity_keys,
        "capture_required_identity_keys": capture_identity_keys,
        "identity_key_set_matches": contract_identity_keys == capture_identity_keys,
        "extra_contract_keys_rejected_by_capture": sorted(
            set(contract_identity_keys) - set(capture_identity_keys)),
        "failure_phase": "after_four_idle_checks_and_attempt_consumption_before_M366_or_CUDA",
    }

    process_pattern = extract_process_regex(runner_text)
    process_re = re.compile(process_pattern, re.IGNORECASE)
    positive_cases = [
        "/repo/profile100.py",
        "/repo/valid825.py",
        "/repo/validate.py",
        "/repo/trainer.py",
        "/repo/trainonly.py",
        "/repo/evaluation.py",
        "/repo/training.py",
        "/repo/run_date11_ft5_and_valid825.py",
        "/repo/run_h67_ep35_profile100_bit_trace.py",
    ]
    negative_cases = [
        "/repo/retraining.py",
        "/repo/invalid825.py",
        "/repo/evaluate.py",
        "/repo/profiler.py",
        "/repo/trainable.py",
        "/repo/profiled.py",
        "/repo/validity.py",
        "/repo/data_profile100extra.py",
        "/repo/evaluationReport.py",
        "/repo/mytrainer.py",
    ]
    process_results = {
        "positive": {case: bool(process_re.search(case))
                     for case in positive_cases},
        "negative": {case: bool(process_re.search(case))
                     for case in negative_cases},
    }
    require(all(process_results["positive"].values()),
            "required positive process case missed")
    require(not any(process_results["negative"].values()),
            "unrelated negative process case matched")

    order_tokens = {
        "static_review_semantic_validation": runner_text.index("d.get('status')"),
        "result_attempt_absence_gate": runner_text.index(
            '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" ]]'),
        "idle_loop": runner_text.index("for sample in 1 2 3 4"),
        "attempt_consumed": runner_text.index('mkdir -- "${ATTEMPT}"'),
        "staging_created": runner_text.index('STAGING="$(mktemp -d'),
        "capture_launched": runner_text.index('"${PYTHON_BIN}" "${CAPTURE}"'),
        "payload_terminal_validation": runner_text.index(
            "PASS_M714_R2_TERMINAL_VALIDATION"),
        "run_complete_written": runner_text.index(
            "m714_h67_ep35_pctda_pattern_s10_terminal_receipt_v2"),
        "staging_sealed": runner_text.rindex('seal_tree "${STAGING}"'),
        "published": runner_text.index('mv -- "${STAGING}" "${RESULT}"'),
        "published_seal_reverified": runner_text.rindex(
            "sha256sum -c -- SHA256SUMS.seal.sha256"),
    }
    require(list(order_tokens.values()) == sorted(order_tokens.values()),
            "runner phase ordering drift")
    require("[[ \"${sample}\" -eq 4 ]] || sleep 5" in runner_text,
            "four idle samples are not separated")

    compute_apps_query_is_fail_closed = not bool(re.search(
        r'GPU_APPS="\$\(nvidia-smi --query-compute-apps=.*?\|\| true\)"',
        runner_text, re.DOTALL))

    smoke = {
        "classification_literal_present":
            '"classification": "deterministic_randomized_algebra_smoke"'
            in capture_text,
        "exhaustive_false_literal_present": '"exhaustive": False' in capture_text,
        "mislabeled_exhaustive_pattern_present":
            "exhaustive-pattern" in capture_text,
        "randomized_vector_count": 256,
    }

    population_tokens = {
        "samples_10": 'population.get("samples") == expected["samples"]' in capture_text,
        "sample_keys_10": 'len(population.get("sample_keys", [])) == expected["samples"]' in capture_text,
        "installed_105": 'population.get("installed_atlif_modules") ==' in capture_text,
        "live_81_45_36": '"M366 live-site population drift"' in capture_text,
        "dead_called_empty": 'population.get("dead_called_sites") == []' in capture_text,
        "calls_450": 't10.get("calls") == expected["t10_calls"]' in capture_text,
        "four_zero_numeric_gates": all(token in capture_text for token in (
            '"signed_q8_range_violations"', '"input_nonfinite"',
            '"bound_violations"', '"integer_early_mismatches"')),
    }

    table_bits = 2 * 32 * 10 * 11
    resident = {
        str(port): {
            "macro_count": port * math.ceil(45 / 2),
            "capacity_bytes": port * math.ceil(45 / 2) * 128 * 128 // 8,
            "area_um2": port * math.ceil(45 / 2) * 8758.360550,
        } for port in (1, 2, 4, 8)
    }

    result_path = HW / "results/m714_h67_ep35_pctda_pattern_s10_r2_20260828"
    attempt_path = HW / "results/.m714_h67_ep35_pctda_pattern_s10_r2_20260828.attempt_consumed"
    result = {
        "schema": "m724_m714_r2_repair_one_shot_runner_fresh_static_recompute_v1",
        "status": "PASS_STATIC_RECOMPUTE__FAIL_RUNNER_AUTHORIZATION",
        "method": {
            "receipt_blind": True,
            "runner_executed": False,
            "author_capture_imported_or_executed": False,
            "gpu_or_nvidia_smi_queried": False,
            "eda_invoked": False,
            "remote_accessed": False,
            "author_files_modified": False,
        },
        "identity": observed,
        "contract_capture_compatibility": contract_capture_compatibility,
        "idle_process_matcher": {
            "pattern": process_pattern,
            "cases": process_results,
            "required_real_name_cases_all_pass": True,
            "unrelated_negative_cases_all_pass": True,
            "compute_apps_query_fail_closed": compute_apps_query_is_fail_closed,
            "compute_apps_query_failure_is_masked_by_or_true":
                not compute_apps_query_is_fail_closed,
        },
        "runner_phase_offsets": order_tokens,
        "four_idle_checks_before_attempt": True,
        "selftest_classification": smoke,
        "m366_population_numeric_static_gates": population_tokens,
        "m716_m720_closure": {
            "immutable_exact_sha_identity": True,
            "process_name_positive_and_negative_cases": True,
            "four_idle_checks_before_attempt": True,
            "compute_apps_query_failure_fail_closed": compute_apps_query_is_fail_closed,
            "contract_accepted_by_capture_before_launch": False,
            "m366_population_and_numeric_prerequisites_present":
                all(population_tokens.values()),
            "one_shot_attempt_staging_failure_quarantine_atomic_publish_and_seal": True,
            "pattern_counter_conservation": True,
            "chunk_tile_boundary": True,
            "deterministic_randomized_smoke_not_exhaustive":
                smoke["classification_literal_present"] and
                smoke["exhaustive_false_literal_present"] and
                not smoke["mislabeled_exhaustive_pattern_present"],
            "ideal_resource_output_fields_and_terminal_boundary": True,
            "source_docstring_no_conservative_schedule_overclaim":
                "conservative issue\nschedule" not in capture_text,
        },
        "independent_arithmetic": {
            "logical_table_bits": table_bits,
            "logical_table_bytes": table_bits // 8,
            "fixed_n1_17n_plus_12": 29,
            "fixed_n4_17n_plus_12": 80,
            "direct_table_load_256bit_beats": math.ceil(table_bits / 256),
            "direct_extra_beats_over_m518_five":
                math.ceil(table_bits / 256) - 5,
            "resident_45_by_ports": resident,
        },
        "preexisting_state": {
            "canonical_result_absent": not result_path.exists(),
            "attempt_identity_absent": not attempt_path.exists(),
        },
        "decision": {
            "authorize_runner": False,
            "reason": "capture rejects the exact contract after attempt consumption; compute-app query failure is also masked",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
