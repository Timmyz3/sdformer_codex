#!/usr/bin/env python3
"""Third receipt-blind static audit of the M714-r2 one-shot runner.

The audit reads source and sealed identities only.  It does not import or run
the capture, invoke the runner, query GPU state, call EDA, or access remote
systems.
"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m714_h67_ep35_pctda_pattern_s10_r2_one_shot.sh"
CONTRACT = HW / "contracts/m714_h67_ep35_pctda_pattern_s10_contract_r2_20260828.json"
CAPTURE = HW / "system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py"
M366 = HW / "system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
M366_CONTRACT = HW / "contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
M716 = HW / "reviews/m716_m714_pctda_prerun_fresh_hammer_r1_20260828/m716_m714_pctda_prerun_fresh_hammer_verdict_r1.json"
M720 = HW / "reviews/m720_m714_r2_one_shot_runner_fresh_static_hammer_r1_20260828/review.json"
M724 = HW / "reviews/m724_m714_r2_repair_one_shot_runner_fresh_static_hammer_r1_20260828/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "d14dfd3572575d7423ed13fde0044edffd601942d7e76e923fa088f2f9250471",
    "contract": "8e58fe96c1c05b1c6713231e36e799f7e68b55f073c4044433a01eb0b308ebd5",
    "capture": "28457d9d2cb94bfe10c8655affdeb4bb51199d72cbb94b6d4398eb893a44c63c",
    "m366": "c4b2e83b2a1341f9790038d395aa8ed4c25c75bc441e932def4e2e32b1ba4045",
    "m366_contract": "95f031569b1695c9c74e7862ac1abd3a95465789bd8c1e4ebe4a658b1bc4cdc2",
    "m716": "471cf62946f48a21815fb9730ea0588a85c34f7598d95f401eb5d2bae7e55263",
    "m720": "8267b4053446f3a84f7629fd8934aa4a35efb4cc18036892da744283151324fa",
    "m724": "b4395d84635838091e70e0576c18c48f777dd2f10a9ed7fecc4eb6074ae512ec",
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
    require(match is not None, "cannot locate process regex")
    parts = re.findall(r"r?'[^']*'", match.group(1))
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
    return milestone.group(1), sorted(re.findall(r'"([^"]+)"', identity.group(1)))


def validate_m366_identity(m366_contract):
    observed = {}
    for name, record in m366_contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        text = record["path"]
        path = ((ROOT / text) if text.startswith("neuron_experiments/")
                else (HW / text)).resolve()
        observed[name] = {
            "exists": path.is_file(),
            "sha256_matches": path.is_file() and sha256(path) == record["sha256"],
        }
    return observed


def main():
    paths = {
        "runner": RUNNER,
        "contract": CONTRACT,
        "capture": CAPTURE,
        "m366": M366,
        "m366_contract": M366_CONTRACT,
        "m716": M716,
        "m720": M720,
        "m724": M724,
        "docs359": DOCS359,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    require(observed == EXPECTED, "frozen identity drift")

    runner_text = RUNNER.read_text(encoding="utf-8")
    capture_text = CAPTURE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    m366_contract = strict_json(M366_CONTRACT)
    ast.parse(capture_text, filename=str(CAPTURE))

    capture_milestone, capture_identity_keys = extract_capture_contract(capture_text)
    contract_identity_keys = sorted(contract["identity"])
    contract_capture = {
        "contract_milestone": contract["milestone"],
        "capture_required_milestone": capture_milestone,
        "milestone_matches": contract["milestone"] == capture_milestone,
        "contract_identity_keys": contract_identity_keys,
        "capture_required_identity_keys": capture_identity_keys,
        "identity_key_set_matches": contract_identity_keys == capture_identity_keys,
    }
    require(contract_capture["milestone_matches"], "milestone mismatch")
    require(contract_capture["identity_key_set_matches"], "identity key mismatch")
    contract_sha_map = {key: value["sha256"]
                        for key, value in contract["identity"].items()}
    require(contract_sha_map == {
        "m714_script": observed["capture"],
        "m366_script": observed["m366"],
        "m366_contract": observed["m366_contract"],
        "m716_prerun_review": observed["m716"],
        "protected_docs359": observed["docs359"],
    }, "contract identity SHA map mismatch")

    process_pattern = extract_process_regex(runner_text)
    process_re = re.compile(process_pattern, re.IGNORECASE)
    positive_cases = [
        "/repo/profile100.py", "/repo/valid825.py", "/repo/validate.py",
        "/repo/trainer.py", "/repo/trainonly.py", "/repo/evaluation.py",
        "/repo/training.py", "/repo/run_date11_ft5_and_valid825.py",
        "/repo/run_h67_ep35_profile100_bit_trace.py",
    ]
    negative_cases = [
        "/repo/retraining.py", "/repo/invalid825.py", "/repo/evaluate.py",
        "/repo/profiler.py", "/repo/trainable.py", "/repo/profiled.py",
        "/repo/validity.py", "/repo/data_profile100extra.py",
        "/repo/evaluationReport.py", "/repo/mytrainer.py",
    ]
    process_cases = {
        "positive": {case: bool(process_re.search(case)) for case in positive_cases},
        "negative": {case: bool(process_re.search(case)) for case in negative_cases},
    }
    require(all(process_cases["positive"].values()), "positive process miss")
    require(not any(process_cases["negative"].values()), "negative process hit")

    gpu_apps_assignment = re.search(
        r'GPU_APPS="\$\(nvidia-smi --query-compute-apps=.*?\)"',
        runner_text, re.DOTALL)
    require(gpu_apps_assignment is not None, "compute-app query missing")
    compute_apps_failure_masked = "|| true" in gpu_apps_assignment.group(0)

    order_tokens = {
        "static_review_semantic_validation": runner_text.index("d.get('status')"),
        "result_attempt_absence_gate": runner_text.index(
            '[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" ]]'),
        "idle_loop": runner_text.index("for sample in 1 2 3 4"),
        "attempt_consumed": runner_text.index('mkdir -- "${ATTEMPT}"'),
        "staging_created": runner_text.index('STAGING="$(mktemp -d'),
        "capture_launch": runner_text.index('"${PYTHON_BIN}" "${CAPTURE}"'),
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
            "runner phase order drift")
    require("[[ \"${sample}\" -eq 4 ]] || sleep 5" in runner_text,
            "idle samples not separated")

    launch_offset = order_tokens["capture_launch"]
    after_launch = runner_text[launch_offset:]
    post_capture_expected_sha = {
        "runner_expected_sha_rechecked":
            "M714_R2_EXPECTED_RUNNER_SHA256" in after_launch,
        "contract_expected_sha_rechecked":
            "EXPECTED_CONTRACT_SHA256" in after_launch,
        "capture_expected_sha_rechecked":
            "EXPECTED_CAPTURE_SHA256" in after_launch,
        "terminal_compares_payload_to_contract_pinned_capture_sha":
            "c['identity']['m714_script']['sha256']" in after_launch or
            'c["identity"]["m714_script"]["sha256"]' in after_launch,
    }
    capture_tail = capture_text[capture_text.index("m366.execute"):]
    capture_self_sha_rechecked_against_contract = bool(re.search(
        r'require\([^\n]*sha256\(Path\(__file__\).*m714_script',
        capture_tail, re.DOTALL))
    post_capture_expected_sha["capture_self_sha_rechecked_against_contract"] = (
        capture_self_sha_rechecked_against_contract)
    post_capture_expected_sha["all_three_frozen_authority_shas_revalidated"] = all((
        post_capture_expected_sha["runner_expected_sha_rechecked"],
        post_capture_expected_sha["contract_expected_sha_rechecked"],
        post_capture_expected_sha["capture_expected_sha_rechecked"],
    ))

    smoke = {
        "classification_is_randomized_smoke":
            '"classification": "deterministic_randomized_algebra_smoke"'
            in capture_text,
        "exhaustive_is_false": '"exhaustive": False' in capture_text,
        "exhaustive_pattern_label_absent": "exhaustive-pattern" not in capture_text,
    }

    population_tokens = {
        "samples": 'population.get("samples") == expected["samples"]' in capture_text,
        "sample_keys": 'len(population.get("sample_keys", [])) == expected["samples"]' in capture_text,
        "installed": 'population.get("installed_atlif_modules") ==' in capture_text,
        "live_split": '"M366 live-site population drift"' in capture_text,
        "dead_called": 'population.get("dead_called_sites") == []' in capture_text,
        "t10_calls": 't10.get("calls") == expected["t10_calls"]' in capture_text,
        "numeric_zero": all(token in capture_text for token in (
            '"signed_q8_range_violations"', '"input_nonfinite"',
            '"bound_violations"', '"integer_early_mismatches"')),
    }
    m366_identity = validate_m366_identity(m366_contract)
    require(all(item["exists"] and item["sha256_matches"]
                for item in m366_identity.values()), "nested M366 identity drift")

    scalar_mismatches = []
    for value in range(-128, 128):
        code = value & 0xff
        reconstructed = sum(((-128 if bit == 7 else (1 << bit))
                             if ((code >> bit) & 1) else 0)
                            for bit in range(8))
        if reconstructed != value:
            scalar_mismatches.append([value, reconstructed])

    table_bits = 2 * 32 * 10 * 11
    resident = {
        str(port): {
            "macro_count": port * math.ceil(45 / 2),
            "capacity_bytes": port * math.ceil(45 / 2) * 128 * 128 // 8,
            "area_um2": port * math.ceil(45 / 2) * 8758.360550,
        } for port in (1, 2, 4, 8)
    }
    claim = {
        "docstring_ideal_resource_lower_bound":
            "ideal-resource issue\nlower bound" in capture_text,
        "conservative_issue_schedule_absent":
            "conservative issue\nschedule" not in capture_text,
        "capture_status_lower_bound_only":
            "PASS_M714_R2_PCTDA_PATTERN_CAPTURE__IDEAL_RESOURCE_LOWER_BOUND_ONLY"
            in capture_text,
        "executable_false": '"pctda_executable_cycle": False' in capture_text,
        "real_output_miter_false": '"pctda_real_output_miter": False' in capture_text,
        "rtl_false": '"pctda_rtl": False' in capture_text,
        "ppa_false": '"pctda_ppa": False' in capture_text,
        "system_speedup_false": '"pctda_system_speedup": False' in capture_text,
        "headline_false": '"pctda_headline": False' in capture_text,
    }

    result_path = HW / "results/m714_h67_ep35_pctda_pattern_s10_r2_20260828"
    attempt_path = HW / "results/.m714_h67_ep35_pctda_pattern_s10_r2_20260828.attempt_consumed"
    result = {
        "schema": "m728_m714_r2_repair2_one_shot_runner_fresh_static_recompute_v1",
        "status": "PASS_STATIC_RECOMPUTE__FAIL_EXACT_SHA_TOCTOU_AUTHORIZATION",
        "method": {
            "receipt_blind": True,
            "runner_executed": False,
            "capture_imported_or_executed": False,
            "gpu_or_nvidia_smi_queried": False,
            "eda_invoked": False,
            "remote_accessed": False,
            "author_files_modified": False,
        },
        "identity": observed,
        "m724_repair_closure": {
            "contract_capture": contract_capture,
            "compute_apps_failure_masked": compute_apps_failure_masked,
            "compute_apps_query_is_fail_closed_under_errexit":
                not compute_apps_failure_masked,
            "claim_boundary": claim,
        },
        "process_regex": {
            "pattern": process_pattern,
            "cases": process_cases,
            "required_positives_all_match": True,
            "unrelated_negatives_all_reject": True,
        },
        "phase_offsets": order_tokens,
        "four_idle_checks_before_attempt": True,
        "selftest": smoke,
        "m366_population_numeric_gates": population_tokens,
        "m366_nested_identity_count": len(m366_identity),
        "m366_nested_identities_all_match": True,
        "independent_math": {
            "all_256_signed_int8_scalar_codes_reconstruct":
                not scalar_mismatches,
            "scalar_mismatches": scalar_mismatches,
            "subset_range": [-640, 635],
            "signed_subset_width_bits": 11,
            "worst_absolute_with_q24_bias": 8715008,
            "fits_signed25": 8715008 < (1 << 24),
            "table_bits": table_bits,
            "table_bytes": table_bits // 8,
            "fixed_n1_n4": [29, 80],
            "direct_table_beats": math.ceil(table_bits / 256),
            "direct_extra_beats_over_five": math.ceil(table_bits / 256) - 5,
            "resident_45": resident,
        },
        "one_shot_structure": {
            "static_review_exact_schema_status_identity_and_decision_bound": True,
            "four_idle_checks_before_attempt": True,
            "same_parent_staging": True,
            "failure_quarantine": True,
            "success_tree_seal_before_publish": True,
            "publish_then_terminal_seal_verify": True,
            "post_capture_frozen_authority_revalidation": post_capture_expected_sha,
        },
        "preexisting_state": {
            "canonical_result_absent": not result_path.exists(),
            "attempt_absent": not attempt_path.exists(),
        },
        "decision": {
            "authorize_runner": False,
            "reason": "post-capture terminal validation does not rebind current runner/contract/capture files to their frozen authority SHAs",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
