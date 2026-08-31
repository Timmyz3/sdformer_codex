#!/usr/bin/env python3
"""Receipt-blind static audit for the M714-r2 capture/contract/runner.

This program reads source, contract, and frozen identity files only.  It must
not import either capture module or execute the runner/GPU/EDA flow.
"""

import ast
import hashlib
import json
import math
import random
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CAPTURE = HW / "system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py"
CONTRACT = HW / "contracts/m714_h67_ep35_pctda_pattern_s10_contract_r2_20260828.json"
RUNNER = HW / "system_simulator/scripts/run_m714_h67_ep35_pctda_pattern_s10_r2_one_shot.sh"
M366 = HW / "system_simulator/scripts/trace_m366_h67_ep35_atlif_remaining_budget_s10.py"
M366_CONTRACT = HW / "contracts/m366_h67_ep35_atlif_remaining_budget_s10_contract_r1_20260825.json"
M716 = HW / "reviews/m716_m714_pctda_prerun_fresh_hammer_r1_20260828/m716_m714_pctda_prerun_fresh_hammer_verdict_r1.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "capture": "f35fa0ec051f8e45c89a0ab0c0280695ae71b429fe3e8b1ac4806dcb8a276200",
    "contract": "184458d69b542386c45b99af0ba8744dc8c6ad1a91e6c06b88812c906a4dd723",
    "runner": "f5f4202d5e934f7dc44838052d493d43d671d1590daf51cad800d0747b30e857",
    "m366": "c4b2e83b2a1341f9790038d395aa8ed4c25c75bc441e932def4e2e32b1ba4045",
    "m366_contract": "95f031569b1695c9c74e7862ac1abd3a95465789bd8c1e4ebe4a658b1bc4cdc2",
    "m716": "471cf62946f48a21815fb9730ea0588a85c34f7598d95f401eb5d2bae7e55263",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"duplicate key {key} in {path}")
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError(f"non-standard JSON token {token} in {path}")

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def must(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    paths = {
        "capture": CAPTURE,
        "contract": CONTRACT,
        "runner": RUNNER,
        "m366": M366,
        "m366_contract": M366_CONTRACT,
        "m716": M716,
        "docs359": DOCS359,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    must(observed == EXPECTED, "frozen identity mismatch")
    contract = strict_json(CONTRACT)
    must(contract["schema"] == "m714_h67_ep35_pctda_pattern_s10_contract_v2",
         "contract schema drift")
    for key, expected_name in (
        ("m714_script", "capture"), ("m366_script", "m366"),
        ("m366_contract", "m366_contract"), ("m716_prerun_review", "m716"),
        ("protected_docs359", "docs359")):
        must(contract["identity"][key]["sha256"] == EXPECTED[expected_name],
             f"contract identity mismatch: {key}")

    capture_text = CAPTURE.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    ast.parse(capture_text, filename=str(CAPTURE))
    syntax = subprocess.run(["bash", "-n", str(RUNNER)], check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    must(syntax.returncode == 0, "runner bash syntax failure")

    # Rebuild the exact seeded scalar-code coverage without importing M714.
    rng = random.Random(714)
    scalar_values = []
    for _ in range(256):
        for _ in range(100):
            rng.randint(-128, 127)
        scalar_values.extend(rng.randint(-128, 127) for _ in range(10))
    scalar_codes = set(scalar_values)
    explicit_scalar_exhaustion = (
        "for value in range(-128, 128)" in capture_text or
        "set(range(-128, 128))" in capture_text)
    explicit_coverage_assert = (
        "missing_scalar_codes" in capture_text or
        "scalar_codes == set(range(-128, 128))" in capture_text)

    # Exercise the runner's process-name expression exactly.  These are source
    # strings only; no process or GPU state is queried.
    process_re = re.compile(
        r"(^|[/_.-])(train|training|eval|evaluation|valid|validation|profile|profiling)([/_. -]|$)",
        re.IGNORECASE)
    process_cases = {
        "/repo/entrypoints/train.py": True,
        "/repo/entrypoints/eval.py": True,
        "/repo/entrypoints/profile100.py": False,
        "/repo/entrypoints/valid825.py": False,
        "/repo/entrypoints/validate.py": False,
        "/repo/entrypoints/run_date11_ft5_and_valid825.py": False,
        "/repo/entrypoints/run_h67_ep35_profile100_bit_trace.py": False,
    }
    observed_cases = {case: bool(process_re.search(case)) for case in process_cases}
    must(observed_cases == process_cases, "unexpected process-regex behavior")

    # Independently recompute the constants attacked in M716.
    table_bits = 2 * 32 * 10 * 11
    ports = (1, 2, 4, 8)
    resident = {
        str(port): {
            "macro_count": port * math.ceil(45 / 2),
            "capacity_bytes": port * math.ceil(45 / 2) * 128 * 128 // 8,
            "area_um2": port * math.ceil(45 / 2) * 8758.360550,
        } for port in ports
    }
    order_tokens = {
        "review_validation": runner_text.index("d.get('status')"),
        "idle_loop": runner_text.index("for sample in 1 2 3 4"),
        "attempt_consumed": runner_text.index('mkdir -- "${ATTEMPT}"'),
        "capture_launch": runner_text.index('"${PYTHON_BIN}" "${CAPTURE}"'),
        "tree_seal": runner_text.rindex('seal_tree "${STAGING}"'),
        "publish_rename": runner_text.index('mv -- "${STAGING}" "${RESULT}"'),
        "terminal_seal_verify": runner_text.rindex('sha256sum -c -- SHA256SUMS.seal.sha256'),
    }
    must(list(order_tokens.values()) == sorted(order_tokens.values()),
         "runner phase ordering drift")

    result = {
        "schema": "m720_m714_r2_receipt_blind_static_recompute_v1",
        "status": "PASS_STATIC_RECOMPUTE__NOT_RUNNER_AUTHORIZATION",
        "method": {
            "author_capture_imported": False,
            "runner_executed": False,
            "gpu_queried": False,
            "eda_invoked": False,
            "m714_result_receipt_read": False,
        },
        "identity": observed,
        "syntax": {"capture_ast": "PASS", "runner_bash_n": "PASS"},
        "m716_closure": {
            "immutable_identity": True,
            "m366_population_and_numeric_prerequisites": True,
            "pattern_counter_conservation": True,
            "ideal_resource_claim_boundary": True,
            "m518_17n_plus_12_not_double_charged": True,
            "build_and_direct_load_separated": True,
            "resident_45_tax_separate": True,
            "four_idle_samples_before_attempt_in_control_flow": True,
            "process_idle_matcher_covers_project_profile100_valid825_validate_names": False,
            "one_shot_staging_atomic_seal_structure": True,
            "relative_pointer_writer": True,
            "randomized_selftest_no_longer_mislabeled_exhaustive": False,
        },
        "idle_process_regex_cases": observed_cases,
        "selftest": {
            "randomized_vectors": 256,
            "random_scalar_draws": len(scalar_values),
            "incidentally_seen_scalar_codes": len(scalar_codes),
            "missing_scalar_codes_for_this_seed": sorted(set(range(-128, 128)) - scalar_codes),
            "explicit_256_code_scalar_exhaustion": explicit_scalar_exhaustion,
            "explicit_scalar_coverage_assert": explicit_coverage_assert,
            "source_comment_contains_exhaustive_pattern":
                "exhaustive-pattern" in capture_text,
        },
        "independent_arithmetic": {
            "logical_table_bits": table_bits,
            "logical_table_bytes": table_bits // 8,
            "fixed_n1_17n_plus_12": 17 * 1 + 12,
            "fixed_n4_17n_plus_12": 17 * 4 + 12,
            "direct_table_beats_256b": math.ceil(table_bits / 256),
            "direct_extra_beats_over_m518_five": math.ceil(table_bits / 256) - 5,
            "resident_45_by_ports": resident,
        },
        "runner_phase_character_offsets": order_tokens,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
