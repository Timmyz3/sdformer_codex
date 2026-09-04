#!/usr/bin/env python3
"""Receipt-blind static audit of the repaired M714-r2 one-shot runner.

This program reads source and frozen review identities only.  It does not
execute/import the M714 capture, invoke the runner, query a GPU, call EDA, or
access a remote system.
"""

import ast
import hashlib
import json
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
M724 = HW / "reviews/m724_m714_r2_repair_one_shot_runner_fresh_static_hammer_r1_20260828/review.json"
M728 = HW / "reviews/m728_m714_r2_repair2_one_shot_runner_fresh_static_hammer_r1_20260828/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m714_h67_ep35_pctda_pattern_s10_r2_20260828"
ATTEMPT = HW / "results/.m714_h67_ep35_pctda_pattern_s10_r2_20260828.attempt_consumed"

EXPECTED = {
    "runner": "350d4bb063f469ecea7729f51c1f23b9a7aaca5f5198abcab95669618df09c28",
    "contract": "8e58fe96c1c05b1c6713231e36e799f7e68b55f073c4044433a01eb0b308ebd5",
    "capture": "28457d9d2cb94bfe10c8655affdeb4bb51199d72cbb94b6d4398eb893a44c63c",
    "m366": "c4b2e83b2a1341f9790038d395aa8ed4c25c75bc441e932def4e2e32b1ba4045",
    "m366_contract": "95f031569b1695c9c74e7862ac1abd3a95465789bd8c1e4ebe4a658b1bc4cdc2",
    "m716": "471cf62946f48a21815fb9730ea0588a85c34f7598d95f401eb5d2bae7e55263",
    "m720": "8267b4053446f3a84f7629fd8934aa4a35efb4cc18036892da744283151324fa",
    "m724": "b4395d84635838091e70e0576c18c48f777dd2f10a9ed7fecc4eb6074ae512ec",
    "m728": "5c0bf22c3ded0ef75223e6211a7ecad36ea7f4dfdb68c007ef4e3485c437091e",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate JSON key {}".format(key))
            out[key] = value
        return out

    def reject(token):
        raise RuntimeError("non-standard JSON token {}".format(token))

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def extract_process_regex(text):
    match = re.search(
        r"if re\.search\(\s*((?:r?'[^']*'\s*)+),\s*joined, re\.IGNORECASE\)",
        text, re.DOTALL)
    require(match is not None, "cannot locate process regex")
    return "".join(ast.literal_eval(part)
                   for part in re.findall(r"r?'[^']*'", match.group(1)))


def main():
    paths = {
        "runner": RUNNER, "contract": CONTRACT, "capture": CAPTURE,
        "m366": M366, "m366_contract": M366_CONTRACT, "m716": M716,
        "m720": M720, "m724": M724, "m728": M728, "docs359": DOCS359,
    }
    observed = {key: sha256(path) for key, path in paths.items()}
    require(observed == EXPECTED, "frozen identity drift")
    require(not RESULT.exists() and not ATTEMPT.exists(),
            "attempt/result already exists during static audit")

    runner = RUNNER.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    ast.parse(capture, filename=str(CAPTURE))

    require("set -Eeuo pipefail" in runner, "strict shell mode missing")
    require('if [[ $# -ne 0 ]]' in runner, "argument rejection missing")
    require(': "${M714_R2_EXPECTED_RUNNER_SHA256:?' in runner,
            "runner admission variable not mandatory")
    require(': "${M714_R2_EXPECTED_STATIC_REVIEW_OUTER_SHA256:?' in runner,
            "review outer-seal variable not mandatory")
    require("m731_m714_r2_terminal_identity_revalidation_fresh_static_hammer_v1" in runner,
            "M731 schema binding missing")
    require("PASS_M731_M714_R2_TERMINAL_IDENTITY_REVALIDATION_STATIC_HAMMER" in runner,
            "M731 status binding missing")
    require("score_100')!=100" in runner and
            "('p0_count','p1_count','p2_count')]!=[0,0,0]" in runner,
            "M731 semantic PASS gate incomplete")

    contract_identity = {key: value["sha256"]
                         for key, value in contract["identity"].items()}
    require(contract["milestone"] == "M714-r2", "contract milestone drift")
    require(contract_identity == {
        "m714_script": EXPECTED["capture"],
        "m366_script": EXPECTED["m366"],
        "m366_contract": EXPECTED["m366_contract"],
        "m716_prerun_review": EXPECTED["m716"],
        "protected_docs359": EXPECTED["docs359"],
    }, "contract identity drift")

    process_pattern = extract_process_regex(runner)
    process_re = re.compile(process_pattern, re.IGNORECASE)
    positives = [
        "/repo/profile100.py", "/repo/valid825.py", "/repo/validate.py",
        "/repo/trainer.py", "/repo/trainonly.py", "/repo/evaluation.py",
        "/repo/training.py", "/repo/run_date11_ft5_and_valid825.py",
        "/repo/run_h67_ep35_profile100_bit_trace.py",
    ]
    negatives = [
        "/repo/retraining.py", "/repo/invalid825.py", "/repo/evaluate.py",
        "/repo/profiler.py", "/repo/trainable.py", "/repo/profiled.py",
        "/repo/validity.py", "/repo/data_profile100extra.py",
        "/repo/evaluationReport.py", "/repo/mytrainer.py",
    ]
    process_cases = {
        "positive": {item: bool(process_re.search(item)) for item in positives},
        "negative": {item: bool(process_re.search(item)) for item in negatives},
    }
    require(all(process_cases["positive"].values()), "process positive miss")
    require(not any(process_cases["negative"].values()), "process false hit")

    gpu_sample = re.search(
        r'GPU_SAMPLE="\$\(nvidia-smi --query-gpu=.*?\)"', runner, re.DOTALL)
    gpu_apps = re.search(
        r'GPU_APPS="\$\(nvidia-smi --query-compute-apps=.*?\)"',
        runner, re.DOTALL)
    require(gpu_sample is not None and gpu_apps is not None,
            "GPU query missing")
    require("|| true" not in gpu_sample.group(0) and
            "|| true" not in gpu_apps.group(0), "GPU query failure masked")
    require("for sample in 1 2 3 4" in runner, "four-check loop missing")
    require('[[ "${sample}" -eq 4 ]] || sleep 5' in runner,
            "idle separation missing")
    require('"${UTIL}" -le 5' in runner and '"${USED}" -le 1024' in runner,
            "GPU numeric idle thresholds missing")
    require('-z "${GPU_APPS}" && -z "${PROCESS_HITS}"' in runner,
            "compute-app/process absence predicate missing")

    function = runner[runner.index("terminal_revalidate_identity()"):
                      runner.index("seal_tree()")]
    terminal_checks = {
        "runner_expected_and_start": (
            '"${M714_R2_EXPECTED_RUNNER_SHA256}"' in function and
            '"${START_RUNNER_SHA256}"' in function),
        "contract_expected_and_start": (
            '"${EXPECTED_CONTRACT_SHA256}"' in function and
            '"${START_CONTRACT_SHA256}"' in function),
        "capture_expected_and_start": (
            '"${EXPECTED_CAPTURE_SHA256}"' in function and
            '"${START_CAPTURE_SHA256}"' in function),
        "attempt_identity_regular": (
            '[[ -f "${ATTEMPT}/IDENTITY" && ! -L "${ATTEMPT}/IDENTITY" ]]'
            in function),
        "attempt_binds_all_three": all(token in function for token in (
            "runner_sha256=${runner_now}", "contract_sha256=${contract_now}",
            "capture_sha256=${capture_now}")),
    }
    require(all(terminal_checks.values()), "terminal identity function incomplete")

    phase = {
        "static_review": runner.index("d.get('status')"),
        "absence_gate": runner.index('[[ ! -e "${RESULT}" && ! -e "${ATTEMPT}" ]]'),
        "idle_loop": runner.index("for sample in 1 2 3 4"),
        "attempt": runner.index('mkdir -- "${ATTEMPT}"'),
        "staging": runner.index('STAGING="$(mktemp -d'),
        "capture": runner.index('"${PYTHON_BIN}" "${CAPTURE}"'),
        "terminal_1": runner.index("terminal_revalidate_identity", runner.index('"${PYTHON_BIN}" "${CAPTURE}"')),
        "payload_validation": runner.index("PASS_M714_R2_TERMINAL_VALIDATION"),
        "terminal_2": runner.index("terminal_revalidate_identity", runner.index("PASS_M714_R2_TERMINAL_VALIDATION")),
        "receipt": runner.index("m714_h67_ep35_pctda_pattern_s10_terminal_receipt_v2"),
        "terminal_3": runner.index("terminal_revalidate_identity", runner.index("m714_h67_ep35_pctda_pattern_s10_terminal_receipt_v2")),
        "seal": runner.rindex('seal_tree "${STAGING}"'),
        "publish": runner.index('mv -- "${STAGING}" "${RESULT}"'),
        "published_verify": runner.rindex("sha256sum -c -- SHA256SUMS.seal.sha256"),
    }
    require(list(phase.values()) == sorted(phase.values()), "phase order drift")
    require(runner.count("terminal_revalidate_identity\n") == 3,
            "expected three post-capture terminal revalidations")

    tail = runner[phase["capture"]:]
    payload_binding = {
        "current_contract_equals_expected":
            "sha(contract_path)!=expected_contract" in tail,
        "current_capture_equals_expected":
            "sha(capture_path)!=expected_capture" in tail,
        "current_runner_equals_expected":
            "sha(runner_path)!=expected_runner" in tail,
        "contract_pins_capture":
            "c.get('identity',{}).get('m714_script',{}).get('sha256')!=expected_capture"
            in tail,
        "payload_contract_equals_current_expected":
            "i.get('m714_contract_sha256')!=sha(contract_path)" in tail,
        "payload_capture_equals_current_expected":
            "i.get('m714_script_sha256')!=sha(capture_path)" in tail,
    }
    require(all(payload_binding.values()), "payload frozen identity binding incomplete")

    require('trap on_exit EXIT' in runner and
            'FAILED_DO_NOT_CITE rc=%s' in runner and
            'failed_or_incomplete' in runner,
            "failure quarantine missing")
    require('find . -type f ! -name SHA256SUMS' in runner and
            'sha256sum -- SHA256SUMS > SHA256SUMS.seal.sha256' in runner,
            "member/outer sealing missing")
    require('mv -- "${STAGING}" "${RESULT}"' in runner,
            "atomic same-parent publication missing")

    require("pctda_ideal_resource_issue_lower_bound" in capture and
            "pctda_executable_cycle\": False" in capture and
            "pctda_real_output_miter\": False" in capture and
            "pctda_system_speedup\": False" in capture and
            "pctda_headline\": False" in capture,
            "capture claim boundary drift")
    require(contract["admission"] == {
        "gpu_run_authorized_by_contract_alone": False,
        "separate_one_shot_runner_and_fresh_static_review_required": True,
        "pattern_capture": False, "ideal_resource_lower_bound": False,
        "real_output_miter": False, "executable_cycle": False,
        "rtl": False, "vcs": False, "synopsys_ppa": False,
        "energy": False, "accuracy": False, "system_speedup": False,
        "headline": False,
    }, "contract admission drift")

    output = {
        "status": "PASS_M731_STATIC_RECOMPUTE",
        "identity": observed,
        "bash_syntax_checked_separately": True,
        "attempt_and_result_absent": True,
        "runner_or_capture_executed": False,
        "gpu_queried": False,
        "eda_or_remote_accessed": False,
        "process_cases": process_cases,
        "terminal_identity_checks": terminal_checks,
        "phase_offsets": phase,
        "payload_binding": payload_binding,
        "claim_boundary": "ideal-resource pattern/lower-bound capture only",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
