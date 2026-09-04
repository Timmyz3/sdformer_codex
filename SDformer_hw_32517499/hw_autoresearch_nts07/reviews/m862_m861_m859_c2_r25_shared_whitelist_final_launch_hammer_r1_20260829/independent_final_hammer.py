#!/usr/bin/env python3
"""Fresh independent M862 final-launch hammer for M859/C2 R25.

This is deliberately source/synthetic-filesystem only.  It never invokes the
released runner, VCS, simv, lmutil, any HDL compiler, any Synopsys EDA tool,
or a CPU/GPU/remote workload.  All launch-chain tests use temporary synthetic
final-hammer directories and the fixed real M861 release.
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


# Dynamic import of the frozen source-hammer program must not add a pycache
# member to its already sealed review directory.
sys.dont_write_bytecode = True


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
sys.path.insert(0, str(HW / "verif_m859"))
import m859_c2_r25_shared_whitelist_guard as guard  # noqa: E402


RUNNER = HW / "dc_handoff/scripts/run_vcs_m859_c2_r25_shared_whitelist_exact_sha.sh"
GUARD = HW / "verif_m859/m859_c2_r25_shared_whitelist_guard.py"
CONTRACT = HW / "contracts/m859_c2_r25_shared_whitelist_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m859_c2_r25_shared_whitelist_vcs_launch_candidate_source_only_r1_20260829.json"
RELEASE = HW / "contracts/m861_m859_c2_r25_shared_whitelist_vcs_launch_admission_r1_20260829.json"
SOURCE_HAMMER = HW / "reviews/m860_m859_c2_r25_shared_whitelist_source_fresh_hammer_r1_20260829"
HANDOFF = HW / "reviews/m861_m859_c2_r25_shared_whitelist_true_release_author_handoff_r1_20260829"
REQUEST = HW / "reviews/m862_m861_m859_c2_r25_shared_whitelist_final_launch_hammer_REQUEST_r1_20260829"
SOURCE_HAMMER_PROGRAM = SOURCE_HAMMER / "hammer_m862.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
R24_RUNNER = HW / "dc_handoff/scripts/run_vcs_m851_c2_r24_recursive_seal_exact_sha.sh"

RUNNER_SHA = "da423b17f6245b0e9af9cc6df05a846e221175da45bfbce9408fe91930a9f8d6"
GUARD_SHA = "5622a5ade16c18091e7f2facd37bcd3d39565c1c7bf6fe694f5c1362fc07e224"
CONTRACT_SHA = "a7458798f11b0ba02d83072d93cf6185508de0e882eb9bf4c02a0b7380e66c5f"
CANDIDATE_SHA = "bf8599efc0ebce9b7e11b6d2ca38061b869c6555bddd620acb93a0ae3332696e"
RELEASE_SHA = "427c09a2da0f41911dcc3ee8c407f7f2ee5717318152ce74d9bb58d6ece3194e"
RELEASE_OUTER = "98ddfe00f2538093c033e4bb8db0e685a5c8ec0830abf9f0b9445a633782449d"
SOURCE_REVIEW_SHA = "7d4af7e1651c6710032db41dc1e3cfca5aff544fb92f343adca52fe0fd8cc4be"
SOURCE_MANIFEST_SHA = "b98d78e6dd3a747ab71bfb509e489b98949fd9a5083874b0f6e56de08fa38ff8"
SOURCE_OUTER = "c4a33ac47c75c226b85256a7f6dae99947280f46e4e62e42b6d2ec0baea4f20b"
HANDOFF_JSON_SHA = "cd24d5bbaf43d8e8a3707b65d7a2a1a3a8446e0a7c1c192628f9a4c90b156165"
HANDOFF_MANIFEST_SHA = "e0660e84b116c7f5adae24375b76a48a2f465f43531b2359630e217bacd0c18b"
HANDOFF_OUTER = "32dfb4ee5e6a6247505db03313702167f437481b53195e00e964c263a1cd8529"
REQUEST_JSON_SHA = "b13c746d69bf24e8e09d07702b392ab04a72b65ddf1bc334b1db6e16d4de9df9"
REQUEST_MANIFEST_SHA = "85d6c47b0b6826addd08447e832f008e2c925ff81ffea4c0260e3ba053928b87"
REQUEST_OUTER = "6c1278b2b322e94cd20d5f0c5bb8410a02095d02d7c54acf2f777b88da808d52"
M856_OUTER = "96fd220ea2061390dccb0563ce3e0592a5d6ea0d7f0b067146032e32eeccda67"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
COMPILE_BLOCK_SHA = "b6f6753be90a2f5c8ab5a3ab2e7acdb1095bc2d31fe033fe7486ad4b998d9ad2"
COMMAND_BLOCK_SHA = "261d47f0a57fd76176c63961b472d353f7e294e9b40a05a8571bed475005ef14"
EXACT_CYCLES = {"k8": [51, 131, 486, 1231, 14],
                "k1x8": [53, 133, 499, 1246, 14]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def clean_env():
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8"}


def run_checked(command, expected_rc=0, env=None):
    completed = subprocess.run(command, env=env or clean_env(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    output = completed.stdout.decode("utf-8", "replace")
    require(completed.returncode == expected_rc,
            "command rc {} expected {}: {}\n{}".format(
                completed.returncode, expected_rc, command, output))
    return output


def formal_population():
    result = HW / "results"
    return sorted(path.name for path in result.iterdir()
                  if path.name.startswith(
                      (".m859_c2_r25_shared_whitelist_vcs_",
                       "m859_c2_r25_shared_whitelist_vcs_r1_20260829")))


def function_block(text, name, next_name):
    start = text.index(name + "() {")
    end = text.index(next_name + "() {", start)
    return text[start:end]


def between(text, begin, end):
    start = text.index(begin)
    stop = text.index(end, start)
    return text[start:stop]


def load_source_hammer():
    spec = importlib.util.spec_from_file_location(
        "m862_r25_source_hammer_replay", SOURCE_HAMMER_PROGRAM)
    require(spec is not None and spec.loader is not None,
            "cannot load frozen M860 source hammer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_fixed_chain():
    fixed = {
        RUNNER: RUNNER_SHA, GUARD: GUARD_SHA, CONTRACT: CONTRACT_SHA,
        CANDIDATE: CANDIDATE_SHA, RELEASE: RELEASE_SHA,
        SOURCE_HAMMER / "review.json": SOURCE_REVIEW_SHA,
        SOURCE_HAMMER / "SHA256SUMS": SOURCE_MANIFEST_SHA,
        SOURCE_HAMMER / "SHA256SUMS.seal.sha256": SOURCE_OUTER,
        HANDOFF / "handoff.json": HANDOFF_JSON_SHA,
        HANDOFF / "SHA256SUMS": HANDOFF_MANIFEST_SHA,
        HANDOFF / "SHA256SUMS.seal.sha256": HANDOFF_OUTER,
        REQUEST / "request.json": REQUEST_JSON_SHA,
        REQUEST / "SHA256SUMS": REQUEST_MANIFEST_SHA,
        REQUEST / "SHA256SUMS.seal.sha256": REQUEST_OUTER,
        Path(str(RELEASE) + ".sha256.seal.sha256"): RELEASE_OUTER,
        DOCS359: DOCS359_SHA,
    }
    for path, expected in fixed.items():
        require(path.is_file() and not path.is_symlink(),
                "fixed identity not regular: " + str(path))
        require(sha256(path) == expected,
                "fixed identity SHA drift: " + str(path))
    for directory, manifest, outer in (
            (SOURCE_HAMMER, SOURCE_MANIFEST_SHA, SOURCE_OUTER),
            (HANDOFF, HANDOFF_MANIFEST_SHA, HANDOFF_OUTER),
            (REQUEST, REQUEST_MANIFEST_SHA, REQUEST_OUTER)):
        identity = guard.base.verify_sealed_directory(directory)
        require(identity["manifest_sha256"] == manifest and
                identity["outer_seal_file_sha256"] == outer,
                "sealed identity drift: " + str(directory))
    guard.base.verify_double_sealed_file(RELEASE)
    guard.base.verify_double_sealed_file(CONTRACT)
    guard.base.verify_double_sealed_file(CANDIDATE)

    source = guard.validate_source(HW, CONTRACT, CANDIDATE, RUNNER)
    require(source["runner_sha256"] == RUNNER_SHA and
            source["contract_sha256"] == CONTRACT_SHA and
            source["candidate_sha256"] == CANDIDATE_SHA and
            source["m856_outer_seal_file_sha256"] == M856_OUTER,
            "live source binding drift")

    release = guard.base.strict_json(RELEASE)
    require(release["schema"] ==
            "m859_c2_r25_shared_whitelist_vcs_launch_admission_v1" and
            release["status"] == guard.RELEASE_STATUS,
            "release schema/status drift")
    guard.require_exact_mapping(release["authorization"], {
        "launch_now": True, "run_vcs": True, "run_simv": True,
        "query_license": True, "run_eda": False, "max_attempts": 1,
    }, "release authorization")
    guard.require_exact_mapping(release["source_binding"], {
        "runner_sha256": RUNNER_SHA,
        "contract_sha256": CONTRACT_SHA,
        "candidate_sha256": CANDIDATE_SHA,
        "source_hammer_outer_seal_file_sha256": SOURCE_OUTER,
        "m856_outer_seal_file_sha256": M856_OUTER,
    }, "release source binding")
    guard.require_exact_mapping(
        release["final_hammer_authorization_exact"]["authorization"],
        guard.FINAL_HAMMER_AUTHORIZATION,
        "release embedded final authorization")
    require(release["final_hammer_authorization_exact"]["key_count"] == 15,
            "release authorization key count drift")

    contract = guard.base.strict_json(CONTRACT)
    for relative, expected in contract["source_sha256"].items():
        path = HW / relative
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected,
                "contract source binding drift: " + relative)
    m803 = [relative for relative in contract["source_sha256"]
            if relative.startswith(("rtl_m803/", "tb_m803/",
                                    "verif_m803/")) or
            relative.startswith("dc_handoff/filelists/date_m803")]
    require(len(m803) == 9, "M803 frozen population is not nine files")

    old = R24_RUNNER.read_text(encoding="utf-8")
    new = RUNNER.read_text(encoding="utf-8")
    compile_block = function_block(new, "compile_and_run",
                                   "publish_failure_receipt")
    command_block = between(new, "log_phase ATTACK_VCS",
                            "log_phase RESULT_STAGE_SEAL")
    require(compile_block == function_block(
        old, "compile_and_run", "publish_failure_receipt"),
        "compile-and-run changed from frozen R24")
    require(command_block == between(
        old, "log_phase ATTACK_VCS", "log_phase RESULT_STAGE_SEAL"),
        "attack/equal-bandwidth command gates changed")
    require(hashlib.sha256(compile_block.encode("utf-8")).hexdigest() ==
            COMPILE_BLOCK_SHA, "compile block anchor drift")
    require(hashlib.sha256(command_block.encode("utf-8")).hexdigest() ==
            COMMAND_BLOCK_SHA, "command/gate anchor drift")
    require(release["frozen_execution_gates"]["exact_cycles"] == EXACT_CYCLES,
            "exact cycles drift")
    gates = release["frozen_execution_gates"]
    require(gates["numeric_mismatches"] == 0 and
            gates["tuple_mismatches"] == 0 and
            gates["weight_mismatches"] == 0 and
            all(gates[key] is True for key in (
                "request_stalls_must_be_nonzero",
                "result_stalls_must_be_nonzero",
                "raw_stalls_must_be_nonzero",
                "full8_requests_must_be_nonzero",
                "k1x8_full_issue_must_be_nonzero",
                "candidate_out_of_order_must_be_nonzero",
                "baseline_out_of_order_must_be_nonzero")),
            "numeric/tuple/weight/stall/full8/issue/order gate drift")
    return source, m803


def replay_source_matrix_and_tests():
    hammer = load_source_hammer()
    source, m803 = hammer.source_checks()
    matrix = hammer.run_matrix()
    failures = [key for key, expected in hammer.EXPECTED.items()
                if matrix.get(key) != expected]
    require(len(hammer.EXPECTED) == 22 and not failures,
            "R25 source adversarial matrix failed: " + ",".join(failures))

    test_outputs = {}
    for python in ("/usr/libexec/platform-python3.6", "/usr/bin/python3.12"):
        output = run_checked([
            python, "-m", "unittest", "discover", "-s",
            str(HW / "verif_m859"), "-p", "test_m859*.py", "-v",
        ])
        require("Ran 5 tests" in output and output.rstrip().endswith("OK"),
                "R25 source tests count/status drift for " + python)
        test_outputs[python] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        validate = run_checked([
            python, str(GUARD), "validate-source", "--hw-root", str(HW),
            "--contract", str(CONTRACT), "--candidate", str(CANDIDATE),
            "--runner", str(RUNNER),
        ])
        require("PASS_M859_R25_SHARED_WHITELIST_SOURCE__NO_VCS_OR_EDA" in
                validate, "validate-source status drift for " + python)

    # Execute the exact clean-environment dry-run boundary once.  The runner
    # itself pins platform-python3.6, so this also exercises its real child.
    dry = hammer.source_dry_run()
    require(dry["return_code"] == 86 and dry["event_count"] == 4 and
            not dry["formal_population_before"] and
            not dry["formal_population_after"],
            "source dry-run did not fail closed at the live-license boundary")
    return source, m803, matrix, test_outputs, dry


def final_review_value(authorization=None, target=None, status=None):
    return {
        "schema": "synthetic_m862_final_hammer_for_adversarial_replay_v1",
        "status": status or guard.FINAL_HAMMER_STATUS,
        "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "authorization": (copy.deepcopy(guard.FINAL_HAMMER_AUTHORIZATION)
                          if authorization is None else authorization),
        "review_target": target or {
            "release_sha256": RELEASE_SHA,
            "runner_sha256": RUNNER_SHA,
            "contract_sha256": CONTRACT_SHA,
            "candidate_sha256": CANDIDATE_SHA,
        },
    }


def validate_synthetic(value=None, raw_text=None, expected_outer=None):
    with tempfile.TemporaryDirectory(prefix="m862_r25_final_chain.") as raw:
        root = Path(raw)
        final = root / "final"
        final.mkdir()
        review = final / "review.json"
        if raw_text is None:
            guard.base.write_json(review, value or final_review_value())
        else:
            review.write_text(raw_text, encoding="utf-8")
        identity = guard.base.seal_directory(final)
        return guard.validate_launch_chain(
            HW, CONTRACT, CANDIDATE, RUNNER, SOURCE_HAMMER, RELEASE, final,
            expected_outer or identity["outer_seal_file_sha256"])


def rejected(callback):
    try:
        callback()
    except (guard.base.Failure, RuntimeError, ValueError, OSError):
        return True
    return False


def authorization_matrix():
    exact = copy.deepcopy(guard.FINAL_HAMMER_AUTHORIZATION)
    require(len(exact) == 15, "final authorization does not have 15 keys")
    positive = validate_synthetic()
    require(positive["status"] == "PASS_M859_R25_EXACT_LAUNCH_CHAIN",
            "positive actual-release synthetic-final chain failed")

    wrong_value = 0
    for key, expected in sorted(exact.items()):
        value = copy.deepcopy(exact)
        if type(expected) is bool:
            value[key] = not expected
        else:
            value[key] = expected + 1
        require(rejected(lambda value=value: validate_synthetic(
            final_review_value(authorization=value))),
            "wrong authorization value accepted: " + key)
        wrong_value += 1

    missing = 0
    for key in sorted(exact):
        value = copy.deepcopy(exact)
        del value[key]
        require(rejected(lambda value=value: validate_synthetic(
            final_review_value(authorization=value))),
            "missing authorization key accepted: " + key)
        missing += 1

    confused = 0
    for key, expected in sorted(exact.items()):
        value = copy.deepcopy(exact)
        value[key] = (1 if expected else 0) if type(expected) is bool else False
        require(rejected(lambda value=value: validate_synthetic(
            final_review_value(authorization=value))),
            "authorization type confusion accepted: " + key)
        confused += 1

    extra = copy.deepcopy(exact)
    extra["unexpected_key"] = "closed-world-bypass"
    require(rejected(lambda: validate_synthetic(
        final_review_value(authorization=extra))),
        "extra authorization key accepted")

    wrong_target = 0
    target = final_review_value()["review_target"]
    for key in sorted(target):
        value = dict(target)
        value[key] = "0" * 64
        require(rejected(lambda value=value: validate_synthetic(
            final_review_value(target=value))),
            "wrong target SHA accepted: " + key)
        wrong_target += 1

    wrong_statuses = ("PASS100_M837_R22_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED",
                      "AUTHORIZED_ONE_M859_R25_SHARED_WHITELIST_CHANNEL_SPLIT_VCS_ATTEMPT",
                      "PASS100_M859_R25_SHARED_WHITELIST_SOURCE__AUTHORIZE_ONE_FRESH_RELEASE_ONLY")
    for status in wrong_statuses:
        require(rejected(lambda status=status: validate_synthetic(
            final_review_value(status=status))),
            "wrong final status accepted: " + status)

    exact_text = json.dumps(final_review_value(), sort_keys=True,
                            allow_nan=False) + "\n"
    malformed = (
        exact_text.replace('"run_vcs": true',
                           '"run_vcs": true, "run_vcs": true', 1),
        exact_text.replace('"status": "' + guard.FINAL_HAMMER_STATUS + '"',
                           '"status": "' + guard.FINAL_HAMMER_STATUS +
                           '", "status": "' + guard.FINAL_HAMMER_STATUS +
                           '"', 1),
        exact_text.replace('"score_out_of_100": 100',
                           '"score_out_of_100": NaN', 1),
        exact_text.replace('"score_out_of_100": 100',
                           '"score_out_of_100": Infinity', 1),
        exact_text.replace('"score_out_of_100": 100',
                           '"score_out_of_100": -Infinity', 1),
        exact_text.replace('"max_attempts": 1',
                           '"max_attempts": NaN', 1),
    )
    for index, raw in enumerate(malformed):
        require(rejected(lambda raw=raw: validate_synthetic(raw_text=raw)),
                "duplicate/nonfinite JSON accepted: {}".format(index))

    require(rejected(lambda: validate_synthetic(expected_outer="0" * 64)),
            "wrong final outer pin accepted")
    return {
        "positive": 1,
        "wrong_value": wrong_value,
        "missing_key": missing,
        "type_confusion": confused,
        "extra_key": 1,
        "wrong_target_sha": wrong_target,
        "wrong_status": len(wrong_statuses),
        "duplicate_and_nonfinite": len(malformed),
        "wrong_outer_pin": 1,
    }


def main():
    before = formal_population()
    require(before == [], "formal M859 population exists before final hammer")
    source, m803 = verify_fixed_chain()
    replay_source, replay_m803, matrix, tests, dry = \
        replay_source_matrix_and_tests()
    require(source == replay_source and m803 == replay_m803,
            "independent source replay identity drift")
    auth = authorization_matrix()
    after = formal_population()
    require(after == [], "formal M859 population created by final hammer")
    require(sha256(DOCS359) == DOCS359_SHA, "docs359 changed during review")

    print("M862/M861/M859 C2 R25 FRESH FINAL-LAUNCH HAMMER")
    print("fixed_release_sha256=" + RELEASE_SHA)
    print("fixed_release_outer_seal_file_sha256=" + RELEASE_OUTER)
    print("runner_sha256=" + RUNNER_SHA)
    print("source_hammer_review_sha256=" + SOURCE_REVIEW_SHA)
    print("source_hammer_outer_seal_file_sha256=" + SOURCE_OUTER)
    print("request_outer_seal_file_sha256=" + REQUEST_OUTER)
    print("source_matrix_passed={}/22".format(len(matrix)))
    print("python36_tests_sha256=" + tests["/usr/libexec/platform-python3.6"])
    print("python312_tests_sha256=" + tests["/usr/bin/python3.12"])
    print("source_dry_run_rc={} events={} formal_before={} formal_after={}".format(
        dry["return_code"], dry["event_count"],
        len(dry["formal_population_before"]),
        len(dry["formal_population_after"])))
    print("m803_frozen_files={}".format(len(m803)))
    print("compile_block_sha256=" + COMPILE_BLOCK_SHA)
    print("commands_gates_sha256=" + COMMAND_BLOCK_SHA)
    print("exact_cycles=51/53,131/133,486/499,1231/1246,14/14")
    for key in sorted(auth):
        print("auth_{}={}".format(key, auth[key]))
    print("formal_population_before={}".format(len(before)))
    print("formal_population_after={}".format(len(after)))
    print("reviewer_runner_executions=0")
    print("vcs_runs=0 simv_runs=0 license_queries=0 eda_runs=0")
    print("formal_attempts_created=0 formal_results_created=0 failure_quarantines_created=0")
    print("docs359_sha256=" + DOCS359_SHA)
    print("status=PASS100_M859_R25_SHARED_WHITELIST_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
