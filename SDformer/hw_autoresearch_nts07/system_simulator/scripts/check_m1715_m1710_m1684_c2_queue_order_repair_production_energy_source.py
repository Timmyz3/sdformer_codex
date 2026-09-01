#!/usr/bin/env python3
"""No-EDA source checker and unchanged runtime parser for M1715."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1710_m1684_c2_runtime_bound_shared_eda_queue_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1710_checker_for_m1715", str(OLD_CHECKER))
OLD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OLD)

RUNNER = HW / "dc_handoff/scripts/run_m1715_m1710_m1684_m1661_c2_queue_order_repair_production_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
CONTRACT = HW / "contracts/m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_author_receipt_r1_20260901"
M1716 = HW / "reviews/m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_hammer_r1_20260901"
M1717 = HW / "contracts/m1717_m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_launch_release_r1_20260901.json"
M1710_RUNNER = OLD.RUNNER
M1710_CHECKER = OLD.CHECKER
M1710_TEST = OLD.TEST
M1710_CONTRACT = OLD.CONTRACT
M1710_AUTHOR = OLD.AUTHOR
M1711 = OLD.M1711
M1712 = OLD.M1712
M1710_FAILURE = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901.failed_or_incomplete.quarantine"
M1710_ATTEMPT = HW / "results/.m1710_c2_shared_eda_queue_production_energy_attempt_consumed"
M1710_RESULT = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901"
M1710_PRIVATE = HW / "results/m1710_c2_shared_eda_queue_production_energy_r1_20260901.private_build.unsealed_do_not_cite"
M1686 = OLD.M1686
M1700 = OLD.M1700
SHARED_LOCK = OLD.SHARED_LOCK

DIRECT_REL = (
    "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv",
    "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv",
    "dc_handoff/scripts/m1684_c2_m1609_fresh_mapped_production_energy.ucli.tcl",
    "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl",
    "dc_handoff/filelists/date_m1684_c2_m1609_k8_fresh_mapped_production_energy.f",
    "dc_handoff/filelists/date_m1684_c2_m1609_k1x8_fresh_mapped_production_energy.f",
)
DIRECT_SOURCES = tuple(HW / rel for rel in DIRECT_REL)
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))
FIXED = {
    M1710_RUNNER: "bf6acc942274fd11bd7731a4120dda833e04ae2cfa0322cd45ac00fbdcafc01b",
    M1710_CHECKER: "2a93e84b01e53f174bd97f8c7ae8e46dfa1c71bd34ac62e9a4e6ede2a31f74c5",
    M1710_TEST: "c4e6be1d08f1d6a64b40ae977579627e999312f2779b1e3107d51f96dfb0cc96",
    M1710_CONTRACT: "9086a6c306b3150e325a6a98b10ada135811fa91d029f100f3d4095bee1da1e5",
    M1710_AUTHOR / "author_receipt.json": "7885281396f3686ca9c076f9cf53e2abdedc164bb421fd0a3dc9051afcb01468",
    M1710_AUTHOR / "SHA256SUMS": "2a817b2dd73d574a1d40ad596c64be5d35793ef417c55206d5e6c214be92be0d",
    M1710_AUTHOR / "SHA256SUMS.seal.sha256": "bf6493dec24b4250c6fd4a9264ab6238f03eac88b8bfbfe3de58687799136254",
    M1711 / "review.json": "f9b867a2f272d6d00f55b3ebc463713f2aa2ac152e44a8d99f761fdbcf898b0e",
    M1711 / "SHA256SUMS": "d33dac5f8d39709c33964b164d011f6944c7a0530147fe1da3955b8abcd76d71",
    M1711 / "SHA256SUMS.seal.sha256": "a04a59fc0311c72fb29d530d9c07f58d6f9e484fab2dc7bee872239a75679fcb",
    M1712: "e5c5371897333962fd372370d4c13a942f56c399b1341d302ece5818fc423a50",
    Path(str(M1712) + ".sha256"): "354efd6661f9633fd6948087329702b8e36657bda1733a0580ad099ed7812d65",
    Path(str(M1712) + ".sha256.seal.sha256"): "dba7b74ca9985fc7c4bfc8e2e1e1a51513e324b7d30c45ef43fa43848a576aa8",
    M1710_FAILURE / "failure.json": "7a334d6a40f7f25ac4152b65a414020a890b1e41a83f62c479589dfd6c8c77ac",
    M1710_FAILURE / "SHA256SUMS": "c3f67b8ae21f828e0fb007b5933fe20a749950652dfa4c514889c9c47810c73e",
    M1710_FAILURE / "SHA256SUMS.seal.sha256": "e8e26336bc59ad54fb781b127e76fb821bc9cfd6e5ee2af5335e86dc4691cf4b",
    OLD.OLD.DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_m1710_failure():
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    need((M1710_FAILURE / "SHA256SUMS").read_text() ==
         FIXED[M1710_FAILURE / "failure.json"] + "  failure.json\n",
         "M1710 failure manifest population drift")
    need((M1710_FAILURE / "SHA256SUMS.seal.sha256").read_text() ==
         FIXED[M1710_FAILURE / "SHA256SUMS"] + "  SHA256SUMS\n",
         "M1710 failure outer seal drift")
    failed = strict_json(M1710_FAILURE / "failure.json")
    need(failed == {
        "attempt_consumed": False,
        "automatic_retry": False,
        "canonical_result": False,
        "counts": {"ptpx_runs": 0, "saif_files": 0,
                   "simv_runs": 0, "vcs_compiles": 0},
        "error": "Failure", "partial_axis_citable": False,
        "phase": "SOURCE_CHAIN", "status": "FAILED_OR_INCOMPLETE"},
        "M1710 pre-attempt failure semantic drift")
    for path in (M1710_ATTEMPT, M1710_RESULT, M1710_PRIVATE):
        need(not os.path.lexists(path), "M1710 retry/residue exists: " + str(path))
    return failed


def validate_queue_source(runner_text=None):
    text = RUNNER.read_text() if runner_text is None else runner_text
    need('LOCK = Path("' + SHARED_LOCK + '")' in text,
         "shared lock path drift")
    blocking = "fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)"
    need(blocking in text, "blocking shared flock absent")
    need("LOCK_NB" not in text, "nonblocking flock resurrected")
    need("def _owned_or_ancestor" in text and "def collision_gate" in text,
         "ancestry-aware collision gate absent")
    run_start = text.index("def run(")
    run_end = text.index("def result_identity", run_start)
    run_body = text[run_start:run_end]
    need('Path(command[0]).name in {"vcs", "pt_shell"}' in run_body,
         "per-launch VCS/PTPX selector absent")
    need("collision_gate()" in run_body and
         run_body.index("collision_gate()") < run_body.index("subprocess.run("),
         "per-launch collision rescan order drift")
    main = text[text.index("def main("):]
    lock = main.index(blocking)
    attempt = main.index("ATTEMPT.mkdir()")
    need("collision_gate()" not in main[:lock],
         "pre-lock collision scan resurrected")
    post_collision = main.index("collision_gate()", lock)
    first_rebind = main.index("runtime_bind_execution_sources()", post_collision)
    first_lexists = main.index("forbidden_release_namespaces_absent()", first_rebind)
    need(lock < post_collision < first_rebind < first_lexists < attempt,
         "blocking-lock/post-collision/rebind/attempt order drift")
    need('state["phase"] = "POST_LOCK_COLLISION"\n        collision_gate()\n        state["phase"] = "POST_LOCK_RUNTIME_REBIND"\n        runtime_bind_execution_sources()' in main,
         "first post-lock collision/rebind adjacency drift")
    need(main[lock:attempt].count("runtime_bind_execution_sources()") >= 2,
         "post-lock exact-SHA/force rebind not repeated before attempt")
    need(main[lock:attempt].count("forbidden_release_namespaces_absent()") >= 2,
         "post-lock lexists gate not repeated before attempt")
    need("def verify_m1710_pre_attempt_failure()" in text and
         "verify_m1710_pre_attempt_failure()" in text,
         "M1710 failure binder absent")
    need("DIRECT_EXECUTION_PATHS = {" in text and
         "for rel in sorted(DIRECT_EXECUTION_PATHS):" in text,
         "six-source runtime binder absent")
    for rel in DIRECT_REL:
        need(rel in text, "direct source missing: " + rel)
    forbid_start = text.index("def forbidden_release_namespaces_absent")
    forbid_end = text.index("def verify_m1710_pre_attempt_failure", forbid_start)
    need("os.path.lexists(path)" in text[forbid_start:forbid_end],
         "forbidden release lexists gate absent")
    failure_start = text.index("def verify_m1710_pre_attempt_failure")
    failure_end = text.index("def runtime_bind_execution_sources", failure_start)
    need("os.path.lexists(path)" in text[failure_start:failure_end],
         "M1710 retry lexists gate absent")


def validate_sources():
    verify_m1710_failure()
    old_contract = strict_json(OLD.M1684_CONTRACT)
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in old_contract.get("source_files", []))
    need(set(DIRECT_REL).issubset(set(mapping)),
         "M1684 direct execution inventory incomplete")
    for rel, path in zip(DIRECT_REL, DIRECT_SOURCES):
        need(path.is_file() and not path.is_symlink() and
             sha(path) == mapping[rel], "runtime source SHA drift: " + rel)
        need("initreg" not in path.read_text().lower(),
             "forbidden initreg: " + rel)
        need(not OLD.active_force_present(path), "active force: " + rel)
    for payload in (M1686, M1700):
        for path in (payload, Path(str(payload) + ".sha256"),
                     Path(str(payload) + ".sha256.seal.sha256")):
            need(not os.path.lexists(path),
                 "forbidden release exists: " + str(path))
    validate_queue_source()
    runner = RUNNER.read_text()
    need(runner.count('for axis in ("k8", "k1x8"):') >= 3,
         "axis geometry drift")
    need(runner.count("for case_id in range(5):") >= 2,
         "case geometry drift")
    for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10'):
        need(token in runner, "execution budget drift")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_contract_r1_v1",
         "contract schema drift")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1716_REVIEW_AND_M1717_RELEASE_REQUIRED__NO_EDA",
         "contract status drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    rows = contract.get("source_files", [])
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    expected = (RUNNER, CHECKER, TEST)
    need(len(mapping) == len(rows) == len(expected),
         "source inventory cardinality")
    for path in expected:
        rel = path.relative_to(HW).as_posix()
        need(mapping.get(rel) == sha(path), "source SHA drift: " + rel)
    for path in (M1716, M1717, Path(str(M1717) + ".sha256"),
                 Path(str(M1717) + ".sha256.seal.sha256"),
                 HW / "results/.m1715_c2_queue_order_repair_production_energy_attempt_consumed",
                 HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901"):
        need(not os.path.lexists(path), "future/result namespace exists: " + str(path))
    return {
        "schema": "m1715_m1710_c2_queue_order_repair_source_check_r1_v1",
        "status": "PASS_M1715_SOURCE_ONLY_NO_EDA",
        "shared_eda_queue": SHARED_LOCK,
        "blocking_flock_before_collision": True,
        "prelock_collision_scan": False,
        "postlock_runtime_rebinds_before_attempt": 2,
        "axes": ["k8", "k1x8"], "cases_per_axis": 5,
        "accepted_sources_per_axis": sum(OLD.EVENTS),
        "runtime_bound_execution_sources": 6,
        "active_force_full_source_scan": True,
        "lexists_gates_preserved": True,
        "m1710_pre_attempt_failure_bound": True,
        "m1710_retry_forbidden": True,
        "claim_boundary": CLAIMS,
    }


active_force_present = OLD.active_force_present
validate_saif = OLD.validate_saif
validate_runtime_log = OLD.validate_runtime_log
parse_power_report = OLD.parse_power_report
aggregate_metrics = OLD.aggregate_metrics
AXES = OLD.AXES
EVENTS = OLD.EVENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "saif", "power"),
                        required=True)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", dest="case_id", type=int)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--power-report", type=Path)
    args = parser.parse_args()
    if args.mode == "source":
        output = validate_sources()
    elif args.mode == "saif":
        need(args.axis is not None and args.case_id is not None and
             args.cycles is not None and args.saif and args.log,
             "saif arguments")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis,
                                                 args.case_id)
    else:
        need(args.power_report is not None, "power report argument")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
