#!/usr/bin/env python3
"""No-EDA source checker and runtime parser for M1698."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tokenize


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
OLD_CHECKER = HW / "system_simulator/scripts/check_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1684_runtime_checker", str(OLD_CHECKER))
OLD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(OLD)

RUNNER = HW / "dc_handoff/scripts/run_m1698_m1684_m1661_c2_shared_eda_queue_production_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1698_m1684_c2_shared_eda_queue_production_energy_source.py"
CONTRACT = HW / "contracts/m1698_m1684_c2_shared_eda_queue_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1698_m1684_c2_shared_eda_queue_production_energy_source_author_receipt_r1_20260901"
M1699 = HW / "reviews/m1699_m1698_m1684_c2_shared_eda_queue_production_energy_source_hammer_r1_20260901"
M1700 = HW / "contracts/m1700_m1699_m1698_m1684_c2_shared_eda_queue_production_energy_launch_release_r1_20260901.json"
M1684_RUNNER = HW / "dc_handoff/scripts/run_m1684_m1661_c2_m1609_fresh_mapped_production_energy_one_shot.py"
M1684_CONTRACT = HW / "contracts/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_20260901.json"
M1684_AUTHOR = HW / "reviews/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_author_receipt_r1_20260901"
M1685 = HW / "reviews/m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_hammer_r1_20260901"
M1686 = HW / "contracts/m1686_m1685_m1684_m1661_c2_m1609_fresh_mapped_production_energy_launch_release_r1_20260901.json"
SHARED_LOCK = "/tmp/date_dual_synopsys_same_uid_eda_queue.lock"

EXECUTION_SOURCES = (
    OLD.MEM, OLD.CASE_TB, OLD.OLD_ASSERT, OLD.ASSERT, OLD.TOP_TB,
    OLD.UCLI, OLD.PT_TCL,
    OLD.FILELISTS["k8"], OLD.FILELISTS["k1x8"], RUNNER,
)
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))
FIXED = {
    M1684_RUNNER: "1c7acc502c010809d56dacd78d857dfb5a44cca74e12025134424c6b9c80b77f",
    M1684_CONTRACT: "7fa827aca2ee236a06010d037ca03dac80fc1491abc59a3162c0092bc84e1683",
    M1684_AUTHOR / "author_receipt.json": "2f1bb4c7ce1a1355c488b75a059e6efe34a305ab01012365a32d0c7370b1724b",
    M1684_AUTHOR / "SHA256SUMS": "5142d91321007cffdcd9f35ab3707f7901a80a91d2d11347224c224ed86b6d30",
    M1684_AUTHOR / "SHA256SUMS.seal.sha256": "dba513a1e26cb9fd18e9a2f8532b2f5f07893c6d3f174f5f2acc523a83afab2a",
    M1685 / "review.json": "d0ee4de559e65eb77053cedd79d95f6d724984d5fe876a8428886675a0bac98f",
    M1685 / "SHA256SUMS": "141a432b28bb18b52c06c31bf58479bafe5b074b9770d6c8d123bd03c8a4195f",
    M1685 / "SHA256SUMS.seal.sha256": "7a6fbaec942122be117e3ea9161f4727689cae67360f48b24490c0f08d0ece33",
    OLD.DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def strip_c_comments_and_strings(text):
    """Remove C/SV comments and quoted strings, preserving line boundaries."""
    out = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                state = "line"; out.extend("  "); index += 2; continue
            if char == "/" and nxt == "*":
                state = "block"; out.extend("  "); index += 2; continue
            if char == '"':
                state = "string"; out.append(" "); index += 1; continue
            out.append(char); index += 1; continue
        if state == "line":
            if char == "\n": state = "code"; out.append("\n")
            else: out.append(" ")
            index += 1; continue
        if state == "block":
            if char == "*" and nxt == "/":
                state = "code"; out.extend("  "); index += 2
            else:
                out.append("\n" if char == "\n" else " "); index += 1
            continue
        if char == "\\" and nxt:
            out.extend("  "); index += 2
        elif char == '"':
            state = "code"; out.append(" "); index += 1
        else:
            out.append("\n" if char == "\n" else " "); index += 1
    return "".join(out)


def active_force_present(path, text=None):
    path = Path(path)
    text = path.read_text() if text is None else text
    if path.suffix == ".py":
        try:
            tokens = tokenize.generate_tokens(io.StringIO(text).readline)
            return any(item.type == tokenize.NAME and item.string == "force"
                       for item in tokens)
        except (tokenize.TokenError, IndentationError):
            return True
    if path.suffix in {".sv", ".v"}:
        return re.search(r"\bforce\b", strip_c_comments_and_strings(text)) is not None
    active = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if line.strip():
            active.append(line)
    return re.search(r"(?m)^\s*force\b", "\n".join(active)) is not None


def validate_queue_source(runner_text=None):
    text = RUNNER.read_text() if runner_text is None else runner_text
    need('LOCK = Path("' + SHARED_LOCK + '")' in text, "shared lock path drift")
    need("fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)" in text,
         "shared flock absent")
    need("def _owned_or_ancestor" in text and "def collision_gate" in text,
         "ancestry-aware collision gate absent")
    run_start = text.index("def run(")
    run_end = text.index("def result_identity", run_start)
    run_body = text[run_start:run_end]
    need('Path(command[0]).name in {"vcs", "pt_shell"}' in run_body,
         "per-launch VCS/PTPX selector absent")
    need("collision_gate()" in run_body, "per-launch collision rescan absent")
    need(run_body.index("collision_gate()") < run_body.index("subprocess.run("),
         "collision rescan not immediately before tool subprocess")
    main = text[text.index("def main("):]
    lock = main.index("fcntl.flock(")
    post_lock = main.index("collision_gate()", lock)
    attempt = main.index("ATTEMPT.mkdir()")
    first_eda = main.index('for axis in ("k8", "k1x8"):')
    need(lock < post_lock < attempt < first_eda, "lock/post-lock/attempt/EDA order drift")
    need("M1686 is permanently denied" in text, "M1686 denial absent")


def validate_sources():
    OLD.validate_predecessors()
    for axis in OLD.AXES:
        OLD.validate_filelist(axis)
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    old_contract = strict_json(M1684_CONTRACT)
    for row in old_contract.get("source_files", []):
        path = HW / row.get("path", "")
        need(path.is_file() and not path.is_symlink()
             and sha(path) == row.get("sha256"),
             "M1684 execution source identity drift: " + str(path))
    review = strict_json(M1685 / "review.json")
    need(review.get("verdict") == "FAIL_CLOSED_NO_M1686_RELEASE",
         "M1685 failure verdict drift")
    need(review.get("authorization", {}).get("m1686_release_authoring") is False,
         "forbidden M1686 release became authorized")
    need(not M1686.exists(), "forbidden M1686 exists")
    validate_queue_source()
    for path in EXECUTION_SOURCES:
        need("initreg" not in path.read_text().lower(),
             "forbidden initreg: " + str(path))
        need(not active_force_present(path), "active force: " + str(path))
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
         "m1698_m1684_c2_shared_eda_queue_production_energy_source_contract_r1_v1",
         "contract schema drift")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1699_REVIEW_AND_M1700_RELEASE_REQUIRED__NO_EDA",
         "contract status drift")
    need(contract.get("claim_boundary") == CLAIMS, "claim promotion")
    rows = contract.get("source_files", [])
    mapping = dict((row.get("path"), row.get("sha256")) for row in rows)
    expected = (RUNNER, CHECKER, TEST)
    need(len(mapping) == len(rows) == len(expected), "source inventory cardinality")
    for path in expected:
        rel = path.relative_to(HW).as_posix()
        need(mapping.get(rel) == sha(path), "source SHA drift: " + rel)
    for path in (M1699, M1700,
                 HW / "results/.m1698_c2_shared_eda_queue_production_energy_attempt_consumed",
                 HW / "results/m1698_c2_shared_eda_queue_production_energy_r1_20260901"):
        need(not os.path.lexists(path), "future/result namespace exists: " + str(path))
    return {"schema": "m1698_m1684_c2_shared_queue_source_check_r1_v1",
            "status": "PASS_M1698_SOURCE_ONLY_NO_EDA",
            "shared_eda_queue": SHARED_LOCK,
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "accepted_sources_per_axis": sum(OLD.EVENTS),
            "active_force_full_source_scan": True,
            "m1686_permanently_denied": True,
            "claim_boundary": CLAIMS}


validate_saif = OLD.validate_saif
validate_runtime_log = OLD.validate_runtime_log
parse_power_report = OLD.parse_power_report
aggregate_metrics = OLD.aggregate_metrics
AXES = OLD.AXES
EVENTS = OLD.EVENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "saif", "power"), required=True)
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
        need(args.axis is not None and args.case_id is not None
             and args.cycles is not None and args.saif and args.log, "saif arguments")
        output = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
        output["runtime"] = validate_runtime_log(args.log, args.axis, args.case_id)
    else:
        need(args.power_report is not None, "power report argument")
        output = parse_power_report(args.power_report)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
