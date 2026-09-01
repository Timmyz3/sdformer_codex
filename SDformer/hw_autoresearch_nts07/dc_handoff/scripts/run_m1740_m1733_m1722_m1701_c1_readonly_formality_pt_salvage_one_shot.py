#!/usr/bin/env python3
"""Zero-EDA canonical salvage of sealed M1722 Formality and M1733 PT evidence."""
from __future__ import print_function

import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_contract_r1_20260901.json"
TEST = HW / "system_simulator/tests/test_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source.py"
M1733_RUNNER = HW / "dc_handoff/scripts/run_m1733_m1722_m1701_c1_formality_reuse_pt_only_one_shot.py"
M1734 = HW / "reviews/m1734_m1733_m1722_m1701_c1_formality_reuse_pt_only_source_hammer_r1_20260901"
M1735 = HW / "contracts/m1735_m1734_m1733_m1722_m1701_c1_formality_reuse_pt_only_launch_release_r1_20260901.json"
M1733_ATTEMPT = HW / "dc_handoff/runs/.m1733_m1722_m1701_c1_formality_reuse_pt_only_attempt_consumed"
M1733_FAILURE = HW / "dc_handoff/runs/m1733_m1722_m1701_c1_formality_reuse_pt_only_r1_20260901.failed_or_incomplete.quarantine"
PTSTA = M1733_FAILURE / "ptsta"
M1722_FORMALITY = HW / "dc_handoff/runs/m1722_m1701_c1_salvage_formality_pt_r1_20260901.failed_or_incomplete.quarantine/formality"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1733_c1_m1701_slowmax_fastmin.tcl"
M1742 = HW / "reviews/m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_hammer_r1_20260901"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
ATTEMPT = HW / "dc_handoff/runs/.m1740_c1_readonly_formality_pt_salvage_attempt_consumed"
RESULT = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
FAILURE = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901.failed_or_incomplete.quarantine"
STAGE = HW / ("dc_handoff/runs/.m1740_c1_readonly_formality_pt_salvage_stage." + str(os.getpid()))

SOURCE_CLAIMS = dict((key, False) for key in (
    "formality", "independent_pt", "dc", "power", "energy",
    "cycle_speedup", "system_speedup", "paper_ppa_ready", "paper_citable",
    "headline"))
RESULT_CLAIMS = {
    "formality": True, "independent_pt": True, "dc": False,
    "power": False, "energy": False, "cycle_speedup": False,
    "system_speedup": False, "paper_ppa_ready": False,
    "paper_citable": True, "headline": False,
}

FIXED_SHA = {
    "m1733_runner": "37723675f3ca3f094cdc747755cfffba41c6899c584c6dd3dbdf2c5ab35a4e9e",
    "m1734_review": "ec5c0c5292e0494b78e3f602236ba1bf6e875c4078ae784cf59db186bfa722f3",
    "m1734_manifest": "aca244bca91c00aea489ac811b7eecfafcf3bfe07d073ec83aa244b10b283081",
    "m1734_outer": "9e4a33e1837672ecc9344963a7cd501787bc8920e22ac56fbd8cb50a013d772c",
    "m1735": "56fa237e5674335646b7fb69f4070060951636c747e11da94d5cc4544826b87a",
    "m1735_sum": "29556f1ae2f26ee1e513a6b29d02da09f50bc79ec681cae8bac82953a009fa6a",
    "m1735_outer": "d2ba9b4f7dcd7f0f0ca1c4a4e968e1bb5ffbd6a88991e278a695777a8e00e906",
    "m1733_attempt_json": "b483911f666ccf25f12fcb6f3925485ea4ba09a86c7f4fd6e6e7ce0fcfdd916f",
    "m1733_attempt_manifest": "ef8aee6ed56642b3c8da7db87ef8981788acc424c78085445735d27c61e5c851",
    "m1733_attempt_outer": "c4c4e0c00a6621880d1af9376b5a9dc9543fa8ce193a3311dcf08831246709f9",
    "m1733_failure_json": "0ae2083bda560ae08394cc48e3430efea3ce59b07a2a53a81fa968f21a9f3cc3",
    "m1733_failure_manifest": "9093eb197b4a837471f5edda6893e8b9806cb734c0995e299fee3aa4909aa614",
    "m1733_failure_outer": "d7b93e1b15e96ca1b3d3a86064931c52a770a59ef62934b72834e8104f9bde3e",
    "marker": "fcdf45d03c8b1c6cee84bf627f27cca01847a0a6547d68f320d28ac2263d1a09",
    "raw_log": "e3ef6c4944c101c7bed30bd9ca89376cf940ae542a1727dbd36ca0c95a3c9034",
    "analysis_coverage": "dde39cd9285268e29a5de0d820e61bca0583e37bc7c797825afd9839a74967a3",
    "check_timing": "7267a999d832f1653b71dd213c0bff91ca848d37d2f11d293eba1c47c8adb024",
    "clock": "dde72ada5698ac675623da0eee78b92b11d7ba5b746a1e1cf5e2e4e99c97903e",
    "constraint_violators": "4ea93e40ff95b12b39199a5d0f63333c87afb949a804c2a7d340789ac62322dc",
    "design": "9b5d5ad06155f86635ac7c99e2d85e83f2110659eac60059355a00113f88252f",
    "exceptions": "23555bb08fba92943d1f8bd49d3d62086ffa3ef9610a4055a1a3cde48c5d8609",
    "global_timing": "00baa573509b5715e40168db6bef1e47f51fd34adecbb041615ea5c2ecf046c0",
    "libraries": "4b172c383eb80b62e4e48905a0922c07d1d0b34c524d6968cda6cfda528fa1a5",
    "runtime_scope": "bc27d8f4a29f9a55d8510efe5b94131353c151a8d3e84db3a64919ef109e10e2",
    "timing_hold": "ddd3510b25587144603a5e601087f9f1b5f4beea49e2f5170bf6c887a65ba938",
    "timing_setup": "82e2b17c43024418cf2c512a12127f1589fb98c870ef771e22f4c3211daffa68",
    "machine": "9e5160a8381fb839aac7e8df409667a5d9486ee6a280d5a8e808d1b40b6d3947",
    "wire_load": "640064e1b6ce37e2936c11fdc15d550d76417fa76199cfcdbd331b3f0c4befbc",
    "pt_tcl": "0fab7432e3806a75241cc4e55699b75d126fa334a3d4f3d7444189ed10001d67",
    "formality_marker": "c3a1837201846cb13e9e45dce3ff36b33e1c319b7136c3479c62677fdcb41f6c",
    "formality_log": "e229b48d3a303656fa3401375e3451b54efeba4046eab0dc82ff41d81a51804b",
    "formality_status": "1ce8c6c17a4890be8a54c86b63b8e36c958398f6b7a92fb9881df2b3b73ba19d",
    "formality_unmatched": "fa9278f6e9d247925ab50f08e121b205a446ccbbd784d27128eccab0ec402ee9",
    "formality_failing": "8b2abbae918a7e0a41de371e480ae8d05654c08b969c69c52b0e18ef4e3e5f4d",
    "formality_aborted": "dff4bb4b4494fbda173974ca91e61b70515502672ca9c44ec39b759831b2fb6d",
    "formality_unverified": "98d817177d460ebd964e1a926cadfa6894184ff771565637de55a99b5794ff5b",
    "formality_black_boxes": "e354863af019743e6fde90686d798b68a0355655846513da9b5617033ebf2c54",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
REPORT_SHA = {
    "analysis_coverage.rpt": "analysis_coverage", "check_timing.rpt": "check_timing",
    "clock.rpt": "clock", "constraint_violators.rpt": "constraint_violators",
    "design.rpt": "design", "exceptions.rpt": "exceptions",
    "global_timing.rpt": "global_timing", "libraries.rpt": "libraries",
    "runtime_scope.rpt": "runtime_scope", "timing_hold_fast.rpt": "timing_hold",
    "timing_setup_slow.rpt": "timing_setup",
    "timing_summary_machine.txt": "machine", "wire_load.rpt": "wire_load",
}
FORMALITY_SHA = {
    "FORMALITY_INTERNAL_COMPLETE.txt": "formality_marker",
    "formality.raw.log": "formality_log",
    "reports/formality_status.rpt": "formality_status",
    "reports/formality_unmatched.rpt": "formality_unmatched",
    "reports/formality_failing.rpt": "formality_failing",
    "reports/formality_aborted.rpt": "formality_aborted",
    "reports/formality_unverified.rpt": "formality_unverified",
    "reports/formality_black_boxes.rpt": "formality_black_boxes",
}


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise Failure("identity drift: " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key: " + key)
            value[key] = item
        return value
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root is not object")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid: " + str(root))
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise Failure("unsafe/duplicate manifest member")
        exact(root / rel, digest)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift: " + str(root))


def verify_file_seal(path, file_sha, sum_sha, outer_sha):
    path = Path(path)
    digest_file = Path(str(path) + ".sha256")
    outer_file = Path(str(path) + ".sha256.seal.sha256")
    exact(path, file_sha); exact(digest_file, sum_sha); exact(outer_file, outer_sha)
    if (digest_file.read_text() != file_sha + "  " + path.name + "\n"
            or outer_file.read_text() != sum_sha + "  " + digest_file.name + "\n"):
        raise Failure("file double seal drift: " + str(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_dir(root):
    root = Path(root)
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n" for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def load_m1733():
    exact(M1733_RUNNER, FIXED_SHA["m1733_runner"])
    spec = importlib.util.spec_from_file_location("m1733_frozen", str(M1733_RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_formality_payload(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise Failure("Formality payload invalid: " + str(root))
    for relative, key in FORMALITY_SHA.items():
        exact(root / relative, FIXED_SHA[key])
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in Formality payload")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(FORMALITY_SHA):
        raise Failure("Formality payload population drift")


def logical_tcl(text):
    commands, current = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        current += stripped
        commands.append(re.sub(r"\s+", " ", current).strip())
        current = ""
    if current:
        raise Failure("unterminated Tcl continuation")
    return commands


def verify_full_pt_tcl_echo(raw_lines):
    exact(PT_TCL, FIXED_SHA["pt_tcl"])
    commands = logical_tcl(PT_TCL.read_text())
    if len(commands) != 89 or commands[0] != (
            "set design_name m935_m912_three_stage_exact_parent_match_product_capture_island"):
        raise Failure("PT Tcl command inventory drift")
    normalized = [re.sub(r"\s+", " ", line).strip() for line in raw_lines]
    expected_counts = {}
    for command in commands:
        expected_counts[command] = expected_counts.get(command, 0) + 1
    for command, count in expected_counts.items():
        if normalized.count(command) != count:
            raise Failure("PT Tcl echo cardinality drift: " + command)
    cursor = -1
    for command in commands:
        try:
            position = normalized.index(command, cursor + 1)
        except ValueError:
            raise Failure("PT Tcl echo absent/reordered: " + command)
        cursor = position
    if normalized[cursor] != "quit":
        raise Failure("PT Tcl did not end at quit")
    return commands


def verify_predecessor_authority():
    verify_seal(M1734, FIXED_SHA["m1734_manifest"], FIXED_SHA["m1734_outer"])
    exact(M1734 / "review.json", FIXED_SHA["m1734_review"])
    if strict_json(M1734 / "review.json").get("status") != (
            "PASS_M1734_M1733_M1722_M1701_C1_FORMALITY_REUSE_PT_ONLY_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_PT_ATTEMPT"):
        raise Failure("M1734 status drift")
    verify_file_seal(M1735, FIXED_SHA["m1735"], FIXED_SHA["m1735_sum"],
                     FIXED_SHA["m1735_outer"])
    release = strict_json(M1735)
    if (release.get("status") != "AUTHORIZE_ONE_M1733_M1722_M1701_C1_FORMALITY_REUSE_PT_ONLY_ATTEMPT"
            or release.get("identity") != {
                "runner_sha256": FIXED_SHA["m1733_runner"],
                "source_contract_sha256": "10e756455f38479aff0b5ec04be0b3479da920b0f1a7afa4da8e6b14d722a43f",
                "m1734_review_sha256": FIXED_SHA["m1734_review"]}
            or release.get("authorization") != {
                "future_m1733_attempts": 1, "automatic_retry": False,
                "formality_runs": 0, "pt_runs": 1, "dc_runs": 0}):
        raise Failure("M1735 semantic drift")


def parse_machine():
    values = {}
    for row in (PTSTA / "reports/timing_summary_machine.txt").read_text().splitlines():
        if row.count("=") != 1:
            raise Failure("machine syntax")
        key, value = row.split("=", 1)
        if key in values:
            raise Failure("duplicate machine key")
        values[key] = value
    expected = {
        "setup_wns_ns": "0.027871", "setup_tns_ns": "0.0",
        "setup_violating_paths": "0", "hold_wns_ns": "0.001827",
        "hold_tns_ns": "0.0", "hold_violating_paths": "0",
        "macro_count": "9", "clock_period_ns": "3.000",
        "setup_uncertainty_ns": "0.200", "hold_uncertainty_ns": "0.050"}
    if values != expected or any(not math.isfinite(float(value)) for value in values.values()):
        raise Failure("machine semantic drift")
    return values


def verify_pt_evidence():
    verify_seal(M1733_ATTEMPT, FIXED_SHA["m1733_attempt_manifest"],
                FIXED_SHA["m1733_attempt_outer"])
    verify_seal(M1733_FAILURE, FIXED_SHA["m1733_failure_manifest"],
                FIXED_SHA["m1733_failure_outer"])
    exact(M1733_ATTEMPT / "attempt.json", FIXED_SHA["m1733_attempt_json"])
    exact(M1733_FAILURE / "failure.json", FIXED_SHA["m1733_failure_json"])
    if strict_json(M1733_ATTEMPT / "attempt.json") != {
            "automatic_retry": False, "dc_runs": 0, "formality_runs": 0,
            "pt_runs": 1,
            "status": "M1733_M1722_C1_FORMALITY_REUSE_PT_ONLY_ATTEMPT_CONSUMED"}:
        raise Failure("M1733 attempt semantic drift")
    if strict_json(M1733_FAILURE / "failure.json") != {
            "attempt_consumed": True, "automatic_retry": False,
            "error": "Failure", "formality_runs": 0, "phase": "PRIMETIME",
            "pt_runs": 1, "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE"}:
        raise Failure("M1733 failure semantic drift")
    exact(PTSTA / "PTSTA_INTERNAL_COMPLETE.txt", FIXED_SHA["marker"])
    exact(PTSTA / "pt.raw.log", FIXED_SHA["raw_log"])
    for name, key in REPORT_SHA.items():
        exact(PTSTA / "reports" / name, FIXED_SHA[key])
    if set(path.name for path in (PTSTA / "reports").iterdir()) != set(REPORT_SHA):
        raise Failure("PT report population drift")
    marker = (PTSTA / "PTSTA_INTERNAL_COMPLETE.txt").read_text().splitlines()
    if marker != ["M1733_C1_M1701_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS",
                  "meaning=REPORTS_COMPLETE_AND_NONNEGATIVE_MAX_MIN_NOT_RESULT_ADMISSION",
                  "paper_claim=false"]:
        raise Failure("PT marker drift")
    machine = parse_machine()
    scope = (PTSTA / "reports/runtime_scope.rpt").read_text().splitlines()
    expected_scope = [
        "milestone=M1733",
        "scope=M1701_C1_salvage_macro_aware_prelayout_independent_PrimeTime",
        "clock_period_ns=3.000", "setup_uncertainty_ns=0.200",
        "hold_uncertainty_ns=0.050",
        "setup_view=std_and_macro_slow_ssg0p9v125c",
        "hold_view=std_and_macro_fast_ffg1p05vm40c",
        "macro_cell=TS1N28HPCPHVTB128X128M4S", "macro_count=9",
        "wireload=ZeroWireload_from_exact_M1701_SDC",
        "parasitics=none_no_read_parasitics_command", "ideal_clock=true",
        "false_path_or_multicycle_added_by_M1733=false", "pt_eco=false"]
    if scope != expected_scope:
        raise Failure("runtime scope drift")
    global_timing = (PTSTA / "reports/global_timing.rpt").read_text()
    setup = (PTSTA / "reports/timing_setup_slow.rpt").read_text()
    hold = (PTSTA / "reports/timing_hold_fast.rpt").read_text()
    if (global_timing.count("No setup violations found.") != 1
            or global_timing.count("No hold violations found.") != 1
            or "slack (VIOLATED)" in setup + hold
            or "slack (MET)" not in setup or "slack (MET)" not in hold
            or "check_timing succeeded." not in (PTSTA / "reports/check_timing.rpt").read_text()):
        raise Failure("timing report semantic drift")
    coverage = (PTSTA / "reports/analysis_coverage.rpt").read_text()
    for pattern in (r"setup\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)",
                    r"hold\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)",
                    r"min_pulse_width\s+78506\s+50526 \( 64%\)\s+0 \(  0%\)\s+27980 \( 36%\)",
                    r"out_setup\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)",
                    r"out_hold\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)"):
        if re.search(pattern, coverage) is None:
            raise Failure("coverage disclosure drift")
    if coverage.count("untested  no_paths") != 2:
        raise Failure("output untested reason/cardinality drift")
    raw_lines = (PTSTA / "pt.raw.log").read_text(errors="replace").splitlines()
    pt063 = "Error: Library Compiler executable path is not set. (PT-063)"
    home = 'Error: can\'t read "::env(HOME)": no such variable'
    cmd013 = "\tUse error_info for more info. (CMD-013)"
    cmd081 = "\tstopped at line 993 due to error. (CMD-081)"
    main_line = "set design_name m935_m912_three_stage_exact_parent_match_product_capture_island"
    for line in (pt063, home, cmd013, cmd081, main_line,
                 "Diagnostics summary: 2 errors, 5 warnings, 30 informationals",
                 "Thank you for using pt_shell!"):
        if raw_lines.count(line) != 1:
            raise Failure("PT raw exact line/cardinality drift: " + line)
    error_lines = [line for line in raw_lines if line.startswith("Error:")]
    if error_lines != [pt063, home]:
        raise Failure("unaccounted PT Error diagnostic")
    main_index = raw_lines.index(main_line)
    tcl_commands = verify_full_pt_tcl_echo(raw_lines)
    if not all(raw_lines.index(line) < main_index for line in (pt063, home, cmd013, cmd081)):
        raise Failure("startup diagnostic not pre-main")
    if (raw_lines.count("quit") != 1
            or raw_lines.index("quit") <= main_index
            or raw_lines.index("Diagnostics summary: 2 errors, 5 warnings, 30 informationals") <= raw_lines.index("quit")
            or raw_lines.index("Thank you for using pt_shell!") <= raw_lines.index("quit")):
        raise Failure("main Tcl completion ordering drift")
    if len(tcl_commands) != 89:
        raise Failure("PT Tcl echo count drift")
    return machine


def verify_contract_sources(contract):
    if (contract.get("schema") != "m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_contract_r1_v1"
            or contract.get("status") != "SOURCE_ONLY__M1742_REVIEW_AND_M1743_RELEASE_REQUIRED__ZERO_EDA"
            or contract.get("claim_boundary") != SOURCE_CLAIMS):
        raise Failure("source contract semantic drift")
    evidence = contract.get("admitted_evidence", {})
    if (evidence.get("coverage_disclosure") != {
            "setup": {"total": 13860, "met": 13851, "violated": 0, "untested": 9},
            "hold": {"total": 13860, "met": 13851, "violated": 0, "untested": 9},
            "out_setup": {"total": 2680, "met": 2679, "violated": 0,
                          "untested_no_paths": 1},
            "out_hold": {"total": 2680, "met": 2679, "violated": 0,
                         "untested_no_paths": 1},
            "min_pulse_width": {"total": 78506, "met": 50526, "violated": 0,
                                "untested_no_clock": 27980}}
            or evidence.get("formality_canonical_payload") != {
                "directory": "formality", "artifact_count": 8,
                "self_contained": True}
            or evidence.get("physical_scope") != {
                "prelayout": True, "ideal_clock": True,
                "wireload": "ZeroWireload", "parasitics": False,
                "paper_ppa_ready": False}):
        raise Failure("source contract evidence boundary drift")
    startup = contract.get("startup_diagnostic_classification", {})
    if (startup.get("line_start_error_count") != 2
            or startup.get("unaccounted_error_diagnostics") != 0
            or startup.get("pt_tcl_sha256") != FIXED_SHA["pt_tcl"]
            or startup.get("logical_tcl_commands_echoed_once_in_order") != 89
            or startup.get("runtime_scope_exact_key_count") != 14):
        raise Failure("source contract startup/Tcl disclosure drift")
    revised = contract.get("revised_from", {})
    if (revised.get("m1737_review_sha256") !=
            "5e12a1f85ee543838caf5c9cb5fcbee0f98aec9adfe2f6bf9633f077261cd329"
            or revised.get("m1737_manifest_sha256") !=
            "2a64581949d32d33c1a9a690033031ed42c022af38e2690670aae4b5b5b329e7"
            or revised.get("m1737_outer_file_sha256") !=
            "e2f218231fd8bfb2c815b141bd6037fce72fbdb26b3d7ec81cf42310132eba79"):
        raise Failure("failed-review repair lineage drift")
    if contract.get("future_execution") != {
            "max_attempts": 1, "automatic_retry": False, "eda_runs": 0,
            "license_queries": 0, "network_calls": 0,
            "result": "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901",
            "attempt": "dc_handoff/runs/.m1740_c1_readonly_formality_pt_salvage_attempt_consumed",
            "failure": "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901.failed_or_incomplete.quarantine"}:
        raise Failure("source contract execution budget drift")
    rows = contract.get("source_files")
    if not isinstance(rows, list):
        raise Failure("source inventory absent")
    mapping = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or row["path"] in mapping:
            raise Failure("source inventory malformed")
        mapping[row["path"]] = row["sha256"]
        exact(HW / row["path"], row["sha256"])
    expected = {path.relative_to(HW).as_posix() for path in (RUNNER, TEST)}
    if set(mapping) != expected:
        raise Failure("source inventory incomplete")


def verify_authority():
    names = ("M1740_EXPECTED_RUNNER_SHA256", "M1740_EXPECTED_SOURCE_CONTRACT_SHA256",
             "M1740_EXPECTED_M1742_REVIEW_SHA256", "M1740_EXPECTED_M1742_MANIFEST_SHA256",
             "M1740_EXPECTED_M1742_OUTER_FILE_SHA256", "M1740_EXPECTED_M1743_RELEASE_SHA256")
    pins = dict((name, os.environ.get(name, "")) for name in names)
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("M1742/M1743 exact SHA authority absent")
    exact(RUNNER, pins[names[0]]); exact(CONTRACT, pins[names[1]])
    verify_contract_sources(strict_json(CONTRACT))
    verify_seal(M1742, pins[names[3]], pins[names[4]])
    exact(M1742 / "review.json", pins[names[2]])
    review = strict_json(M1742 / "review.json")
    release_sum = Path(str(M1743) + ".sha256")
    release_outer = Path(str(M1743) + ".sha256.seal.sha256")
    exact(M1743, pins[names[5]])
    if (not release_sum.is_file() or not release_outer.is_file()
            or release_sum.read_text() != sha(M1743) + "  " + M1743.name + "\n"
            or release_outer.read_text() != sha(release_sum) + "  " + release_sum.name + "\n"):
        raise Failure("M1743 double seal drift")
    release = strict_json(M1743)
    expected_status = "PASS_M1742_M1740_M1733_M1722_M1701_C1_READONLY_FORMALITY_PT_SALVAGE_SOURCE_HAMMER__AUTHORIZE_ONE_ZERO_EDA_CANONICALIZATION"
    if (review.get("status") != expected_status
            or release.get("status") != "AUTHORIZE_ONE_M1740_C1_READONLY_FORMALITY_PT_SALVAGE_CANONICALIZATION"
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER), "source_contract_sha256": sha(CONTRACT),
                "m1742_review_sha256": sha(M1742 / "review.json")}
            or release.get("authorization") != {
                "future_m1740_attempts": 1, "automatic_retry": False,
                "eda_runs": 0, "license_queries": 0, "network_calls": 0}):
        raise Failure("M1743 semantic drift")


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    if next((HW / "dc_handoff/runs").glob(
            ".m1740_c1_readonly_formality_pt_salvage_stage.*"), None) is not None:
        raise Failure("stale M1740 stage")


def main():
    if len(sys.argv) != 1:
        raise Failure("M1740 accepts no arguments")
    state = {"attempt": False, "complete": False, "phase": "SOURCE_CHAIN"}
    try:
        verify_authority()
        verify_predecessor_authority()
        formality = load_m1733().verify_m1722_formality_reuse()
        verify_formality_payload(M1722_FORMALITY)
        machine = verify_pt_evidence()
        exact(HW / "docs/359_DATE终局冻结_20260813.md", FIXED_SHA["docs359"])
        namespaces_fresh()
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1740_C1_READONLY_FORMALITY_PT_SALVAGE_ATTEMPT_CONSUMED",
            "automatic_retry": False, "eda_runs": 0,
            "license_queries": 0, "network_calls": 0})
        seal_dir(ATTEMPT)
        state["phase"] = "CANONICAL_STAGE"
        STAGE.mkdir()
        shutil.copytree(PTSTA, STAGE / "ptsta")
        shutil.copytree(M1722_FORMALITY, STAGE / "formality")
        verify_formality_payload(STAGE / "formality")
        write_json(STAGE / "receipt.json", {
            "schema": "m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_receipt_r1_v1",
            "status": "PASS_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT",
            "formality": {"source": "sealed_M1722_failure_subproof",
                           "canonical_payload": "formality",
                           "artifact_sha256": dict((relative, FIXED_SHA[key])
                                                   for relative, key in FORMALITY_SHA.items()),
                           "verification_succeeded": True,
                           "passing_compare_points": formality["passing_compare_points"],
                           "failing": 0, "aborted": 0, "unverified": 0,
                           "unmatched": 0, "macro_instances_per_side": 9},
            "prime_time": machine,
            "coverage_disclosure": {
                "setup": {"total": 13860, "met": 13851, "violated": 0, "untested": 9},
                "hold": {"total": 13860, "met": 13851, "violated": 0, "untested": 9},
                "out_setup": {"total": 2680, "met": 2679,
                              "violated": 0, "untested_no_paths": 1},
                "out_hold": {"total": 2680, "met": 2679,
                             "violated": 0, "untested_no_paths": 1},
                "min_pulse_width": {"total": 78506, "met": 50526,
                                    "violated": 0, "untested_no_clock": 27980}},
            "startup_diagnostics": {
                "pre_main_only": True, "error_count": 2,
                "exact_ids": ["PT-063", "CMD-013", "CMD-081"],
                "main_tcl_and_all_reports_completed": True},
            "scope": {"prelayout": True, "ideal_clock": True,
                      "wireload": "ZeroWireload", "parasitics": False,
                      "macro_count": 9, "power_or_energy": False},
            "execution": {"eda_runs": 0, "license_queries": 0,
                          "network_calls": 0, "automatic_retry": False},
            "claim_boundary": RESULT_CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1740_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT\n")
        seal_dir(STAGE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1740_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT")
        return 0
    except BaseException as error:
        if state["attempt"] and not state["complete"] and not FAILURE.exists():
            try:
                quarantine = HW / ("dc_handoff/runs/.m1740_c1_readonly_formality_pt_salvage_failure." + str(os.getpid()))
                quarantine.mkdir()
                write_json(quarantine / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": True,
                    "automatic_retry": False, "eda_runs": 0,
                    "license_queries": 0, "network_calls": 0})
                seal_dir(quarantine); publish_no_replace(quarantine, FAILURE)
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
