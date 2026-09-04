#!/usr/bin/env python3
"""Fresh independent no-EDA hammer for the M884 C1 macro-aware DC source."""

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_exact_sha_r1.sh"
TCL = ROOT / "dc_handoff/scripts/run_dc_m884_m528_r21_macro_aware_product_candidate.tcl"
FILELIST = ROOT / "dc_handoff/filelists/date_m884_m528_r21_macro_aware_product_dc.f"
SDC = ROOT / "dc_handoff/constraints/date_m884_m528_r21_macro_aware_product_3ns.sdc"
CONTRACT = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_source_only_contract_r1_20260829.json"
CANDIDATE = ROOT / "contracts/m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_r1_20260829.json"
AUTHOR_TEST = ROOT / "verif_m528_dw1rw/test_m884_m528_r21_macro_dc_source_closure.py"
ADAPTER = ROOT / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
BINDING = ROOT / "rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json"
TOP = ROOT / "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv"
SVA = ROOT / "verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
HANDOFF_DIR = ROOT / "reviews/m884_m528_r21_macro_aware_product_dc_source_author_handoff_r1_20260829"
REQUEST_DIR = ROOT / "reviews/m885_m884_m528_r21_macro_aware_product_dc_source_hammer_REQUEST_r1_20260829"
M881_DIR = ROOT / "reviews/m881_c1_m528_m533_physical_evidence_first_principles_audit_r1_20260829"
M879_DIR = ROOT / "reviews/m879_m863_c1_r21_unit_delay_vcs_result_hammer_r1_20260829"
M863_DIR = ROOT / "results/m863_m533_m528_dead_write_only_1rw_unit_delay_vcs_r21_20260829"
M623_DIR = ROOT / "reviews/m623_m617_m597_m593_parent_scratch_energy_r5_result_hammer_r1_20260828"
CANONICAL = ROOT / "dc_handoff/runs/m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829"
ATTEMPT = ROOT / "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_attempt_consumed"
LOCK = ROOT / "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_launch_lock"


EXPECTED = {
    RUNNER: "b23d53dc45828d3e206d0e37f421f775d585c9cc32c457addeea6b26cc9b4ab2",
    TCL: "f9703e94198f05dbeb9101e12ec4e8dfa993e528212b173fba64cc2a261066e1",
    FILELIST: "610773defff65b1539169f3d0a8158ffd02be10691c581ffd6fee37bf52d7a69",
    SDC: "b0f6bb13d24260a66f81e4fc59c8b58a219b79ad39be749600820f67197f1ed2",
    AUTHOR_TEST: "37a4566ab9206d33157e073142fa965ff645c4bd348e8f125cc86dfab18b853d",
    CONTRACT: "271b6e85119ef0783dc074788c0269a4f5e047c9a2fe572bb8b86fba07fd56fb",
    CANDIDATE: "e89c2d613906412fcf1381ef71261a509f140b2f6d454d3b66e02ad2b5cfe080",
    TOP: "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    SVA: "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b",
    ADAPTER: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    BINDING: "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    HANDOFF_DIR / "handoff.json": "4a40a59b616162b7a4ad2a013e2fc3275cbe86b179cd73a6881aa0dc03b10351",
    HANDOFF_DIR / "SHA256SUMS": "44269aa90121cf598a50bf5e21f84f3135905cd99a05093d508c14e022bae591",
    HANDOFF_DIR / "SHA256SUMS.seal.sha256": "dacac902cae14427f0b03435e62df5f91972c134b3f7c832fad3c24ba63133e3",
    REQUEST_DIR / "request.json": "499edbdc207b33c3aca3b846fd1a169f1382b1e6046127873b62b9ba6cf3d6c4",
    REQUEST_DIR / "SHA256SUMS": "c55896478c0cb327eded5b55ba75627cff5c601db770f03023d5c1590f4c5ea1",
    REQUEST_DIR / "SHA256SUMS.seal.sha256": "016426089501f8dc509f2b71e18cb5fd3009bde7aa1fb55cb37b799d08768db8",
}


REQUIRED_ARTIFACTS = [
    "reports/link.rpt", "reports/macro_binding_audit.txt",
    "reports/check_design_precompile.rpt", "reports/check_design_postcompile.rpt",
    "reports/check_timing_precompile.rpt", "reports/check_timing_postcompile.rpt",
    "reports/resources_precompile.rpt", "reports/resources_postcompile.rpt",
    "reports/references_precompile.rpt", "reports/references_postcompile.rpt",
    "reports/hierarchy_postcompile.rpt", "reports/qor.rpt",
    "reports/area_hierarchy.rpt", "reports/timing_setup.rpt",
    "reports/timing_hold_diagnostic.rpt", "reports/constraint_setup.rpt",
    "reports/constraint_hold_diagnostic.rpt", "reports/constraint_max_capacitance.rpt",
    "reports/constraint_max_transition.rpt", "reports/constraint_max_fanout.rpt",
    "reports/flow_contract.rpt", "reports/precompile_loop_gate.rpt",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.sdc",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2.ddc",
    "netlist/m528_dead_write_only_1rw_product_capture_island_r2.svf",
    "TCL_PASS_TERMINAL.txt",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def strict_load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"),
                      object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def strict_load_bytes(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def verify_tree(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad evidence tree")
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"], cwd=str(path),
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS.seal.sha256"],
                          cwd=str(path), stdout=subprocess.DEVNULL)


def verify_file_seal(path):
    path = Path(path)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)
    subprocess.check_call(["sha256sum", "-c", path.name + ".sha256.seal.sha256"],
                          cwd=str(path.parent), stdout=subprocess.DEVNULL)


def assert_population_absent():
    require(not CANONICAL.exists(), "canonical already exists")
    require(not ATTEMPT.exists(), "attempt already consumed")
    require(not LOCK.exists(), "launch lock exists")
    runs = ROOT / "dc_handoff/runs"
    require(not list(runs.glob(".m884_m528_r21_macro_aware_product_dc_work.*")),
            "work population exists")
    require(not list(runs.glob(
        "m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829.failed_or_incomplete.*")),
        "quarantine population exists")


def expected_claims_false(claims):
    for key in ["fair_K_zero_bit", "throughput_per_mm2", "speedup",
                "system_speedup", "system", "power", "energy", "ppa",
                "physical_route", "paper_ppa_ready", "headline"]:
        require(claims[key] is False, "forbidden claim enabled: " + key)


def validate_contract(contract):
    require(set(contract) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "exact_files", "fairness", "foundry_views", "frozen_authorities",
        "future_release_chain", "physical_point", "schema", "status",
        "tool_identity"}, "contract key closure")
    require(contract["schema"] ==
            "m884_m528_r21_macro_aware_product_dc_source_only_contract_v1",
            "contract schema")
    require(contract["status"] ==
            "SOURCE_ONLY_M884_M528_R21_MACRO_AWARE_PRODUCT_DC__FRESH_HAMMER_REQUIRED__NO_EDA_AUTHORIZED",
            "contract status")
    require(contract["authorization"] == {
        "author_ran_eda": False, "run_dc_now": False,
        "run_vcs_now": False, "run_formality_now": False,
        "run_pt_now": False, "run_ptpx_now": False,
        "run_saif_now": False, "run_remote_now": False}, "contract auth")
    require(contract["fairness"] == {
        "candidate_point_only": True, "fair_K_zero_bit": False,
        "zero_rtl_baseline_present": False, "bit_rtl_baseline_present": False},
        "contract fairness")
    expected_claims_false(contract["claim_boundary"])
    point = contract["physical_point"]
    require(point["candidate"] == "M528 R21 product-capture only", "candidate object")
    require(point["clock_period_ns"] == 3.0 and point["ideal_clock"] is True,
            "clock contract")
    require(point["wireload"] == "ZeroWireload" and
            point["compile_define"] == "SYNTHESIS", "compile contract")
    require(point["macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
            point["macro_count"] == 9, "macro identity/count")
    require(point["macro_physical_capacity_bytes"] == 18432 and
            point["total_capacity_obligation_bytes"] == 213376 and
            point["capacity_ceiling_bytes"] == 245760,
            "capacity obligation")
    require(point["all_storage_foundry_macro_mapped"] is False,
            "all-storage claim escaped")
    require(point["macro_slow_fast_min_pair"] is True and
            point["setup_must_be_met"] is True and
            point["hold_diagnostic_only"] is True, "timing boundary")
    require(point["tim209_required"] == 0 and point["opt150_required"] == 0,
            "loop gate")
    require(point["five_constraint_reports"] == [
        "max_delay", "min_delay_diagnostic", "max_capacitance",
        "max_transition", "max_fanout"], "constraint list")
    require(point["mapped_outputs"] == ["Verilog", "SDC", "DDC", "SVF"],
            "mapped outputs")


def validate_candidate(contract, candidate):
    require(set(candidate) == {
        "authorization", "claim_boundary", "date", "docs359_sha256",
        "fairness", "frozen_authorities", "future_release_chain", "identity",
        "launch_now", "prospective_attempt", "schema", "status"},
        "candidate key closure")
    require(candidate["schema"] ==
            "m884_m528_r21_macro_aware_product_dc_launch_candidate_source_only_v1",
            "candidate schema")
    require(candidate["status"] ==
            "READY_FOR_FRESH_M884_SOURCE_HAMMER__NO_EDA_AUTHORIZED",
            "candidate status")
    require(candidate["launch_now"] is False, "launch enabled")
    require(candidate["authorization"] == {
        "max_attempts": 0, "run_dc": False, "run_vcs": False,
        "run_formality": False, "run_pt": False, "run_ptpx": False,
        "run_saif": False, "run_remote": False}, "candidate auth")
    require(candidate["fairness"] == contract["fairness"], "fairness mismatch")
    expected_claims_false(candidate["claim_boundary"])
    require(candidate["identity"] == {
        "runner_path": RUNNER.relative_to(ROOT).as_posix(),
        "runner_sha256": sha(RUNNER),
        "source_contract_path": CONTRACT.relative_to(ROOT).as_posix(),
        "source_contract_sha256": sha(CONTRACT),
        "result_path": "dc_handoff/runs/m884_m528_r21_macro_aware_product_dc_3p000ns_r1_20260829",
        "attempt_path": "dc_handoff/runs/.m884_m528_r21_macro_aware_product_dc_attempt_consumed"},
        "candidate identity")
    require(candidate["frozen_authorities"] == contract["frozen_authorities"],
            "authority mismatch")
    require(candidate["future_release_chain"] == contract["future_release_chain"],
            "release chain mismatch")
    attempt = candidate["prospective_attempt"]
    require(attempt["clock_period_ns"] == 3.0 and attempt["macro_count"] == 9,
            "prospective physical point")
    for key in ["result_absent_at_authoring", "attempt_absent_at_authoring",
                "canonical_unique", "failure_quarantine_unique"]:
        require(attempt[key] is True, "prospective identity false: " + key)


def mutation_rejected(contract, candidate, mutator):
    trial_contract = copy.deepcopy(contract)
    trial_candidate = copy.deepcopy(candidate)
    mutator(trial_contract, trial_candidate)
    try:
        validate_contract(trial_contract)
        validate_candidate(trial_contract, trial_candidate)
    except (KeyError, RuntimeError, TypeError):
        return True
    return False


def cross_python_strict(paths):
    program = r'''
import json, sys
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate")
        out[key] = value
    return out
def nonfinite(value):
    raise ValueError("nonfinite")
for path in sys.argv[1:]:
    with open(path, "rb") as handle:
        json.loads(handle.read().decode("utf-8"), object_pairs_hook=unique,
                   parse_constant=nonfinite)
for raw in [b'{"a":1,"a":2}', b'{"x":{"k":1,"k":2}}',
            b'{"authorization":{"run_dc":false,"run_dc":true}}',
            b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']:
    try:
        json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                   parse_constant=nonfinite)
    except ValueError:
        pass
    else:
        raise SystemExit(91)
print("STRICT_JSON_PASS")
'''
    versions = []
    for executable in ["/usr/libexec/platform-python3.6",
                       "/opt/anaconda3/envs/pytorch310/bin/python3.10"]:
        completed = subprocess.run([executable, "-c", program] +
                                   [str(p) for p in paths],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"STRICT_JSON_PASS" in completed.stdout,
                "strict JSON failed: " + executable)
        version = subprocess.check_output([executable, "--version"],
                                          stderr=subprocess.STDOUT).decode().strip()
        versions.append(version)
    return versions


def run_author_test_both():
    outputs = []
    for executable in ["/usr/libexec/platform-python3.6",
                       "/opt/anaconda3/envs/pytorch310/bin/python3.10"]:
        completed = subprocess.run([executable, str(AUTHOR_TEST)], cwd=str(ROOT),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 0 and
                b"PASS M884 source closure" in completed.stdout,
                "author source closure failed: " + executable)
        outputs.append(completed.stdout.decode().strip())
    return outputs


def run_full_path(expected_runner_sha):
    tmp = Path(tempfile.mkdtemp(prefix="m885_m884_fullpath.", dir="/tmp"))
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "M884_NO_EDA_FULL_PATH_SELFTEST": "1",
        "M884_NO_EDA_SELFTEST_ROOT": str(tmp),
        "M884_EXPECTED_DC_RUNNER_SHA256": expected_runner_sha,
        "M884_EXPECTED_DC_ADMISSION_SHA256": sha(CANDIDATE),
    }
    completed = subprocess.run([str(RUNNER)], cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp, completed


def populate_fixture(root):
    root = Path(root)
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload\n")
    (root / "reports/macro_binding_audit.txt").write_text(
        "status=PASS_M884_RESOLVED_LIBRARY_MACRO_STRUCTURE\n"
        "macro_count_pre=9\nmacro_count_post=9\n"
        "macro_slow_fast_min_pair=true\n")
    (root / "reports/precompile_loop_gate.rpt").write_text(
        "TIM-209=0\nOPT-150=0\nstatus=PASS_PRECOMPILE_LOOP_GATE\n")
    (root / "reports/timing_setup.rpt").write_text("slack (MET) 0.1234\n")
    for name in ["constraint_setup.rpt", "constraint_max_capacitance.rpt",
                 "constraint_max_transition.rpt", "constraint_max_fanout.rpt"]:
        (root / "reports" / name).write_text(
            "This design has no violated constraints.\n")
    (root / "reports/constraint_hold_diagnostic.rpt").write_text(
        "diagnostic hold may be violated\n")
    (root / "reports/area_hierarchy.rpt").write_text("Total cell area: 123.4\n")
    (root / "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v").write_text(
        "\n".join(["TS1N28HPCPHVTB128X128M4S u%d();" % i for i in range(9)]) + "\n")
    (root / "TCL_PASS_TERMINAL.txt").write_text(
        "status=PASS_M884_M528_R21_MACRO_AWARE_PRODUCT_DC_TCL_TERMINAL\n")


def validate_fixture(root):
    root = Path(root)
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                "missing/empty/symlink artifact: " + relative)
    audit = (root / "reports/macro_binding_audit.txt").read_text()
    for line in ["status=PASS_M884_RESOLVED_LIBRARY_MACRO_STRUCTURE",
                 "macro_count_pre=9", "macro_count_post=9",
                 "macro_slow_fast_min_pair=true"]:
        require(line in audit, "macro audit gate: " + line)
    loop = (root / "reports/precompile_loop_gate.rpt").read_text()
    for line in ["TIM-209=0", "OPT-150=0", "status=PASS_PRECOMPILE_LOOP_GATE"]:
        require(line in loop, "loop gate: " + line)
    setup = (root / "reports/timing_setup.rpt").read_text()
    require("slack (VIOLATED)" not in setup and "slack (MET)" in setup,
            "setup gate")
    for name in ["constraint_setup.rpt", "constraint_max_capacitance.rpt",
                 "constraint_max_transition.rpt", "constraint_max_fanout.rpt"]:
        require("This design has no violated constraints." in
                (root / "reports" / name).read_text(), "constraint gate: " + name)
    netlist = (root / "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v").read_text()
    require(len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", netlist)) == 9,
            "mapped macro count")
    all_text = "\n".join(p.read_text(errors="replace") for p in root.rglob("*")
                          if p.is_file() and not p.is_symlink())
    require(not re.search(r"unresolved reference|unable to resolve reference|"
                          r"inferred.*parent|parent.*inferred|register.array fallback",
                          all_text, flags=re.I), "forbidden unresolved/inferred evidence")


def artifact_attacks(tmp):
    base = Path(tmp) / "artifact_base"
    populate_fixture(base)
    validate_fixture(base)
    negatives = 0
    # Every mandatory output must fail closed when absent and when empty.
    for index, relative in enumerate(REQUIRED_ARTIFACTS):
        for mode in ["missing", "zero"]:
            trial = Path(tmp) / ("artifact_%s_%02d" % (mode, index))
            shutil.copytree(base, trial)
            target = trial / relative
            if mode == "missing":
                target.unlink()
            else:
                target.write_bytes(b"")
            try:
                validate_fixture(trial)
            except (OSError, RuntimeError):
                negatives += 1
    semantic = [
        ("pre8", "reports/macro_binding_audit.txt", "macro_count_pre=9", "macro_count_pre=8"),
        ("post8", "reports/macro_binding_audit.txt", "macro_count_post=9", "macro_count_post=8"),
        ("pair_false", "reports/macro_binding_audit.txt", "macro_slow_fast_min_pair=true", "macro_slow_fast_min_pair=false"),
        ("tim209", "reports/precompile_loop_gate.rpt", "TIM-209=0", "TIM-209=1"),
        ("opt150", "reports/precompile_loop_gate.rpt", "OPT-150=0", "OPT-150=1"),
        ("setup", "reports/timing_setup.rpt", "slack (MET)", "slack (VIOLATED)"),
        ("setup_constraint", "reports/constraint_setup.rpt", "This design has no violated constraints.", "VIOLATED"),
        ("cap_constraint", "reports/constraint_max_capacitance.rpt", "This design has no violated constraints.", "VIOLATED"),
        ("tran_constraint", "reports/constraint_max_transition.rpt", "This design has no violated constraints.", "VIOLATED"),
        ("fanout_constraint", "reports/constraint_max_fanout.rpt", "This design has no violated constraints.", "VIOLATED"),
        ("macro8", "netlist/m528_dead_write_only_1rw_product_capture_island_r2_mapped.v", "TS1N28HPCPHVTB128X128M4S u8();\n", ""),
        ("unresolved", "reports/link.rpt", "payload", "unresolved reference"),
    ]
    for index, item in enumerate(semantic):
        label, relative, before, after = item
        trial = Path(tmp) / ("artifact_sem_%02d_%s" % (index, label))
        shutil.copytree(base, trial)
        path = trial / relative
        path.write_text(path.read_text().replace(before, after))
        try:
            validate_fixture(trial)
        except (OSError, RuntimeError):
            negatives += 1
    expected = 2 * len(REQUIRED_ARTIFACTS) + len(semantic)
    require(negatives == expected, "artifact mutation escaped")
    return negatives


def check_source_structure(contract):
    tcl = TCL.read_text()
    sdc = SDC.read_text()
    runner = RUNNER.read_text()
    filelist = [line.strip() for line in FILELIST.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    require(filelist == [ADAPTER.relative_to(ROOT).as_posix(),
                         TOP.relative_to(ROOT).as_posix()], "filelist closure/order")
    require(len(filelist) == len(set(filelist)), "duplicate filelist source")
    require(not any(path.endswith(".v") for path in filelist),
            "foundry behavioral Verilog entered DC filelist")
    for token in [
        "analyze -format sverilog -define SYNTHESIS",
        "set_min_library $std_slow_db -min_version $std_fast_db",
        "set_min_library $macro_slow_db -min_version $macro_fast_db",
        "set_wire_load_model -name ZeroWireload",
        "get_lib_cells -quiet */$macro_cell",
        "get_cells -hierarchical -filter \"ref_name == $macro_cell\"",
        "macro_count_pre", "macro_count_post", "compile_ultra -no_autoungroup",
        "report_area -hierarchy", "report_timing -delay_type max",
        "report_timing -delay_type min", "report_constraint -max_delay",
        "report_constraint -min_delay", "report_constraint -max_capacitance",
        "report_constraint -max_transition", "report_constraint -max_fanout",
        "write_file -format verilog", "write_sdc", "write -format ddc", "set_svf",
    ]:
        require(token in tcl, "missing Tcl gate: " + token)
    require(len(re.findall(r"^compile_ultra\b", tcl, flags=re.M)) == 1,
            "compile_ultra command count")
    require("set_fix_hold" not in tcl and "-only_hold_time" not in tcl,
            "hold optimization entered diagnostic flow")
    require("create_clock -name core_clk -period 3.000" in sdc,
            "3 ns SDC absent")
    require("set_propagated_clock" not in tcl + sdc, "ideal clock violated")
    adapter = re.sub(r"//.*?$|/\*.*?\*/", "", ADAPTER.read_text(),
                     flags=re.M | re.S)
    require(len(re.findall(r"\bTS1N28HPCPHVTB128X128M4S\b", adapter)) == 1,
            "adapter template macro count")
    require("slice < 9" in adapter and "{1'b0, address}" in adapter and
            "slice*128 +: 128" in adapter, "adapter binding")
    require(not re.search(r"\b(?:reg|logic)\b[^;]*\[[^]]+\]\s*\[[^]]+\]", adapter),
            "adapter register-array fallback")
    for token in [
        "macro_count_pre=9", "macro_count_post=9",
        "macro_slow_fast_min_pair=true", "TIM-209=0", "OPT-150=0",
        "slack \\(MET\\)", "constraint_setup.rpt",
        "constraint_max_capacitance.rpt", "constraint_max_transition.rpt",
        "constraint_max_fanout.rpt", "TS1N28HPCPHVTB128X128M4S",
        "fair_K_zero_bit=false", "speedup=false", "system_speedup=false",
        "paper_ppa_ready=false"]:
        require(token in runner, "runner postcondition absent: " + token)
    require("reports/timing_hold_diagnostic.rpt" in runner and
            "constraint_hold_diagnostic.rpt" in runner,
            "hold diagnostic artifacts absent")
    require("This design has no violated constraints." in runner,
            "non-hold constraint hard gate absent")
    require("m884_forbidden_macro_v" in runner and
            "m884_filelist" in runner, "behavioral view exclusion absent")
    require(contract["foundry_views"]["macro_slow_path"].endswith("ssg0p9v125c.db") and
            contract["foundry_views"]["macro_fast_path"].endswith("ffg1p05vm40c.db"),
            "macro corner pair")


def main():
    assert_population_absent()
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "identity nonregular: " + str(path))
        require(sha(path) == digest, "identity SHA drift: " + str(path))
    # Only newly authored M884 payloads carry adjacent two-level file seals.
    # R21 RTL/SVA and the older binding sources are instead pinned in the
    # closed exact_files map and their already sealed evidence authorities.
    for path in [RUNNER, TCL, FILELIST, SDC, AUTHOR_TEST, CONTRACT, CANDIDATE]:
        verify_file_seal(path)
    for tree in [HANDOFF_DIR, REQUEST_DIR, M881_DIR, M879_DIR, M863_DIR, M623_DIR]:
        verify_tree(tree)

    contract = strict_load(CONTRACT)
    candidate = strict_load(CANDIDATE)
    handoff = strict_load(HANDOFF_DIR / "handoff.json")
    request = strict_load(REQUEST_DIR / "request.json")
    m881 = strict_load(M881_DIR / "review.json")
    m879 = strict_load(M879_DIR / "review.json")
    m863 = strict_load(M863_DIR / "RUN_COMPLETE.json")
    m623 = strict_load(M623_DIR / "review.json")
    binding = strict_load(BINDING)
    validate_contract(contract)
    validate_candidate(contract, candidate)

    cited_json = [CONTRACT, CANDIDATE, HANDOFF_DIR / "handoff.json",
                  REQUEST_DIR / "request.json", M881_DIR / "review.json",
                  M879_DIR / "review.json", M863_DIR / "RUN_COMPLETE.json",
                  M623_DIR / "review.json", BINDING]
    versions = cross_python_strict(cited_json)
    author_outputs = run_author_test_both()

    require(handoff["status"] ==
            "AUTHOR_M884_SOURCE_ONLY_COMPLETE__FRESH_INDEPENDENT_HAMMER_REQUIRED__NO_EDA_AUTHORIZED",
            "handoff status")
    require(request["status"] ==
            "REQUEST_FRESH_INDEPENDENT_M884_SOURCE_HAMMER__NO_EDA_OR_RELEASE",
            "request status")
    require(request["required_output"]["score_required"] == 100 and
            request["required_output"]["launch_from_reviewer"] is False,
            "request boundary")

    require(m881["status"] ==
            "PASS_AUDIT__NO_CURRENT_M528_DC_STA_FORMALITY__FRESH_R21_MACRO_AWARE_SUCCESSOR_REQUIRED" and
            m881["verdict"] == "PASS_AUDIT" and m881["score_100"] == 100,
            "M881 authority")
    require(m881["macro_binding_semantics"]["fast_macro_db_available"] is True and
            m881["claim_boundary"]["current_m528_dc_sta"] is False,
            "M881 boundary")
    require(m879["status"] ==
            "PASS100_M863_C1_R21_SYNOPSYS_VCS_E3_FUNCTIONAL_RESULT_ADMITTED" and
            m879["score_out_of_100"] == 100 and
            [m879["p0_count"], m879["p1_count"], m879["p2_count"]] == [0, 0, 0],
            "M879 authority")
    require(m879["claim_boundary"]["directed_component_synopsys_vcs_e3_functional_citable"] is True and
            m879["claim_boundary"]["timing_verified"] is False,
            "M879 boundary")
    require(m863["claim_boundary"]["functional_vcs_only"] is True and
            m863["claim_boundary"]["speedup"] is False and
            m863["claim_boundary"]["ppa"] is False,
            "M863 functional boundary")
    require(m623["status"] ==
            "PASS_M623_M617_R5_BOUNDED_GENERATED_MACRO_COMPONENT_RESULT" and
            m623["claim_boundary"]["rtl_integrated_macro_ppa"] is False,
            "M623 macro boundary")
    require(binding["cell"] == "TS1N28HPCPHVTB128X128M4S" and
            binding["instance_count"] == 9 and
            binding["rtl_adapter"]["synthesizable_register_array_fallback"] is False and
            binding["claim_boundary"]["dc_sta"] is False, "binding plan")

    require(len(contract["exact_files"]) == 10, "exact file count")
    for relative, digest in contract["exact_files"].items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "exact source drift: " + relative)
    for group in ["tool_identity", "foundry_views"]:
        for key, value in contract[group].items():
            if key.endswith("_path"):
                digest_key = key[:-5] + "_sha256"
                if digest_key in contract[group]:
                    path = Path(value)
                    require(path.is_file() and sha(path) == contract[group][digest_key],
                            "external identity drift: " + key)
    macro_manifest = Path(contract["foundry_views"]["macro_manifest_path"])
    subprocess.check_call(["sha256sum", "-c", "SHA256SUMS"],
                          cwd=str(macro_manifest.parent), stdout=subprocess.DEVNULL)

    check_source_structure(contract)

    semantic_mutators = [
        lambda c, a: c.update({"unknown": 1}),
        lambda c, a: a.update({"unknown": 1}),
        lambda c, a: c["authorization"].update({"run_dc_now": True}),
        lambda c, a: a["authorization"].update({"run_dc": True}),
        lambda c, a: a["authorization"].update({"max_attempts": 1}),
        lambda c, a: a.update({"launch_now": True}),
        lambda c, a: c["fairness"].update({"fair_K_zero_bit": True}),
        lambda c, a: a["fairness"].update({"fair_K_zero_bit": True}),
        lambda c, a: c["claim_boundary"].update({"speedup": True}),
        lambda c, a: a["claim_boundary"].update({"throughput_per_mm2": True}),
        lambda c, a: a["claim_boundary"].update({"ppa": True}),
        lambda c, a: c["physical_point"].update({"clock_period_ns": 2.5}),
        lambda c, a: c["physical_point"].update({"macro_count": 8}),
        lambda c, a: c["physical_point"].update({"hold_diagnostic_only": False}),
        lambda c, a: c["physical_point"].update({"tim209_required": 1}),
        lambda c, a: c["physical_point"].update({"all_storage_foundry_macro_mapped": True}),
        lambda c, a: a["identity"].update({"runner_sha256": "0" * 64}),
        lambda c, a: a["identity"].update({"result_path": "dc_handoff/runs/collision"}),
        lambda c, a: a["prospective_attempt"].update({"canonical_unique": False}),
        lambda c, a: a["frozen_authorities"].update({"m879_review_sha256": "0" * 64}),
    ]
    semantic_negatives = sum(1 for mutator in semantic_mutators
                             if mutation_rejected(contract, candidate, mutator))
    require(semantic_negatives == len(semantic_mutators), "semantic mutation escaped")

    strict_negatives = 0
    for raw in [b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}',
                b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']:
        try:
            strict_load_bytes(raw)
        except ValueError:
            strict_negatives += 1
    require(strict_negatives == 5, "strict JSON mutation escaped")

    tmp = Path(tempfile.mkdtemp(prefix="m885_m884_artifacts.", dir="/tmp"))
    try:
        artifact_negatives = artifact_attacks(tmp)
    finally:
        shutil.rmtree(str(tmp))

    full_tmp, full = run_full_path(sha(RUNNER))
    try:
        require(full.returncode == 0, "full no-EDA runner path failed")
        marker = (full_tmp / "FULL_PATH_PASS.txt").read_text()
        for line in ["status=PASS_M884_FULL_ADMISSION_CONTRACT_PATH_NO_EDA",
                     "admission_launch_now=false", "candidate_only=true",
                     "fair_K_zero_bit=false", "attempt_consumed=false",
                     "license_query_started=false", "dc_shell_started=false"]:
            require(line in marker, "no-EDA terminal marker: " + line)
    finally:
        shutil.rmtree(str(full_tmp))
    wrong_tmp, wrong = run_full_path("0" * 64)
    try:
        require(wrong.returncode == 3 and not (wrong_tmp / "FULL_PATH_PASS.txt").exists(),
                "wrong runner SHA did not fail pre-attempt")
    finally:
        shutil.rmtree(str(wrong_tmp))

    assert_population_absent()
    print(json.dumps({
        "status": "PASS100_M884_SOURCE_FRESH_HAMMER",
        "python_versions": versions,
        "author_source_closure_runs": len(author_outputs),
        "strict_json_negatives": strict_negatives,
        "semantic_negatives": semantic_negatives,
        "artifact_negatives": artifact_negatives,
        "full_path_no_eda": 1,
        "wrong_runner_sha_pre_attempt": 1,
        "exact_files": len(contract["exact_files"]),
        "macro_count": 9,
        "dc_runs": 0,
        "vcs_runs": 0,
        "license_queries": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
