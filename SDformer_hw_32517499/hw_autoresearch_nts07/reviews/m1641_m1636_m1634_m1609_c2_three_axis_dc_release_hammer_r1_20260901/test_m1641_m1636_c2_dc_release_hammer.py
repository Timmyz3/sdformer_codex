#!/usr/bin/env python3
"""Read-only M1641 hammer for the M1636 three-axis DC release."""

from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import stat
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RELEASE = HW / (
    "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_"
    "r1_20260901.json")
RUNNER = HW / (
    "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_"
    "axis_logic_only_exact_sha_r1.sh")
CONTRACT = HW / (
    "contracts/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_"
    "source_contract_r1_20260901.json")
HAMMER = HW / (
    "reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_r1_20260901")
FILELIST = HW / (
    "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_"
    "axis_logic_only_dc.f")
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RUNS = HW / "dc_handoff/runs"

RELEASE_SHA = "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088"
RUNNER_SHA = "da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7"
CONTRACT_SHA = "9f5e5b1cb40da5cd403270ba48ceac9b5a7d6aecd79b7ad98cf3d644d0f8f030"
HAMMER_REVIEW_SHA = "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620"
HAMMER_MANIFEST_SHA = "e47e87a00975172451069984073e83487f1cb97cdd101a240437ed789fac66aa"
HAMMER_OUTER_SHA = "9dbcef360c8038403174bbfe05e3c0f3e3f09a7235c78cac1c47ae1a94707614"
FILELIST_SHA = "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f"
TCL_SHA = "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe"
SDC_SHA = "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(rows):
        value = {}
        for key, item in rows:
            assert key not in value
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite " + token)))


def strict_json(path):
    return strict_json_text(Path(path).read_text(encoding="utf-8"))


def verify_file_seal(path):
    path = Path(path)
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for member in (path, sums, outer):
        assert stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink()
    assert sums.read_text(encoding="ascii") == sha256(path) + "  " + path.name + "\n"
    assert outer.read_text(encoding="ascii") == sha256(sums) + "  " + sums.name + "\n"


def verify_dir_seal(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert stat.S_ISREG(sums.lstat().st_mode) and not sums.is_symlink()
    assert stat.S_ISREG(outer.lstat().st_mode) and not outer.is_symlink()
    assert outer.read_text(encoding="ascii") == sha256(sums) + "  SHA256SUMS\n"
    listed = set()
    for row in sums.read_text(encoding="utf-8").splitlines():
        fields = row.split("  ", 1)
        assert len(fields) == 2 and len(fields[0]) == 64
        relative = Path(fields[1])
        assert not relative.is_absolute() and ".." not in relative.parts
        member = root / relative
        assert stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink()
        assert sha256(member) == fields[0]
        assert relative.as_posix() not in listed
        listed.add(relative.as_posix())
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    assert actual == listed


def audit_release(value):
    assert value["schema"] == (
        "m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_v1")
    assert value["milestone"] == "M1636"
    assert value["status"] == (
        "AUTHORIZE_ONE_M1634_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_ATTEMPT")
    assert value["authorization"] == {
        "dc_shell_runs": 3, "all_other_eda_runs": 0}
    identity = value["identity"]
    assert identity == {
        "release_path": (
            "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_"
            "release_r1_20260901.json"),
        "runner_path": (
            "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_"
            "three_axis_logic_only_exact_sha_r1.sh"),
        "runner_sha256": RUNNER_SHA,
        "source_contract_path": (
            "contracts/m1634_m1609_c2_registered_fault_three_axis_logic_"
            "only_dc_source_contract_r1_20260901.json"),
        "source_contract_sha256": CONTRACT_SHA,
        "hammer_review_path": (
            "reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_"
            "r1_20260901/review.json"),
        "hammer_review_sha256": HAMMER_REVIEW_SHA,
        "hammer_manifest_sha256": HAMMER_MANIFEST_SHA,
        "hammer_outer_seal_file_sha256": HAMMER_OUTER_SHA,
        "docs359_sha256": DOCS359_SHA,
    }
    assert value["axes"] == [
        {"name": "k1", "arch_mode": 0},
        {"name": "k8", "arch_mode": 1},
        {"name": "k1x8", "arch_mode": 2},
    ]
    assert value["fairness"] == {
        "fresh_rtl_synthesis_each_axis": True,
        "same_top_filelist_tcl_sdc_libraries_clock": True,
        "m1609_registered_fault_seam_in_every_axis": True,
        "old_m872_netlist_reuse": False,
        "clock_period_ns": 3.0,
        "setup_uncertainty_ns": 0.2,
        "hold_uncertainty_ns": 0.05,
        "ideal_clock": True,
        "wireload": "ZeroWireload",
        "logic_only_pre_macro": True,
        "macro_count": 0,
        "compile_ultra_per_axis": 1,
        "hold_diagnostic_only": True,
    }
    assert value["canonical_namespace"] == {
        "result": (
            "dc_handoff/runs/m1634_m1609_c2_registered_fault_three_axis_"
            "logic_only_dc_3p000ns_r1_20260901"),
        "attempt": (
            "dc_handoff/runs/.m1634_m1609_c2_registered_fault_three_axis_"
            "logic_only_dc_attempt_consumed"),
        "lock": (
            "dc_handoff/runs/.m1634_m1609_c2_registered_fault_three_axis_"
            "logic_only_dc_launch_lock"),
        "fresh_at_release_authoring": True,
        "consume_attempt_before_first_dc": True,
        "retry": False,
        "publish_no_replace": True,
    }
    assert value["post_run_gate"] == {
        "different_author_result_hammer_required": True,
        "production_saif_ptpx_requires_new_release": True,
        "frozen_cycles_1913_1945_not_refreshed_by_this_run": True,
    }
    assert value["claim_boundary"] == {
        "release_authored": True,
        "launch_executed": False,
        "fresh_physical_axes": False,
        "hold_closed": False,
        "power": False,
        "energy": False,
        "cycle_refresh": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    }
    launch = value["launch_instruction"]
    assert launch["automatic_launch"] is False
    assert launch["runner_path"] == identity["runner_path"]
    assert launch["caller_pin_runner"] == (
        "M1634_EXPECTED_DC_RUNNER_SHA256=" + RUNNER_SHA)
    assert launch["caller_pin_release"] == (
        "M1634_EXPECTED_DC_RELEASE_SHA256="
        "CALLER_MUST_SUPPLY_EXACT_DOUBLE_SEALED_M1636_FILE_SHA256")
    assert value["author_execution_receipt"] == {
        "dc_shell_runs": 0, "all_other_eda_runs": 0,
        "attempt_or_result_created": False, "docs359_modified": False}


def audit_runner(text, tcl, filelist):
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert '[[ $# -eq 0 ]]' in text
    assert text.count('"${DC_SHELL}" -f "${TCL}"') == 1
    assert "axis_names=(k1 k8 k1x8)" in text
    assert "axis_modes=(0 1 2)" in text
    assert "for index in 0 1 2; do" in text
    assert text.count("compile_ultra_per_axis':1") == 1
    assert "dc_shell_runs':3" in text
    assert "'vcs_runs':0,'pt_runs':0,'ptpx_runs':0,'formality_runs':0" in text
    assert "automatic_retry':False" in text

    release_gate = text.index('verify_file_seal "${RELEASE}"')
    runner_pin = text.index('M1634_EXPECTED_DC_RUNNER_SHA256')
    release_pin = text.index('M1634_EXPECTED_DC_RELEASE_SHA256')
    freshness = text.index('[[ ! -e "${RESULT}"')
    lock = text.index('mkdir -- "${LOCK}"')
    attempt = text.index('mkdir -- "${ATTEMPT}"')
    work = text.index('mkdir -- "${WORK}"')
    dc = text.index('"${DC_SHELL}" -f "${TCL}"')
    assert release_gate < runner_pin < release_pin < freshness < lock < attempt < work < dc
    assert text.count('mkdir -- "${ATTEMPT}"') == 1
    assert 'rm -rf' not in text and 'rm -r' not in text
    assert 'rmdir -- "${ATTEMPT}"' not in text
    assert 'retry=false' in text and 'automatic_retry' in text
    assert 'mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"' in text
    assert 'mv -T -n -- "${WORK}" "${RESULT}"' in text

    assert 'cp -- "${FILELIST}" "${WORK}/input_filelist.f"' in text
    assert not re.search(r'\b(read_ddc|read_verilog)\b', text)
    assert 'cp -- "${old_artifact}"' not in text
    assert text.count('old_m872_netlist_reuse\':False') == 1
    assert 'DESIGN_NAME="${DESIGN}" HW_ROOT="${HW_ROOT}" RTL_FILELIST="${FILELIST}"' in text
    assert 'LIB_DB="${SLOW_DB}" MIN_LIB_DB="${FAST_DB}" SDC_FILE="${SDC}"' in text
    assert 'CLOCK_PERIOD_NS=3.000 ELAB_PARAMETERS="ARCH_MODE=${mode}"' in text
    assert "'axis_order':['k1','k8','k1x8']" in text
    assert "'hold_closed':False,'hold_report_present':True" in text
    assert "'hold_diagnostic_only':True" in text
    assert "'system_speedup':False,'paper_headline':False" in text
    assert "cycle_refresh" not in text and "1913" not in text and "1945" not in text

    rows = [row for row in filelist.splitlines() if row.strip()]
    assert len(rows) == 12
    assert rows[0] == (
        "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_"
        "compactor_registered_fault_successor.sv")
    assert not any(row.startswith("rtl_m214/") for row in rows)
    compile_commands = re.findall(r"(?m)^\s*compile_ultra\s*$", tcl)
    assert len(compile_commands) == 1
    assert "compile_ultra -incremental" not in tcl
    assert "compile -incremental" not in tcl
    assert "set_fix_hold" not in tcl
    assert "ELAB_PARAMETERS" in tcl and "elaborate $design_name -parameters" in tcl


def mutated_release_rejections(text):
    replacements = [
        ('"dc_shell_runs": 3', '"dc_shell_runs": 2'),
        ('"all_other_eda_runs": 0', '"all_other_eda_runs": 1'),
        ('"arch_mode": 0', '"arch_mode": 1'),
        ('"arch_mode": 2', '"arch_mode": 3'),
        ('"fresh_rtl_synthesis_each_axis": true', '"fresh_rtl_synthesis_each_axis": false'),
        ('"same_top_filelist_tcl_sdc_libraries_clock": true', '"same_top_filelist_tcl_sdc_libraries_clock": false'),
        ('"m1609_registered_fault_seam_in_every_axis": true', '"m1609_registered_fault_seam_in_every_axis": false'),
        ('"old_m872_netlist_reuse": false', '"old_m872_netlist_reuse": true'),
        ('"clock_period_ns": 3.0', '"clock_period_ns": 4.0'),
        ('"setup_uncertainty_ns": 0.2', '"setup_uncertainty_ns": 0.0'),
        ('"hold_uncertainty_ns": 0.05', '"hold_uncertainty_ns": 0.0'),
        ('"ideal_clock": true', '"ideal_clock": false'),
        ('"wireload": "ZeroWireload"', '"wireload": "enclosed"'),
        ('"logic_only_pre_macro": true', '"logic_only_pre_macro": false'),
        ('"macro_count": 0', '"macro_count": 1'),
        ('"compile_ultra_per_axis": 1', '"compile_ultra_per_axis": 2'),
        ('"hold_diagnostic_only": true', '"hold_diagnostic_only": false'),
        ('"fresh_at_release_authoring": true', '"fresh_at_release_authoring": false'),
        ('"consume_attempt_before_first_dc": true', '"consume_attempt_before_first_dc": false'),
        ('"retry": false', '"retry": true'),
        ('"publish_no_replace": true', '"publish_no_replace": false'),
        ('"frozen_cycles_1913_1945_not_refreshed_by_this_run": true',
         '"frozen_cycles_1913_1945_not_refreshed_by_this_run": false'),
        ('"cycle_refresh": false', '"cycle_refresh": true'),
        ('"automatic_launch": false', '"automatic_launch": true'),
        ('"attempt_or_result_created": false', '"attempt_or_result_created": true'),
    ]
    rejected = 0
    survivors = []
    for index, (old, new) in enumerate(replacements):
        mutant = text.replace(old, new)
        assert mutant != text, old
        try:
            audit_release(strict_json_text(mutant))
        except (AssertionError, KeyError, ValueError):
            rejected += 1
        else:
            survivors.append(index)
    return rejected, len(replacements), survivors


def mutated_runner_rejections(text, tcl, filelist):
    replacements = [
        ("set -euo pipefail", "set -eo pipefail"),
        ('[[ $# -eq 0 ]]', '[[ $# -ge 0 ]]'),
        ("axis_names=(k1 k8 k1x8)", "axis_names=(k1 k8)"),
        ("axis_modes=(0 1 2)", "axis_modes=(0 1 1)"),
        ("for index in 0 1 2; do", "for index in 0 1; do"),
        ('verify_file_seal "${RELEASE}"', ': # release seal removed'),
        ('M1634_EXPECTED_DC_RUNNER_SHA256', 'M1634_UNPINNED_DC_RUNNER_SHA256'),
        ('M1634_EXPECTED_DC_RELEASE_SHA256', 'M1634_UNPINNED_DC_RELEASE_SHA256'),
        ('mkdir -- "${ATTEMPT}"', ': # attempt removed'),
        ('mkdir -- "${WORK}"', 'mkdir -p -- "${WORK}"'),
        ('mv -T -n -- "${WORK}" "${RESULT}"', 'mv -T -- "${WORK}" "${RESULT}"'),
        ('mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"',
         'mv -T -- "${WORK}" "${RESULT}"'),
        ('cp -- "${FILELIST}" "${WORK}/input_filelist.f"',
         'cp -- "${old_artifact}" "${WORK}/input.ddc"'),
        ('old_m872_netlist_reuse\':False', 'old_m872_netlist_reuse\':True'),
        ('CLOCK_PERIOD_NS=3.000', 'CLOCK_PERIOD_NS=4.000'),
        ('ELAB_PARAMETERS="ARCH_MODE=${mode}"', 'ELAB_PARAMETERS="ARCH_MODE=1"'),
        ("'hold_diagnostic_only':True", "'hold_diagnostic_only':False"),
        ("'hold_closed':False,'hold_report_present':True",
         "'hold_closed':True,'hold_report_present':True"),
        ("'dc_shell_runs':3", "'dc_shell_runs':2"),
        ("'vcs_runs':0,'pt_runs':0,'ptpx_runs':0,'formality_runs':0",
         "'vcs_runs':1,'pt_runs':0,'ptpx_runs':0,'formality_runs':0"),
        ("'automatic_retry':False", "'automatic_retry':True"),
        ("'system_speedup':False,'paper_headline':False",
         "'system_speedup':True,'paper_headline':True"),
        ('"${DC_SHELL}" -f "${TCL}"',
         '"${DC_SHELL}" -f "${TCL}"\n  "${DC_SHELL}" -f "${TCL}"'),
        ('[[ ! -e "${RESULT}"', '[[ -e "${RESULT}"'),
        ('retry=false', 'retry=true'),
    ]
    rejected = 0
    survivors = []
    for index, (old, new) in enumerate(replacements):
        mutant = text.replace(old, new)
        assert mutant != text, old
        try:
            audit_runner(mutant, tcl, filelist)
        except (AssertionError, ValueError):
            rejected += 1
        else:
            survivors.append(index)
    return rejected, len(replacements), survivors


class M1641ReleaseHammer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release_text = RELEASE.read_text(encoding="utf-8")
        cls.release = strict_json_text(cls.release_text)
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.filelist = FILELIST.read_text(encoding="utf-8")

    def test_01_release_identity_and_double_seal(self):
        verify_file_seal(RELEASE)
        self.assertEqual(sha256(RELEASE), RELEASE_SHA)
        self.assertEqual(sha256(RUNNER), RUNNER_SHA)
        self.assertEqual(sha256(CONTRACT), CONTRACT_SHA)

    def test_02_m1635_identity_and_double_seal(self):
        verify_dir_seal(HAMMER)
        self.assertEqual(sha256(HAMMER / "review.json"), HAMMER_REVIEW_SHA)
        self.assertEqual(sha256(HAMMER / "SHA256SUMS"), HAMMER_MANIFEST_SHA)
        self.assertEqual(sha256(HAMMER / "SHA256SUMS.seal.sha256"), HAMMER_OUTER_SHA)

    def test_03_release_schema_authority_and_claims(self):
        audit_release(self.release)

    def test_04_exact_common_sources_tcl_sdc_and_docs(self):
        self.assertEqual(sha256(FILELIST), FILELIST_SHA)
        self.assertEqual(sha256(TCL), TCL_SHA)
        self.assertEqual(sha256(SDC), SDC_SHA)
        self.assertEqual(sha256(DOCS359), DOCS359_SHA)

    def test_05_exact_three_fresh_axes_and_no_old_netlist(self):
        audit_runner(self.runner, self.tcl, self.filelist)

    def test_06_attempt_and_all_result_namespaces_are_fresh(self):
        exact = [
            RUNS / "m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_3p000ns_r1_20260901",
            RUNS / ".m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_attempt_consumed",
            RUNS / ".m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_launch_lock",
        ]
        for path in exact:
            self.assertFalse(path.exists(), str(path))
        self.assertEqual(list(RUNS.glob(
            ".m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_work.*")), [])
        self.assertEqual(list(RUNS.glob(
            "m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_3p000ns_"
            "r1_20260901.failed_or_incomplete.*.quarantine")), [])

    def test_07_only_three_dc_shell_runs_are_released(self):
        self.assertEqual(self.release["authorization"], {
            "dc_shell_runs": 3, "all_other_eda_runs": 0})
        self.assertEqual(self.runner.count('"${DC_SHELL}" -f "${TCL}"'), 1)
        self.assertIn("for index in 0 1 2; do", self.runner)

    def test_08_same_resource_axis_mapping(self):
        fair = self.release["fairness"]
        self.assertTrue(fair["same_top_filelist_tcl_sdc_libraries_clock"])
        self.assertEqual(fair["clock_period_ns"], 3.0)
        self.assertTrue(fair["logic_only_pre_macro"])
        self.assertEqual(fair["macro_count"], 0)

    def test_09_no_retry_and_atomic_no_replace(self):
        namespace = self.release["canonical_namespace"]
        self.assertFalse(namespace["retry"])
        self.assertTrue(namespace["consume_attempt_before_first_dc"])
        self.assertTrue(namespace["publish_no_replace"])
        self.assertNotIn("while true", self.runner)
        self.assertNotIn("until ", self.runner)

    def test_10_hold_is_diagnostic_and_cycles_are_not_refreshed(self):
        self.assertTrue(self.release["fairness"]["hold_diagnostic_only"])
        self.assertFalse(self.release["claim_boundary"]["hold_closed"])
        self.assertFalse(self.release["claim_boundary"]["cycle_refresh"])
        self.assertTrue(self.release["post_run_gate"][
            "frozen_cycles_1913_1945_not_refreshed_by_this_run"])

    def test_11_release_mutations_fail_closed(self):
        rejected, total, survivors = mutated_release_rejections(self.release_text)
        self.assertEqual(total, 25)
        self.assertEqual(rejected, total, repr(survivors))

    def test_12_runner_mutations_fail_closed(self):
        rejected, total, survivors = mutated_runner_rejections(
            self.runner, self.tcl, self.filelist)
        self.assertEqual(total, 25)
        self.assertEqual(rejected, total, repr(survivors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
