#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
HELPER = HW / "dc_handoff/scripts/check_m1311_python_symlink_entity.sh"
WRAPPER = HW / "dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.sh"
ORCHESTRATOR = HW / "dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.py"
ADMISSION = HW / "contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_launch_admission_r1_20260831.json"
CONTRACT = HW / "contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_source_contract_r1_20260831.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(obj, keys):
    assert type(obj) is dict and set(obj) == set(keys)


def sealed_payload(path):
    assert Path(str(path) + ".sha256").read_text().split() == [sha(path), path.name]
    side = Path(str(path) + ".sha256")
    assert Path(str(path) + ".sha256.seal.sha256").read_text().split() == [sha(side), side.name]


def load_orchestrator():
    spec = importlib.util.spec_from_file_location("m1311_orchestrator", str(ORCHESTRATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_helper_fixture(attack=None):
    with tempfile.TemporaryDirectory(prefix="m1311_entity.") as tmp:
        root = Path(tmp)
        entity = root / "entity"
        link3 = root / "python3.6"
        link2 = root / "alternative"
        link1 = root / "python3"
        if attack == "nonregular":
            entity.mkdir()
        else:
            shutil.copy2("/bin/true", str(entity))
            entity.chmod(0o755)
        link3.symlink_to(str(entity))
        link2.symlink_to(str(link3))
        link1.symlink_to(str(link2))
        if attack == "target_swap":
            other = root / "other"
            shutil.copy2("/bin/false", str(other)); other.chmod(0o755)
            link1.unlink(); link1.symlink_to(str(other))
        elif attack == "dangling":
            link3.unlink(); link3.symlink_to(str(root / "missing"))
        if entity.is_file():
            st = entity.stat(); digest = sha(entity)
        else:
            st = entity.stat(); digest = "0" * 64
        if attack == "sha_drift":
            digest = "0" * 64
        cmd = ["/usr/bin/bash", str(HELPER), str(link1), str(link2), str(link2),
               str(link3), str(link3), str(entity), str(entity), str(st.st_dev),
               str(st.st_ino), "%o" % stat.S_IMODE(st.st_mode), str(st.st_size), digest]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


def write_reports(root, hold=0.0, unconstrained=0, untested=0):
    reports = root / "reports"; reports.mkdir(parents=True)
    (reports / "timing_setup_slow.rpt").write_text("slack (MET) 0.100000\n")
    state = "MET" if hold >= 0 else "VIOLATED"
    (reports / "timing_hold_fast.rpt").write_text("slack (%s) %.6f\n" % (state, hold))
    rows = []
    for name in ("setup", "hold", "out_setup", "out_hold"):
        rows.append("%s 10 10 (100.00%%) 0 (0.00%%) %d (0.00%%)\n" % (name, untested))
    (reports / "analysis_coverage.rpt").write_text("".join(rows))
    if unconstrained:
        check = ("Warning: There are %d input ports with no input delay specified. "
                 "The paths from such ports will be unconstrained.\n" % unconstrained)
    else:
        check = "No endpoint warning.\n"
    (reports / "check_timing.rpt").write_text(check)
    (reports / "constraint_violators.rpt").write_text("No violators.\n")


def receipt_mock(module, hold=0.0, unconstrained=0, untested=0):
    with tempfile.TemporaryDirectory(prefix="m1311_receipt.") as tmp:
        old = module.M1288_CANONICAL
        module.M1288_CANONICAL = Path(tmp)
        try:
            write_reports(module.M1288_CANONICAL, hold, unconstrained, untested)
            return module.result_receipt()
        finally:
            module.M1288_CANONICAL = old


def main():
    checks = {}
    subprocess.run(["bash", "-n", str(HELPER)], check=True)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    subprocess.run(["/usr/libexec/platform-python3.6", "-m", "py_compile", str(ORCHESTRATOR)], check=True)
    checks["syntax"] = True

    assert run_helper_fixture() == 0
    assert run_helper_fixture("target_swap") != 0
    assert run_helper_fixture("dangling") != 0
    assert run_helper_fixture("nonregular") != 0
    assert run_helper_fixture("sha_drift") != 0
    checks["python_entity_attacks"] = True

    source = WRAPPER.read_text()
    assert "exec 9</usr/libexec/platform-python3.6" in source
    assert 'stat -Lc \'%d:%i:%a:%s\' "/proc/${BASHPID}/fd/9"' in source
    assert 'm1311_sha "/proc/${BASHPID}/fd/9"' in source
    assert 'exec /usr/bin/env -i' in source
    checks["fd_bound_entity_recheck"] = True

    admission = json.loads(ADMISSION.read_text())
    contract = json.loads(CONTRACT.read_text())
    exact(admission, ("schema", "date", "milestone", "status", "objective",
        "identity", "exact_files", "tool", "python_entity", "preflight",
        "authorization", "result_adjudication", "claim_boundary"))
    assert admission["identity"]["wrapper_sha256"] == sha(WRAPPER)
    assert admission["identity"]["orchestrator_sha256"] == sha(ORCHESTRATOR)
    assert admission["identity"]["helper_sha256"] == sha(HELPER)
    assert admission["identity"]["contract_sha256"] == sha(CONTRACT)
    assert admission["authorization"] == {
        "launch_now": False, "launch_after_independent_hammer": True,
        "run_pt": True, "run_dc": False, "run_vcs": False,
        "run_formality": False, "run_ptpx": False, "run_remote": False,
        "query_license": True, "max_attempts_is_one": True,
        "strict_result_adjudication": True}
    assert admission["claim_boundary"]["launch_admission_only"] is True
    assert all(v is False for k, v in admission["claim_boundary"].items()
               if k != "launch_admission_only")
    for relative, digest in admission["exact_files"].items():
        target = HW / relative
        assert target.is_file() and not target.is_symlink() and sha(target) == digest
    sealed_payload(ADMISSION); sealed_payload(CONTRACT)
    checks["admission_contract_exact_closed"] = True

    assert contract["status"] == "M1311_SOURCE_ONLY__ZERO_LICENSE_PT_EDA"
    assert contract["authorization"] == {
        "launch_now": False, "max_attempts_now": 0, "run_pt_now": False,
        "run_dc_now": False, "run_vcs_now": False, "run_formality_now": False,
        "run_ptpx_now": False, "run_remote_now": False,
        "query_license_now": False, "independent_receipt_blind_hammer_required": True}
    checks["source_authority_zero"] = True

    orchestrator_text = ORCHESTRATOR.read_text()
    order = [orchestrator_text.index("validate_admission(data, args)"),
             orchestrator_text.index("first_collisions = collisions()"),
             orchestrator_text.index("mem = meminfo()"),
             orchestrator_text.index("license_run = subprocess.run("),
             orchestrator_text.index("ATTEMPT.mkdir()"),
             orchestrator_text.index('subprocess.run(["/usr/bin/bash", str(M1288_RUNNER)]')]
    assert order == sorted(order)
    checks["preflight_attempt_launch_order"] = True

    module = load_orchestrator()
    assert module.process_is_repo_scoped(str(module.REPO / "hw_autoresearch_nts07"),
                                         ["/opt/synopsys/pt_shell"])
    assert module.process_is_repo_scoped("/tmp", [str(module.REPO / "run.tcl")])
    assert not module.process_is_repo_scoped("/home/fangyl/Work/project",
                                             ["/opt/synopsys/vcs", "simv"])
    checks["repo_scoped_collision_positive_negative"] = True

    good = receipt_mock(module, 0.0, 0, 0)
    neg = receipt_mock(module, -0.05, 0, 0)
    unconstrained = receipt_mock(module, 0.0, 2, 0)
    uncovered = receipt_mock(module, 0.0, 0, 1)
    assert good["strict_timing_gate_pass"] is True
    assert neg["strict_timing_gate_pass"] is False
    assert unconstrained["strict_timing_gate_pass"] is False
    assert uncovered["strict_timing_gate_pass"] is False
    for receipt in (good, neg, unconstrained, uncovered):
        assert receipt["claim_boundary"] == {
            "fresh_result_hammer_required": True, "power": False,
            "energy": False, "speedup": False, "system": False,
            "paper_ppa_ready": False, "headline": False}
    checks["strict_timing_mocks"] = True

    frozen = {
        "dc_handoff/scripts/run_m1302_m1288_fixed_t10_ptsta_exact_closed_one_shot.sh": "3f24a7d38df4e5c9df6b5316cc747b272fc4161d09b9a1580ea07f9998f18446",
        "contracts/m1289_m1288_c3_m917_fixed_t10_ptsta_launch_admission_r1_20260830.json": "1ea53ea55a8cc2bbc992aa932f73e7865561f7dde16e53f5d74efe3a7b146e3e",
        "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"}
    for relative, digest in frozen.items():
        assert sha(HW / relative) == digest
    checks["old_files_unchanged"] = True

    for path in (module.M1288_CANONICAL, module.M1288_WORK, module.M1288_ATTEMPT,
                 module.M1302_CANONICAL, module.M1302_WORK, module.M1302_ATTEMPT,
                 module.CANONICAL, module.WORK, module.ATTEMPT):
        assert not path.exists() and not path.is_symlink()
    checks["zero_attempt_namespaces"] = True

    print(json.dumps({"schema": "m1311_author_static_test_v1", "status": "PASS",
        "checks": checks, "license_pt_eda_calls": 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
