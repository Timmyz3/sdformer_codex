#!/opt/anaconda3/bin/python3
"""Read-only M2179 hammer for the consumed M2178 failed SAIF acquisition."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m2178_m2176_ordinary_native_saif_reset_semantics_preflight_attempt_consumed"
QUARANTINE = RESULTS / "m2178_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_r1_20260904.failed.3022955.quarantine"
CANONICAL = RESULTS / "m2178_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_r1_20260904"
LOCK = RESULTS / ".m2178_m2176_ordinary_native_saif_reset_semantics_preflight_launch_lock"
UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M2177 = HW / "reviews/m2177_m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_hammer_r1_20260904"
OFFICIAL_SAIF = Path("/opt/synopsys/vcs/V-2023.12-SP1/doc/UserGuide/html/vcs_user_guide/saif_support/saif_support.html")
OFFICIAL_SV = Path("/opt/synopsys/vcs/V-2023.12-SP1/doc/UserGuide/html/vcs_mx_lca_features/saif_features/saif_features.html")
M1046 = RESULTS / "m1046_m1001_c2_ucli_power_preflight.2027456.sealed"
M1044_UCLI = HW / "dc_handoff/scripts/m1044_vcs_ucli_power_saif_preflight.ucli.tcl"
M2139_Q = RESULTS / "m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904.failed.2153005.quarantine"
M2125_UCLI = HW / "dc_handoff/scripts/m2125_m2018_tsbg_ordinary_rtl_saif_window_diagnostic.ucli.tcl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_exhaustive_seal(directory: Path) -> dict[str, object]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    entries: dict[str, str] = {}
    for raw in manifest.read_text().splitlines():
        digest, name = raw.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        assert name not in entries
        path = directory / name
        assert path.is_file() and not path.is_symlink()
        assert sha256(path) == digest, path
        entries[name] = digest
    actual = {
        str(path.relative_to(directory)) for path in directory.rglob("*")
        if path.is_file() and path not in {manifest, outer}
    }
    assert set(entries) == actual, (set(entries) - actual, actual - set(entries))
    outer_digest, outer_name = outer.read_text().split()
    assert outer_name.lstrip("*") == "SHA256SUMS"
    assert outer_digest == sha256(manifest)
    return {
        "member_count": len(entries), "exhaustive": True,
        "manifest_sha256": sha256(manifest), "outer_sha256": sha256(outer),
    }


def html_text(path: Path) -> str:
    value = path.read_text(errors="ignore")
    value = re.sub(r"<script.*?</script>|<style.*?</style>", " ", value,
                   flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def saif_profile(path: Path) -> dict[str, object]:
    text = path.read_text()
    lines = text.splitlines()
    durations = re.findall(r"\(DURATION\s+([^)]+)\)", text)
    return {
        "sha256": sha256(path), "bytes": path.stat().st_size, "lines": len(lines),
        "duration": durations[0].strip() if len(durations) == 1 else None,
        "instance_count": sum(line.lstrip().startswith("(INSTANCE ") for line in lines),
        "net_block_count": sum(line.strip() == "(NET" for line in lines),
        "record_count": sum(line.lstrip().startswith("(T0 ") for line in lines),
        "default_monitoring_header": "default behavior is to monitor internal nets" in text,
        "explicit_all_monitoring_header": "explicitly turns" in text,
    }


def main() -> None:
    assert sha256(DOC359) == DOC359_SHA256
    attempt_seal = verify_exhaustive_seal(ATTEMPT)
    quarantine_seal = verify_exhaustive_seal(QUARANTINE)
    m2177_seal = verify_exhaustive_seal(M2177)
    attempt = json.loads((ATTEMPT / "attempt.json").read_text())
    counts = json.loads((QUARANTINE / "execution_counts.json").read_text())
    commands = json.loads((QUARANTINE / "execution_commands.json").read_text())
    assert attempt["status"] == "M2178_ATTEMPT_CONSUMED"
    assert attempt["automatic_retry"] is False
    assert counts == {
        "admitted_measurement_saif_files": 0, "admitted_saif_files": 0,
        "dc_runs": 0, "diagnostic_saif_files_written": 1, "gpu_runs": 0,
        "icc2_runs": 0, "license_queries": 1, "ptpx_runs": 0,
        "raw_saif_files_written": 2, "simv_runs": 1, "vcs_compiles": 1,
    }
    assert not CANONICAL.exists() and not LOCK.exists()
    m2178_names = sorted(path.name for path in RESULTS.iterdir()
                         if "m2178" in path.name.lower())
    assert m2178_names == [ATTEMPT.name, QUARANTINE.name]

    compile_log = (QUARANTINE / "vcs_compile.log").read_text()
    runtime_log = (QUARANTINE / "rtl_sim.log").read_text()
    runtime_parse = json.loads((QUARANTINE / "runtime_parse.log").read_text())
    prehistory_failure = (QUARANTINE / "prehistory_saif_parse.log").read_text().strip()
    failed = (QUARANTINE / "FAILED_DO_NOT_CITE.txt").read_text()
    assert "simv up to date" in compile_log and "Top Level Modules:" in compile_log
    assert "Error-" not in compile_log
    assert "PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT" in runtime_log
    assert runtime_parse["completion_ledger"] == {
        "bundles": 1788, "commits": 24, "cycles": 20292, "issues": 1278,
        "products": 29472, "reads": 14304, "rows": 149,
    }
    assert runtime_parse["duration_ns"] == 60876.0
    assert prehistory_failure == "M2176_PARSE_FAIL_CLOSED: target INSTANCE dut_ordinary count 0 != 1"
    assert "status=FAILED_DO_NOT_CITE" in failed and "automatic_retry=false" in failed
    assert commands["vcs_compile"].count("-debug_access+r") == 1

    prehistory = saif_profile(QUARANTINE / "rtl_prehistory.saif")
    measurement = saif_profile(QUARANTINE / "rtl_measurement.saif")
    assert prehistory["duration"] == "1167.01"
    assert measurement["duration"] == "60876.00"
    for profile in (prehistory, measurement):
        assert profile["instance_count"] == 0 and profile["net_block_count"] == 0
        assert profile["record_count"] == 0 and profile["default_monitoring_header"]
        assert not profile["explicit_all_monitoring_header"]
    for name in ("rtl_prehistory.saif", "rtl_measurement.saif"):
        sidecar = QUARANTINE / f"{name}.sha256"
        outer = QUARANTINE / f"{name}.sha256.seal.sha256"
        assert sidecar.read_text().split() == [sha256(QUARANTINE / name), name]
        assert outer.read_text().split() == [sha256(sidecar), sidecar.name]

    ucli = UCLI.read_text()
    assert "power -gate_level" not in ucli
    assert "power tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary" in ucli
    assert ucli.count("power -enable") == 2 and ucli.count("power -report") == 2

    official = html_text(OFFICIAL_SAIF)
    official_sv = html_text(OFFICIAL_SV)
    assert "power -gate_level on mda" in official
    assert "power -gate_level on sv" in official_sv
    assert "must pass the sv string to power -gate_level command" in official_sv
    assert "must use the -debug_access+pp compile-time option" in official_sv

    m1046_seal = verify_exhaustive_seal(M1046)
    m1046_receipt = json.loads((M1046 / "preflight.json").read_text())
    m1046_saif = saif_profile(M1046 / "tiny.saif")
    assert m1046_receipt["status"] == "PASS_M1044_TINY_UCLI_POWER_SAIF_PREFLIGHT"
    assert m1046_receipt["debug_flag"] == "-debug_access+r"
    assert m1046_receipt["saif_nonempty"] is True
    assert "power -gate_level all mda sv" in M1044_UCLI.read_text()
    assert m1046_saif["instance_count"] >= 2 and m1046_saif["record_count"] > 0
    assert m1046_saif["explicit_all_monitoring_header"]

    m2139_saif = saif_profile(M2139_Q / "ordinary_lru4/rtl_execute.saif")
    m2139_commands = json.loads((M2139_Q / "execution_commands.json").read_text())
    assert "power -gate_level all mda sv" in M2125_UCLI.read_text()
    assert m2139_commands["vcs_compile"].count("-debug_access+r") == 1
    assert m2139_saif["duration"] == "60876.00"
    assert m2139_saif["record_count"] == 93971
    assert m2139_saif["instance_count"] > 0 and m2139_saif["net_block_count"] > 0
    assert m2139_saif["explicit_all_monitoring_header"]

    output = {
        "status": "PASS_M2179_READ_ONLY_FAILURE_ISOLATION",
        "execution_invoked": {
            "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
            "raw_saif_files_written": 0, "dc_runs": 0, "ptpx_runs": 0,
            "icc2_runs": 0, "gpu_runs": 0,
        },
        "sealed_inputs": {
            "attempt": attempt_seal, "quarantine": quarantine_seal,
            "m2177": m2177_seal, "m1046_success_control": m1046_seal,
        },
        "m2178": {
            "attempt_status": attempt["status"], "execution_counts": counts,
            "vcs_compile_completed": True, "runtime_functional_pass": True,
            "runtime_ledger": runtime_parse["completion_ledger"],
            "runtime_duration_ns": runtime_parse["duration_ns"],
            "prehistory_saif": prehistory, "measurement_saif": measurement,
            "real_failure_point": prehistory_failure,
            "canonical_result_absent": True, "lock_absent": True,
            "identity_permanently_consumed": True,
        },
        "root_cause_evidence": {
            "m2178_ucli_gate_level_command_absent": True,
            "official_ucli_requires_gate_level_for_mda": True,
            "official_sv_mode_requires_sv_keyword": True,
            "sealed_m1046_same_debug_r_with_gate_level_nonempty": m1046_saif,
            "same_m2018_workload_debug_r_with_gate_level_records": m2139_saif,
            "dut_activity_absent_ruled_out": True,
            "scope_typo_or_resolution_failure_observed": False,
            "monitoring_policy_missing_high_confidence": True,
        },
        "official_help": {
            "saif_support_path": str(OFFICIAL_SAIF),
            "saif_support_sha256": sha256(OFFICIAL_SAIF),
            "sv_saif_path": str(OFFICIAL_SV),
            "sv_saif_sha256": sha256(OFFICIAL_SV),
        },
        "successor": {
            "source_identity": "M2185", "source_review_identity": "M2186",
            "future_run_identity": "M2187", "future_result_review_identity": "M2188",
            "minimum_new_ucli_first_command": "power -gate_level all mda sv",
            "position": "before power <dut_ordinary_scope> and before power -enable",
            "m2178_retry": False, "m2187_authorized_now": False,
        },
        "identity": {
            "docs359_sha256": sha256(DOC359), "ucli_sha256": sha256(UCLI),
            "attempt_manifest_sha256": sha256(ATTEMPT / "SHA256SUMS"),
            "quarantine_manifest_sha256": sha256(QUARANTINE / "SHA256SUMS"),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
