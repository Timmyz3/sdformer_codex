#!/opt/anaconda3/bin/python3
"""Independent, no-EDA M2161 source hammer for the M2160 campaign."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
PARSER = HW / "system_simulator/scripts/parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py"
TB = HW / "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2160_m2018_ordinary_native_saif_report_reset_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2160_m2018_ordinary_native_saif_report_reset_preflight_one_shot.py"
CONTRACT = HW / "contracts/m2160_m2018_ordinary_native_saif_report_reset_preflight_source_contract_r1_20260904.json"
AUTHOR = HW / "reviews/m2160_m2018_ordinary_native_saif_report_reset_preflight_source_author_receipt_r1_20260904"
M2152 = HW / "reviews/m2152_m2151_m2149_ordinary_single_axis_native_saif_preflight_failure_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parser():
    spec = importlib.util.spec_from_file_location("m2160_parser_for_m2161", PARSER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_parser()


def must_reject(fn) -> None:
    try:
        fn()
    except M.Failure:
        return
    raise AssertionError("declared fail-closed mutation was accepted")


def must_accept(fn) -> object:
    return fn()


def seal_file(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n")
    outer = Path(str(sidecar) + ".seal.sha256")
    outer.write_text(f"{sha256(sidecar)}  {sidecar.name}\n")


def verify_sealed_directory(root: Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    assert outer.read_text().split() == [sha256(manifest), "SHA256SUMS"]
    members: dict[str, str] = {}
    for row in manifest.read_text().splitlines():
        digest, rel = row.split(maxsplit=1)
        path = root / rel
        assert path.is_file() and not path.is_symlink() and sha256(path) == digest
        members[rel] = digest
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    assert sorted(members) == actual
    return members


def runtime_text() -> str:
    return "\n".join([
        "M2160_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=single_ordinary_dut",
        "M2160_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=192/192 row_live_one=149 cache_valid=4/4 cache_valid_one=0 slot_valid=8/8 slot_valid_one=0 bridge_overflow=16/16 bridge_overflow_one=0 rsp_shape_legal=8/8 rsp_shape_legal_one=8 total=228/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
        "M2160_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=1167.01 next_ucli_action=disable_report_prehistory_then_reset",
        "M2160_UCLI_PHASE order=2 action=first_run_returned census_and_begin_preceded_return=1",
        "M2160_UCLI_PHASE order=3 action=prehistory_power_disable timing=before_diagnostic_report",
        "M2160_UCLI_PHASE order=4 action=prehistory_power_report scope=single_ordinary_dut diagnostic_only=1",
        "M2160_UCLI_PHASE order=5 action=power_reset_requested timing=after_prehistory_report_before_measurement_enable",
        "M2160_UCLI_PHASE order=6 action=measurement_power_enable timing=after_reset_before_second_run",
        "M2160_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 rows=149 issues=1278 products=29472 commits=24 bundles=1788 scalar_weight_reads=14304 duration_ns=60876.00",
        "PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT ledger_exact=1 arithmetic_scoreboard_exact=1 internal_census_exact=1 enable_before_reset_preload=1 prehistory_report_requested=1 power_reset_requested=1 frontends=1 schedule_mode=0 second_axis=0 initreg_diagnostic_only=1 paper_citable=0",
        "M2160_UCLI_PHASE order=7 action=second_run_returned end_and_pass_preceded_return=1",
        "M2160_UCLI_PHASE order=8 action=measurement_power_disable timing=before_measurement_report",
        "M2160_UCLI_PHASE order=9 action=measurement_power_report scope=single_ordinary_dut admitted_candidate=1",
        "",
    ])


def saif_text(*, role: str, tx_first: int = 0,
              duration: float | None = None, records: int = 93971,
              mute_first_critical: bool = False,
              instance: str = "dut_ordinary", records_in_instance: bool = True) -> str:
    if duration is None:
        duration = 60876 if role == "measurement" else 1167.01
    names = list(M.CRITICAL) if role == "measurement" else ["load_valid"]
    rows: list[str] = []
    for index in range(records):
        name = names[index] if index < len(names) else f"filler_{index}"
        tx = tx_first if index == 0 else 0
        tc = 0 if mute_first_critical and index == 0 else 2
        rows.append(
            f"({name} (T0 {duration - 1 - tx:.2f}) (T1 1) (TX {tx}) (TC {tc}))")
    prefix = ["(SAIFILE", "(TIMESCALE 1 ns)", f"(DURATION {duration})",
              f"(INSTANCE {instance}"]
    if records_in_instance:
        return "\n".join([*prefix, *rows, ")", ")", ""])
    return "\n".join([*prefix, ")", *rows, ")", ""])


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    tb = TB.read_text()
    filelist = FILELIST.read_text()
    ucli = UCLI.read_text()
    runner = RUNNER.read_text()

    # Frozen identities and exhaustive source-author evidence.
    assert sha256(DOC359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    assert sha256(CONTRACT) == "411d7b5f53a91372d72a91c8319123bc62beca0bfff2d6f8c53d128c803747d0"
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    assert sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name]
    assert outer.read_text().split() == [sha256(sidecar), sidecar.name]
    author_members = verify_sealed_directory(AUTHOR)
    m2152_members = verify_sealed_directory(M2152)
    assert len(author_members) == 5 and len(m2152_members) == 4
    assert sha256(M2152 / "review.json") == contract["m2152_disposition"]["review_sha256"]
    for rel, digest in contract["source_inventory"].items():
        path = REPO / rel
        assert path.is_file() and not path.is_symlink() and sha256(path) == digest

    # Positive topology and actual UCLI run/$stop semantics.
    topology = M.audit_single_axis_source(tb, filelist)
    assert topology == contract["single_axis_topology"]
    assert tb.count("$stop;") == 2
    assert ucli.count("\nrun\n") == 2
    assert ucli.count("power -report") == 2
    assert ucli.count("power -enable") == 2
    assert ucli.index("power -disable") < ucli.index("power -report") < ucli.index("power -reset")
    assert ucli.index("power -reset") < ucli.rindex("power -enable") < ucli.rindex("\nrun\n")
    uncommented_tb = "\n".join(line for line in tb.splitlines()
                                 if not line.lstrip().startswith("//"))
    assert "force " not in uncommented_tb and "deposit " not in uncommented_tb
    assert "power_reset_requested=1" in tb
    assert "power_reset_accepted=1" not in tb and "power_reset_at_first_stop" not in tb
    assert runner.index("seal_file(path)") < runner.index('str(PARSER), "runtime"')
    assert 'str(prehistory_saif), "--role", "diagnostic_prehistory"' in runner
    assert 'str(measurement_saif), "--role", "measurement"' in runner
    assert 'str(VCS), "-full64"' in runner
    assert '[str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]' in runner
    assert 'run(["dc_shell"' not in runner
    assert 'run(["pt_shell"' not in runner
    assert 'run(["icc2_shell"' not in runner

    # Independently repeat every mutation declared by the author matrix.
    topology_mutations = {
        "parent_dual_axis_testbench": tb + "\ntb_m2051 injected_parent();\n",
        "second_dut_symbol": tb + "\nlogic dut_tsbg;\n",
        "second_load_valid_symbol": tb + "\nlogic load_valid_tsbg;\n",
        "schedule_mode_one": tb.replace(".SCHEDULE_MODE(0)", ".SCHEDULE_MODE(1)"),
        "second_completion_wait": tb + "\ninitial wait (tsbg_done_cycle >= 0);\n",
        "second_hierarchical_path": tb + "\ninitial if (tsbg.busy) $fatal;\n",
    }
    for mutated in topology_mutations.values():
        must_reject(lambda mutated=mutated: M.audit_single_axis_source(mutated, filelist))
    must_reject(lambda: M.audit_single_axis_source(
        tb, filelist + "hw_autoresearch_nts07/tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv\n"))
    must_reject(lambda: M.audit_single_axis_source(
        tb, filelist + "hw_autoresearch_nts07/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv\n"))

    with tempfile.TemporaryDirectory(prefix="m2161_hammer.") as raw:
        root = Path(raw)
        runtime = root / "rtl_sim.log"
        prehistory = root / "rtl_prehistory.saif"
        measurement = root / "rtl_measurement.saif"
        runtime.write_text(runtime_text())
        must_accept(lambda: M.parse_runtime(runtime))

        runtime_mutations = {
            "power_reset_request_marker_corrupted": runtime_text().replace(
                "order=5 action=power_reset_requested",
                "order=5 action=power_reset_requested_BAD"),
            "228_element_census_incomplete": runtime_text().replace(
                "row_live=192/192", "row_live=191/192"),
            "product_ledger_drift": runtime_text().replace(
                "products=29472", "products=29471"),
            "second_axis_marker_flipped": runtime_text().replace(
                "second_axis=0", "second_axis=1"),
            "SAIF_REPORT_BEFORE_RESET_warning_injected": runtime_text()
                + "Warning-[SAIF_REPORT_BEFORE_RESET] Toggle reporting not done\n",
            "ignored_reset_notice_injected": runtime_text()
                + "This request to reset power information will be ignored.\n",
        }
        for text in runtime_mutations.values():
            runtime.write_text(text)
            must_reject(lambda: M.parse_runtime(runtime))

        prehistory.write_text(saif_text(role="diagnostic_prehistory"))
        measurement.write_text(saif_text(role="measurement"))
        seal_file(prehistory)
        seal_file(measurement)
        assert M.parse_saif(prehistory, role="diagnostic_prehistory")["record_count"] == 93971
        assert M.parse_saif(measurement, role="measurement")["record_count"] == 93971

        saif_mutations = [
            ("measurement_TX_nonzero", measurement, "measurement",
             saif_text(role="measurement", tx_first=1)),
            ("measurement_duration_drift", measurement, "measurement",
             saif_text(role="measurement", duration=60877)),
            ("measurement_record_count_drift", measurement, "measurement",
             saif_text(role="measurement", records=93970)),
            ("critical_cone_activity_removed", measurement, "measurement",
             saif_text(role="measurement", mute_first_critical=True)),
            ("diagnostic_prehistory_duration_drift", prehistory,
             "diagnostic_prehistory",
             saif_text(role="diagnostic_prehistory", duration=1168.01)),
        ]
        for _, path, role, text in saif_mutations:
            path.write_text(text)
            seal_file(path)
            must_reject(lambda path=path, role=role: M.parse_saif(path, role=role))
        measurement.write_text(saif_text(role="measurement"))
        seal_file(measurement)
        Path(str(measurement) + ".sha256").write_text(
            "0" * 64 + f"  {measurement.name}\n")
        must_reject(lambda: M.parse_saif(measurement, role="measurement"))

        # New adversarial attacks.  These are required to reject, but M2160
        # accepts them.  Detecting both admissions is the review result.
        warning_bypasses = [
            "Warning: This reset request was ignored.",
            "Warning: Power information reset request ignored.",
            "Warning-[POWER_RESET_IGNORED] Switching counters were not cleared.",
            "Warning: request to reset switching activity has been ignored.",
        ]
        accepted_warning_bypasses: list[str] = []
        for index, warning in enumerate(warning_bypasses):
            candidate = root / f"reset_warning_bypass_{index}.log"
            candidate.write_text(runtime_text() + warning + "\n")
            try:
                M.parse_runtime(candidate)
            except M.Failure:
                pass
            else:
                accepted_warning_bypasses.append(warning)
        assert accepted_warning_bypasses == warning_bypasses

        accepted_scope_bypasses: list[str] = []
        for name, instance in (("records_outside_empty_target", "dut_ordinary"),
                               ("records_outside_wrong_instance", "impostor")):
            candidate = root / f"{name}.saif"
            candidate.write_text(saif_text(
                role="measurement", instance=instance, records_in_instance=False))
            seal_file(candidate)
            try:
                parsed = M.parse_saif(candidate, role="measurement")
            except M.Failure:
                pass
            else:
                assert parsed["record_count"] == 93971
                accepted_scope_bypasses.append(name)
        assert accepted_scope_bypasses == [
            "records_outside_empty_target", "records_outside_wrong_instance"]

    result = {
        "status": "PASS_M2161_HAMMER_DETECTED_M2160_BLOCKERS__M2162_NOT_AUTHORIZED",
        "no_eda": True,
        "author_mutation_matrix": {
            "topology_rejected": 8,
            "runtime_rejected": 6,
            "saif_rejected": 6,
        },
        "positive_contracts": {
            "single_frontend_schedule_mode_zero": True,
            "true_ucli_run_stop_order": True,
            "prehistory_report_before_reset": True,
            "both_files_sealed_before_parse": True,
            "slot42_ledgers_and_scoreboard_present": True,
            "one_shot_no_retry": True,
        },
        "blockers": {
            "reset_warning_synonyms_accepted": 4,
            "out_of_instance_saif_forgeries_accepted": 2,
        },
        "authorization": {
            "m2162": False,
            "license_queries": 0,
            "vcs_compiles": 0,
            "simv_runs": 0,
            "raw_saif_files_written": 0,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
