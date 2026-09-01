#!/usr/bin/env python3
"""Different-author no-EDA hammer for the M1684 production-energy source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
CHECKER = HW / "system_simulator/scripts/check_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1684_m1661_c2_m1609_fresh_mapped_production_energy_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
CONTRACT = HW / "contracts/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1684_m1661_c2_m1609_fresh_mapped_production_energy_source_author_receipt_r1_20260901"

EXPECTED = {
    CHECKER: "f9f5ccf9623bc1a036e2a76a3d1db6c9d90075d385326c5871eab91d48e438a5",
    RUNNER: "1c7acc502c010809d56dacd78d857dfb5a44cca74e12025134424c6b9c80b77f",
    TEST: "d23a44875563cfb3b86762752f011bd01b25cafb5baf643e7983c7bea0ac53f6",
    CONTRACT: "7fa827aca2ee236a06010d037ca03dac80fc1491abc59a3162c0092bc84e1683",
    AUTHOR / "author_receipt.json": "2f1bb4c7ce1a1355c488b75a059e6efe34a305ab01012365a32d0c7370b1724b",
    AUTHOR / "SHA256SUMS": "5142d91321007cffdcd9f35ab3707f7901a80a91d2d11347224c224ed86b6d30",
    AUTHOR / "SHA256SUMS.seal.sha256": "dba513a1e26cb9fd18e9a2f8532b2f5f07893c6d3f174f5f2acc523a83afab2a",
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


def load_module():
    spec = importlib.util.spec_from_file_location("m1684_checker_hammer", str(CHECKER))
    need(spec is not None and spec.loader is not None, "checker import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(action):
    try:
        action()
    except (RuntimeError, ValueError, ZeroDivisionError):
        return True
    return False


def runtime_log(axis, case_id, accepted=None, endpoint=None, extra=""):
    events = [20, 41, 90, 110, 0][case_id]
    packets = [1, 2, 4, 8, 1][case_id]
    cycles = {"k8": [51, 131, 486, 1231, 14],
              "k1x8": [53, 133, 499, 1246, 14]}[axis][case_id]
    display = "K8" if axis == "k8" else "K1x8"
    accepted = events if accepted is None else accepted
    endpoint = (17 if case_id < 4 else 0) if endpoint is None else endpoint
    return (
        "PASS M1334 coverage case={0} source=1 endpoint={1} commit=6 stall=3 done=1 unknown=0 fatal=0\n"
        "PASS M1684 M1609 binary-clean production case={0} accepted_sources={2} "
        "source_packets={3} endpoint_accepts={1} result_accepts=6 done_accepts=1 "
        "fault_binary_clean=1 registered_fault_public_zero=1\n"
        "PASS M979 mapped replay axis={4} case={0} events={5} cycles={6} "
        "saif_duration_ns={7} numeric_mismatches=0 tuple_mismatches=0 "
        "weight_mismatches=0 accepted_unknowns=0 protocol_errors=0\n{8}"
    ).format(case_id, endpoint, accepted, packets, display, events, cycles,
             cycles * 3, extra)


def saif_text(cycles, endpoint_tc=5, tx=0, fault_tc=0, reset_tc=0):
    signals = {
        "clk_core": 10, "raw_valid": 2, "raw_accept": 2,
        "raw_bitmap[0]": 2, "mem_req_valid": endpoint_tc,
        "mem_req_accept": endpoint_tc, "mem_rsp_valid": endpoint_tc,
        "mem_rsp_accept": endpoint_tc, "result_valid": 2,
        "result_accept": 2, "result_accumulator[0]": 2,
        "token_done_valid": 2, "token_done_accept": 2,
        "protocol_error": fault_tc, "numeric_overflow": 0,
        "stale_response_seen": 0, "rst_core": reset_tc,
    }
    body = "\n".join("          ({0} (T0 1) (T1 1) (TX {1}) (TC {2}))".format(
        name, tx if name == "raw_valid" else 0, tc) for name, tc in signals.items())
    return """(SAIFILE
  (DURATION {duration})
  (INSTANCE tb_m1684_c2_m1609_fresh_mapped_production_energy
    (INSTANCE core
      (INSTANCE dut
        (NET
{body}
        )
      )
    )
  )
)
""".format(duration=cycles * 3, body=body)


def power_text(switch="1.0", internal="2.0", leakage="0.1", total="3.1", extra=""):
    return """Report : Averaged Power
    -unit mW
Net Switching Power = {0}
Cell Internal Power = {1}
Cell Leakage Power = {2}
Total Power = {3}
{4}""".format(switch, internal, leakage, total, extra)


def static_runner_checks(text):
    need('for axis in ("k8", "k1x8"):' in text, "axis loop")
    need(text.count('for axis in ("k8", "k1x8"):') >= 3, "axis geometry")
    need(text.count("for case_id in range(5):") >= 2, "case geometry")
    need(text.index("all ten mapped production SAIF gates")
         < text.index('state["phase"] = "PTPX_"'), "SAIF-first order")
    for token in ('"vcs_compiles": 2', '"simv_runs": 10',
                  '"saif_files": 10', '"ptpx_runs": 10',
                  '"k8": [51, 131, 486, 1231, 14]',
                  '"k1x8": [53, 133, 499, 1246, 14]',
                  "EVENTS = [20, 41, 90, 110, 0]",
                  '"automatic_retry": False'):
        need(token in text, "runner token: " + token)
    need("initreg" not in text.lower(), "initreg")
    active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    need(re.search(r"(?m)^\s*force(?:\s|$)", active, re.I) is None, "active force")


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "identity drift: " + str(path))
    module = load_module()
    # The review directory is necessarily present when this sealed hammer is
    # rerun. Redirect only the checker's future-review-absence sentinel; all
    # source, predecessor, release and result gates remain live.
    old_review = module.M1685
    with tempfile.TemporaryDirectory() as directory:
        module.M1685 = Path(directory) / "future_review_sentinel"
        source = module.validate_sources()
    module.M1685 = old_review
    need(source["status"] == "PASS_M1684_SOURCE_ONLY_NO_EDA", "source check")

    contract = module.strict_json(CONTRACT)
    need(contract["fair_campaign"]["accepted_sources_per_axis"] == 261,
         "261-source contract")
    need(contract["future_execution"] == {
        "authorized_now": False, "attempts": 1, "vcs_compiles": 2,
        "simv_runs": 10, "production_saif_files": 10, "ptpx_runs": 10,
        "all_ten_binary_clean_functional_and_saif_gates_before_first_ptpx": True,
        "axis_execution": "sequential", "automatic_retry": False,
        "fresh_result_namespace": "results/m1684_c2_mapped_production_energy_r1_20260901",
        "fresh_attempt_namespace": "results/.m1684_c2_mapped_production_energy_attempt_consumed"},
         "execution contract")
    runner = RUNNER.read_text()
    static_runner_checks(runner)
    need("m872" not in "\n".join(module.active_lines(module.FILELISTS["k8"])).lower(),
         "old M872 K8")
    need("m872" not in "\n".join(module.active_lines(module.FILELISTS["k1x8"])).lower(),
         "old M872 K1x8")

    mutations = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bad = root / "bad.f"
        filelist = module.FILELISTS["k8"].read_text()
        bad.write_text(filelist.replace("m1661_m1652", "m872_m803", 1))
        mutations["old_m872_filelist"] = rejected(lambda: module.validate_filelist("k8", bad))
        bad.write_text(filelist.replace("M979_AXIS_K8", "M979_AXIS_K1X8", 1))
        mutations["axis_define_swap"] = rejected(lambda: module.validate_filelist("k8", bad))

        log = root / "run.log"
        log.write_text(runtime_log("k8", 0, accepted=19))
        mutations["accepted_source_denominator"] = rejected(
            lambda: module.validate_runtime_log(log, "k8", 0))
        log.write_text(runtime_log("k8", 0, extra="fault asserted\n"))
        mutations["runtime_fault_token"] = rejected(
            lambda: module.validate_runtime_log(log, "k8", 0))
        log.write_text(runtime_log("k8", 0) + runtime_log("k8", 0))
        mutations["duplicate_runtime_pass"] = rejected(
            lambda: module.validate_runtime_log(log, "k8", 0))
        log.write_text(runtime_log("k8", 4, endpoint=1))
        mutations["zero_case_runtime_endpoint"] = rejected(
            lambda: module.validate_runtime_log(log, "k8", 4))

        saif = root / "run.saif"
        saif.write_text(saif_text(51))
        module.validate_saif(saif, "k8", 0, 51)
        saif.write_text(saif_text(50))
        mutations["saif_duration"] = rejected(
            lambda: module.validate_saif(saif, "k8", 0, 51))
        saif.write_text(saif_text(51, tx=1))
        mutations["saif_unknown_tx"] = rejected(
            lambda: module.validate_saif(saif, "k8", 0, 51))
        saif.write_text(saif_text(51, fault_tc=1))
        mutations["saif_fault_toggle"] = rejected(
            lambda: module.validate_saif(saif, "k8", 0, 51))
        saif.write_text(saif_text(51, reset_tc=1))
        mutations["saif_reset_in_window"] = rejected(
            lambda: module.validate_saif(saif, "k8", 0, 51))
        saif.write_text(saif_text(14, endpoint_tc=1))
        mutations["zero_case_saif_endpoint"] = rejected(
            lambda: module.validate_saif(saif, "k8", 4, 14))
        saif.write_text(saif_text(51, endpoint_tc=0))
        mutations["active_case_empty_endpoint"] = rejected(
            lambda: module.validate_saif(saif, "k8", 0, 51))

        power = root / "power.rpt"
        power.write_text(power_text(total="3.2"))
        mutations["power_subtotal"] = rejected(lambda: module.parse_power_report(power))
        power.write_text(power_text(switch="-1.0", total="1.1"))
        mutations["negative_power"] = rejected(lambda: module.parse_power_report(power))
        power.write_text(power_text(total="0.0", switch="0", internal="0", leakage="0"))
        mutations["zero_total_power"] = rejected(lambda: module.parse_power_report(power))
        power.write_text(power_text(extra="Total Power = 3.1\n"))
        mutations["duplicate_power_field"] = rejected(lambda: module.parse_power_report(power))
        power.write_text(power_text(total="NaN"))
        mutations["nonfinite_power"] = rejected(lambda: module.parse_power_report(power))

        rows = []
        for axis, total in (("k8", 2.0), ("k1x8", 4.0)):
            for case_id in range(5):
                rows.append({"axis": axis, "case": case_id,
                             "cycles": module.AXES[axis]["cycles"][case_id],
                             "accepted_sources": module.EVENTS[case_id],
                             "total_mw": total})
        metrics = module.aggregate_metrics(rows)
        need(metrics["axes"]["k8"]["accepted_sources"] == 261, "aggregate source")
        need(abs(metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"]
                 - 1945.0 / 1913.0) < 1e-12, "aggregate cycle")
        mutations["missing_metric_coordinate"] = rejected(
            lambda: module.aggregate_metrics(rows[:-1]))
        changed = [dict(row) for row in rows]
        changed[0]["cycles"] += 1
        mutations["metric_cycle_anchor"] = rejected(
            lambda: module.aggregate_metrics(changed))
        changed = [dict(row) for row in rows]
        changed[0]["accepted_sources"] += 1
        mutations["metric_source_anchor"] = rejected(
            lambda: module.aggregate_metrics(changed))
        changed = [dict(row) for row in rows]
        changed[-1]["axis"], changed[-1]["case"] = "k8", 0
        mutations["duplicate_metric_coordinate"] = rejected(
            lambda: module.aggregate_metrics(changed))

        js = root / "bad.json"
        js.write_text('{"x":1,"x":2}')
        mutations["duplicate_json_key"] = rejected(lambda: module.strict_json(js))
        js.write_text('{"x":NaN}')
        mutations["nonfinite_json"] = rejected(lambda: module.strict_json(js))

    mutated = runner.replace('"ptpx_runs": 10', '"ptpx_runs": 9')
    mutations["execution_count"] = rejected(lambda: static_runner_checks(mutated))
    mutated = runner.replace("# all ten mapped production SAIF gates must close before any PTPX call.",
                             "force dut.state = 1;", 1)
    mutations["active_force"] = rejected(lambda: static_runner_checks(mutated))
    mutated = runner.replace("from __future__ import annotations",
                             "from __future__ import annotations\n# initreg bypass", 1)
    mutations["initreg"] = rejected(lambda: static_runner_checks(mutated))
    need(all(mutations.values()), "mutation escaped: " + repr(
        sorted(name for name, value in mutations.items() if not value)))

    collision_calls = [match.start() for match in re.finditer(r"collision_gate\(\)", runner)]
    first_compile = runner.index('state["phase"] = "COMPILE_"')
    first_ptpx = runner.index('state["phase"] = "PTPX_"')
    queue = {
        "lock": "/tmp/m1684_c2_mapped_production_energy.lock",
        "task_private_lock_only": True,
        "collision_gate_call_count_including_definition": len(collision_calls),
        "collision_gate_after_first_compile": any(pos > first_compile for pos in collision_calls),
        "collision_gate_after_first_ptpx": any(pos > first_ptpx for pos in collision_calls),
        "shared_cross_campaign_eda_lock": False,
        "one_eda_queue_proven": False,
    }
    need(queue["collision_gate_after_first_compile"] is False, "queue reproduction drift")

    result = {
        "schema": "m1685_m1684_c2_production_energy_source_independent_hammer_r1_v1",
        "status": "FAIL_M1685_M1684_C2_PRODUCTION_ENERGY_SOURCE__QUEUE_REPAIR_REQUIRED",
        "score_over_100": 94,
        "p0_count": 0, "p1_count": 1, "p2_count": 1,
        "verified": {
            "fresh_m1661_k8_k1x8_mapped_v_sdc_bound": True,
            "old_m872_excluded": True,
            "m1609_m1627_registered_fault_semantics_bound": True,
            "same_five_cases_3ns_261_sources": True,
            "future_geometry_2_compile_10_sim_10_saif_10_ptpx": True,
            "all_saif_before_ptpx": True,
            "direct_dut_saif_and_binary_x_clean": True,
            "zero_event_case_integrated": True,
            "energy_math_mw_pj_cycle_pj_source_ratio_throughput_w": True,
            "attempt_seal_no_retry": True,
            "no_initreg_or_active_force_in_exact_sources": True,
            "claim_boundary_all_false": True,
            "eda_executed": False, "release_created": False,
        },
        "queue_reproduction": queue,
        "mutation_count": len(mutations),
        "mutations_rejected": sorted(mutations),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
