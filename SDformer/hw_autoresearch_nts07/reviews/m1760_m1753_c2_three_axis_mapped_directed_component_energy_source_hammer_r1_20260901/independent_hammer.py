#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent CPU-only hammer for the M1753 C2 energy source.

This review does not launch VCS, simv, PrimeTime, or any other EDA tool.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
CASE_TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
K1_FILELIST = HW / "dc_handoff/filelists/iscas_m1753_c2_m1609_k1_mapped_directed_energy.f"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


SPEC = importlib.util.spec_from_file_location("m1753_independent", CHECKER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

checks: list[str] = []
mutations: list[str] = []


def check(name: str, condition: bool) -> None:
    if not condition:
        raise RuntimeError("independent check failed: " + name)
    checks.append(name)


def reject(name: str, action) -> None:
    try:
        action()
    except (RuntimeError, ValueError, json.JSONDecodeError):
        mutations.append(name)
        return
    raise RuntimeError("negative mutation accepted: " + name)


def runtime_log(axis: str, case_id: int, *, cycle_delta: int = 0,
                display_override: str | None = None,
                events_delta: int = 0, duplicate: bool = False) -> str:
    display = display_override or {"k1": "K1", "k8": "K8", "k1x8": "K1x8"}[axis]
    events = M.EVENTS[case_id] + events_delta
    endpoint = 0 if case_id == 4 else 1
    m1684 = (f"PASS M1684 M1609 binary-clean production case={case_id} "
             f"accepted_sources={events} source_packets={M.PACKETS[case_id]} "
             f"endpoint_accepts={endpoint} result_accepts=1 done_accepts=1 "
             "fault_binary_clean=1 registered_fault_public_zero=1")
    m979 = (f"PASS M979 mapped replay axis={display} case={case_id} "
            f"events={M.EVENTS[case_id]} "
            f"cycles={M.AXES[axis]['cycles'][case_id] + cycle_delta}")
    text = m1684 + "\n" + m979 + "\n" + f"PASS M1334 coverage case={case_id}\n"
    if duplicate:
        text += m979 + "\n"
    return text


def rows() -> list[dict]:
    answer = []
    power = {"k1": 3.0, "k8": 2.0, "k1x8": 8.0}
    for axis in M.AXES:
        for case_id, cycles in enumerate(M.AXES[axis]["cycles"]):
            total = power[axis] + case_id * 0.1
            answer.append({
                "axis": axis, "case": case_id, "cycles": cycles,
                "accepted_sources": M.EVENTS[case_id],
                "net_switching_mw": total * 0.3,
                "cell_internal_mw": total * 0.6,
                "cell_leakage_mw": total * 0.1,
                "total_mw": total,
            })
    return answer


contract = M.strict_json(CONTRACT)
check("runner identity", sha(RUNNER) == "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e")
check("checker identity", sha(CHECKER) == "b9bb417be8786b69a3d476d75b2a49c0a99b46518ed76b0bad9a572937160312")
check("contract identity", sha(CONTRACT) == "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21")
check("docs359 frozen", sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
check("source-only status", contract["status"] == "SOURCE_ONLY__M1760_REVIEW_AND_M1761_RELEASE_REQUIRED__NO_EDA")
check("no release", not M1761.exists() and not Path(str(M1761) + ".sha256").exists())
check("claim boundary all false", contract["claim_boundary"] == M.CLAIMS)
check("directed class", contract["workload_boundary"]["class"] == "DIRECTED_COMPONENT_NOT_PRODUCTION")
check("five cases", contract["workload_boundary"]["cases_per_axis"] == 5)
check("261 accepted sources", contract["workload_boundary"]["accepted_sources_per_axis"] == 261)
check("3 ns clock", contract["workload_boundary"]["clock_period_ns"] == 3.0)
check("three exact axes", tuple(contract["mapped_axes"]) == ("k1", "k8", "k1x8"))
check("whole mapped component", contract["measurement_contract"]["ptpx_report_power_scope"] == "WHOLE_MAPPED_COMPONENT")
check("public-port mapped activity", contract["measurement_contract"]["mapped_public_port_activity"] is True)
check("fifteen SAIF and PTPX coordinates", contract["future_execution"]["saif_files"] == 15 and contract["future_execution"]["ptpx_runs"] == 15)
check("no partial axis citation", contract["future_execution"]["partial_axis_citable"] is False)
check("no automatic retry", contract["future_execution"]["automatic_retry"] is False)
check("joint disclosure", contract["mandatory_joint_disclosure"]["same_table_and_sentence"] is True)
check("forbid K8-vs-K1 headline", contract["mandatory_joint_disclosure"]["k8_vs_single_k1_headline_forbidden"] is True)
check("external exclusions exact", contract["measurement_contract"]["external_exclusions"] == ["weight_sram", "testbench_memory_model", "io_phy", "clock_tree", "postlayout_parasitics"])

runner_text = RUNNER.read_text()
check("three axis loop", 'AXES = ("k1", "k8", "k1x8")' in runner_text and "for axis in AXES:" in runner_text)
check("all SAIF before PTPX", runner_text.index("all fifteen SAIF coordinates required before PTPX") < runner_text.index('state["phase"] = "PTPX_"'))
check("fresh per-axis build", 'axis_dir = WORK / "build" / axis' in runner_text and '"-Mdir=csrc"' in runner_text)
check("DUT-only UCLI scope", contract["measurement_contract"]["dut_only_saif_scope"] ==
      "tb_m1684_c2_m1609_fresh_mapped_production_energy.core.dut")
check("whole report power", '"whole_component_report_power": True' in runner_text)
check("joint receipt fields", '"must_be_same_table_and_sentence": True' in runner_text)

k1_lines = [line.strip() for line in K1_FILELIST.read_text().splitlines() if line.strip()]
check("K1 define", k1_lines[0] == "+define+M979_AXIS_K1")
check("K1 mapped netlist", any(line.endswith(contract["mapped_axes"]["k1"]["netlist"]["path"])
                               for line in k1_lines))
check("same wrapper", k1_lines[-1].endswith("tb_m1684_c2_m1609_fresh_mapped_production_energy.sv"))

with tempfile.TemporaryDirectory() as temp_name:
    temp = Path(temp_name)
    for axis in M.AXES:
        for case_id in range(5):
            path = temp / f"{axis}_{case_id}.log"
            path.write_text(runtime_log(axis, case_id))
            value = M.validate_runtime_log(path, axis, case_id)
            check(f"runtime exact {axis}/{case_id}", value["accepted_sources"] == M.EVENTS[case_id])
    good = temp / "good_k1.log"
    good.write_text(runtime_log("k1", 0))
    for name, kwargs in (
        ("k1 cycle substitution", {"cycle_delta": 1}),
        ("k1 axis substitution", {"display_override": "K8"}),
        ("k1 denominator substitution", {"events_delta": 1}),
        ("k1 duplicate pass", {"duplicate": True}),
    ):
        mutated = temp / (name.replace(" ", "_") + ".log")
        mutated.write_text(runtime_log("k1", 0, **kwargs))
        reject(name, lambda p=mutated: M.validate_runtime_log(p, "k1", 0))
    missing = temp / "missing_m1684.log"
    missing.write_text("PASS M979 mapped replay axis=K1 case=0 events=20 cycles=259\nPASS M1334 coverage case=0\n")
    reject("missing binary-clean pass", lambda: M.validate_runtime_log(missing, "k1", 0))

metric = M.aggregate_metrics(rows())
check("K8 cycles 1913", metric["axes"]["k8"]["cycles"] == 1913)
check("K1x8 cycles 1945", metric["axes"]["k1x8"]["cycles"] == 1945)
check("cycle speedup", math.isclose(metric["equal_bandwidth_cycle_speedup_k8_vs_k1x8"], 1.0167276529012024, rel_tol=0.0, abs_tol=1e-15))
check("throughput/mm2", math.isclose(metric["equal_bandwidth_throughput_per_mm2_k8_vs_k1x8"], 4.562720096484654, rel_tol=0.0, abs_tol=1e-15))

bad = rows(); bad.pop()
reject("missing Cartesian coordinate", lambda: M.aggregate_metrics(bad))
bad = rows(); bad[0]["axis"] = "k8"
reject("duplicate Cartesian coordinate", lambda: M.aggregate_metrics(bad))
bad = rows(); bad[0]["cycles"] += 1
reject("cycle anchor", lambda: M.aggregate_metrics(bad))
bad = rows(); bad[0]["accepted_sources"] += 1
reject("accepted source anchor", lambda: M.aggregate_metrics(bad))
bad = rows(); bad[0]["total_mw"] += 1.0
reject("power decomposition", lambda: M.aggregate_metrics(bad))

with tempfile.TemporaryDirectory() as temp_name:
    path = Path(temp_name) / "power.rpt"
    good_power = ("Report : Averaged Power\n-unit mW\n"
                  "Net Switching Power = 1.00000000\n"
                  "Cell Internal Power = 2.00000000\n"
                  "Cell Leakage Power = 0.10000000\n"
                  "Total Power = 3.10000000\n")
    path.write_text(good_power)
    check("four-field power parse", M.parse_power_report(path)["total_mw"] == 3.1)
    path.write_text(good_power.replace("Total Power = 3.10000000\n", ""))
    reject("missing total power", lambda: M.parse_power_report(path))
    path.write_text(good_power + "Total Power = 3.10000000\n")
    reject("duplicate total power", lambda: M.parse_power_report(path))
    path.write_text(good_power.replace("Total Power = 3.10000000", "Total Power = 4.10000000"))
    reject("power subtotal", lambda: M.parse_power_report(path))

with tempfile.TemporaryDirectory() as temp_name:
    duplicate = Path(temp_name) / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n')
    reject("duplicate JSON key", lambda: M.strict_json(duplicate))
    nonfinite = Path(temp_name) / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n')
    reject("nonfinite JSON", lambda: M.strict_json(nonfinite))

tb_text = CASE_TB.read_text()
defense_depth = {
    "k1_tb_internal_cycle_assertion_present": bool(re.search(r"if\s*\(axis\s*==\s*0\)", tb_text)),
    "k1_post_sim_unique_axis_case_event_cycle_gate_present": True,
    "blocking": False,
    "reason": "AXIS_ID=0 returns -1 inside the reused TB, but the immediately following exact-SHA M1753 checker requires one unique K1 PASS line with the frozen cycle and no result is publishable if that checker fails.",
}
check("K1 missing internal anchor observed", defense_depth["k1_tb_internal_cycle_assertion_present"] is False)

output = {
    "schema": "m1760_m1753_c2_source_independent_hammer_r1_v1",
    "status": "PASS_SOURCE_ONLY_NO_EDA",
    "score": 98,
    "checks": len(checks),
    "negative_mutations_rejected": len(mutations),
    "mutation_names": mutations,
    "defense_depth_advisory": defense_depth,
    "source_review_execution": {
        "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
        "ptpx_runs": 0, "eda_runs": 0, "network_calls": 0,
    },
}
print(json.dumps(output, sort_keys=True, indent=2))
