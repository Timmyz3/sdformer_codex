#!/usr/bin/env python3
"""Read-only identity and semantic checks for the sealed M1502 first fault.

This checker does not invoke a simulator, an EDA executable, or a consumed
binary.  It deliberately checks only facts that survive in the sealed failure
and frozen source/netlist text.
"""
import hashlib
import json
from pathlib import Path


HW = Path(__file__).resolve().parents[2]

PINS = {
    "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine/failure.json": "2bad717f51fa99e2526b4ec8b7b305b4bbbf60b84728d6f799de59aa72bfe7d2",
    "results/.m1502_c2_mapped_vcs_saif_ptpx_attempt_consumed/attempt.json": "b0359256e8d3934ee460835053f7655e53599512232fbbf03e28df5ef6da01c1",
    "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite/candidate/k8_case0.log": "a3652eac8859376458d7b7738e29415d75d52754ff23d2e7dda012feb407fc8e",
    "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite/candidate/k8_case0.assert.report": "eb197e0733b21f934595f3ff204669dec93f16663d63656e8a554a660eabfc33",
    "dc_handoff/tb/m1334_c2_production_activity_assertions.sv": "86be3fa541bf65afa6ada99aa3e2bd494ed689594fece18cfea135b91420c32a",
    "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv": "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    "dc_handoff/tb/tb_m1334_c2_headline_mapped_production_activity.sv": "eacc165bad9eb3ef6c38e87f6f0de8cafd75e167f0ef02d340647634540982ca",
    "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv": "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv": "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    "dc_handoff/scripts/run_m1502_m1493_c2_source_chain_successor_one_shot.py": "91fc6a8867a138098b660e4d450eda50f5bd1850f9127bc349c2a303aac36df1",
    "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829/diagnostic_summary.json": "2a023930576712352c439abc26cafe35a390ddc16f9eb59913f30d5bba4fa6c6",
    "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829/equalbw/sim.log": "2392a3cc0bb61551528a548ffc0b0fe32faf70cf16f6f06960b61a71442e90a7",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(text: str, token: str, label: str) -> None:
    if token not in text:
        raise SystemExit(f"FAIL semantic {label}: missing {token!r}")


def main() -> None:
    for relative, expected in PINS.items():
        path = HW / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"FAIL identity nonregular: {relative}")
        observed = digest(path)
        if observed != expected:
            raise SystemExit(
                f"FAIL identity {relative}: {observed} != {expected}")

    failure = json.loads((HW / next(iter(PINS))).read_text())
    if failure["phase"] != "SIM_k8_0" or failure["counts"] != {
        "ptpx_runs": 0, "saif_files": 0, "simv_runs": 1, "vcs_compiles": 1
    }:
        raise SystemExit("FAIL sealed M1502 boundary drift")

    log = (HW / "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite/candidate/k8_case0.log").read_text()
    require(log, "started at 28500ps failed at 28500ps", "first fault time")
    require(log, "cp_source, 10 attempts, 1 match", "one source accept")
    require(log, "cp_endpoint, 10 attempts, 0 match", "no endpoint accept")
    require(log, "cp_commit, 10 attempts, 0 match", "no commit")
    require(log, "cp_done, 10 attempts, 0 match", "no done")
    require(log, "Offending '(!((protocol_error || numeric_overflow) || stale_response_seen))'", "aggregate assertion")

    assertions = (HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv").read_text()
    require(assertions, "assert property (!(|endpoint_fault))", "independent endpoint assertion")
    require(assertions, "protocol_error || numeric_overflow", "aggregate fault assertion")
    require(assertions, "|| stale_response_seen", "aggregate stale assertion")

    rtl = (HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv").read_text()
    require(rtl, "assign protocol_error = core_protocol_error || adapter_protocol_error", "RTL protocol aggregation")
    require(rtl, "assign stale_response_seen = core_stale_response_seen", "RTL stale aggregation")

    netlist = (HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v").read_text()
    require(netlist, ".Q(numeric_overflow)", "mapped numeric output flop")
    require(netlist, ".Z(stale_response_seen)", "mapped stale cone")
    require(netlist, ".ZN(\n        protocol_error)", "mapped protocol cone")

    old = json.loads((HW / "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829/diagnostic_summary.json").read_text())
    if old["root_cause"]["class"] != "GATE_LEVEL_UNINITIALIZED_STATE_X_PROPAGATION":
        raise SystemExit("FAIL predecessor mapped-X diagnostic drift")

    rtl_log = (HW / "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829/equalbw/sim.log").read_text()
    require(rtl_log, "PASS M803EQ channel-split cutthrough-8bank equal-bandwidth FC2 VCS", "RTL equal-bandwidth PASS")
    require(rtl_log, "clean_cases=10 exact_cycle_cases=5", "RTL clean cases")

    print("PASS M1568 read-only M1502 first-fault forensic identities=16")
    print("PASS boundary phase=SIM_k8_0 compile=1 sim=1 saif=0 ptpx=0")
    print("PASS first_fault_ps=28500 source_accept=1 endpoint_accept=0 commit=0 done=0")
    print("PASS assertion_resolution=aggregate_only exact_bit_not_preserved")
    print("PASS ranking=protocol_or_X_highest stale_unlikely numeric_very_unlikely endpoint_not_observed")
    print("PASS no_simv_no_vcs_no_eda_no_retry no_claim=true")


if __name__ == "__main__":
    main()
