#!/usr/bin/env python3
"""M979 static/source and future per-case mapped-gate SAIF validator.

The default mode is read-only and executes no VCS/PT/PTPX or other EDA tool.
The ``--saif`` mode validates an already-created SAIF without modifying it.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
UCLI = HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m993_m979_c2_mapped_gate_saif_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m979_c2_mapped_gate_saif_source.py"
CONTRACT = HW / "contracts/m979_m974_c2_three_axis_mapped_gate_saif_source_contract_r1_20260829.json"
BASE = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
NET = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
SDC = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.sdc"
AXES = {
    "k1": {"define": "M979_AXIS_K1", "module": "ARCH_MODE0",
           "net_sha": "060e7cd00e5a0f79860430c823439424ae88211cd2ff0d71bc787c9e6691d6b3",
           "sdc_sha": "df2b08e2c8a8faa87f7ab8f738888589f7b7595b386b905388b9428204c5a9bd",
           "cycles": None},
    "k8": {"define": "M979_AXIS_K8", "module": "ARCH_MODE1",
           "net_sha": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
           "sdc_sha": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
           "cycles": [51, 131, 486, 1231, 14]},
    "k1x8": {"define": "M979_AXIS_K1X8", "module": "ARCH_MODE2",
             "net_sha": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
             "sdc_sha": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
             "cycles": [53, 133, 499, 1246, 14]},
}
M974 = HW / "reviews/m974_m903_m872_c2_three_axis_pt_saif_ptpx_first_principles_r1_20260829"
UPSTREAM = {
    M974 / "review.json": "da9a246a62244edab5839044978ae8b98af930bb35757d03f10ea390847b7893",
    M974 / "SHA256SUMS": "e74ea811dca4adfd766e31826728344b69d35be5be01418c928ef5004e9e7bf9",
    M974 / "SHA256SUMS.seal.sha256": "7bafbba0e740060b9366743547eb7adb0f3f1724446ea3512055ffa960b3bbfc",
    HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv": "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _must(condition, message):
    if not condition:
        raise RuntimeError(message)


def _activity(text):
    """Return aggregate TC by SAIF object name for ordinary scalar/vector records."""
    out = {}
    record = re.compile(
        r"\((\\?[^\s()]+)\s+((?:\((?:T0|T1|TX|TC|IG)\s+[-+0-9.eE]+\)\s*)+)\)",
        re.MULTILINE)
    for match in record.finditer(text):
        name = match.group(1).replace("\\", "")
        tc = re.search(r"\(TC\s+([-+0-9.eE]+)\)", match.group(2))
        if tc:
            out[name] = out.get(name, 0.0) + float(tc.group(1))
    return out


def _cone(activity, prefixes):
    return sum(v for name, v in activity.items()
               if any(name == p or name.startswith(p + "[") for p in prefixes))


def validate_saif(path, axis, case_id, cycles):
    _must(axis in AXES, "unknown axis")
    _must(0 <= case_id <= 4, "case outside 0..4")
    anchor = AXES[axis]["cycles"]
    if anchor is not None:
        _must(cycles == anchor[case_id], "M867 cycle anchor mismatch")
    text = Path(path).read_text(errors="strict")
    durations = re.findall(r"\(DURATION\s+([-+0-9.eE]+)\)", text)
    _must(len(durations) == 1, "SAIF must contain one DURATION")
    duration = float(durations[0])
    _must(abs(duration - cycles * 3.0) <= 1e-6,
          "SAIF duration is not measured_cycles*3ns")
    tx = [float(x) for x in re.findall(r"\(TX\s+([-+0-9.eE]+)\)", text)]
    _must(tx and all(x == 0.0 for x in tx), "SAIF TX entries absent or nonzero")
    _must(re.search(r"\(INSTANCE\s+tb_m979_c2_three_axis_mapped_gate_case_saif\b", text),
          "testbench instance path absent")
    _must(re.search(r"\(INSTANCE\s+dut\b", text), "exact DUT scope absent")
    activity = _activity(text)
    _must(activity, "no per-object TC activity parsed")
    cones = {
        "clock": _cone(activity, ("clk_core",)),
        "header_raw": _cone(activity, ("header_accept", "raw_accept", "raw_valid")),
        "memory": _cone(activity, ("mem_req_accept", "mem_rsp_accept",
                                    "mem_req_valid", "mem_rsp_valid")),
        "accumulator_result": _cone(activity, ("result_accumulator", "result_accept",
                                                "result_valid")),
        "token_done": _cone(activity, ("token_done_accept", "token_done_valid")),
    }
    for name in ("clock", "header_raw", "accumulator_result", "token_done"):
        _must(cones[name] > 0.0, "zero major cone: " + name)
    if case_id < 4:
        _must(cones["memory"] > 0.0, "zero memory cone on nonzero workload")
    reset_tc = _cone(activity, ("rst_core",))
    _must(reset_tc == 0.0, "reset toggled inside capture window")
    nonzero = sum(v > 0.0 for v in activity.values())
    return {
        "schema": "m979_per_case_mapped_gate_saif_check_v1",
        "status": "PASS_M979_PER_CASE_MAPPED_GATE_SAIF",
        "axis": axis, "case": case_id, "cycles": cycles,
        "duration_ns": duration, "tx_entries": len(tx), "tx_nonzero": 0,
        "reset_tc": reset_tc, "major_cone_tc": cones,
        "zero_case_memory_nonzero_required": case_id < 4,
        "activity_objects": len(activity), "nonzero_activity_objects": nonzero,
        "nonzero_toggle_percent": 100.0 * nonzero / len(activity),
    }


def validate_static(contract=CONTRACT):
    for path, expected in UPSTREAM.items():
        _must(path.is_file() and not path.is_symlink() and sha(path) == expected,
              "upstream identity drift: " + str(path))
    for axis, spec in AXES.items():
        net, sdc = BASE / axis / NET, BASE / axis / SDC
        _must(sha(net) == spec["net_sha"] and sha(sdc) == spec["sdc_sha"],
              "mapped input drift: " + axis)
        _must(spec["module"] in net.read_text()[:2000], "mapped module suffix drift")
    tb, ucli, runner = TB.read_text(), UCLI.read_text(), RUNNER.read_text()
    for token in ("M979_AXIS_K1", "M979_AXIS_K8", "M979_AXIS_K1X8",
                  "M979_CASE=%d", "51", "131", "486", "1231",
                  "53", "133", "499", "1246", "numeric_mismatches",
                  "tuple_mismatches", "accepted_unknowns"):
        _must(token in tb, "TB missing token: " + token)
    _must("power tb_m979_c2_three_axis_mapped_gate_case_saif.dut" in ucli,
          "UCLI is not DUT-only")
    _must(ucli.index("run") < ucli.index("power -enable") <
          ucli.index("power -disable") < ucli.index("power -report"),
          "UCLI window ordering invalid")
    for token in ("k1 k8 k1x8", "0 1 2 3 4", "case${case_id}.saif",
                  "M979_UCLI_SAIF", "check_m979_c2_mapped_gate_saif_source.py",
                  "M990", "M991", "M992", "M993", "ATTEMPT_ATOMIC_CONSUME",
                  "attempt already consumed or incomplete", "seal_dir"):
        _must(token in runner, "runner missing token: " + token)
    _must('mkdir "${attempt}"' in runner and "attempt_stage" not in runner,
          "runner does not consume the canonical attempt directly")
    for forbidden in ("old_simv", "reuse_simv", "results/.m859"):
        _must(forbidden not in runner, "runner permits stale simulation artifact")
    data = json.loads(Path(contract).read_text())
    _must(data["status"] == "PASS_M979_SOURCE_ONLY__NO_EDA_EXECUTED",
          "contract status drift")
    _must(data["axes"] == ["k1", "k8", "k1x8"] and data["case_count_per_axis"] == 5,
          "contract geometry drift")
    _must(data["claim_boundary"]["vcs_executed"] is False and
          data["claim_boundary"]["ptpx_executed"] is False,
          "contract falsely admits execution")
    return {
        "schema": "m979_c2_mapped_gate_saif_source_static_check_v1",
        "status": "PASS_M979_STATIC_SOURCE__NO_EDA",
        "tb_sha256": sha(TB), "ucli_sha256": sha(UCLI),
        "runner_sha256": sha(RUNNER), "test_sha256": sha(TEST),
        "contract_sha256": sha(contract), "axis_count": 3,
        "case_count": 15, "mapped_port_orientation_tool_proven": False,
        "vcs_pt_ptpx_executed": False, "gpu_remote_used": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, default=CONTRACT)
    p.add_argument("--saif", type=Path)
    p.add_argument("--axis", choices=sorted(AXES))
    p.add_argument("--case", type=int, dest="case_id")
    p.add_argument("--cycles", type=int)
    a = p.parse_args()
    if a.saif:
        _must(a.axis is not None and a.case_id is not None and a.cycles is not None,
              "--saif requires --axis --case --cycles")
        value = validate_saif(a.saif, a.axis, a.case_id, a.cycles)
    else:
        value = validate_static(a.contract)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
