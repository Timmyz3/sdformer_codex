#!/usr/bin/env python3
"""Static/source and future-SAIF checks for M1332; never launches EDA."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
NET = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
SDC = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.sdc"
TB_OLD = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
MEM = HW / "dc_handoff/tb/m1332_c2_production_activity_reset_safe_memory_model.sv"
SVA = HW / "dc_handoff/tb/m1332_c2_production_activity_assertions.sv"
TB = HW / "dc_handoff/tb/tb_m1332_c2_headline_mapped_production_activity.sv"
UCLI = HW / "dc_handoff/scripts/m1332_c2_headline_mapped_production_activity.ucli.tcl"
CONTRACT = HW / "contracts/m1332_c2_headline_mapped_production_activity_source_contract_r1_20260831.json"
FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1332_c2_k8_mapped_production_activity.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1332_c2_k1x8_mapped_production_activity.f",
}
AXES = {
    "k8": {
        "define": "M979_AXIS_K8", "opposite": "M979_AXIS_K1X8",
        "net_sha": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "sdc_sha": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
        "cycles": [51, 131, 486, 1231, 14],
    },
    "k1x8": {
        "define": "M979_AXIS_K1X8", "opposite": "M979_AXIS_K8",
        "net_sha": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "sdc_sha": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
        "cycles": [53, 133, 499, 1246, 14],
    },
}
FROZEN = {
    TB_OLD: "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/SHA256SUMS.seal.sha256": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    HW / "reviews/m1331_c2_production_saif_ptpx_readonly_gap_audit_r1_20260831/SHA256SUMS.seal.sha256": "3d3c556d8ead1e2be729ddcd20285cc542aed560363905c4bac3b66fd0592d75",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def must(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_filelist(text, axis):
    spec = AXES[axis]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    must(lines.count("+define+" + spec["define"]) == 1,
         axis + " filelist lacks its unique axis define")
    must(spec["opposite"] not in text and "M979_AXIS_K1\n" not in text,
         axis + " filelist admits another or diagnostic K1 axis")
    must(lines.count("+define+SVA_RUNTIME_ENABLED") == 1,
         axis + " filelist lacks live SVA")
    must(sum("m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v" in line
             for line in lines) == 1, axis + " mapped netlist count is not one")
    must(any(("/" + axis + "/netlist/") in line for line in lines),
         axis + " filelist binds the wrong mapped coordinate")
    must(sum("m1332_c2_production_activity_reset_safe_memory_model.sv" in line
             for line in lines) == 1, axis + " reset-safe endpoint absent")
    must(not any("tb_m349/m349_fc2_scalar_bank_memory_model.sv" in line
                 for line in lines), axis + " silently falls back to old memory")
    for leaf in ("tb_m979_c2_three_axis_mapped_gate_case_saif.sv",
                 "m1332_c2_production_activity_assertions.sv",
                 "tb_m1332_c2_headline_mapped_production_activity.sv"):
        must(sum(leaf in line for line in lines) == 1,
             axis + " filelist missing/duplicates " + leaf)
    return lines


def _activity(text):
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
    return sum(value for name, value in activity.items()
               if any(name == p or name.startswith(p + "[") for p in prefixes))


def validate_saif(path, axis, case_id, cycles):
    must(axis in AXES, "only headline k8/k1x8 axes are admitted")
    must(0 <= case_id < 5, "case outside frozen 0..4 set")
    must(cycles == AXES[axis]["cycles"][case_id], "M903 cycle anchor mismatch")
    text = Path(path).read_text(errors="strict")
    durations = re.findall(r"\(DURATION\s+([-+0-9.eE]+)\)", text)
    must(len(durations) == 1 and abs(float(durations[0]) - cycles*3.0) <= 1e-6,
         "SAIF duration is not cycle anchor times 3 ns")
    tx = [float(x) for x in re.findall(r"\(TX\s+([-+0-9.eE]+)\)", text)]
    must(tx and all(x == 0.0 for x in tx), "SAIF has absent/nonzero TX")
    must(re.search(r"\(INSTANCE\s+tb_m1332_c2_headline_mapped_production_activity\b", text),
         "M1332 top scope absent")
    must(re.search(r"\(INSTANCE\s+core\b", text)
         and re.search(r"\(INSTANCE\s+dut\b", text), "exact core.dut scope absent")
    activity = _activity(text)
    must(activity, "no per-object activity parsed")
    cones = {
        "clock": _cone(activity, ("clk_core",)),
        "source": _cone(activity, ("raw_valid", "raw_accept", "raw_bitmap")),
        "endpoint": _cone(activity, ("mem_req_valid", "mem_req_accept",
                                      "mem_rsp_valid", "mem_rsp_accept")),
        "commit": _cone(activity, ("result_valid", "result_accept",
                                    "result_accumulator")),
        "done": _cone(activity, ("token_done_valid", "token_done_accept")),
    }
    for name in ("clock", "source", "commit", "done"):
        must(cones[name] > 0.0, "zero production cone: " + name)
    if case_id < 4:
        must(cones["endpoint"] > 0.0, "nonzero case has zero endpoint activity")
    reset_tc = _cone(activity, ("rst_core",))
    must(reset_tc == 0.0, "reset toggled inside production activity window")
    return {"schema": "m1332_c2_production_saif_check_r1",
            "status": "PASS_M1332_HEADLINE_AXIS_PRODUCTION_SAIF",
            "axis": axis, "case": case_id, "cycles": cycles,
            "duration_ns": float(durations[0]), "tx_nonzero": 0,
            "reset_tc": reset_tc, "major_cone_tc": cones}


def validate_static(contract=CONTRACT):
    for path, expected in FROZEN.items():
        must(path.is_file() and not path.is_symlink() and sha(path) == expected,
             "frozen identity drift: " + str(path))
    m903 = json.loads((HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json").read_text())
    must(m903["status"] == "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED",
         "M903 admission status drift")
    for axis, spec in AXES.items():
        net, sdc = BASE / axis / NET, BASE / axis / SDC
        must(sha(net) == spec["net_sha"] and sha(sdc) == spec["sdc_sha"],
             axis + " mapped identity drift")
        validate_filelist(FILELISTS[axis].read_text(), axis)

    mem, sva, tb, ucli = (MEM.read_text(), SVA.read_text(), TB.read_text(),
                           UCLI.read_text())
    for token in ("request_payload_known", "mem_req_valid === 1'b0",
                  "mem_req_accept === 1'b1", "mem_rsp_accept === 1'b1",
                  "endpoint_protocol_fault_q", "epoch_q[slot] <= '0",
                  "channel_q[slot] <= '0"):
        must(token in mem, "reset-safe endpoint missing token: " + token)
    for token in ("ap_request_payload_known", "ap_response_payload_known",
                  "ap_result_stable_under_stall", "cp_source", "cp_endpoint",
                  "cp_commit", "cp_stall", "M1332 coverage/fault gate failed",
                  "case_id < 4 && endpoint_count == 0",
                  "case_id == 4 && endpoint_count != 0",
                  "M1332 assertion absolute watchdog"):
        must(token in sva, "assertion/cover source missing token: " + token)
    must("tb_m979_c2_three_axis_mapped_gate_case_saif core" in tb,
         "wrapper does not preserve frozen M979 workload driver")
    must("core.g_memory[bank].memory.endpoint_protocol_fault_q" in tb,
         "endpoint fault is not observable")
    scope = "tb_m1332_c2_headline_mapped_production_activity.core.dut"
    must(ucli.count("power " + scope) == 1
         and ucli.count("power -report $::env(M1332_SAIF_FILE) 1e-9 " + scope) == 1,
         "UCLI is not exact DUT-only scope")
    must(ucli.index("run") < ucli.index("power -enable")
         < ucli.index("power -disable") < ucli.index("power -report"),
         "UCLI activity window ordering invalid")

    data = json.loads(Path(contract).read_text())
    must(data["status"] == "PASS_M1332_SOURCE_ONLY__NO_EDA_EXECUTED",
         "source contract status drift")
    must(data["axes"] == ["k8", "k1x8"] and data["cases_per_axis"] == 5,
         "headline-axis geometry drift")
    must(data["claim_boundary"]["vcs"] is False
         and data["claim_boundary"]["saif"] is False
         and data["claim_boundary"]["ptpx"] is False,
         "source contract falsely admits execution")
    for item in data["source_files"]:
        path = HW / item["path"]
        must(sha(path) == item["sha256"], "source contract SHA drift: " + item["path"])
    return {"schema": "m1332_c2_headline_mapped_production_activity_static_check_r1",
            "status": "PASS_M1332_SOURCE_ONLY__NO_EDA",
            "axes": ["k8", "k1x8"], "cases": 10,
            "same_frozen_workloads": True, "headline_rtl_modified": False,
            "eda_executed": False, "contract_sha256": sha(contract)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--saif", type=Path)
    parser.add_argument("--axis", choices=sorted(AXES))
    parser.add_argument("--case", type=int, dest="case_id")
    parser.add_argument("--cycles", type=int)
    args = parser.parse_args()
    if args.saif:
        must(args.axis is not None and args.case_id is not None
             and args.cycles is not None, "SAIF mode requires axis/case/cycles")
        out = validate_saif(args.saif, args.axis, args.case_id, args.cycles)
    else:
        out = validate_static(args.contract)
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
