#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only hammer for the consumed M1146R6 mapped-case0 failure.

No EDA executable, launcher, producer, or failed namespace is invoked.  The
program verifies fixed bytes and derives the first-X/cascade facts from the
frozen log and source/netlist text only.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
ATTEMPT = HW / "results/.m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_attempt_consumed"
WORK = HW / "results/.m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_work.1949316.1788053754703494967"
CASE = WORK / "mapped_vcs/case0.log"
COMPILE = WORK / "mapped_vcs/compile.log"
FAILURE = WORK / "failure.json"
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
WRAPPER = HW / "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
CORE_TOP = HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv"
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/"
            "digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    CASE: "35dbbdb708b7f5be182a6d285c0f899b8f70200d76f01319ac46e2ea1ff3c394",
    TB: "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b",
    WRAPPER: "86df0f7be383e6ba8ee17c1e27fc25fd18eb6fecc01329c41a976cd836004dd0",
    NETLIST: "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

NAMES = [
    "obs_header_accept", "obs_raw_accept", "obs_busy", "obs_protocol_error",
    "obs_numeric_overflow", "obs_stale_response", "obs_fault",
    "obs_bank_request_accept", "obs_bank_response_accept",
    "obs_service_fifo_count", "obs_service_outstanding_count",
    "obs_service_group_count", "obs_service_request_count",
    "obs_service_response_count", "obs_service_context_count",
    "obs_service_result_count", "obs_service_active_bank_read_count",
    "obs_adapter_live_slots", "obs_adapter_bundle_request_count",
    "obs_adapter_bank_request_count", "obs_adapter_bank_response_count",
    "obs_adapter_bundle_response_count",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def regular(path: Path) -> bool:
    mode = path.lstat().st_mode
    return stat.S_ISREG(mode) and not path.is_symlink()


def verify_attempt() -> dict:
    assert ATTEMPT.is_dir() and not ATTEMPT.is_symlink()
    manifest = ATTEMPT / "SHA256SUMS"
    outer = ATTEMPT / "SHA256SUMS.seal.sha256"
    assert regular(manifest) and regular(outer)
    assert text(outer).split() == [sha(manifest), "SHA256SUMS"]
    listed = {}
    for line in text(manifest).splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        listed[name] = digest
        assert regular(ATTEMPT / name) and sha(ATTEMPT / name) == digest
    actual = {p.name for p in ATTEMPT.iterdir()
              if p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert actual == set(listed)
    attempt = json.loads(text(ATTEMPT / "attempt.json"))
    assert attempt["status"] == "M1146R6_SINGLE_ATTEMPT_CONSUMED__NO_RETRY"
    assert attempt["compile_attempts"] == attempt["case0_attempts"] == 1
    assert attempt["dc_attempts"] == 0 and attempt["automatic_retry"] is False
    return {"manifest_members": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def main() -> None:
    for path, digest in EXPECTED.items():
        assert regular(path) and sha(path) == digest, path
    attempt_seal = verify_attempt()
    case = text(CASE)
    tb = text(TB)
    wrapper = text(WRAPPER)
    core = text(CORE_TOP)
    memory = text(MEMORY)
    netlist = text(NETLIST)
    compile_log = text(COMPILE)
    failure = json.loads(text(FAILURE))

    first = re.search(r"M1112_FIRST_X cycle=(\d+) bitmap=([0-9a-f]{6})", case)
    summary = re.search(r"M1112_UNKNOWN_SUMMARY first_cycle=(\d+) first_bitmap=([0-9a-f]{6}) union_bitmap=([0-9a-f]{6}) cycles=(\d+)", case)
    assert first and summary
    assert first.groups() == ("3", "000048")
    assert summary.groups() == ("3", "000048", "2b96cc", "128")
    first_bits = [i for i in range(22) if int(first.group(2), 16) & (1 << i)]
    union_bits = [i for i in range(22) if int(summary.group(3), 16) & (1 << i)]
    assert first_bits == [3, 6]
    assert union_bits == [2, 3, 6, 7, 9, 10, 12, 15, 16, 17, 19, 21]
    stages = {int(cycle): int(bitmap, 16) for cycle, bitmap in
              re.findall(r"^M1112_STAGE cycle=(\d+) unknown=([0-9a-f]{6})", case, re.M)}
    assert len(stages) == 128
    assert stages[3] == 0x48 and stages[4] == 0xC8
    assert stages[5] == 0x0914C8 and stages[9] == 0x2B96C8 and stages[12] == 0x2B96CC

    predicates = re.findall(r"sample_unknown_bitmap\[(\d+)\]=\$isunknown\((obs_\w+)\);", tb)
    assert [(int(index), name) for index, name in predicates] == list(enumerate(NAMES))
    assert "repeat(5)@(posedge clk_core);" in tb
    assert "@(negedge clk_core);rst_core=0;header_valid=1;" in tb
    assert "always #1.5 clk_core=~clk_core;" in tb
    assert "obs_fault=protocol_error|numeric_overflow|stale_response_seen;" in wrapper
    assert "obs_bank_request_accept=mem_req_accept;" in wrapper

    updates = {
        9: ("raw_accept", "result_accept"),
        10: ("request_accept_count", "response_accept_count"),
        12: ("request_accept_count",),
        15: ("result_accept",),
        16: ("request_accept_count",),
        17: ("raw_accept", "result_accept"),
        19: ("request_accept_count",),
        21: ("result_accept",),
    }
    for bit, tokens in updates.items():
        signal = NAMES[bit].replace("obs_", "shadow_") + "_q"
        assert signal in wrapper
        for token in tokens:
            assert token in wrapper

    async_flops = re.findall(
        r"DFCNQD1BWP35P140\s+(shadow_\S+?)\s*\(.*?\.CDN\(([^)]+)\)",
        netlist, re.S)
    assert len(async_flops) == 337
    assert all(clear.startswith("n") for _, clear in async_flops)
    assert netlist.count("DFCNQD1BWP35P140") == 337
    assert "module DFCNQD1BWP35P140 (D, CP, CDN, Q);" in text(CELL)
    assert "(negedge CDN => (Q+:1'b0))" in text(CELL)
    assert "UNIT_DELAY" not in compile_log and "SDF" not in compile_log

    # The first independent X is functional protocol control.  The direct fault
    # alias is not a second root; later shadow Xs follow accepted-event taps.
    assert "if (core_mem_req_accept != adapter_core_mem_req_accept)" in core
    assert "if (core_mem_rsp_accept != adapter_core_mem_rsp_accept)" in core
    assert "|| consistency_fault_q || consistency_fault_now" in core
    assert "!pending_q[mem_req_slot]" in memory
    assert failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE"
    assert failure["phase"] == "FROZEN_NETLIST_CASE0_128_ONCE"
    assert "PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW" not in case

    result = {
        "schema": "m1151r6_m1146r6_c2_case0_x_failure_mechanical_v1",
        "status": "PASS_READ_ONLY_FAILURE_HAMMER__M1146R6_DO_NOT_RETRY__ROOT_NOT_UNIQUELY_LOCALIZED",
        "scope": {"vcs": False, "simv": False, "dc": False, "rtl_or_netlist_modified": False,
                  "failed_namespace_modified": False, "docs359_modified": False},
        "attempt": attempt_seal,
        "first_x": {"cycle": 3, "bitmap": "000048", "bits": first_bits,
                    "signals": [NAMES[i] for i in first_bits],
                    "independent_root_signal": "obs_protocol_error",
                    "obs_fault_is_direct_alias": True},
        "cascade": {"cycle4_added": ["obs_bank_request_accept"],
                    "cycle5_added": [NAMES[i] for i in [10, 12, 16, 19]],
                    "cycle9_added": [NAMES[i] for i in [9, 15, 17, 21]],
                    "cycle12_added": ["obs_busy"],
                    "union_bitmap": "2b96cc", "union_signals": [NAMES[i] for i in union_bits]},
        "reset_and_cell_exclusions": {"reset_active_posedges": 5,
                    "reset_release_edge": "negedge", "release_margin_ns": 1.5,
                    "shadow_async_flop_bits": 337, "active_low_clear_cell": "DFCNQD1BWP35P140",
                    "shadow_clear_polarity_matches_active_high_rst_through_inverter": True,
                    "unit_delay_or_sdf": False,
                    "shadow_reset_chain_is_first_root": False,
                    "tb_reset_timing_is_supported_as_root": False},
        "root_cause_boundary": {"localized_to": "FUNCTIONAL_PROTOCOL_CONTROL_CONE_BEFORE_SHADOW_COUNTERS",
                    "candidate_x_amplifier": "protocol_error includes consistency_fault_now from four-state != of paired accepts",
                    "candidate_upstream": "request-ready/accept X can be created by invalid-payload-indexed TB ready or a functional synchronous-reset/data reconvergence",
                    "unique_internal_register_proven_from_22_port_log": False,
                    "classification": "FUNCTIONAL_HANDSHAKE_X__EXACT_UPSTREAM_CONE_REQUIRES_ADDITIVE_INTERNAL_OBSERVATION"},
        "identity": {str(path.relative_to(HW) if HW in path.parents else path): digest
                     for path, digest in EXPECTED.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
