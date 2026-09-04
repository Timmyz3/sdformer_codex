#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1154R6 conditional dual-DUT VCS root diagnostic; source only.

The exact frozen M1133R6 netlist is inspected for stable semantic internal
taps before any attempt or tool invocation.  The current netlist lacks the
paired accept, consistency-fault, and component protocol-error tap names, so
the real preflight intentionally stops.  Bounded self-test exercises only a
synthetic declaration fixture and source templates; it never runs VCS or DC.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HW = SOURCE_FILE.parent.parent.parent
RESULTS = HW / "results"
CONTRACT = HW / "contracts/m1154r6_c2_dual_dut_vcs_root_diagnostic_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "52a27a267f2064efc3db8cffe3c705775eb97887b639114ba6e83cab3537df6e",
    "f1e3173dea034247627c31fe444b43f7bc0094b1a162c7e9cf4f450160cb107c",
    "75121250728947363d63ca9c8ed562961eb3dbb1fd9c3207806570304d8b2aa0",
)
M1151 = HW / "reviews/m1151r6_m1146r6_c2_case0_x_failure_audit_r1_20260830"
M1151_ID = (
    "08f9041acc9671f76f5c94a87c5ceba4797c8bfe9f8cdae41bbe9647ea7d3411",
    "8fab9695b27004337c1993d016a078d6523dcc78654f4a79ad6dbcda3316ee41",
    "72bf8c7500a45961aefada1cb3b720bfc0b357eb7e08257379015fb6c1288c5f",
)
NETLIST = HW / ("results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830."
                "failed_or_incomplete.1172090.quarantine/dc/netlist/"
                "m1129r5_c2_k1_async_observation_shadow_wrapper_mapped.v")
NETLIST_SHA = "362e855cd3b4391d31dc7a08e5388d9545f289c81d291c512d25294a8539cbc4"
OLD_TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
OLD_TB_SHA = "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b"
OLD_MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
OLD_MEMORY_SHA = "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa"
RTL_REFERENCE = HW / "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
RTL_REFERENCE_SHA = "86df0f7be383e6ba8ee17c1e27fc25fd18eb6fecc01329c41a976cd836004dd0"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
            "TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
            "tcbn28hpcplusbwp35p140.v")
CELL_SHA = "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
VCS_SHA = "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = RESULTS / "m1154r6_c2_dual_dut_vcs_root_diagnostic_r1_20260830"
ATTEMPT = RESULTS / ".m1154r6_c2_dual_dut_vcs_root_diagnostic_attempt_consumed"
WORK_PREFIX = ".m1154r6_c2_dual_dut_vcs_root_diagnostic_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
LOCK = Path("/tmp/m1154r6_c2_dual_dut_vcs_root_diagnostic.lock")
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"

RETAINED_FAULT_TAPS = (
    "implementation_core_frontend_compactor_fault_q",
    "implementation_core_frontend_paired_sink_fault_q",
    "implementation_core_adapter_fault_q",
    "implementation_core_g_k1_service_fault_q",
    "implementation_memory_adapter_fault_q",
)
PAIRED_ACCEPT_TAPS = (
    "implementation_core_mem_req_accept",
    "implementation_adapter_core_mem_req_accept",
    "implementation_core_mem_rsp_accept",
    "implementation_adapter_core_mem_rsp_accept",
)
CONSISTENCY_TAPS = (
    "implementation_consistency_fault_now",
    "implementation_consistency_fault_q",
)
PROTOCOL_TAPS = (
    "implementation_core_protocol_error",
    "implementation_adapter_protocol_error",
)
REQUIRED_TAPS = RETAINED_FAULT_TAPS + PAIRED_ACCEPT_TAPS + CONSISTENCY_TAPS + PROTOCOL_TAPS


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal drift")


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"; manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "M1151 outer drift")
    listed = {}
    for row in manifest.read_text(encoding="utf-8").splitlines():
        fields = row.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "M1151 manifest row")
        name = fields[1].lstrip("*"); relative = Path(name)
        require(name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "M1151 manifest member")
        listed[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1151 symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "M1151 special member")
    require(actual == set(listed), "M1151 member set drift")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


def namespace_fresh() -> bool:
    paths = (RESULT, ATTEMPT, LOCK)
    variable = tuple(RESULTS.glob(WORK_PREFIX + "*")) + tuple(RESULTS.glob(FAILURE_PREFIX + "*"))
    return not any(path.exists() or path.is_symlink() for path in paths + variable)


def declaration_prefix(path: Path) -> str:
    """Read only declarations; stop before the first mapped cell instance."""
    lines = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if re.match(r"^\s*[A-Z][A-Z0-9]*BWP35P140\s+", line):
                break
            lines.append(line)
    value = "".join(lines)
    require("module m1129r5_c2_k1_async_observation_shadow_wrapper" in value and
            "wire" in value, "mapped declaration prefix drift")
    return value


def stable_tap_census_from_text(declarations: str) -> dict[str, Any]:
    def declared(name: str) -> bool:
        # Only the declaration prefix is searched. Anonymous n* cones and cell
        # pin occurrences can never satisfy this semantic-name gate.
        return re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) +
                         r"(?![A-Za-z0-9_$])", declarations) is not None
    present = tuple(name for name in REQUIRED_TAPS if declared(name))
    missing = tuple(name for name in REQUIRED_TAPS if not declared(name))
    return {"present": present, "missing": missing,
            "required": len(REQUIRED_TAPS), "present_count": len(present),
            "missing_count": len(missing), "anonymous_n_net_allowed": False}


def valid_qualified_endpoint_template() -> str:
    return r'''`timescale 1ns/1ps
`default_nettype none
// New diagnostic-only endpoint; the original M349 source is unchanged.
module m1154r6_valid_qualified_scalar_bank_endpoint #(
    parameter int BANK_ID=0,TAG_BITS=24,CHANNEL_BITS=12,EPOCH_BITS=16,
    GENERATION_BITS=32,SLICE_LANES=16,LATENCY=4
)(
    input logic clk_core,rst_core,enable,request_allow,newest_first,spurious_valid,
    input logic mem_req_valid,input logic [EPOCH_BITS-1:0] mem_req_epoch,
    input logic [2:0] mem_req_slot,input logic [GENERATION_BITS-1:0] mem_req_generation,
    input logic [TAG_BITS-1:0] mem_req_tag,input logic [2:0] mem_req_output_block,
    input logic [2:0] mem_req_slice,input logic [CHANNEL_BITS-1:0] mem_req_source_channel,
    input logic mem_req_accept,output logic mem_req_ready,
    output logic endpoint_protocol_fault_now,
    output logic mem_rsp_valid,input logic mem_rsp_ready,
    output logic [EPOCH_BITS-1:0] mem_rsp_epoch,output logic [2:0] mem_rsp_slot,
    output logic [GENERATION_BITS-1:0] mem_rsp_generation,
    output logic [TAG_BITS-1:0] mem_rsp_tag,
    output logic signed [7:0] mem_rsp_weight[0:SLICE_LANES-1],input logic mem_rsp_accept,
    output logic [31:0] request_count,response_count,output logic [3:0] pending_count,
    output logic live_slot_reuse_error
);
    logic request_payload_known,inner_req_ready,inner_rsp_valid;
    always_comb begin
        request_payload_known=!$isunknown({mem_req_slot,mem_req_epoch,
            mem_req_generation,mem_req_tag,mem_req_output_block,mem_req_slice,
            mem_req_source_channel});
        endpoint_protocol_fault_now=mem_req_valid&&!request_payload_known;
        mem_req_ready=(mem_req_valid&&request_payload_known)?inner_req_ready:1'b0;
        mem_rsp_valid=inner_rsp_valid; // only accepted qualified requests were stored
    end
    m349_fc2_scalar_bank_memory_model #(.BANK_ID(BANK_ID),.TAG_BITS(TAG_BITS),
        .CHANNEL_BITS(CHANNEL_BITS),.EPOCH_BITS(EPOCH_BITS),
        .GENERATION_BITS(GENERATION_BITS),.SLICE_LANES(SLICE_LANES),.LATENCY(LATENCY)) inner(
        .clk_core(clk_core),.rst_core(rst_core),.enable(enable),
        .request_allow(request_allow&&mem_req_valid&&request_payload_known),
        .newest_first(newest_first),.spurious_valid(spurious_valid),
        .mem_req_valid(mem_req_valid&&request_payload_known),.mem_req_ready(inner_req_ready),
        .mem_req_epoch(mem_req_epoch),.mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation),.mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block),.mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(mem_req_accept&&mem_req_valid&&request_payload_known),
        .mem_rsp_valid(inner_rsp_valid),.mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_epoch(mem_rsp_epoch),.mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation),.mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight),.mem_rsp_accept(mem_rsp_accept),
        .request_count(request_count),.response_count(response_count),
        .pending_count(pending_count),.live_slot_reuse_error(live_slot_reuse_error));
endmodule
`default_nettype wire
'''


def dual_dut_probe_template() -> str:
    tap_rows = "\n".join(
        f"// atomic_probe orig.dut_orig.{tap} qualified.dut_qualified.{tap}"
        for tap in REQUIRED_TAPS)
    return f'''`timescale 1ns/1ps
`default_nettype none
// Design specification only. A successor may elaborate this only after the
// stable-tap gate passes and a different-author hammer authorizes one VCS run.
module m1154r6_dual_dut_atomic_probe_spec;
// DUT_A: frozen netlist + unchanged M349 endpoint.
// DUT_B: same frozen netlist/stimulus + M1154 valid-qualified endpoint.
// Neither DUT may use force, initreg, SDF, X coercion, or delayed checking.
{tap_rows}
// first-X snapshot is an atomic bitmap captured before control decisions;
// union collection continues through the exact 128-cycle window.
endmodule
`default_nettype wire
'''


def source_preflight() -> dict[str, Any]:
    verify_double(CONTRACT, CONTRACT_ID)
    review = verify_tree(M1151, M1151_ID)
    for path, digest in ((NETLIST, NETLIST_SHA), (OLD_TB, OLD_TB_SHA),
                         (OLD_MEMORY, OLD_MEMORY_SHA),
                         (RTL_REFERENCE, RTL_REFERENCE_SHA), (CELL, CELL_SHA),
                         (VCS, VCS_SHA), (DOCS359, DOCS359_SHA)):
        verify_regular(path, digest)
    require(review["status"] ==
            "PASS_M1151R6_READ_ONLY_FAILURE_HAMMER__M1146R6_DO_NOT_RETRY__ONE_ADDITIVE_VCS_DIAGNOSTIC_AUTHORIZED" and
            review["authorization"]["retry_m1146r6"] is False and
            review["authorization"]["dc"] is False and
            review["authorization"]["different_author_hammer_before_execution"] is True,
            "M1151 authorization drift")
    require(namespace_fresh(), "M1154R6 namespace is not fresh")
    census = stable_tap_census_from_text(declaration_prefix(NETLIST))
    require(set(RETAINED_FAULT_TAPS).issubset(census["present"]),
            "retained fault tap identity drift")
    expected_missing = set(PAIRED_ACCEPT_TAPS + CONSISTENCY_TAPS + PROTOCOL_TAPS)
    require(set(census["missing"]) == expected_missing,
            "unexpected stable-tap census drift")
    return {
        "status": "STOP_M1154R6_FROZEN_NETLIST_LACKS_REQUIRED_STABLE_SEMANTIC_TAPS__NO_ATTEMPT_NO_VCS",
        "stable_tap_census": census,
        "attempt_created": False,
        "vcs_calls": 0,
        "dc_calls": 0,
        "anonymous_n_net_substitution_allowed": False,
        "recommendation": "stop mapped-observation expansion; retain M903 logic-only claim",
    }


def source_bounded_mock_self_test() -> dict[str, Any]:
    before = namespace_fresh()
    real = source_preflight()
    mock_declarations = ("module m1129r5_c2_k1_async_observation_shadow_wrapper;\n  wire "
                         + ",\n       ".join(REQUIRED_TAPS) + ";\nendmodule\n")
    mock = stable_tap_census_from_text(mock_declarations)
    endpoint = valid_qualified_endpoint_template()
    probe = dual_dut_probe_template()
    require(mock["missing"] == () and mock["present_count"] == len(REQUIRED_TAPS),
            "bounded all-tap mock rejected")
    require("mem_req_ready=(mem_req_valid&&request_payload_known)?inner_req_ready:1'b0" in endpoint and
            "mem_req_accept&&mem_req_valid&&request_payload_known" in endpoint and
            "endpoint_protocol_fault_now=mem_req_valid&&!request_payload_known" in endpoint and
            "force" not in endpoint.lower() and "+initreg" not in endpoint.lower(),
            "valid-qualified endpoint contract drift")
    require(all(("orig.dut_orig." + tap) in probe and
                ("qualified.dut_qualified." + tap) in probe for tap in REQUIRED_TAPS),
            "dual-DUT atomic probe contract drift")
    require(before and namespace_fresh() and real["attempt_created"] is False and
            real["vcs_calls"] == 0 and real["dc_calls"] == 0,
            "bounded self-test mutated real namespace or invoked tool")
    return {
        "schema": "m1154r6_c2_dual_dut_vcs_root_diagnostic_source_bounded_mock_v1",
        "status": "PASS_SOURCE_AND_BOUNDED_MOCK__REAL_STABLE_TAP_GATE_STOP",
        "real_preflight": real,
        "bounded_mock_required_taps": len(REQUIRED_TAPS),
        "bounded_mock_missing_taps": 0,
        "endpoint_valid_and_payload_qualified": True,
        "paired_dut_atomic_probe_specified": True,
        "attempt_created": False,
        "vcs_calls": 0,
        "dc_calls": 0,
    }


def production_main() -> dict[str, Any]:
    preflight = source_preflight()
    raise Failure(preflight["status"] +
                  ": paired accept/consistency/component protocol taps were optimized away; "
                  "anonymous n* cone binding is not reproducible")


def main() -> int:
    require(len(sys.argv) == 1, "M1154R6 accepts zero arguments")
    # The zero-argument path is deliberately a stop path for this exact frozen
    # netlist. It can never consume an attempt or invoke VCS/DC.
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
