`timescale 1ns/1ps
`default_nettype none

// Source-draft adapter for the frozen M979 five-case workload.  M979 names the
// old derived tops, so this zero-logic simulation-only shell maps that public
// boundary to the fresh M1809-derived mapped top.  UCLI scopes the child
// `implementation` instance, not this shell, so the emitted SAIF names match
// the mapped design loaded by PrimeTime PX.
`ifdef M1831_AXIS_K8
  `define M979_AXIS_K8
  `define M1831_LEGACY_TOP m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE1
  `define M1831_NEW_TOP m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0
`elsif M1831_AXIS_K1X8
  `define M979_AXIS_K1X8
  `define M1831_LEGACY_TOP m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE2
  `define M1831_NEW_TOP m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1
`else
  `error "M1831 requires exactly one axis define"
`endif

module `M1831_LEGACY_TOP (
    input logic clk_core, input logic rst_core,
    input logic header_valid, output logic header_ready,
    input logic [23:0] header_tag,
    input logic [5:0] header_raw_beat_count,
    input logic [3:0] header_window_depth,
    input logic [3:0] header_output_blocks,
    output logic header_accept,
    input logic raw_valid, output logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [19:0] raw_beat_index,
    input logic [383:0] raw_bitmap,
    input logic raw_last, output logic raw_accept,
    output logic [7:0] mem_req_valid,
    input logic [7:0] mem_req_ready,
    output logic [127:0] mem_req_epoch,
    output logic [23:0] mem_req_slot,
    output logic [255:0] mem_req_generation,
    output logic [191:0] mem_req_tag,
    output logic [23:0] mem_req_output_block,
    output logic [23:0] mem_req_slice,
    output logic [95:0] mem_req_source_channel,
    output logic [7:0] mem_req_accept,
    input logic [7:0] mem_rsp_valid,
    output logic [7:0] mem_rsp_ready,
    input logic [127:0] mem_rsp_epoch,
    input logic [23:0] mem_rsp_slot,
    input logic [255:0] mem_rsp_generation,
    input logic [191:0] mem_rsp_tag,
    input logic [1023:0] mem_rsp_weight,
    output logic [7:0] mem_rsp_accept,
    output logic result_valid, input logic result_ready,
    output logic [23:0] result_tag,
    output logic [2:0] result_output_block,
    output logic [2:0] result_slice,
    output logic [383:0] result_accumulator,
    output logic result_last, output logic result_accept,
    output logic token_done_valid, input logic token_done_ready,
    output logic [23:0] token_done_tag,
    output logic token_done_had_event,
    output logic token_done_accept,
    output logic protocol_error, output logic numeric_overflow,
    output logic stale_response_seen, output logic busy
);
    `M1831_NEW_TOP implementation (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(header_accept), .raw_valid(raw_valid),
        .raw_ready(raw_ready), .raw_lane_valid(raw_lane_valid),
        .raw_beat_index(raw_beat_index), .raw_bitmap(raw_bitmap),
        .raw_last(raw_last), .raw_accept(raw_accept),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation), .mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(mem_req_accept), .mem_rsp_valid(mem_rsp_valid),
        .mem_rsp_ready(mem_rsp_ready), .mem_rsp_epoch(mem_rsp_epoch),
        .mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight), .mem_rsp_accept(mem_rsp_accept),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_tag(result_tag),
        .result_output_block(result_output_block),
        .result_slice(result_slice),
        .result_accumulator(result_accumulator), .result_last(result_last),
        .result_accept(result_accept), .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),
        .token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),
        .protocol_error(protocol_error),
        .numeric_overflow(numeric_overflow),
        .stale_response_seen(stale_response_seen), .busy(busy));
endmodule

`include "tb_m979_c2_three_axis_mapped_gate_case_saif.sv"

`undef M1831_LEGACY_TOP
`undef M1831_NEW_TOP
`ifdef M1831_AXIS_K8
  `undef M979_AXIS_K8
`elsif M1831_AXIS_K1X8
  `undef M979_AXIS_K1X8
`endif

`default_nettype wire
