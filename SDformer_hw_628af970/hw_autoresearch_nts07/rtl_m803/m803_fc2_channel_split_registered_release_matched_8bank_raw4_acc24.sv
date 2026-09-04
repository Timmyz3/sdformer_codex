`timescale 1ns/1ps
`default_nettype none

// M803/C2 R16 is the additive physical-comparison shell for frozen canonical K1,
// channel-split shared-state K8,
// and replicated K1x8. All three elaboration points expose identical
// functional and eight-bank SRAM
// pins.  Debug counters are deliberately sunk inside the shell so DC can
// remove observation-only state symmetrically before matched PPA.
module m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24 #(
    // 0: canonical K1, 1: shared-state K8, 2: replicated K1x8.
    parameter int ARCH_MODE = 1,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         header_valid,
    output logic                         header_ready,
    input  logic [TAG_BITS-1:0]          header_tag,
    input  logic [5:0]                   header_raw_beat_count,
    input  logic [3:0]                   header_window_depth,
    input  logic [3:0]                   header_output_blocks,
    output logic                         header_accept,
    input  logic                         raw_valid,
    output logic                         raw_ready,
    input  logic [3:0]                   raw_lane_valid,
    input  logic [4:0]                   raw_beat_index [0:3],
    input  logic [95:0]                  raw_bitmap [0:3],
    input  logic                         raw_last,
    output logic                         raw_accept,
    output logic [7:0]                   mem_req_valid,
    input  logic [7:0]                   mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch [0:7],
    output logic [2:0]                   mem_req_slot [0:7],
    output logic [GENERATION_BITS-1:0]   mem_req_generation [0:7],
    output logic [TAG_BITS-1:0]          mem_req_tag [0:7],
    output logic [2:0]                   mem_req_output_block [0:7],
    output logic [2:0]                   mem_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel [0:7],
    output logic [7:0]                   mem_req_accept,
    input  logic [7:0]                   mem_rsp_valid,
    output logic [7:0]                   mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch [0:7],
    input  logic [2:0]                   mem_rsp_slot [0:7],
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation [0:7],
    input  logic [TAG_BITS-1:0]          mem_rsp_tag [0:7],
    input  logic signed [7:0]            mem_rsp_weight
                                                   [0:7][0:SLICE_LANES-1],
    output logic [7:0]                   mem_rsp_accept,
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic [2:0]                   result_output_block,
    output logic [2:0]                   result_slice,
    output logic signed [23:0]           result_accumulator
                                                   [0:SLICE_LANES-1],
    output logic                         result_last,
    output logic                         result_accept,
    output logic                         token_done_valid,
    input  logic                         token_done_ready,
    output logic [TAG_BITS-1:0]          token_done_tag,
    output logic                         token_done_had_event,
    output logic                         token_done_accept,
    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         stale_response_seen,
    output logic                         busy
);
`define M519_COMMON_PORTS \
        .clk_core(clk_core), .rst_core(rst_core), \
        .header_valid(header_valid), .header_ready(header_ready), \
        .header_tag(header_tag), \
        .header_raw_beat_count(header_raw_beat_count), \
        .header_window_depth(header_window_depth), \
        .header_output_blocks(header_output_blocks), \
        .header_accept(header_accept), \
        .raw_valid(raw_valid), .raw_ready(raw_ready), \
        .raw_lane_valid(raw_lane_valid), \
        .raw_beat_index(raw_beat_index), .raw_bitmap(raw_bitmap), \
        .raw_last(raw_last), .raw_accept(raw_accept), \
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready), \
        .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot), \
        .mem_req_generation(mem_req_generation), \
        .mem_req_tag(mem_req_tag), \
        .mem_req_output_block(mem_req_output_block), \
        .mem_req_slice(mem_req_slice), \
        .mem_req_source_channel(mem_req_source_channel), \
        .mem_req_accept(mem_req_accept), \
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready), \
        .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot), \
        .mem_rsp_generation(mem_rsp_generation), \
        .mem_rsp_tag(mem_rsp_tag), .mem_rsp_weight(mem_rsp_weight), \
        .mem_rsp_accept(mem_rsp_accept), \
        .result_valid(result_valid), .result_ready(result_ready), \
        .result_tag(result_tag), \
        .result_output_block(result_output_block), \
        .result_slice(result_slice), \
        .result_accumulator(result_accumulator), \
        .result_last(result_last), .result_accept(result_accept), \
        .token_done_valid(token_done_valid), \
        .token_done_ready(token_done_ready), \
        .token_done_tag(token_done_tag), \
        .token_done_had_event(token_done_had_event), \
        .token_done_accept(token_done_accept), \
        .protocol_error(protocol_error), \
        .numeric_overflow(numeric_overflow), \
        .stale_response_seen(stale_response_seen), .busy(busy)

    generate
        if (ARCH_MODE == 0) begin : g_k1
            logic [5:0] unused_fifo_count;
            logic [6:0] unused_outstanding_count;
            logic [31:0] unused_group_count, unused_request_count;
            logic [31:0] unused_response_count, unused_context_count;
            logic [31:0] unused_result_count, unused_active_read_count;
            logic [3:0] unused_adapter_live_slots;
            logic [31:0] unused_adapter_bundle_request_count;
            logic [31:0] unused_adapter_bank_request_count;
            logic [31:0] unused_adapter_bank_response_count;
            logic [31:0] unused_adapter_bundle_response_count;

            // M499 is the exact-VCS canonical K1 endpoint.  Its outer adapter
            // waits one edge before reusing a retired slot, avoiding the
            // K1-only three-layer combinational ready loop found in M497.
            m519_fc2_k1_registered_release_8bank_raw4_acc24 #(
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS),
                .GENERATION_BITS(GENERATION_BITS),
                .SLICE_LANES(SLICE_LANES)
            ) implementation (
                `M519_COMMON_PORTS,
                .debug_fifo_count(unused_fifo_count),
                .debug_outstanding_count(unused_outstanding_count),
                .debug_group_accept_count(unused_group_count),
                .debug_request_accept_count(unused_request_count),
                .debug_response_accept_count(unused_response_count),
                .debug_context_write_count(unused_context_count),
                .debug_result_accept_count(unused_result_count),
                .debug_active_bank_read_count(unused_active_read_count),
                .debug_adapter_live_slots(unused_adapter_live_slots),
                .debug_adapter_bundle_request_count(
                    unused_adapter_bundle_request_count),
                .debug_adapter_bank_request_count(
                    unused_adapter_bank_request_count),
                .debug_adapter_bank_response_count(
                    unused_adapter_bank_response_count),
                .debug_adapter_bundle_response_count(
                    unused_adapter_bundle_response_count));
        end else if (ARCH_MODE == 1) begin : g_k8
            logic [5:0] unused_fifo_count;
            logic [6:0] unused_outstanding_count;
            logic [31:0] unused_group_count, unused_request_count;
            logic [31:0] unused_response_count, unused_context_count;
            logic [31:0] unused_result_count, unused_active_read_count;
            logic [3:0] unused_adapter_live_slots;
            logic [31:0] unused_adapter_bundle_request_count;
            logic [31:0] unused_adapter_bank_request_count;
            logic [31:0] unused_adapter_bank_response_count;
            logic [31:0] unused_adapter_bundle_response_count;

            m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24 #(
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS),
                .GENERATION_BITS(GENERATION_BITS),
                .SLICE_LANES(SLICE_LANES)
            ) implementation (
                `M519_COMMON_PORTS,
                .debug_fifo_count(unused_fifo_count),
                .debug_outstanding_count(unused_outstanding_count),
                .debug_group_accept_count(unused_group_count),
                .debug_request_accept_count(unused_request_count),
                .debug_response_accept_count(unused_response_count),
                .debug_context_write_count(unused_context_count),
                .debug_result_accept_count(unused_result_count),
                .debug_active_bank_read_count(unused_active_read_count),
                .debug_adapter_live_slots(unused_adapter_live_slots),
                .debug_adapter_bundle_request_count(
                    unused_adapter_bundle_request_count),
                .debug_adapter_bank_request_count(
                    unused_adapter_bank_request_count),
                .debug_adapter_bank_response_count(
                    unused_adapter_bank_response_count),
                .debug_adapter_bundle_response_count(
                    unused_adapter_bundle_response_count));
        end else if (ARCH_MODE == 2) begin : g_k1x8
            logic [5:0] unused_fifo_count;
            logic [6:0] unused_outstanding_count;
            logic [31:0] unused_group_count, unused_request_count;
            logic [31:0] unused_response_count, unused_context_count;
            logic [31:0] unused_result_count, unused_active_read_count;

            m519_fc2_k1x8_registered_release_raw4_acc24 #(
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS),
                .GENERATION_BITS(GENERATION_BITS),
                .SLICE_LANES(SLICE_LANES)
            ) implementation (
                `M519_COMMON_PORTS,
                .debug_fifo_count(unused_fifo_count),
                .debug_outstanding_count(unused_outstanding_count),
                .debug_group_accept_count(unused_group_count),
                .debug_request_accept_count(unused_request_count),
                .debug_response_accept_count(unused_response_count),
                .debug_context_write_count(unused_context_count),
                .debug_result_accept_count(unused_result_count),
                .debug_active_bank_read_count(unused_active_read_count));
        end else begin : g_illegal_mode
            initial $fatal(1, "M803 ARCH_MODE must be 0, 1, or 2");
        end
    endgenerate
`undef M519_COMMON_PORTS
endmodule

`default_nettype wire
