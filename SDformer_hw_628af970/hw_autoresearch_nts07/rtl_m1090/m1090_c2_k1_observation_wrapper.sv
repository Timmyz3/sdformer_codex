`timescale 1ns/1ps
`default_nettype none

// Observation-only shell around the frozen M1058 K1 implementation.  Every
// obs_* output is a fanout of an existing functional/debug signal and no
// observation signal feeds the implementation or any functional output.
module m1090_c2_k1_observation_wrapper #(
    parameter int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16,
    parameter int GENERATION_BITS=32, SLICE_LANES=16
) (
    input logic clk_core,input logic rst_core,
    input logic header_valid,output logic header_ready,
    input logic [TAG_BITS-1:0] header_tag,
    input logic [5:0] header_raw_beat_count,
    input logic [3:0] header_window_depth,input logic [3:0] header_output_blocks,
    output logic header_accept,
    input logic raw_valid,output logic raw_ready,input logic [3:0] raw_lane_valid,
    input logic [4:0] raw_beat_index[0:3],input logic [95:0] raw_bitmap[0:3],
    input logic raw_last,output logic raw_accept,
    output logic [7:0] mem_req_valid,input logic [7:0] mem_req_ready,
    output logic [EPOCH_BITS-1:0] mem_req_epoch[0:7],
    output logic [2:0] mem_req_slot[0:7],
    output logic [GENERATION_BITS-1:0] mem_req_generation[0:7],
    output logic [TAG_BITS-1:0] mem_req_tag[0:7],
    output logic [2:0] mem_req_output_block[0:7],output logic [2:0] mem_req_slice[0:7],
    output logic [CHANNEL_BITS-1:0] mem_req_source_channel[0:7],
    output logic [7:0] mem_req_accept,
    input logic [7:0] mem_rsp_valid,output logic [7:0] mem_rsp_ready,
    input logic [EPOCH_BITS-1:0] mem_rsp_epoch[0:7],
    input logic [2:0] mem_rsp_slot[0:7],
    input logic [GENERATION_BITS-1:0] mem_rsp_generation[0:7],
    input logic [TAG_BITS-1:0] mem_rsp_tag[0:7],
    input logic signed [7:0] mem_rsp_weight[0:7][0:SLICE_LANES-1],
    output logic [7:0] mem_rsp_accept,
    output logic result_valid,input logic result_ready,
    output logic [TAG_BITS-1:0] result_tag,
    output logic [2:0] result_output_block,output logic [2:0] result_slice,
    output logic signed [23:0] result_accumulator[0:SLICE_LANES-1],
    output logic result_last,output logic result_accept,
    output logic token_done_valid,input logic token_done_ready,
    output logic [TAG_BITS-1:0] token_done_tag,
    output logic token_done_had_event,output logic token_done_accept,
    output logic protocol_error,output logic numeric_overflow,
    output logic stale_response_seen,output logic busy,

    output logic obs_header_accept,output logic obs_raw_accept,
    output logic obs_busy,output logic obs_protocol_error,
    output logic obs_numeric_overflow,output logic obs_stale_response,
    output logic obs_fault,
    output logic [7:0] obs_bank_request_accept,
    output logic [7:0] obs_bank_response_accept,
    output logic [5:0] obs_service_fifo_count,
    output logic [6:0] obs_service_outstanding_count,
    output logic [31:0] obs_service_group_count,
    output logic [31:0] obs_service_request_count,
    output logic [31:0] obs_service_response_count,
    output logic [31:0] obs_service_context_count,
    output logic [31:0] obs_service_result_count,
    output logic [31:0] obs_service_active_bank_read_count,
    output logic [3:0] obs_adapter_live_slots,
    output logic [31:0] obs_adapter_bundle_request_count,
    output logic [31:0] obs_adapter_bank_request_count,
    output logic [31:0] obs_adapter_bank_response_count,
    output logic [31:0] obs_adapter_bundle_response_count
);
    logic [5:0] debug_fifo_count;
    logic [6:0] debug_outstanding_count;
    logic [31:0] debug_group_count,debug_request_count,debug_response_count;
    logic [31:0] debug_context_count,debug_result_count,debug_active_read_count;
    logic [3:0] debug_adapter_live_slots;
    logic [31:0] debug_adapter_bundle_request_count;
    logic [31:0] debug_adapter_bank_request_count;
    logic [31:0] debug_adapter_bank_response_count;
    logic [31:0] debug_adapter_bundle_response_count;

    m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24 #(
        .TAG_BITS(TAG_BITS),.CHANNEL_BITS(CHANNEL_BITS),.EPOCH_BITS(EPOCH_BITS),
        .GENERATION_BITS(GENERATION_BITS),.SLICE_LANES(SLICE_LANES)
    ) implementation (
        .clk_core(clk_core),.rst_core(rst_core),
        .header_valid(header_valid),.header_ready(header_ready),.header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),.header_accept(header_accept),
        .raw_valid(raw_valid),.raw_ready(raw_ready),.raw_lane_valid(raw_lane_valid),
        .raw_beat_index(raw_beat_index),.raw_bitmap(raw_bitmap),.raw_last(raw_last),
        .raw_accept(raw_accept),.mem_req_valid(mem_req_valid),
        .mem_req_ready(mem_req_ready),.mem_req_epoch(mem_req_epoch),
        .mem_req_slot(mem_req_slot),.mem_req_generation(mem_req_generation),
        .mem_req_tag(mem_req_tag),.mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),.mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(mem_req_accept),.mem_rsp_valid(mem_rsp_valid),
        .mem_rsp_ready(mem_rsp_ready),.mem_rsp_epoch(mem_rsp_epoch),
        .mem_rsp_slot(mem_rsp_slot),.mem_rsp_generation(mem_rsp_generation),
        .mem_rsp_tag(mem_rsp_tag),.mem_rsp_weight(mem_rsp_weight),
        .mem_rsp_accept(mem_rsp_accept),.result_valid(result_valid),
        .result_ready(result_ready),.result_tag(result_tag),
        .result_output_block(result_output_block),.result_slice(result_slice),
        .result_accumulator(result_accumulator),.result_last(result_last),
        .result_accept(result_accept),.token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),.token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),.protocol_error(protocol_error),
        .numeric_overflow(numeric_overflow),.stale_response_seen(stale_response_seen),
        .busy(busy),.debug_fifo_count(debug_fifo_count),
        .debug_outstanding_count(debug_outstanding_count),
        .debug_group_accept_count(debug_group_count),
        .debug_request_accept_count(debug_request_count),
        .debug_response_accept_count(debug_response_count),
        .debug_context_write_count(debug_context_count),
        .debug_result_accept_count(debug_result_count),
        .debug_active_bank_read_count(debug_active_read_count),
        .debug_adapter_live_slots(debug_adapter_live_slots),
        .debug_adapter_bundle_request_count(debug_adapter_bundle_request_count),
        .debug_adapter_bank_request_count(debug_adapter_bank_request_count),
        .debug_adapter_bank_response_count(debug_adapter_bank_response_count),
        .debug_adapter_bundle_response_count(debug_adapter_bundle_response_count));

    always_comb begin
        obs_header_accept=header_accept; obs_raw_accept=raw_accept;
        obs_busy=busy; obs_protocol_error=protocol_error;
        obs_numeric_overflow=numeric_overflow;
        obs_stale_response=stale_response_seen;
        obs_fault=protocol_error|numeric_overflow|stale_response_seen;
        obs_bank_request_accept=mem_req_accept;
        obs_bank_response_accept=mem_rsp_accept;
        obs_service_fifo_count=debug_fifo_count;
        obs_service_outstanding_count=debug_outstanding_count;
        obs_service_group_count=debug_group_count;
        obs_service_request_count=debug_request_count;
        obs_service_response_count=debug_response_count;
        obs_service_context_count=debug_context_count;
        obs_service_result_count=debug_result_count;
        obs_service_active_bank_read_count=debug_active_read_count;
        obs_adapter_live_slots=debug_adapter_live_slots;
        obs_adapter_bundle_request_count=debug_adapter_bundle_request_count;
        obs_adapter_bank_request_count=debug_adapter_bank_request_count;
        obs_adapter_bank_response_count=debug_adapter_bank_response_count;
        obs_adapter_bundle_response_count=debug_adapter_bundle_response_count;
    end
endmodule

`default_nettype wire
