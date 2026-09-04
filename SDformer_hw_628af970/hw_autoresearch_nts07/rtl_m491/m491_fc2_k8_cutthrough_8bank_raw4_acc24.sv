`timescale 1ns/1ps
`default_nettype none

// M491 adds a response cut-through path to the canonical shared-state FC2 top.
// The arithmetic/control core is the frozen M342 SOURCE_CAP=8 composition.
// M490 lowers its atomic bank bundle into eight independent scalar SRAM ports,
// reassembles responses by slot identity, and forwards a newly completed bundle
// on its last-bank-arrival edge when the core is ready.
// Thus this top and M349 K1x8 expose the same physical memory-port shape.
module m491_fc2_k8_cutthrough_8bank_raw4_acc24 #(
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
    output logic                         busy,
    output logic [5:0]                   debug_fifo_count,
    output logic [6:0]                   debug_outstanding_count,
    output logic [31:0]                  debug_group_accept_count,
    output logic [31:0]                  debug_request_accept_count,
    output logic [31:0]                  debug_response_accept_count,
    output logic [31:0]                  debug_context_write_count,
    output logic [31:0]                  debug_result_accept_count,
    output logic [31:0]                  debug_active_bank_read_count,
    output logic [3:0]                   debug_adapter_live_slots,
    output logic [31:0]                  debug_adapter_bundle_request_count,
    output logic [31:0]                  debug_adapter_bank_request_count,
    output logic [31:0]                  debug_adapter_bank_response_count,
    output logic [31:0]                  debug_adapter_bundle_response_count
);
    localparam bit PARAMETERS_LEGAL = SLICE_LANES == 16;

    logic core_mem_req_valid, core_mem_req_ready, core_mem_req_accept;
    logic [EPOCH_BITS-1:0] core_mem_req_epoch;
    logic [2:0] core_mem_req_slot;
    logic [GENERATION_BITS-1:0] core_mem_req_generation;
    logic [TAG_BITS-1:0] core_mem_req_tag;
    logic [2:0] core_mem_req_output_block, core_mem_req_slice;
    logic [3:0] core_mem_req_source_count;
    logic [7:0] core_mem_req_bank_valid;
    logic [CHANNEL_BITS-1:0] core_mem_req_source_channel [0:7];
    logic adapter_core_mem_req_accept;

    logic core_mem_rsp_valid, core_mem_rsp_ready, core_mem_rsp_accept;
    logic [EPOCH_BITS-1:0] core_mem_rsp_epoch;
    logic [2:0] core_mem_rsp_slot;
    logic [GENERATION_BITS-1:0] core_mem_rsp_generation;
    logic [TAG_BITS-1:0] core_mem_rsp_tag;
    logic [7:0] core_mem_rsp_bank_valid;
    logic signed [7:0] core_mem_rsp_weight [0:7][0:SLICE_LANES-1];
    logic adapter_core_mem_rsp_accept;

    logic core_protocol_error, core_stale_response_seen, core_busy;
    logic adapter_protocol_error, adapter_stale_response_seen, adapter_busy;
    logic consistency_fault_q, consistency_fault_now;
    logic [2:0] core_debug_fifo_count;
    logic [3:0] core_debug_outstanding_count;

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M491 supports only SLICE_LANES=16");
        end
    endgenerate

    always_comb begin
        consistency_fault_now = 0;
        if (core_mem_req_accept != adapter_core_mem_req_accept)
            consistency_fault_now = 1;
        if (core_mem_rsp_accept != adapter_core_mem_rsp_accept)
            consistency_fault_now = 1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) consistency_fault_q <= 0;
        else if (consistency_fault_now) consistency_fault_q <= 1;
    end

    assign protocol_error = core_protocol_error || adapter_protocol_error
        || consistency_fault_q || consistency_fault_now;
    assign stale_response_seen = core_stale_response_seen
        || adapter_stale_response_seen;
    assign busy = core_busy || adapter_busy;
    assign debug_fifo_count = {3'b000, core_debug_fifo_count};
    assign debug_outstanding_count = {3'b000, core_debug_outstanding_count};

    m342_fc2_standalone_raw4_acc24 #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SOURCE_CAP(8), .SLICE_LANES(SLICE_LANES)
    ) core (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index),
        .raw_bitmap(raw_bitmap), .raw_last(raw_last),
        .raw_accept(raw_accept),
        .mem_req_valid(core_mem_req_valid),
        .mem_req_ready(core_mem_req_ready),
        .mem_req_epoch(core_mem_req_epoch),
        .mem_req_slot(core_mem_req_slot),
        .mem_req_generation(core_mem_req_generation),
        .mem_req_tag(core_mem_req_tag),
        .mem_req_output_block(core_mem_req_output_block),
        .mem_req_slice(core_mem_req_slice),
        .mem_req_source_count(core_mem_req_source_count),
        .mem_req_bank_valid(core_mem_req_bank_valid),
        .mem_req_source_channel(core_mem_req_source_channel),
        .mem_req_accept(core_mem_req_accept),
        .mem_rsp_valid(core_mem_rsp_valid),
        .mem_rsp_ready(core_mem_rsp_ready),
        .mem_rsp_epoch(core_mem_rsp_epoch),
        .mem_rsp_slot(core_mem_rsp_slot),
        .mem_rsp_generation(core_mem_rsp_generation),
        .mem_rsp_tag(core_mem_rsp_tag),
        .mem_rsp_bank_valid(core_mem_rsp_bank_valid),
        .mem_rsp_weight(core_mem_rsp_weight),
        .mem_rsp_accept(core_mem_rsp_accept),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_tag(result_tag),
        .result_output_block(result_output_block),
        .result_slice(result_slice),
        .result_accumulator(result_accumulator),
        .result_last(result_last), .result_accept(result_accept),
        .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),
        .token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),
        .protocol_error(core_protocol_error),
        .numeric_overflow(numeric_overflow),
        .stale_response_seen(core_stale_response_seen),
        .busy(core_busy), .debug_fifo_count(core_debug_fifo_count),
        .debug_outstanding_count(core_debug_outstanding_count),
        .debug_group_accept_count(debug_group_accept_count),
        .debug_request_accept_count(debug_request_accept_count),
        .debug_response_accept_count(debug_response_accept_count),
        .debug_context_write_count(debug_context_write_count),
        .debug_result_accept_count(debug_result_accept_count),
        .debug_active_bank_read_count(debug_active_bank_read_count));

    m490_fc2_bundle_to_8bank_cutthrough_adapter #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .OUTSTANDING(8), .SLICE_LANES(SLICE_LANES)
    ) memory_adapter (
        .clk_core(clk_core), .rst_core(rst_core),
        .core_req_valid(core_mem_req_valid),
        .core_req_ready(core_mem_req_ready),
        .core_req_epoch(core_mem_req_epoch),
        .core_req_slot(core_mem_req_slot),
        .core_req_generation(core_mem_req_generation),
        .core_req_tag(core_mem_req_tag),
        .core_req_output_block(core_mem_req_output_block),
        .core_req_slice(core_mem_req_slice),
        .core_req_source_count(core_mem_req_source_count),
        .core_req_bank_valid(core_mem_req_bank_valid),
        .core_req_source_channel(core_mem_req_source_channel),
        .core_req_accept(adapter_core_mem_req_accept),
        .bank_req_valid(mem_req_valid), .bank_req_ready(mem_req_ready),
        .bank_req_epoch(mem_req_epoch), .bank_req_slot(mem_req_slot),
        .bank_req_generation(mem_req_generation),
        .bank_req_tag(mem_req_tag),
        .bank_req_output_block(mem_req_output_block),
        .bank_req_slice(mem_req_slice),
        .bank_req_source_channel(mem_req_source_channel),
        .bank_req_accept(mem_req_accept),
        .bank_rsp_valid(mem_rsp_valid), .bank_rsp_ready(mem_rsp_ready),
        .bank_rsp_epoch(mem_rsp_epoch), .bank_rsp_slot(mem_rsp_slot),
        .bank_rsp_generation(mem_rsp_generation),
        .bank_rsp_tag(mem_rsp_tag), .bank_rsp_weight(mem_rsp_weight),
        .bank_rsp_accept(mem_rsp_accept),
        .core_rsp_valid(core_mem_rsp_valid),
        .core_rsp_ready(core_mem_rsp_ready),
        .core_rsp_epoch(core_mem_rsp_epoch),
        .core_rsp_slot(core_mem_rsp_slot),
        .core_rsp_generation(core_mem_rsp_generation),
        .core_rsp_tag(core_mem_rsp_tag),
        .core_rsp_bank_valid(core_mem_rsp_bank_valid),
        .core_rsp_weight(core_mem_rsp_weight),
        .core_rsp_accept(adapter_core_mem_rsp_accept),
        .protocol_error(adapter_protocol_error),
        .stale_response_seen(adapter_stale_response_seen),
        .busy(adapter_busy),
        .debug_live_slots(debug_adapter_live_slots),
        .debug_bundle_request_count(
            debug_adapter_bundle_request_count),
        .debug_bank_request_count(debug_adapter_bank_request_count),
        .debug_bank_response_count(debug_adapter_bank_response_count),
        .debug_bundle_response_count(
            debug_adapter_bundle_response_count));
endmodule

`default_nettype wire
