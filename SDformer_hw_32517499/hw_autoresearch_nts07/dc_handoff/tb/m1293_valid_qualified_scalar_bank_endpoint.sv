`timescale 1ns/1ps
`default_nettype none

// Additive M1293 repair of the M1279 diagnostic endpoint.  Functional behavior
// is intentionally unchanged; the complete always_comb block is protected by
// the M1293 comment-insensitive token/block guard.
module m1293_valid_qualified_scalar_bank_endpoint #(
    parameter int BANK_ID = 0,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16,
    parameter int LATENCY = 4
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         enable,
    input  logic                         request_allow,
    input  logic                         newest_first,
    input  logic                         spurious_valid,
    input  logic                         mem_req_valid,
    output logic                         mem_req_ready,
    input  logic [EPOCH_BITS-1:0]        mem_req_epoch,
    input  logic [2:0]                   mem_req_slot,
    input  logic [GENERATION_BITS-1:0]   mem_req_generation,
    input  logic [TAG_BITS-1:0]          mem_req_tag,
    input  logic [2:0]                   mem_req_output_block,
    input  logic [2:0]                   mem_req_slice,
    input  logic [CHANNEL_BITS-1:0]      mem_req_source_channel,
    input  logic                         mem_req_accept,
    output logic                         endpoint_protocol_fault_now,
    output logic                         mem_rsp_valid,
    input  logic                         mem_rsp_ready,
    output logic [EPOCH_BITS-1:0]        mem_rsp_epoch,
    output logic [2:0]                   mem_rsp_slot,
    output logic [GENERATION_BITS-1:0]   mem_rsp_generation,
    output logic [TAG_BITS-1:0]          mem_rsp_tag,
    output logic signed [7:0]            mem_rsp_weight [0:SLICE_LANES-1],
    input  logic                         mem_rsp_accept,
    output logic [31:0]                  request_count,
    output logic [31:0]                  response_count,
    output logic [3:0]                   pending_count,
    output logic                         live_slot_reuse_error
);
    logic request_payload_known;
    logic qualified_request_valid;
    logic qualified_request_accept;
    logic inner_req_ready;

    always_comb begin : valid_qualified_guard
        request_payload_known = !$isunknown({mem_req_epoch, mem_req_slot,
            mem_req_generation, mem_req_tag, mem_req_output_block,
            mem_req_slice, mem_req_source_channel});
        endpoint_protocol_fault_now = 1'b0;
        qualified_request_valid = 1'b0;
        qualified_request_accept = 1'b0;
        mem_req_ready = 1'b0;

        if (mem_req_valid === 1'b1) begin
            if (request_payload_known) begin
                qualified_request_valid = 1'b1;
                mem_req_ready = inner_req_ready;
                if (mem_req_accept === 1'b1) begin
                    qualified_request_accept = 1'b1;
                end else if (mem_req_accept !== 1'b0) begin
                    endpoint_protocol_fault_now = 1'b1;
                end
            end else begin
                endpoint_protocol_fault_now = 1'b1;
            end
        end else if (mem_req_valid !== 1'b0) begin
            endpoint_protocol_fault_now = 1'b1;
        end
    end

    m349_fc2_scalar_bank_memory_model #(
        .BANK_ID(BANK_ID), .TAG_BITS(TAG_BITS),
        .CHANNEL_BITS(CHANNEL_BITS), .EPOCH_BITS(EPOCH_BITS),
        .GENERATION_BITS(GENERATION_BITS), .SLICE_LANES(SLICE_LANES),
        .LATENCY(LATENCY)
    ) inner (
        .clk_core(clk_core), .rst_core(rst_core), .enable(enable),
        .request_allow(request_allow), .newest_first(newest_first),
        .spurious_valid(spurious_valid),
        .mem_req_valid(qualified_request_valid),
        .mem_req_ready(inner_req_ready), .mem_req_epoch(mem_req_epoch),
        .mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation), .mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(qualified_request_accept),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight), .mem_rsp_accept(mem_rsp_accept),
        .request_count(request_count), .response_count(response_count),
        .pending_count(pending_count),
        .live_slot_reuse_error(live_slot_reuse_error));
endmodule

`default_nettype wire
