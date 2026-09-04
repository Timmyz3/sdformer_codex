`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_descriptor_resident_engine_assertions #(
    parameter int TILE_BITS = 256,
    parameter int MAX_CHUNKS = 12,
    parameter int MAX_LANE_TILES = 32,
    parameter int ISSUE_WIDTH = 16,
    parameter int CONTEXTS = 4,
    parameter int REDUCE_SLOTS = 4,
    parameter int OUT_LANES = 96,
    parameter int TAG_W = 32,
    parameter int OBJECT_W = 64,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int BANK_ADDR_W = 4,
    parameter int CTX_W = 2,
    parameter int CTX_COUNT_W = 3,
    parameter int CHUNK_W = 4,
    parameter int LANE_TILE_W = 5,
    parameter int SLOT_W = 2,
    parameter int SOURCE_COUNT_W = 13
) (
    input logic clk_core,
    input logic rst_core,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic descriptor_use_motion,
    input logic [TILE_BITS-1:0] descriptor_source_bits,
    input logic [TILE_BITS-1:0] descriptor_negative_bits,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [OBJECT_W-1:0] weight_request_object_tag,
    input logic [CHUNK_W-1:0] weight_request_chunk_index,
    input logic [LANE_TILE_W-1:0] weight_request_lane_tile,
    input logic [ISSUE_WIDTH-1:0] weight_request_bank_valid,
    input logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr,
    input logic [ISSUE_WIDTH*CTX_W-1:0] weight_request_bank_context,
    input logic [ISSUE_WIDTH*SLOT_W-1:0] weight_request_bank_slot,
    input logic [ISSUE_WIDTH-1:0] weight_request_bank_negative,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [OBJECT_W-1:0] output_object_tag,
    input logic [LANE_TILE_W-1:0] output_lane_tile,
    input logic output_use_motion,
    input logic [SOURCE_COUNT_W-1:0] output_source_count,
    input logic [OUT_LANES*ACC_W-1:0] output_acc,
    input logic [CTX_COUNT_W-1:0] resident_contexts,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    assert property (
        weight_request_valid && !weight_request_ready
        |=> weight_request_valid && $stable({
            weight_request_object_tag, weight_request_chunk_index,
            weight_request_lane_tile, weight_request_bank_valid,
            weight_request_bank_addr, weight_request_bank_context,
            weight_request_bank_slot, weight_request_bank_negative
        })
    ) else $error("M4 weight request changed under backpressure");

    assert property (
        output_valid && !output_ready
        |=> output_valid && $stable({
            output_tag, output_object_tag, output_lane_tile,
            output_use_motion, output_source_count, output_acc
        })
    ) else $error("M4 output changed under backpressure");

    assert property (
        descriptor_valid && descriptor_ready
        |-> (descriptor_negative_bits & ~descriptor_source_bits) == '0
    ) else $error("M4 negative bitmap is not a selected-source subset");

    assert property (
        descriptor_valid && descriptor_ready && !descriptor_use_motion
        |-> descriptor_negative_bits == '0
    ) else $error("M4 Local descriptor carried negative sources");

    // The engine exposes one strictly ordered outstanding response slot.
    // A producer must not return a zero-latency or unsolicited response while
    // ready is low; response bank identity is checked by the RTL itself.
    assert property (
        weight_response_valid |-> weight_response_ready
    ) else $error("M4 weight response violated outstanding/latency contract");

    assert property (
        protocol_error |-> !descriptor_ready && !weight_request_valid && !output_valid
    ) else $error("M4 fail-stop outputs remained enabled");

    generate
        for (genvar bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin : g_bank
            assert property (
                weight_request_valid && weight_request_bank_valid[bank]
                |-> weight_request_bank_context[bank*CTX_W +: CTX_W] < resident_contexts
                    && weight_request_bank_slot[bank*SLOT_W +: SLOT_W] < REDUCE_SLOTS
            ) else $error("M4 request used out-of-range context/slot");
            assert property (
                weight_request_valid && weight_request_bank_negative[bank]
                |-> weight_request_bank_valid[bank]
            ) else $error("M4 negative flag appeared without a valid bank request");
            for (genvar other = bank + 1; other < ISSUE_WIDTH; other = other + 1) begin : g_other
                assert property (
                    weight_request_valid
                    && weight_request_bank_valid[bank]
                    && weight_request_bank_valid[other]
                    && weight_request_bank_context[bank*CTX_W +: CTX_W]
                        == weight_request_bank_context[other*CTX_W +: CTX_W]
                    |-> weight_request_bank_slot[bank*SLOT_W +: SLOT_W]
                        != weight_request_bank_slot[other*SLOT_W +: SLOT_W]
                ) else $error("M4 reducer slot was assigned twice in one context");
            end
        end
    endgenerate

    cover property (weight_request_valid && $countones(weight_request_bank_valid) >= 4);
    cover property (weight_request_valid && |weight_request_bank_negative);
    cover property (output_valid && output_lane_tile != '0);
    cover property (output_valid && !output_ready ##1 output_valid && !output_ready);
    cover property (descriptor_valid && !descriptor_ready ##1 descriptor_valid && !descriptor_ready);
endmodule

bind qfit_dual_line_descriptor_resident_engine
    qfit_dual_line_descriptor_resident_engine_assertions #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W), .BANK_ADDR_W(BANK_ADDR_W),
        .CTX_W(CTX_W), .CTX_COUNT_W(CTX_COUNT_W), .CHUNK_W(CHUNK_W),
        .LANE_TILE_W(LANE_TILE_W), .SLOT_W(SLOT_W),
        .SOURCE_COUNT_W(SOURCE_COUNT_W)
    ) u_qfit_dual_line_descriptor_resident_engine_assertions (
        .clk_core, .rst_core,
        .descriptor_valid, .descriptor_ready, .descriptor_use_motion,
        .descriptor_source_bits, .descriptor_negative_bits,
        .weight_request_valid, .weight_request_ready,
        .weight_request_object_tag, .weight_request_chunk_index,
        .weight_request_lane_tile, .weight_request_bank_valid,
        .weight_request_bank_addr, .weight_request_bank_context,
        .weight_request_bank_slot, .weight_request_bank_negative,
        .weight_response_valid, .weight_response_ready,
        .output_valid, .output_ready, .output_tag, .output_object_tag,
        .output_lane_tile, .output_use_motion, .output_source_count,
        .output_acc, .resident_contexts, .protocol_error
    );

`default_nettype wire
