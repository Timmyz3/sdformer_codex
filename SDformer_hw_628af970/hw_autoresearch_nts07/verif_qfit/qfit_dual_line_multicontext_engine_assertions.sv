`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_multicontext_engine_assertions #(
    parameter int TILE_BITS = 256,
    parameter int ISSUE_WIDTH = 16,
    parameter int CONTEXTS = 4,
    parameter int OUT_LANES = 96,
    parameter int TAG_W = 32,
    parameter int OBJECT_W = 64,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int BANK_ADDR_W = 4,
    parameter int CTX_W = 2,
    parameter int COUNT_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic [TILE_BITS-1:0] command_source_bits,
    input logic [TILE_BITS-1:0] command_negative_bits,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [OBJECT_W-1:0] weight_request_object_tag,
    input logic [ISSUE_WIDTH-1:0] weight_request_bank_valid,
    input logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr,
    input logic [ISSUE_WIDTH*CTX_W-1:0] weight_request_bank_context,
    input logic [ISSUE_WIDTH-1:0] weight_request_bank_negative,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [OBJECT_W-1:0] output_object_tag,
    input logic output_use_motion,
    input logic [COUNT_W-1:0] output_source_count,
    input logic [OUT_LANES*ACC_W-1:0] output_acc,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    assert property (
        weight_request_valid && !weight_request_ready
        |=> weight_request_valid
            && $stable(weight_request_object_tag)
            && $stable(weight_request_bank_valid)
            && $stable(weight_request_bank_addr)
            && $stable(weight_request_bank_context)
            && $stable(weight_request_bank_negative)
    ) else $error("M3 weight request changed under backpressure");

    assert property (
        output_valid && !output_ready
        |=> output_valid && $stable({
            output_tag, output_object_tag, output_use_motion,
            output_source_count, output_acc
        })
    ) else $error("M3 output changed under backpressure");

    assert property (
        command_valid && command_ready
        |-> (command_negative_bits & ~command_source_bits) == '0
    ) else $error("M3 negative bitmap is not a selected-source subset");

    assert property (
        protocol_error |-> !command_ready && !weight_request_valid && !output_valid
    ) else $error("M3 fail-stop outputs remained enabled");

    generate
        for (genvar bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin : g_bank
            assert property (
                weight_request_valid && weight_request_bank_valid[bank]
                |-> weight_request_bank_context[bank*CTX_W +: CTX_W] < CONTEXTS
            ) else $error("M3 request used out-of-range context");
            assert property (
                weight_request_valid && weight_request_bank_negative[bank]
                |-> weight_request_bank_valid[bank]
            ) else $error("M3 negative flag appeared without a valid bank request");
        end
    endgenerate

    cover property (
        weight_request_valid && $countones(weight_request_bank_valid) >= 12
    );
    cover property (
        weight_request_valid && |weight_request_bank_negative
    );
    cover property (
        output_valid && !output_ready ##1 output_valid && !output_ready
    );
    cover property (
        command_valid && !command_ready ##1 command_valid && !command_ready
    );
endmodule

bind qfit_dual_line_multicontext_engine
    qfit_dual_line_multicontext_engine_assertions #(
        .TILE_BITS(TILE_BITS), .ISSUE_WIDTH(ISSUE_WIDTH), .CONTEXTS(CONTEXTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W), .BANK_ADDR_W(BANK_ADDR_W),
        .CTX_W(CTX_W), .COUNT_W(COUNT_W)
    ) u_qfit_dual_line_multicontext_engine_assertions (
        .clk_core, .rst_core,
        .command_valid, .command_ready,
        .command_source_bits, .command_negative_bits,
        .weight_request_valid, .weight_request_ready,
        .weight_request_object_tag, .weight_request_bank_valid,
        .weight_request_bank_addr, .weight_request_bank_context,
        .weight_request_bank_negative,
        .output_valid, .output_ready, .output_tag, .output_object_tag,
        .output_use_motion, .output_source_count, .output_acc,
        .protocol_error
    );

`default_nettype wire
