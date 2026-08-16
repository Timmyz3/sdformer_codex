`ifndef GATESTACK_BIAS_SRAM_MODEL_SV
`define GATESTACK_BIAS_SRAM_MODEL_SV
`default_nettype none

module gatestack_bias_sram_model #(
    parameter int TAG_W = 32,
    parameter int OUTPUT_TILE_W = 8,
    parameter int TOKEN_ID_W = 8,
    parameter int OUT_TILE = 8,
    parameter int ACC_W = 32
) (
    input  logic                            clk_core,
    input  logic                            rst_core,
    input  logic                            req_allow,
    input  logic                            bias_req_valid,
    output logic                            bias_req_ready,
    input  logic [TAG_W-1:0]                bias_req_tag,
    input  logic [OUTPUT_TILE_W-1:0]        bias_req_output_tile,
    input  logic [TOKEN_ID_W-1:0]           bias_req_token_id,
    input  logic [(OUT_TILE*ACC_W)-1:0]     lookup_values,
    output logic                            bias_rsp_valid,
    input  logic                            bias_rsp_ready,
    output logic [TAG_W-1:0]                bias_rsp_tag,
    output logic [TOKEN_ID_W-1:0]           bias_rsp_token_id,
    output logic [(OUT_TILE*ACC_W)-1:0]     bias_rsp_values
);
    logic pending_q;
    logic delay_q;

    assign bias_req_ready = req_allow && !pending_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pending_q <= 1'b0;
            delay_q <= 1'b0;
            bias_rsp_valid <= 1'b0;
            bias_rsp_tag <= '0;
            bias_rsp_token_id <= '0;
            bias_rsp_values <= '0;
        end else begin
            if (bias_rsp_valid && bias_rsp_ready) begin
                pending_q <= 1'b0;
                bias_rsp_valid <= 1'b0;
            end
            if (delay_q) begin
                delay_q <= 1'b0;
                bias_rsp_valid <= 1'b1;
            end
            if (bias_req_valid && bias_req_ready) begin
                assert (!$isunknown(bias_req_output_tile));
                pending_q <= 1'b1;
                delay_q <= 1'b1;
                bias_rsp_tag <= bias_req_tag;
                bias_rsp_token_id <= bias_req_token_id;
                bias_rsp_values <= lookup_values;
            end
        end
    end
endmodule

`default_nettype wire
`endif
