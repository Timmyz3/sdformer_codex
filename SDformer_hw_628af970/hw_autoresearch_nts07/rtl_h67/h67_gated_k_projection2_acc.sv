`default_nettype none

// Two-channel projection checker for the gated-K stream. This module is a
// synthesizable numeric boundary, not the throughput-oriented DCTF backend.
module h67_gated_k_projection2_acc #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9
) (
    input  logic                         clk,
    input  logic                         rst_core,
    input  logic                         row_start,
    input  logic                         row_done,
    input  logic                         in_fire,
    input  logic                         in_last,
    input  logic [HEAD_DIM-1:0]          in_k_bits,
    input  logic [GATE_W-1:0]            in_gate_q17,
    input  logic [HEAD_DIM*8-1:0]        weight0_flat,
    input  logic [HEAD_DIM*8-1:0]        weight1_flat,
    output logic                         result_valid,
    output logic signed [31:0]           result0_acc32,
    output logic signed [31:0]           result1_acc32
);
    logic signed [31:0] acc0_q;
    logic signed [31:0] acc1_q;
    logic signed [31:0] contribution0_w;
    logic signed [31:0] contribution1_w;
    logic signed [31:0] next_acc0_w;
    logic signed [31:0] next_acc1_w;
    logic saw_input_q;
    logic result_emitted_q;
    integer lane;

    always_comb begin
        contribution0_w = '0;
        contribution1_w = '0;
        for (lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            if (in_k_bits[lane]) begin
                contribution0_w = contribution0_w
                    + $signed(weight0_flat[lane*8 +: 8])
                    * $signed({1'b0, in_gate_q17});
                contribution1_w = contribution1_w
                    + $signed(weight1_flat[lane*8 +: 8])
                    * $signed({1'b0, in_gate_q17});
            end
        end
        next_acc0_w = acc0_q + contribution0_w;
        next_acc1_w = acc1_q + contribution1_w;
    end

    always_ff @(posedge clk) begin
        if (rst_core) begin
            acc0_q <= '0;
            acc1_q <= '0;
            saw_input_q <= 1'b0;
            result_emitted_q <= 1'b0;
            result_valid <= 1'b0;
            result0_acc32 <= '0;
            result1_acc32 <= '0;
        end else begin
            result_valid <= 1'b0;
            if (row_start) begin
                acc0_q <= '0;
                acc1_q <= '0;
                saw_input_q <= 1'b0;
                result_emitted_q <= 1'b0;
            end else if (in_fire) begin
                acc0_q <= next_acc0_w;
                acc1_q <= next_acc1_w;
                saw_input_q <= 1'b1;
                if (in_last) begin
                    result_valid <= 1'b1;
                    result_emitted_q <= 1'b1;
                    result0_acc32 <= next_acc0_w;
                    result1_acc32 <= next_acc1_w;
                end
            end else if (row_done && !saw_input_q && !result_emitted_q) begin
                result_valid <= 1'b1;
                result_emitted_q <= 1'b1;
                result0_acc32 <= acc0_q;
                result1_acc32 <= acc1_q;
            end
        end
    end
endmodule

`default_nettype wire
