`timescale 1ns/1ps
`default_nettype none

// Motion时间对在归一化域取商，在projection域按active mask重新展开。
// q_pair/k_pair低32位为t0，高32位为t1。
module h67_temporal_score_quotient #(
    parameter int HEAD_DIM = 32,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 9,
    parameter int COUNT_W = $clog2(HEAD_DIM + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [PAIR_ID_W-1:0]       in_pair_id,
    input  logic [2*HEAD_DIM-1:0]      in_q_pair,
    input  logic [2*HEAD_DIM-1:0]      in_k_pair,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic [PAIR_ID_W-1:0]       out_pair_id,
    output logic signed [SCORE_W-1:0]  out_score_q7,
    output logic [1:0]                 out_temporal_mask,
    output logic [1:0]                 out_active_mask,
    output logic                       out_last,

    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_equal_pairs
);
    logic signed [SCORE_W-1:0] score0_w;
    logic signed [SCORE_W-1:0] score1_w;
    logic [COUNT_W-1:0] unused_overlap0;
    logic [COUNT_W-1:0] unused_same_zero0;
    logic [COUNT_W-1:0] unused_motion0;
    logic [COUNT_W-1:0] unused_overlap1;
    logic [COUNT_W-1:0] unused_same_zero1;
    logic [COUNT_W-1:0] unused_motion1;

    logic valid_q;
    logic second_pending_q;
    logic [PAIR_ID_W-1:0] pair_id_q;
    logic signed [SCORE_W-1:0] score_q;
    logic signed [SCORE_W-1:0] second_score_q;
    logic [1:0] temporal_mask_q;
    logic [1:0] active_mask_q;
    logic second_active_q;
    logic [31:0] pairs_q;
    logic [31:0] descriptors_q;
    logic [31:0] equal_pairs_q;

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(1'b1)
    ) u_score0 (
        .q_bits(in_q_pair[HEAD_DIM-1:0]),
        .k_current_bits(in_k_pair[HEAD_DIM-1:0]),
        .k_peer_bits(in_k_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .overlap(unused_overlap0),
        .same_zero(unused_same_zero0),
        .motion_xor(unused_motion0),
        .score_q7(score0_w)
    );

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(1'b1)
    ) u_score1 (
        .q_bits(in_q_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_current_bits(in_k_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_peer_bits(in_k_pair[HEAD_DIM-1:0]),
        .overlap(unused_overlap1),
        .same_zero(unused_same_zero1),
        .motion_xor(unused_motion1),
        .score_q7(score1_w)
    );

    assign in_ready = !valid_q;
    assign out_valid = valid_q;
    assign out_pair_id = pair_id_q;
    assign out_score_q7 = score_q;
    assign out_temporal_mask = temporal_mask_q;
    assign out_active_mask = active_mask_q;
    assign out_last = valid_q && !second_pending_q;
    assign perf_pairs = pairs_q;
    assign perf_descriptors = descriptors_q;
    assign perf_equal_pairs = equal_pairs_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_q <= 1'b0;
            second_pending_q <= 1'b0;
            pair_id_q <= '0;
            score_q <= '0;
            second_score_q <= '0;
            temporal_mask_q <= '0;
            active_mask_q <= '0;
            second_active_q <= 1'b0;
            pairs_q <= '0;
            descriptors_q <= '0;
            equal_pairs_q <= '0;
        end else begin
            if (out_valid && out_ready) begin
                descriptors_q <= descriptors_q + 1'b1;
                if (second_pending_q) begin
                    score_q <= second_score_q;
                    temporal_mask_q <= 2'b10;
                    active_mask_q <= {second_active_q, 1'b0};
                    second_pending_q <= 1'b0;
                end else begin
                    valid_q <= 1'b0;
                end
            end

            if (in_valid && in_ready) begin
                valid_q <= 1'b1;
                pair_id_q <= in_pair_id;
                score_q <= score0_w;
                pairs_q <= pairs_q + 1'b1;
                if (score0_w == score1_w) begin
                    temporal_mask_q <= 2'b11;
                    active_mask_q <= {
                        |in_k_pair[2*HEAD_DIM-1:HEAD_DIM],
                        |in_k_pair[HEAD_DIM-1:0]
                    };
                    second_pending_q <= 1'b0;
                    second_score_q <= '0;
                    second_active_q <= 1'b0;
                    equal_pairs_q <= equal_pairs_q + 1'b1;
                end else begin
                    temporal_mask_q <= 2'b01;
                    active_mask_q <= {
                        1'b0,
                        |in_k_pair[HEAD_DIM-1:0]
                    };
                    second_pending_q <= 1'b1;
                    second_score_q <= score1_w;
                    second_active_q <= |in_k_pair[2*HEAD_DIM-1:HEAD_DIM];
                end
            end
        end
    end
endmodule

`default_nettype wire
