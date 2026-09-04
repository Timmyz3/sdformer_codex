`timescale 1ns/1ps
`default_nettype none

// 将严格顺序的T=2 temporal pair编码为一条或两条16-bit slot。
// QUOTIENT_ENABLE=0固定发两条；=1在两个score相等时可逆合并为一条。
module h67_temporal_slot_encoder #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int COUNT_W = $clog2(HEAD_DIM + 1),
    parameter bit QUOTIENT_ENABLE = 1'b1
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,

    input  logic                       pair_valid,
    output logic                       pair_ready,
    input  logic [PAIR_ID_W-1:0]       pair_id,
    input  logic [2*HEAD_DIM-1:0]      q_pair,
    input  logic [2*HEAD_DIM-1:0]      k_pair,

    output logic                       packet_valid,
    input  logic                       packet_ready,
    output logic [1:0]                 packet_slot_count,
    output logic [15:0]                packet_slot0,
    output logic [15:0]                packet_slot1,
    output logic                       pair_commit,

    output logic                       protocol_error,
    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_slots,
    output logic [31:0]                perf_equal_pairs
);
    localparam int NEXT_W = $clog2(PAIRS + 1);

    logic signed [SCORE_W-1:0] score0_w;
    logic signed [SCORE_W-1:0] score1_w;
    logic [COUNT_W-1:0] unused_overlap0;
    logic [COUNT_W-1:0] unused_same_zero0;
    logic [COUNT_W-1:0] unused_motion0;
    logic [COUNT_W-1:0] unused_overlap1;
    logic [COUNT_W-1:0] unused_same_zero1;
    logic [COUNT_W-1:0] unused_motion1;
    logic [NEXT_W-1:0] next_pair_q;
    logic id_legal;
    logic score_legal;
    logic pair_legal;
    logic score_equal;
    logic active0;
    logic active1;
    logic protocol_error_q;
    logic [31:0] pairs_q;
    logic [31:0] slots_q;
    logic [31:0] equal_pairs_q;

    h67_motionxor_score_q7 #(
        .HEAD_DIM(HEAD_DIM),
        .SCORE_W(SCORE_W),
        .COUNT_W(COUNT_W),
        .ENABLE_MOTION_XOR(1'b1)
    ) u_score0 (
        .q_bits(q_pair[HEAD_DIM-1:0]),
        .k_current_bits(k_pair[HEAD_DIM-1:0]),
        .k_peer_bits(k_pair[2*HEAD_DIM-1:HEAD_DIM]),
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
        .q_bits(q_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_current_bits(k_pair[2*HEAD_DIM-1:HEAD_DIM]),
        .k_peer_bits(k_pair[HEAD_DIM-1:0]),
        .overlap(unused_overlap1),
        .same_zero(unused_same_zero1),
        .motion_xor(unused_motion1),
        .score_q7(score1_w)
    );

    assign id_legal = 32'(next_pair_q) < 32'(PAIRS)
                   && 32'(pair_id) == 32'(next_pair_q);
    assign score_legal = score0_w >= 0
                      && score0_w <= $signed(SCORE_W'(255))
                      && score1_w >= 0
                      && score1_w <= $signed(SCORE_W'(255));
    assign pair_legal = id_legal && score_legal;
    assign score_equal = score0_w == score1_w;
    assign active0 = |k_pair[HEAD_DIM-1:0];
    assign active1 = |k_pair[2*HEAD_DIM-1:HEAD_DIM];

    assign packet_slot_count = (QUOTIENT_ENABLE && score_equal) ? 2'd1 : 2'd2;
    assign packet_slot0 = (QUOTIENT_ENABLE && score_equal)
        ? {3'b000, 1'b1, active1, active0, 2'b11, score0_w[7:0]}
        : {3'b000, 1'b0, 1'b0, active0, 2'b01, score0_w[7:0]};
    assign packet_slot1 = {3'b000, 1'b1, active1, 1'b0, 2'b10, score1_w[7:0]};
    assign packet_valid = pair_valid && pair_legal && !window_start;
    // 非法pair被握手并丢弃，避免错误输入把共享调度器永久堵死。
    assign pair_ready = !window_start
                      && (pair_legal ? packet_ready : 1'b1);
    assign pair_commit = packet_valid && packet_ready;
    assign protocol_error = protocol_error_q;
    assign perf_pairs = pairs_q;
    assign perf_slots = slots_q;
    assign perf_equal_pairs = equal_pairs_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            next_pair_q <= '0;
            protocol_error_q <= 1'b0;
            pairs_q <= '0;
            slots_q <= '0;
            equal_pairs_q <= '0;
        end else if (window_start) begin
            next_pair_q <= '0;
            protocol_error_q <= 1'b0;
            pairs_q <= '0;
            slots_q <= '0;
            equal_pairs_q <= '0;
        end else begin
            if (pair_valid && pair_ready && !pair_legal)
                protocol_error_q <= 1'b1;
            if (pair_commit) begin
                next_pair_q <= next_pair_q + 1'b1;
                pairs_q <= pairs_q + 1'b1;
                slots_q <= slots_q + 32'(packet_slot_count);
                if (score_equal)
                    equal_pairs_q <= equal_pairs_q + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
