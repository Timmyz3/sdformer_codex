`timescale 1ns/1ps
`default_nettype none

// RQTB packet encoder with a joint five-sufficient-statistic score front end.
// Packet semantics match h67_temporal_slot_encoder exactly.
module h67_mssb5_temporal_slot_encoder #(
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
    logic [COUNT_W-1:0] unused_o0;
    logic [COUNT_W-1:0] unused_z0;
    logic [COUNT_W-1:0] unused_o1;
    logic [COUNT_W-1:0] unused_z1;
    logic [COUNT_W-1:0] unused_motion;
    logic [NEXT_W-1:0] next_pair_q;
    logic legal_w;
    logic equal_w;
    logic protocol_error_q;
    logic [31:0] pairs_q;
    logic [31:0] slots_q;
    logic [31:0] equal_q;

    h67_mssb5_score_pair #(
        .HEAD_DIM(HEAD_DIM), .SCORE_W(SCORE_W), .COUNT_W(COUNT_W)
    ) u_score (
        .q_pair(q_pair), .k_pair(k_pair),
        .overlap0(unused_o0), .same_zero0(unused_z0),
        .overlap1(unused_o1), .same_zero1(unused_z1),
        .motion(unused_motion), .score0_q7(score0_w),
        .score1_q7(score1_w)
    );

    assign legal_w = 32'(next_pair_q) < 32'(PAIRS)
                  && 32'(pair_id) == 32'(next_pair_q)
                  && score0_w >= 0
                  && score0_w <= $signed(SCORE_W'(255))
                  && score1_w >= 0
                  && score1_w <= $signed(SCORE_W'(255));
    assign equal_w = score0_w == score1_w;
    assign packet_slot_count = QUOTIENT_ENABLE && equal_w ? 2'd1 : 2'd2;
    assign packet_slot0 = QUOTIENT_ENABLE && equal_w
        ? {3'b000, 1'b1, |k_pair[2*HEAD_DIM-1:HEAD_DIM],
            |k_pair[HEAD_DIM-1:0], 2'b11,
            score0_w[7:0]}
        : {3'b000, 1'b0, 1'b0, |k_pair[HEAD_DIM-1:0], 2'b01,
            score0_w[7:0]};
    assign packet_slot1 = {3'b000, 1'b1,
        |k_pair[2*HEAD_DIM-1:HEAD_DIM], 1'b0, 2'b10, score1_w[7:0]};
    assign packet_valid = pair_valid && legal_w && !window_start;
    assign pair_ready = !window_start
                      && (legal_w ? packet_ready : 1'b1);
    assign pair_commit = packet_valid && packet_ready;
    assign protocol_error = protocol_error_q;
    assign perf_pairs = pairs_q;
    assign perf_slots = slots_q;
    assign perf_equal_pairs = equal_q;

    always_ff @(posedge clk_core) begin
        if (rst_core || window_start) begin
            next_pair_q <= '0;
            protocol_error_q <= 1'b0;
            pairs_q <= '0;
            slots_q <= '0;
            equal_q <= '0;
        end else begin
            if (pair_valid && pair_ready && !legal_w)
                protocol_error_q <= 1'b1;
            if (pair_commit) begin
                next_pair_q <= next_pair_q + 1'b1;
                pairs_q <= pairs_q + 1'b1;
                slots_q <= slots_q + 32'(packet_slot_count);
                if (equal_w)
                    equal_q <= equal_q + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
