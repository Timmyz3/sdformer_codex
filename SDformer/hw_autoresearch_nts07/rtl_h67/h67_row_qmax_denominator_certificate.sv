`timescale 1ns/1ps
`default_nettype none

// Load-time sufficient condition for the frozen H67 T450 denominator shift.
// The raw-Q wrapper can be replaced by the count-input core when an upstream
// metadata builder already computes both Q popcounts.
module h67_row_qmax_denominator_certificate_core #(
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int QCOUNT_W = 6,
    parameter int QCOUNT_LIMIT = 15,
    parameter int CERTIFIED_SHIFT = 17
) (
    input  logic                    clk_core,
    input  logic                    rst_core,
    input  logic                    row_load_start,
    input  logic                    load_accept,
    input  logic [PAIR_ID_W-1:0]    load_pair_id,
    input  logic [QCOUNT_W-1:0]     load_qcount0,
    input  logic [QCOUNT_W-1:0]     load_qcount1,
    output logic                    certificate_valid,
    output logic                    certificate_pass,
    output logic [5:0]              denominator_shift,
    output logic [QCOUNT_W-1:0]     row_qcount_max,
    output logic [$clog2(PAIRS+1)-1:0] accepted_pairs,
    output logic                    protocol_error
);
    localparam int NEXT_W = $clog2(PAIRS + 1);

    logic [NEXT_W-1:0] next_pair_q;
    logic [QCOUNT_W-1:0] row_qcount_max_q;
    logic certificate_valid_q;
    logic certificate_pass_q;
    logic protocol_error_q;
    logic [NEXT_W-1:0] expected_pair_w;
    logic [QCOUNT_W-1:0] base_qcount_max_w;
    logic [QCOUNT_W-1:0] pair_qcount_max_w;
    logic [QCOUNT_W-1:0] next_qcount_max_w;
    logic load_id_legal;
    logic previous_row_incomplete;

    assign expected_pair_w = row_load_start ? '0 : next_pair_q;
    assign base_qcount_max_w = row_load_start ? '0 : row_qcount_max_q;
    assign pair_qcount_max_w = load_qcount0 >= load_qcount1
                             ? load_qcount0 : load_qcount1;
    assign next_qcount_max_w = base_qcount_max_w >= pair_qcount_max_w
                             ? base_qcount_max_w : pair_qcount_max_w;
    assign load_id_legal = !certificate_valid_q
                        && 32'(expected_pair_w) < 32'(PAIRS)
                        && 32'(load_pair_id) == 32'(expected_pair_w);
    assign previous_row_incomplete = next_pair_q != '0 && !certificate_valid_q;

    assign certificate_valid = certificate_valid_q;
    assign certificate_pass = certificate_valid_q && certificate_pass_q;
    assign denominator_shift = certificate_valid_q && certificate_pass_q
                             ? 6'(CERTIFIED_SHIFT) : 6'd0;
    assign row_qcount_max = row_qcount_max_q;
    assign accepted_pairs = next_pair_q;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            next_pair_q <= '0;
            row_qcount_max_q <= '0;
            certificate_valid_q <= 1'b0;
            certificate_pass_q <= 1'b0;
            protocol_error_q <= 1'b0;
        end else begin
            if (row_load_start) begin
                next_pair_q <= '0;
                row_qcount_max_q <= '0;
                certificate_valid_q <= 1'b0;
                certificate_pass_q <= 1'b0;
                protocol_error_q <= previous_row_incomplete;
            end

            if (load_accept) begin
                if (!load_id_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    next_pair_q <= expected_pair_w + 1'b1;
                    row_qcount_max_q <= next_qcount_max_w;
                    if (32'(load_pair_id) == 32'(PAIRS - 1)) begin
                        certificate_valid_q <= 1'b1;
                        certificate_pass_q <=
                            32'(next_qcount_max_w) <= QCOUNT_LIMIT;
                    end
                end
            end
        end
    end
endmodule

module h67_row_qkm_denominator_certificate_core #(
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int COUNT_W = 6,
    parameter int SCORE_BOUND_W = 8,
    parameter int SCORE_BOUND_LIMIT = 96,
    parameter int CERTIFIED_SHIFT = 17
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         row_load_start,
    input  logic                         load_accept,
    input  logic [PAIR_ID_W-1:0]         load_pair_id,
    input  logic [COUNT_W-1:0]           load_qcount0,
    input  logic [COUNT_W-1:0]           load_qcount1,
    input  logic [COUNT_W-1:0]           load_kcount0,
    input  logic [COUNT_W-1:0]           load_kcount1,
    input  logic [COUNT_W-1:0]           load_motion_count,
    output logic                         certificate_valid,
    output logic                         certificate_pass,
    output logic [5:0]                   denominator_shift,
    output logic [SCORE_BOUND_W-1:0]     row_score_upper_bound,
    output logic [$clog2(PAIRS+1)-1:0]   accepted_pairs,
    output logic                         protocol_error
);
    logic [SCORE_BOUND_W-1:0] score_upper0_w;
    logic [SCORE_BOUND_W-1:0] score_upper1_w;

    function automatic [SCORE_BOUND_W-1:0] score_upper_bound(
        input logic [COUNT_W-1:0] q_count,
        input logic [COUNT_W-1:0] k_count,
        input logic [COUNT_W-1:0] motion_count
    );
        logic [COUNT_W-1:0] overlap_upper;
        logic [COUNT_W-1:0] same_zero_upper;
        logic [COUNT_W-1:0] silence_integer;
        logic [3:0] silence_remainder;
        logic [SCORE_BOUND_W-1:0] score_integer;
        logic silence_increment;
        begin
            overlap_upper = q_count <= k_count ? q_count : k_count;
            same_zero_upper = COUNT_W'(32) - q_count - k_count
                            + overlap_upper;
            silence_integer = same_zero_upper >> 4;
            silence_remainder = same_zero_upper[3:0];
            score_integer = SCORE_BOUND_W'(overlap_upper * 4)
                          + SCORE_BOUND_W'(motion_count)
                          + SCORE_BOUND_W'(silence_integer);
            silence_increment = (silence_remainder > 4'd8)
                             || ((silence_remainder == 4'd8)
                                 && score_integer[0]);
            score_upper_bound = score_integer
                              + SCORE_BOUND_W'(silence_increment);
        end
    endfunction

    assign score_upper0_w = score_upper_bound(
        load_qcount0, load_kcount0, load_motion_count
    );
    assign score_upper1_w = score_upper_bound(
        load_qcount1, load_kcount1, load_motion_count
    );

    h67_row_qmax_denominator_certificate_core #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .QCOUNT_W(SCORE_BOUND_W),
        .QCOUNT_LIMIT(SCORE_BOUND_LIMIT),
        .CERTIFIED_SHIFT(CERTIFIED_SHIFT)
    ) u_bound_tracker (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_qcount0(score_upper0_w),
        .load_qcount1(score_upper1_w),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_qcount_max(row_score_upper_bound),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );
endmodule

module h67_row_qkm_denominator_certificate #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int SCORE_BOUND_W = 8,
    parameter int SCORE_BOUND_LIMIT = 96,
    parameter int CERTIFIED_SHIFT = 17
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         row_load_start,
    input  logic                         load_accept,
    input  logic [PAIR_ID_W-1:0]         load_pair_id,
    input  logic [2*HEAD_DIM-1:0]        load_q_pair,
    input  logic [2*HEAD_DIM-1:0]        load_k_pair,
    output logic                         certificate_valid,
    output logic                         certificate_pass,
    output logic [5:0]                   denominator_shift,
    output logic [SCORE_BOUND_W-1:0]     row_score_upper_bound,
    output logic [$clog2(PAIRS+1)-1:0]   accepted_pairs,
    output logic                         protocol_error
);
    logic [5:0] qcount0_w;
    logic [5:0] qcount1_w;
    logic [5:0] kcount0_w;
    logic [5:0] kcount1_w;
    logic [5:0] motion_count_w;

    initial begin
        if (HEAD_DIM != 32)
            $error("H67 QKM denominator certificate currently requires HEAD_DIM=32");
    end

    h67_balanced_popcount32 u_qcount0 (
        .bits(load_q_pair[31:0]), .count(qcount0_w)
    );
    h67_balanced_popcount32 u_qcount1 (
        .bits(load_q_pair[63:32]), .count(qcount1_w)
    );
    h67_balanced_popcount32 u_kcount0 (
        .bits(load_k_pair[31:0]), .count(kcount0_w)
    );
    h67_balanced_popcount32 u_kcount1 (
        .bits(load_k_pair[63:32]), .count(kcount1_w)
    );
    h67_balanced_popcount32 u_motion_count (
        .bits(load_k_pair[31:0] ^ load_k_pair[63:32]),
        .count(motion_count_w)
    );

    h67_row_qkm_denominator_certificate_core #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .SCORE_BOUND_W(SCORE_BOUND_W),
        .SCORE_BOUND_LIMIT(SCORE_BOUND_LIMIT),
        .CERTIFIED_SHIFT(CERTIFIED_SHIFT)
    ) u_core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_qcount0(qcount0_w),
        .load_qcount1(qcount1_w),
        .load_kcount0(kcount0_w),
        .load_kcount1(kcount1_w),
        .load_motion_count(motion_count_w),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_score_upper_bound(row_score_upper_bound),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );
endmodule

// Realistic integration point for the existing preload metadata builders:
// both Q counts are reused, while K0/K1 and temporal motion need three trees.
module h67_row_qkm_denominator_certificate_reuse_qcounts #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int SCORE_BOUND_W = 8,
    parameter int SCORE_BOUND_LIMIT = 96,
    parameter int CERTIFIED_SHIFT = 17
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         row_load_start,
    input  logic                         load_accept,
    input  logic [PAIR_ID_W-1:0]         load_pair_id,
    input  logic [5:0]                   load_qcount0,
    input  logic [5:0]                   load_qcount1,
    input  logic [2*HEAD_DIM-1:0]        load_k_pair,
    output logic                         certificate_valid,
    output logic                         certificate_pass,
    output logic [5:0]                   denominator_shift,
    output logic [SCORE_BOUND_W-1:0]     row_score_upper_bound,
    output logic [$clog2(PAIRS+1)-1:0]   accepted_pairs,
    output logic                         protocol_error
);
    logic [5:0] kcount0_w;
    logic [5:0] kcount1_w;
    logic [5:0] motion_count_w;

    initial begin
        if (HEAD_DIM != 32)
            $error("H67 reused-Q-count certificate currently requires HEAD_DIM=32");
    end

    h67_balanced_popcount32 u_kcount0 (
        .bits(load_k_pair[31:0]), .count(kcount0_w)
    );
    h67_balanced_popcount32 u_kcount1 (
        .bits(load_k_pair[63:32]), .count(kcount1_w)
    );
    h67_balanced_popcount32 u_motion_count (
        .bits(load_k_pair[31:0] ^ load_k_pair[63:32]),
        .count(motion_count_w)
    );

    h67_row_qkm_denominator_certificate_core #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .SCORE_BOUND_W(SCORE_BOUND_W),
        .SCORE_BOUND_LIMIT(SCORE_BOUND_LIMIT),
        .CERTIFIED_SHIFT(CERTIFIED_SHIFT)
    ) u_core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_qcount0(load_qcount0),
        .load_qcount1(load_qcount1),
        .load_kcount0(kcount0_w),
        .load_kcount1(kcount1_w),
        .load_motion_count(motion_count_w),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_score_upper_bound(row_score_upper_bound),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );
endmodule

module h67_row_qmax_denominator_certificate #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int QCOUNT_W = $clog2(HEAD_DIM + 1),
    parameter int QCOUNT_LIMIT = 15,
    parameter int CERTIFIED_SHIFT = 17
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       row_load_start,
    input  logic                       load_accept,
    input  logic [PAIR_ID_W-1:0]       load_pair_id,
    input  logic [2*HEAD_DIM-1:0]      load_q_pair,
    output logic                       certificate_valid,
    output logic                       certificate_pass,
    output logic [5:0]                 denominator_shift,
    output logic [QCOUNT_W-1:0]        row_qcount_max,
    output logic [$clog2(PAIRS+1)-1:0] accepted_pairs,
    output logic                       protocol_error
);
    logic [5:0] qcount0_w;
    logic [5:0] qcount1_w;

    initial begin
        if (HEAD_DIM != 32)
            $error("H67 denominator certificate currently requires HEAD_DIM=32");
        if (QCOUNT_W != 6)
            $error("H67 denominator certificate currently requires QCOUNT_W=6");
    end

    h67_balanced_popcount32 u_qcount0 (
        .bits(load_q_pair[31:0]),
        .count(qcount0_w)
    );
    h67_balanced_popcount32 u_qcount1 (
        .bits(load_q_pair[63:32]),
        .count(qcount1_w)
    );

    h67_row_qmax_denominator_certificate_core #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .QCOUNT_W(QCOUNT_W),
        .QCOUNT_LIMIT(QCOUNT_LIMIT),
        .CERTIFIED_SHIFT(CERTIFIED_SHIFT)
    ) u_core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .load_qcount0(qcount0_w),
        .load_qcount1(qcount1_w),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_qcount_max(row_qcount_max),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );
endmodule

`default_nettype wire
