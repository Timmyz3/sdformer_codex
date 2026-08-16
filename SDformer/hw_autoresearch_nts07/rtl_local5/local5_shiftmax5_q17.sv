`timescale 1ns/1ps
`default_nettype none

// 5-candidate RTL Shiftmax (matches software _rtl_shiftmax_gate_q17):
// Q7 scores -> row max -> 16-entry Q8 exp2 LUT -> integer sum -> ceil-pow2
// normalize -> unsigned Q1.7 RNE gate saturated to [0, 2].
// preserve_mean is fixed 0 for H66d (Local-5).
module local5_shiftmax5_q17 #(
    parameter int N_CAND     = 5,
    parameter int SCORE_W    = 16,
    parameter int GATE_W     = 9
)(
    input  logic [N_CAND*SCORE_W-1:0] score_q7,
    input  logic [N_CAND-1:0]         valid,
    output logic [N_CAND*GATE_W-1:0]  gate_q17
);
    localparam int SCORE_MIN_Q7 = -256; // -2.0 in Q7

    integer i;
    logic signed [SCORE_W-1:0] score_q7_i [0:N_CAND-1];
    logic signed [SCORE_W-1:0] score_use [0:N_CAND-1];
    logic signed [SCORE_W-1:0] row_max;
    logic signed [SCORE_W:0]   delta;
    logic [SCORE_W:0]          abs_delta;
    logic [3:0]                integer_shift;
    logic [6:0]                frac_q7;
    logic [3:0]                frac_idx;
    logic [7:0]                frac_idx_full;
    logic [8:0]                lut_val;
    logic [8:0]                exp_q8 [0:N_CAND-1];
    logic [15:0]               row_sum_q8;
    logic [5:0]                den_shift;
    logic [15:0]               probe;
    logic [23:0]               scaled;
    logic [23:0]               quotient;
    logic [23:0]               remainder;
    logic [23:0]               half;
    logic [23:0]               gate_rounded;

    // 16-entry Q8 exp2 LUT (same as TTX/H67 RTL)
    function automatic logic [8:0] exp2_lut(input logic [3:0] idx);
        case (idx)
            4'd0:  exp2_lut = 9'd256;
            4'd1:  exp2_lut = 9'd245;
            4'd2:  exp2_lut = 9'd234;
            4'd3:  exp2_lut = 9'd224;
            4'd4:  exp2_lut = 9'd215;
            4'd5:  exp2_lut = 9'd205;
            4'd6:  exp2_lut = 9'd196;
            4'd7:  exp2_lut = 9'd188;
            4'd8:  exp2_lut = 9'd181;
            4'd9:  exp2_lut = 9'd173;
            4'd10: exp2_lut = 9'd165;
            4'd11: exp2_lut = 9'd158;
            4'd12: exp2_lut = 9'd152;
            4'd13: exp2_lut = 9'd145;
            4'd14: exp2_lut = 9'd139;
            default: exp2_lut = 9'd133;
        endcase
    endfunction

    always_comb begin
        row_max = SCORE_W'(SCORE_MIN_Q7);
        for (i = 0; i < N_CAND; i = i + 1) begin
            score_q7_i[i] = signed'(score_q7[i*SCORE_W +: SCORE_W]);
            score_use[i] = valid[i] ? score_q7_i[i] : SCORE_W'(SCORE_MIN_Q7);
            if (valid[i] && score_use[i] > row_max)
                row_max = score_use[i];
        end

        row_sum_q8 = '0;
        for (i = 0; i < N_CAND; i = i + 1) begin
            delta = SCORE_W'(score_use[i]) - SCORE_W'(row_max);
            abs_delta = (-delta);
            // Keep the 8.0-Q7 threshold wider than narrow SCORE_W instances.
            if (abs_delta >= 11'd1024)
                integer_shift = 4'd8;
            else
                integer_shift = 4'(abs_delta >> 7);
            frac_q7 = abs_delta[6:0];
            frac_idx_full = ({1'b0, frac_q7} + 8'd7) >> 3;
            if (frac_idx_full > 8'd15)
                frac_idx = 4'd15;
            else
                frac_idx = frac_idx_full[3:0];
            lut_val = exp2_lut(frac_idx);
            exp_q8[i] = valid[i] ? (lut_val >> integer_shift) : 9'd0;
            row_sum_q8 = row_sum_q8 + {7'b0, exp_q8[i]};
        end

        // ceil_log2(row_sum): den_shift = floor(log2(row_sum-1))+1 for row_sum>0
        probe = (row_sum_q8 == 0) ? 16'd0 : (row_sum_q8 - 16'd1);
        den_shift = '0;
        for (i = 0; i < 16; i = i + 1) begin
            if (probe != 0) begin
                den_shift = den_shift + 6'd1;
                probe = probe >> 1;
            end
        end

        for (i = 0; i < N_CAND; i = i + 1) begin
            // gate = exp / 2^den_shift * 128, RNE
            scaled = {8'b0, exp_q8[i], 7'b0};
            if (den_shift == 0) begin
                quotient = scaled;
                remainder = '0;
                half = '0;
            end else begin
                quotient = scaled >> den_shift;
                remainder = scaled - (quotient << den_shift);
                half = 24'd1 << (den_shift - 1);
            end
            gate_rounded = quotient;
            if (den_shift != 0
                && (remainder > half
                    || (remainder == half && quotient[0]))) begin
                gate_rounded = quotient + 24'd1;
            end
            if (!valid[i])
                gate_q17[i*GATE_W +: GATE_W] = '0;
            else if (gate_rounded > 24'd256)
                gate_q17[i*GATE_W +: GATE_W] = GATE_W'(256);
            else
                gate_q17[i*GATE_W +: GATE_W] = GATE_W'(gate_rounded);
        end
    end
endmodule

`default_nettype wire
