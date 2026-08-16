`default_nettype none

module ttx_exp2_lut_q8 #(
    parameter int SCORE_W    = 16,
    parameter int SCORE_FRAC = 7
)(
    input  logic signed [SCORE_W-1:0] delta_q7,
    output logic [15:0] exp_q8
);
    logic [SCORE_W-1:0] abs_delta;
    logic [8:0] integer_shift;
    logic [3:0] fraction_index;
    logic [4:0] fraction_round;
    logic [15:0] fraction_value;
    logic [7:0] shift_amount;

    always_comb begin
        abs_delta = '0;
        integer_shift = '0;
        fraction_index = '0;
        fraction_round = '0;
        fraction_value = 16'd256;
        shift_amount = '0;

        if (delta_q7 >= 0) begin
            exp_q8 = 16'd256;
        end else begin
            abs_delta = -delta_q7;
            integer_shift = abs_delta[SCORE_W-1:SCORE_FRAC];
            fraction_round = {1'b0, abs_delta[SCORE_FRAC-1:SCORE_FRAC-4]}
                           + {{4{1'b0}}, |abs_delta[SCORE_FRAC-5:0]};
            fraction_index = fraction_round[4] ? 4'd15 : fraction_round[3:0];
            unique case (fraction_index)
                4'd0:  fraction_value = 16'd256;
                4'd1:  fraction_value = 16'd245;
                4'd2:  fraction_value = 16'd234;
                4'd3:  fraction_value = 16'd224;
                4'd4:  fraction_value = 16'd215;
                4'd5:  fraction_value = 16'd205;
                4'd6:  fraction_value = 16'd196;
                4'd7:  fraction_value = 16'd188;
                4'd8:  fraction_value = 16'd181;
                4'd9:  fraction_value = 16'd173;
                4'd10: fraction_value = 16'd165;
                4'd11: fraction_value = 16'd158;
                4'd12: fraction_value = 16'd152;
                4'd13: fraction_value = 16'd145;
                4'd14: fraction_value = 16'd139;
                default: fraction_value = 16'd133;
            endcase
            shift_amount = (integer_shift > 9'd8) ? 8'd8 : integer_shift[7:0];
            exp_q8 = fraction_value >> shift_amount;
        end
    end
endmodule

`default_nettype wire
