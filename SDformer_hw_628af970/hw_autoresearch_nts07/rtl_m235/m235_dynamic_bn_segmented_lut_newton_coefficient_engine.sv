`timescale 1ns/1ps
`default_nettype none

// M235 consumes finalized current-batch mean/variance plus checkpoint gamma/beta
// and emits invstd/alpha/offset.  One scalar multiplier is reused across three
// Newton products and two affine products.  Moment sum/sumsq division is an
// explicit upstream boundary.
module m235_dynamic_bn_segmented_lut_newton_coefficient_engine #(
    parameter int TAG_BITS = 24
) (
    input  logic                    clk_core,
    input  logic                    rst_core,

    input  logic                    request_valid,
    output logic                    request_ready,
    input  logic [TAG_BITS-1:0]     request_tag,
    input  logic [21:0]             variance_plus_epsilon_uq6p16,
    input  logic signed [17:0]      mean_sq3p14,
    input  logic signed [15:0]      gamma_sq1p14,
    input  logic signed [15:0]      beta_sq1p14,
    output logic                    request_accept,

    output logic                    result_valid,
    input  logic                    result_ready,
    output logic [TAG_BITS-1:0]     result_tag,
    output logic [19:0]             invstd_uq4p16,
    output logic signed [19:0]      alpha_sq3p16,
    output logic signed [19:0]      offset_sq3p16,
    output logic                    result_accept,

    output logic                    protocol_error,
    output logic                    busy,
    output logic [3:0]              debug_state,
    output logic [31:0]             debug_request_count,
    output logic [31:0]             debug_result_count
);
    localparam logic [3:0] S_IDLE   = 4'd0;
    localparam logic [3:0] S_LUT    = 4'd1;
    localparam logic [3:0] S_Y2     = 4'd2;
    localparam logic [3:0] S_MY2    = 4'd3;
    localparam logic [3:0] S_YNEW   = 4'd4;
    localparam logic [3:0] S_SCALE  = 4'd5;
    localparam logic [3:0] S_ALPHA  = 4'd6;
    localparam logic [3:0] S_OFFSET = 4'd7;
    localparam logic [3:0] S_RESULT = 4'd8;

    logic [3:0] state_q;
    logic fault_q, result_valid_q;
    logic [TAG_BITS-1:0] tag_q, result_tag_q;
    logic signed [5:0] even_exponent_q;
    logic [17:0] mantissa_q;
    logic [5:0] lut_index_q;
    logic [19:0] y_q;
    logic [21:0] y2_q, my2_q;
    logic [19:0] invstd_q;
    logic signed [19:0] alpha_q, offset_q;
    logic signed [17:0] mean_q;
    logic signed [15:0] gamma_q, beta_q;
    logic [31:0] request_count_q, result_count_q;

    logic request_legal, illegal_request;
    logic signed [21:0] multiplier_a, multiplier_b;
    wire signed [43:0] multiplier_product = multiplier_a * multiplier_b;
    logic [21:0] half_my2_w, term_w;
    logic [19:0] scaled_invstd_w;
    logic signed [19:0] alpha_w, offset_w;
    logic [63:0] rounded_half_my2_w;
    logic signed [43:0] rounded_y2_w, rounded_my2_w, rounded_ynew_w;

    logic signed [5:0] normalized_even_exponent_w;
    logic [17:0] normalized_mantissa_w;
    logic [5:0] normalized_lut_index_w;

    function automatic logic [63:0] rshift_rne_u64(
        input logic [63:0] value, input integer shift);
        logic [63:0] quotient, remainder, mask, half;
        begin
            if (shift <= 0) begin
                rshift_rne_u64 = value;
            end else begin
                mask = (64'h1 << shift) - 1'b1;
                half = 64'h1 << (shift-1);
                quotient = value >> shift;
                remainder = value & mask;
                rshift_rne_u64 = quotient
                    + ((remainder > half)
                       || ((remainder == half) && quotient[0]));
            end
        end
    endfunction

    function automatic logic signed [43:0] rshift_rne_s44(
        input logic signed [43:0] value, input integer shift);
        logic negative;
        logic [43:0] magnitude, quotient, remainder, mask, half, rounded;
        begin
            negative = value < 0;
            magnitude = negative ? $unsigned(-value) : $unsigned(value);
            if (shift <= 0) begin
                rounded = magnitude;
            end else begin
                mask = (44'h1 << shift) - 1'b1;
                half = 44'h1 << (shift-1);
                quotient = magnitude >> shift;
                remainder = magnitude & mask;
                rounded = quotient
                    + ((remainder > half)
                       || ((remainder == half) && quotient[0]));
            end
            rshift_rne_s44 = negative ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    function automatic logic [19:0] saturate_u20(
        input logic signed [43:0] value);
        begin
            if (value < 0)
                saturate_u20 = '0;
            else if ($unsigned(value) > 44'h000000fffff)
                saturate_u20 = 20'hfffff;
            else
                saturate_u20 = value[19:0];
        end
    endfunction

    function automatic logic signed [19:0] saturate_s20(
        input logic signed [43:0] value);
        begin
            if (value > 44'sd524287)
                saturate_s20 = 20'sh7ffff;
            else if (value < -44'sd524288)
                saturate_s20 = 20'sh80000;
            else
                saturate_s20 = value[19:0];
        end
    endfunction

    function automatic logic [18:0] rsqrt_lut_uq1p18(
        input logic [5:0] index);
        begin
            case (index)
                6'd0: rsqrt_lut_uq1p18=19'h3f818;
                6'd1: rsqrt_lut_uq1p18=19'h3e8d0;
                6'd2: rsqrt_lut_uq1p18=19'h3da33;
                6'd3: rsqrt_lut_uq1p18=19'h3cc36;
                6'd4: rsqrt_lut_uq1p18=19'h3becd;
                6'd5: rsqrt_lut_uq1p18=19'h3b1ee;
                6'd6: rsqrt_lut_uq1p18=19'h3a591;
                6'd7: rsqrt_lut_uq1p18=19'h399ac;
                6'd8: rsqrt_lut_uq1p18=19'h38e39;
                6'd9: rsqrt_lut_uq1p18=19'h38330;
                6'd10: rsqrt_lut_uq1p18=19'h3788c;
                6'd11: rsqrt_lut_uq1p18=19'h36e46;
                6'd12: rsqrt_lut_uq1p18=19'h3645a;
                6'd13: rsqrt_lut_uq1p18=19'h35ac1;
                6'd14: rsqrt_lut_uq1p18=19'h35179;
                6'd15: rsqrt_lut_uq1p18=19'h3487b;
                6'd16: rsqrt_lut_uq1p18=19'h33fc6;
                6'd17: rsqrt_lut_uq1p18=19'h33754;
                6'd18: rsqrt_lut_uq1p18=19'h32f22;
                6'd19: rsqrt_lut_uq1p18=19'h3272f;
                6'd20: rsqrt_lut_uq1p18=19'h31f75;
                6'd21: rsqrt_lut_uq1p18=19'h317f3;
                6'd22: rsqrt_lut_uq1p18=19'h310a7;
                6'd23: rsqrt_lut_uq1p18=19'h3098d;
                6'd24: rsqrt_lut_uq1p18=19'h302a3;
                6'd25: rsqrt_lut_uq1p18=19'h2fbe8;
                6'd26: rsqrt_lut_uq1p18=19'h2f55a;
                6'd27: rsqrt_lut_uq1p18=19'h2eef6;
                6'd28: rsqrt_lut_uq1p18=19'h2e8ba;
                6'd29: rsqrt_lut_uq1p18=19'h2e2a6;
                6'd30: rsqrt_lut_uq1p18=19'h2dcb7;
                6'd31: rsqrt_lut_uq1p18=19'h2d6ec;
                6'd32: rsqrt_lut_uq1p18=19'h2ce7c;
                6'd33: rsqrt_lut_uq1p18=19'h2c3ae;
                6'd34: rsqrt_lut_uq1p18=19'h2b959;
                6'd35: rsqrt_lut_uq1p18=19'h2af75;
                6'd36: rsqrt_lut_uq1p18=19'h2a5fa;
                6'd37: rsqrt_lut_uq1p18=19'h29ce0;
                6'd38: rsqrt_lut_uq1p18=19'h29421;
                6'd39: rsqrt_lut_uq1p18=19'h28bb8;
                6'd40: rsqrt_lut_uq1p18=19'h283a0;
                6'd41: rsqrt_lut_uq1p18=19'h27bd3;
                6'd42: rsqrt_lut_uq1p18=19'h2744c;
                6'd43: rsqrt_lut_uq1p18=19'h26d09;
                6'd44: rsqrt_lut_uq1p18=19'h26604;
                6'd45: rsqrt_lut_uq1p18=19'h25f3b;
                6'd46: rsqrt_lut_uq1p18=19'h258ab;
                6'd47: rsqrt_lut_uq1p18=19'h2524f;
                6'd48: rsqrt_lut_uq1p18=19'h24c27;
                6'd49: rsqrt_lut_uq1p18=19'h2462e;
                6'd50: rsqrt_lut_uq1p18=19'h24063;
                6'd51: rsqrt_lut_uq1p18=19'h23ac4;
                6'd52: rsqrt_lut_uq1p18=19'h2354d;
                6'd53: rsqrt_lut_uq1p18=19'h22ffe;
                6'd54: rsqrt_lut_uq1p18=19'h22ad5;
                6'd55: rsqrt_lut_uq1p18=19'h225d0;
                6'd56: rsqrt_lut_uq1p18=19'h220ec;
                6'd57: rsqrt_lut_uq1p18=19'h21c2a;
                6'd58: rsqrt_lut_uq1p18=19'h21787;
                6'd59: rsqrt_lut_uq1p18=19'h21302;
                6'd60: rsqrt_lut_uq1p18=19'h20e9a;
                6'd61: rsqrt_lut_uq1p18=19'h20a4e;
                6'd62: rsqrt_lut_uq1p18=19'h2061c;
                default: rsqrt_lut_uq1p18=19'h20203;
            endcase
        end
    endfunction

    always_comb begin : normalize_request
        integer most_significant_bit;
        integer exponent;
        integer even_exponent;
        logic [63:0] wide_value;
        logic [63:0] normalized;
        most_significant_bit = -1;
        for (integer bit_index=0; bit_index<22; bit_index=bit_index+1)
            if (variance_plus_epsilon_uq6p16[bit_index])
                most_significant_bit = bit_index;
        exponent = most_significant_bit - 16;
        even_exponent = exponent - (exponent & 1);
        wide_value = {42'b0,variance_plus_epsilon_uq6p16};
        if (even_exponent < 0)
            normalized = wide_value << (-even_exponent);
        else
            normalized = rshift_rne_u64(wide_value,even_exponent);
        normalized_even_exponent_w = even_exponent;
        normalized_mantissa_w = normalized[17:0];
        if (normalized < 64'd131072)
            normalized_lut_index_w = (normalized-64'd65536)>>11;
        else
            normalized_lut_index_w = 6'd32
                + ((normalized-64'd131072)>>12);
        request_legal = variance_plus_epsilon_uq6p16 != 0
            && normalized >= 64'd65536 && normalized < 64'd262144;
    end

    always_comb begin : multiplier_schedule
        multiplier_a = '0;
        multiplier_b = '0;
        case (state_q)
            S_Y2: begin
                multiplier_a = $signed({2'b0,y_q});
                multiplier_b = $signed({2'b0,y_q});
            end
            S_MY2: begin
                multiplier_a = $signed({4'b0,mantissa_q});
                multiplier_b = $signed(y2_q);
            end
            S_YNEW: begin
                multiplier_a = $signed({2'b0,y_q});
                multiplier_b = $signed(term_w);
            end
            S_ALPHA: begin
                multiplier_a = {{6{gamma_q[15]}},gamma_q};
                multiplier_b = $signed({2'b0,invstd_q});
            end
            S_OFFSET: begin
                multiplier_a = {{2{alpha_q[19]}},alpha_q};
                multiplier_b = {{4{mean_q[17]}},mean_q};
            end
            default: begin end
        endcase
    end

    always_comb begin : arithmetic_finish
        logic [63:0] scaled;
        integer scale_shift;
        logic signed [43:0] rounded_alpha;
        logic signed [43:0] rounded_alpha_mean;
        logic signed [43:0] beta_q16;
        rounded_half_my2_w = rshift_rne_u64({42'b0,my2_q},1);
        half_my2_w = rounded_half_my2_w[21:0];
        term_w = 22'd393216 - half_my2_w;
        scale_shift = -2 - (even_exponent_q / 2);
        if (scale_shift >= 0)
            scaled = {44'b0,y_q} << scale_shift;
        else
            scaled = rshift_rne_u64({44'b0,y_q},-scale_shift);
        scaled_invstd_w = scaled > 64'hfffff
            ? 20'hfffff : scaled[19:0];
        rounded_alpha = rshift_rne_s44(multiplier_product,14);
        alpha_w = saturate_s20(rounded_alpha);
        rounded_alpha_mean = rshift_rne_s44(multiplier_product,14);
        beta_q16 = {{26{beta_q[15]}},beta_q,2'b0};
        offset_w = saturate_s20(beta_q16-rounded_alpha_mean);
        rounded_y2_w = rshift_rne_s44(multiplier_product,18);
        rounded_my2_w = rshift_rne_s44(multiplier_product,16);
        rounded_ynew_w = rshift_rne_s44(multiplier_product,18);
    end

    always_comb begin : interfaces
        illegal_request = request_valid && !request_legal;
        protocol_error = fault_q || illegal_request;
        request_ready = !protocol_error && state_q == S_IDLE
            && !result_valid_q;
        request_accept = request_valid && request_ready;
        result_valid = result_valid_q && !protocol_error;
        result_accept = result_valid && result_ready;
        result_tag = result_valid ? result_tag_q : '0;
        invstd_uq4p16 = result_valid ? invstd_q : '0;
        alpha_sq3p16 = result_valid ? alpha_q : '0;
        offset_sq3p16 = result_valid ? offset_q : '0;
        busy = state_q != S_IDLE || result_valid_q;
        debug_state = state_q;
        debug_request_count = request_count_q;
        debug_result_count = result_count_q;
    end

    always_ff @(posedge clk_core) begin : state
        if (rst_core) begin
            state_q <= S_IDLE;
            fault_q <= 1'b0;
            result_valid_q <= 1'b0;
            tag_q <= '0;
            result_tag_q <= '0;
            even_exponent_q <= '0;
            mantissa_q <= '0;
            lut_index_q <= '0;
            y_q <= '0;
            y2_q <= '0;
            my2_q <= '0;
            invstd_q <= '0;
            alpha_q <= '0;
            offset_q <= '0;
            mean_q <= '0;
            gamma_q <= '0;
            beta_q <= '0;
            request_count_q <= '0;
            result_count_q <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;
            if (!protocol_error) begin
                case (state_q)
                    S_IDLE: if (request_accept) begin
                        tag_q <= request_tag;
                        even_exponent_q <= normalized_even_exponent_w;
                        mantissa_q <= normalized_mantissa_w;
                        lut_index_q <= normalized_lut_index_w;
                        mean_q <= mean_sq3p14;
                        gamma_q <= gamma_sq1p14;
                        beta_q <= beta_sq1p14;
                        request_count_q <= request_count_q + 1'b1;
                        state_q <= S_LUT;
                    end
                    S_LUT: begin
                        y_q <= {1'b0,rsqrt_lut_uq1p18(lut_index_q)};
                        state_q <= S_Y2;
                    end
                    S_Y2: begin
                        y2_q <= rounded_y2_w[21:0];
                        state_q <= S_MY2;
                    end
                    S_MY2: begin
                        my2_q <= rounded_my2_w[21:0];
                        state_q <= S_YNEW;
                    end
                    S_YNEW: begin
                        y_q <= rounded_ynew_w[19:0];
                        state_q <= S_SCALE;
                    end
                    S_SCALE: begin
                        invstd_q <= scaled_invstd_w;
                        state_q <= S_ALPHA;
                    end
                    S_ALPHA: begin
                        alpha_q <= alpha_w;
                        state_q <= S_OFFSET;
                    end
                    S_OFFSET: begin
                        offset_q <= offset_w;
                        result_tag_q <= tag_q;
                        result_valid_q <= 1'b1;
                        result_count_q <= result_count_q + 1'b1;
                        state_q <= S_RESULT;
                    end
                    S_RESULT: if (result_accept) begin
                        result_valid_q <= 1'b0;
                        state_q <= S_IDLE;
                    end
                    default: fault_q <= 1'b1;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
