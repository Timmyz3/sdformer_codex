`timescale 1ns/1ps
`default_nettype none

// M27 fixed-shape ATLIF factor tile arithmetic core.
//
// Frozen arithmetic contract:
//   * X, left factor, and right factor are signed INT8.
//   * Stage 1 forms 48 exact Q24 dot products (rank=3, lanes=16, T=10).
//     It consumes two temporal terms per accumulator and therefore instantiates
//     exactly 48*2 = 96 signed INT8 multiplications in each of five cycles.
//   * At the end of stage 1, Q24 is shifted by the compile-time
//     REQUANT_SHIFT power-of-two scale (frozen to eight in this milestone),
//     rounded to nearest with ties to even (RNE), saturated to signed INT8,
//     and registered as the 48-value intermediate tile.
//   * Stage 2 forms two temporal rows (32 lane outputs) per cycle.  Each output
//     uses three signed INT8 multiplications, so the same 96 multiplier slots
//     are occupied for each of five more cycles.  A dynamic signed Q24 bias is
//     added and the result is saturated to signed Q24.
//
// The wide request port is a compute-core boundary, not a claim that all
// operands can be loaded from SRAM in one cycle.  Input/factor/bias loading,
// double buffering, output DMA, and any system-level speedup are out of scope.
module qfit_atlif_rank3_exact96_core #(
    parameter int TAG_W = 48,
    parameter int REQUANT_SHIFT = 8,
    localparam int T = 10,
    localparam int RANK = 3,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int OUTPUTS_PER_BEAT = 32,
    localparam int MULTIPLIERS = 96
) (
    input  logic                                    clk_core,
    input  logic                                    rst_core,

    input  logic                                    request_valid,
    output logic                                    request_ready,
    output logic                                    request_legal,
    input  logic [TAG_W-1:0]                        request_tag,
    input  logic [(T*LANES*IN_W)-1:0]               request_x,
    input  logic [(RANK*T*IN_W)-1:0]                request_right_factor,
    input  logic [(T*RANK*IN_W)-1:0]                request_left_factor,
    input  logic [(T*LANES*ACC_W)-1:0]              request_bias,

    output logic                                    result_valid,
    input  logic                                    result_ready,
    output logic [TAG_W-1:0]                        result_tag,
    output logic [2:0]                              result_beat,
    output logic [(OUTPUTS_PER_BEAT*ACC_W)-1:0]     result_values,
    output logic                                    done,
    output logic [TAG_W-1:0]                        done_tag,
    output logic                                    protocol_error,
    output logic                                    busy,

    // Verification/accounting observability.  arithmetic_active is high for
    // every non-stalled arithmetic cycle and multiplier_active_mask is all
    // ones exactly then.  stage_select is zero for stage 1 and one for stage 2.
    output logic                                    arithmetic_active,
    output logic                                    stage_select,
    output logic [2:0]                              phase_cycle,
    output logic [MULTIPLIERS-1:0]                  multiplier_active_mask
);
    localparam logic [4:0] REQUANT_SHIFT_BITS = REQUANT_SHIFT;
    typedef enum logic [1:0] {IDLE, STAGE1, STAGE2, DRAIN} state_t;
    state_t state_q;

    logic [TAG_W-1:0] tag_q;
    logic signed [IN_W-1:0] x_q [0:(T*LANES)-1];
    logic signed [IN_W-1:0] right_q [0:(RANK*T)-1];
    logic signed [IN_W-1:0] left_q [0:(T*RANK)-1];
    logic signed [ACC_W-1:0] bias_q [0:(T*LANES)-1];
    logic signed [ACC_W-1:0] stage1_acc_q [0:(RANK*LANES)-1];
    logic signed [IN_W-1:0] intermediate_q [0:(RANK*LANES)-1];

    logic signed [IN_W-1:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [IN_W-1:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [(2*IN_W)-1:0] multiplier_product [0:MULTIPLIERS-1];

    logic result_valid_q;
    logic [TAG_W-1:0] result_tag_q;
    logic [2:0] result_beat_q;
    logic signed [ACC_W-1:0] result_q [0:OUTPUTS_PER_BEAT-1];
    logic done_q;
    logic [TAG_W-1:0] done_tag_q;
    logic protocol_error_q;
    logic [2:0] phase_cycle_q;

    logic request_fire;
    logic result_fire;
    logic output_slot_available;
    logic execute_stage2;

    function automatic logic signed [IN_W-1:0] rne_sat_q24_to_q8(
        input logic signed [ACC_W-1:0] value,
        input logic [4:0] shift
    );
        logic negative;
        logic [ACC_W-1:0] magnitude;
        logic [ACC_W-1:0] quotient;
        logic [ACC_W-1:0] remainder;
        logic [ACC_W-1:0] remainder_mask;
        logic [ACC_W-1:0] half;
        logic round_up;
        logic [ACC_W:0] rounded_magnitude;
        begin
            negative = value[ACC_W-1];
            magnitude = negative ? (~$unsigned(value) + 1'b1) : $unsigned(value);
            if (shift == 0) begin
                quotient = magnitude;
                remainder = '0;
                half = '0;
                round_up = 1'b0;
            end else begin
                remainder_mask = ({ACC_W{1'b1}} >> (ACC_W-shift));
                remainder = magnitude & remainder_mask;
                half = {{(ACC_W-1){1'b0}}, 1'b1} << (shift-1'b1);
                quotient = magnitude >> shift;
                round_up = (remainder > half)
                    || ((remainder == half) && quotient[0]);
            end
            rounded_magnitude = {1'b0, quotient} + round_up;
            if (!negative && rounded_magnitude > 127)
                rne_sat_q24_to_q8 = 8'sd127;
            else if (negative && rounded_magnitude > 128)
                rne_sat_q24_to_q8 = -8'sd128;
            else if (negative)
                rne_sat_q24_to_q8 = -$signed(rounded_magnitude[IN_W-1:0]);
            else
                rne_sat_q24_to_q8 = $signed(rounded_magnitude[IN_W-1:0]);
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] sat_q26_to_q24(
        input logic signed [ACC_W+1:0] value
    );
        logic signed [ACC_W+1:0] maximum;
        logic signed [ACC_W+1:0] minimum;
        begin
            maximum = ({{(ACC_W+1){1'b0}}, 1'b1} << (ACC_W-1)) - 1'b1;
            minimum = -({{(ACC_W+1){1'b0}}, 1'b1} << (ACC_W-1));
            if (value > maximum)
                sat_q26_to_q24 = {1'b0, {(ACC_W-1){1'b1}}};
            else if (value < minimum)
                sat_q26_to_q24 = {1'b1, {(ACC_W-1){1'b0}}};
            else
                sat_q26_to_q24 = value[ACC_W-1:0];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (T != 10 || RANK != 3 || LANES != 16)
            $fatal(1, "M27 shape must remain T=10 rank=3 lanes=16");
        if (IN_W != 8 || ACC_W != 24 || OUTPUTS_PER_BEAT != 32)
            $fatal(1, "M27 arithmetic widths must remain INT8/Q24/32 outputs");
        if (MULTIPLIERS != 96)
            $fatal(1, "M27 requires exactly 96 multiplier slots");
        if (REQUANT_SHIFT < 0 || REQUANT_SHIFT > 23)
            $fatal(1, "M27 REQUANT_SHIFT must be in [0,23]");
    end
`endif

    assign request_ready = !rst_core && state_q == IDLE && !protocol_error_q;
    assign request_legal = request_ready;
    assign request_fire = request_valid && request_ready;
    assign result_fire = result_valid_q && result_ready;
    assign output_slot_available = !result_valid_q || result_ready;
    assign execute_stage2 = state_q == STAGE2 && output_slot_available;

    assign result_valid = result_valid_q;
    assign result_tag = result_tag_q;
    assign result_beat = result_beat_q;
    assign done = done_q;
    assign done_tag = done_tag_q;
    assign protocol_error = protocol_error_q;
    assign busy = state_q != IDLE;
    assign stage_select = state_q == STAGE2 || state_q == DRAIN;
    assign phase_cycle = phase_cycle_q;
    assign arithmetic_active = state_q == STAGE1 || execute_stage2;
    assign multiplier_active_mask = arithmetic_active ? {MULTIPLIERS{1'b1}} : '0;

    for (genvar output_index = 0; output_index < OUTPUTS_PER_BEAT;
         output_index++) begin : gen_result_pack
        assign result_values[(output_index*ACC_W) +: ACC_W] = result_q[output_index];
    end

    // This is the sole multiplication site in the module.  The generate loop
    // creates exactly 96 signed 8x8 products, shared by both arithmetic stages.
    for (genvar multiplier = 0; multiplier < MULTIPLIERS;
         multiplier++) begin : gen_exact_96_multipliers
        assign multiplier_product[multiplier]
            = multiplier_a[multiplier] * multiplier_b[multiplier];
    end

    always_comb begin : select_shared_multiplier_operands
        for (int multiplier = 0; multiplier < MULTIPLIERS; multiplier++) begin
            multiplier_a[multiplier] = '0;
            multiplier_b[multiplier] = '0;
        end
        if (state_q == STAGE1) begin
            for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    for (int pair = 0; pair < 2; pair++) begin
                        multiplier_a[(((rank_index*LANES)+lane)*2)+pair]
                            = x_q[(((phase_cycle_q*2)+pair)*LANES)+lane];
                        multiplier_b[(((rank_index*LANES)+lane)*2)+pair]
                            = right_q[(rank_index*T)+(phase_cycle_q*2)+pair];
                    end
                end
            end
        end else if (state_q == STAGE2) begin
            for (int row_in_beat = 0; row_in_beat < 2; row_in_beat++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                        multiplier_a[(((row_in_beat*LANES)+lane)*RANK)+rank_index]
                            = intermediate_q[(rank_index*LANES)+lane];
                        multiplier_b[(((row_in_beat*LANES)+lane)*RANK)+rank_index]
                            = left_q[(((phase_cycle_q*2)+row_in_beat)*RANK)+rank_index];
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin : core_state
        logic signed [ACC_W:0] stage1_sum;
        logic signed [ACC_W+1:0] stage2_sum;
        if (rst_core) begin
            state_q <= IDLE;
            tag_q <= '0;
            result_valid_q <= 1'b0;
            result_tag_q <= '0;
            result_beat_q <= '0;
            done_q <= 1'b0;
            done_tag_q <= '0;
            protocol_error_q <= 1'b0;
            phase_cycle_q <= '0;
            for (int index = 0; index < RANK*LANES; index++) begin
                stage1_acc_q[index] <= '0;
                intermediate_q[index] <= '0;
            end
            for (int output_index = 0; output_index < OUTPUTS_PER_BEAT;
                 output_index++)
                result_q[output_index] <= '0;
        end else begin
            done_q <= 1'b0;

            if (result_fire)
                result_valid_q <= 1'b0;

            case (state_q)
                IDLE: begin
                    phase_cycle_q <= '0;
                    if (request_fire) begin
                        tag_q <= request_tag;
                        for (int index = 0; index < T*LANES; index++) begin
                            x_q[index] <= $signed(request_x[(index*IN_W) +: IN_W]);
                            bias_q[index]
                                <= $signed(request_bias[(index*ACC_W) +: ACC_W]);
                        end
                        for (int index = 0; index < RANK*T; index++)
                            right_q[index]
                                <= $signed(request_right_factor[(index*IN_W) +: IN_W]);
                        for (int index = 0; index < T*RANK; index++)
                            left_q[index]
                                <= $signed(request_left_factor[(index*IN_W) +: IN_W]);
                        for (int index = 0; index < RANK*LANES; index++) begin
                            stage1_acc_q[index] <= '0;
                            intermediate_q[index] <= '0;
                        end
                        state_q <= STAGE1;
                    end
                end

                STAGE1: begin
                    for (int accumulator = 0; accumulator < RANK*LANES;
                         accumulator++) begin
                        // Explicit extension freezes the addition width instead
                        // of relying on assignment-context sizing rules.
                        stage1_sum = {
                            stage1_acc_q[accumulator][ACC_W-1],
                            stage1_acc_q[accumulator]
                        } + {
                            {(ACC_W+1-(2*IN_W)){
                                multiplier_product[accumulator*2][(2*IN_W)-1]}},
                            multiplier_product[accumulator*2]
                        } + {
                            {(ACC_W+1-(2*IN_W)){
                                multiplier_product[(accumulator*2)+1][(2*IN_W)-1]}},
                            multiplier_product[(accumulator*2)+1]
                        };
                        if (phase_cycle_q == 4)
                            intermediate_q[accumulator]
                                <= rne_sat_q24_to_q8(stage1_sum[ACC_W-1:0],
                                                    REQUANT_SHIFT_BITS);
                        else
                            stage1_acc_q[accumulator] <= stage1_sum[ACC_W-1:0];
                    end
                    if (phase_cycle_q == 4) begin
                        // No transition state: immediately after this edge the
                        // shared multiplier bank presents stage-2 cycle zero.
                        state_q <= STAGE2;
                        phase_cycle_q <= '0;
                    end else begin
                        phase_cycle_q <= phase_cycle_q + 1'b1;
                    end
                end

                STAGE2: begin
                    if (output_slot_available) begin
                        for (int output_index = 0;
                             output_index < OUTPUTS_PER_BEAT; output_index++) begin
                            // The Q26 full-precision sum is explicit so a Q24
                            // bias near either rail cannot wrap before the
                            // saturating Q26-to-Q24 function observes it.
                            stage2_sum = {{2{bias_q[
                                (((phase_cycle_q*2)+(output_index/LANES))*LANES)
                                +(output_index%LANES)][ACC_W-1]}}, bias_q[
                                (((phase_cycle_q*2)+(output_index/LANES))*LANES)
                                +(output_index%LANES)]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)+1][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)+1]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)+2][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)+2]};
                            result_q[output_index] <= sat_q26_to_q24(stage2_sum);
                        end
                        result_valid_q <= 1'b1;
                        result_tag_q <= tag_q;
                        result_beat_q <= phase_cycle_q;
                        if (phase_cycle_q == 4) begin
                            state_q <= DRAIN;
                        end else begin
                            phase_cycle_q <= phase_cycle_q + 1'b1;
                        end
                    end
                end

                DRAIN: begin
                    if (result_fire) begin
                        done_q <= 1'b1;
                        done_tag_q <= tag_q;
                        state_q <= IDLE;
                        phase_cycle_q <= '0;
                    end
                end

                default: begin
                    protocol_error_q <= 1'b1;
                    state_q <= IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
