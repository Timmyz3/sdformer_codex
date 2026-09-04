`timescale 1ns/1ps
`default_nettype none

// M31 temporal-polymorphic ATLIF operator engine.
//
// A single, auditable 96x signed-INT8 multiplier pool is reused by:
//   mode 0: T10 rank-3, 16 lanes, five reduction + five reconstruction beats;
//   mode 1: T2 dense 2x2, 24 lanes, one packet per cycle.
//
// The context mode is immutable from load through drain/release.  A held input
// always belongs to the current context and therefore has priority over a
// simultaneous release request.  One shared 48-bit result FIFO stores either
// five 32-bit T10 beats or one 48-bit T2 packet.
module qfit_atlif_unified_t10_t2_stream_core #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T10_T = 10,
    localparam int T10_PHASES = T10_T/2,
    localparam int T10_RANK = 3,
    localparam int T10_LANES = 16,
    localparam int T2_LANES = 24,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int MULTIPLIERS = 96,
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,

    input  logic                                  parameter_valid,
    output logic                                  parameter_ready,
    input  logic                                  parameter_mode,
    input  logic [(T10_RANK*T10_T*IN_W)-1:0]      parameter_t10_right_factor,
    input  logic [(T10_T*T10_RANK*IN_W)-1:0]      parameter_t10_left_factor,
    input  logic [(T10_T*ACC_W)-1:0]              parameter_t10_bias,
    input  logic signed [ACC_W-1:0]               parameter_t10_threshold,
    input  logic [4:0]                            parameter_t10_requant_shift,
    input  logic [(4*IN_W)-1:0]                   parameter_t2_weight,
    input  logic [(2*ACC_W)-1:0]                  parameter_t2_bias,
    input  logic signed [ACC_W-1:0]               parameter_t2_threshold,
    output logic                                  parameter_loaded,
    output logic                                  loaded_mode,
    input  logic                                  parameter_release_valid,
    output logic                                  parameter_release_ready,

    input  logic                                  input_valid,
    output logic                                  input_ready,
    input  logic [TAG_W-1:0]                      input_tag,
    input  logic [2:0]                            input_beat,
    input  logic [T2_LANES-1:0]                   input_lane_valid,
    input  logic [255:0]                          input_port0_values,
    input  logic [255:0]                          input_port1_values,

    output logic                                  result_valid,
    input  logic                                  result_ready,
    output logic                                  result_mode,
    output logic [TAG_W-1:0]                      result_tag,
    output logic [2:0]                            result_beat,
    output logic [47:0]                           result_valid_bits,
    output logic [47:0]                           result_bits,
    output logic                                  done,
    output logic                                  done_mode,
    output logic [TAG_W-1:0]                      done_tag,
    output logic                                  protocol_error,
    output logic                                  busy,

    output logic                                  arithmetic_active,
    output logic [1:0]                            issue_kind,
    output logic [2:0]                            phase_cycle,
    output logic [MULTIPLIERS-1:0]                multiplier_active_mask,
    output logic [FIFO_COUNT_W-1:0]               result_fifo_occupancy
);
    typedef enum logic [1:0] {
        T10_IDLE, T10_STAGE1, T10_WAIT_STAGE2, T10_STAGE2
    } t10_state_t;
    typedef enum logic [1:0] {
        BANK_EMPTY, BANK_FILL, BANK_READY, BANK_ACTIVE
    } bank_state_t;

    t10_state_t t10_state_q;
    bank_state_t bank_state_q [0:1];
    logic active_bank_q;
    logic fill_active_q;
    logic fill_bank_q;
    logic [2:0] expected_input_beat_q;
    logic [TAG_W-1:0] bank_tag_q [0:1];
    logic signed [IN_W-1:0] x_bank_q [0:1][0:(T10_T*T10_LANES)-1];

    logic parameter_loaded_q;
    logic parameter_mode_q;
    logic signed [IN_W-1:0] t10_right_q [0:(T10_RANK*T10_T)-1];
    logic signed [IN_W-1:0] t10_left_q [0:(T10_T*T10_RANK)-1];
    logic signed [ACC_W-1:0] t10_bias_q [0:T10_T-1];
    logic signed [ACC_W-1:0] t10_threshold_q;
    logic [4:0] t10_requant_shift_q;
    logic signed [IN_W-1:0] t2_weight_q [0:3];
    logic signed [ACC_W-1:0] t2_bias_q [0:1];
    logic signed [ACC_W-1:0] t2_threshold_q;

    logic signed [ACC_W-1:0] t10_stage1_acc_q
        [0:(T10_RANK*T10_LANES)-1];
    logic signed [IN_W-1:0] t10_intermediate_q
        [0:(T10_RANK*T10_LANES)-1];

    logic signed [IN_W-1:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [IN_W-1:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [(2*IN_W)-1:0] multiplier_product [0:MULTIPLIERS-1];

    logic fifo_mode_q [0:FIFO_DEPTH-1];
    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_valid_bits_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_bits_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_write_pointer_q;
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;

    logic done_q;
    logic done_mode_q;
    logic [TAG_W-1:0] done_tag_q;
    logic protocol_error_q;
    logic [2:0] phase_cycle_q;
    logic input_fire;
    logic input_bank;
    logic input_protocol_ok;
    logic result_fire;
    logic t10_push_result;
    logic t2_push_result;
    logic push_result;
    logic t2_fifo_credit;
    logic [FIFO_COUNT_W-1:0] fifo_free_slots;
    logic safe_parameter_boundary;

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
            if (shift > 23) begin
                rne_sat_q24_to_q8 = '0;
            end else begin
                negative = value[ACC_W-1];
                magnitude = negative ? (~$unsigned(value) + 1'b1)
                    : $unsigned(value);
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
                    rne_sat_q24_to_q8
                        = -$signed(rounded_magnitude[IN_W-1:0]);
                else
                    rne_sat_q24_to_q8
                        = $signed(rounded_magnitude[IN_W-1:0]);
            end
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] sat_q25_to_q24(
        input logic signed [ACC_W:0] value
    );
        logic signed [ACC_W:0] maximum;
        logic signed [ACC_W:0] minimum;
        begin
            maximum = ({{ACC_W{1'b0}}, 1'b1} << (ACC_W-1)) - 1'b1;
            minimum = -({{ACC_W{1'b0}}, 1'b1} << (ACC_W-1));
            if (value > maximum)
                sat_q25_to_q24 = {1'b0, {(ACC_W-1){1'b1}}};
            else if (value < minimum)
                sat_q25_to_q24 = {1'b1, {(ACC_W-1){1'b0}}};
            else
                sat_q25_to_q24 = value[ACC_W-1:0];
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
        if (FIFO_DEPTH < 10 || (1 << FIFO_PTR_W) != FIFO_DEPTH)
            $fatal(1, "M31 FIFO_DEPTH must be power-of-two and at least ten");
        if (T10_T != 10 || T10_RANK != 3 || T10_LANES != 16
            || T2_LANES != 24 || MULTIPLIERS != 96)
            $fatal(1, "M31 shape/resource contract drift");
    end
`endif

    assign safe_parameter_boundary = t10_state_q == T10_IDLE
        && bank_state_q[0] == BANK_EMPTY
        && bank_state_q[1] == BANK_EMPTY
        && fifo_count_q == 0 && !fill_active_q;
    assign parameter_ready = !rst_core && !parameter_loaded_q
        && safe_parameter_boundary && !protocol_error_q;
    assign parameter_release_ready = !rst_core && parameter_loaded_q
        && safe_parameter_boundary && !protocol_error_q && !input_valid;
    assign parameter_loaded = parameter_loaded_q;
    assign loaded_mode = parameter_mode_q;

    assign input_bank = fill_active_q ? fill_bank_q
        : (bank_state_q[0] == BANK_EMPTY ? 1'b0 : 1'b1);
    assign t2_fifo_credit = fifo_count_q < FIFO_DEPTH || result_fire;
    always_comb begin
        if (!parameter_mode_q) begin
            input_ready = !rst_core && parameter_loaded_q && !protocol_error_q
                && (fill_active_q || bank_state_q[0] == BANK_EMPTY
                    || bank_state_q[1] == BANK_EMPTY);
            input_protocol_ok = input_port1_values == '0
                && input_lane_valid == 24'h00ffff
                && (fill_active_q
                    ? (input_beat == expected_input_beat_q
                        && input_tag == bank_tag_q[fill_bank_q])
                    : (input_beat == 0));
        end else begin
            input_ready = !rst_core && parameter_loaded_q && !protocol_error_q
                && t2_fifo_credit;
            input_protocol_ok = input_beat == 0
                && input_lane_valid == {T2_LANES{1'b1}}
                && input_port0_values[255:192] == '0
                && input_port1_values[255:192] == '0;
        end
    end
    assign input_fire = input_valid && input_ready;

    assign result_valid = fifo_count_q != 0;
    assign result_mode = fifo_mode_q[fifo_read_pointer_q];
    assign result_tag = fifo_tag_q[fifo_read_pointer_q];
    assign result_beat = fifo_beat_q[fifo_read_pointer_q];
    assign result_valid_bits = fifo_valid_bits_q[fifo_read_pointer_q];
    assign result_bits = fifo_bits_q[fifo_read_pointer_q];
    assign result_fire = result_valid && result_ready;
    assign fifo_free_slots = FIFO_DEPTH - fifo_count_q;
    assign t10_push_result = !parameter_mode_q
        && t10_state_q == T10_STAGE2 && fifo_count_q < FIFO_DEPTH;
    assign t2_push_result = parameter_mode_q && input_fire
        && input_protocol_ok;
    assign push_result = t10_push_result || t2_push_result;

    assign done = done_q;
    assign done_mode = done_mode_q;
    assign done_tag = done_tag_q;
    assign protocol_error = protocol_error_q;
    assign busy = t10_state_q != T10_IDLE
        || bank_state_q[0] != BANK_EMPTY
        || bank_state_q[1] != BANK_EMPTY
        || fifo_count_q != 0 || fill_active_q;
    assign phase_cycle = phase_cycle_q;
    always_comb begin
        arithmetic_active = 1'b0;
        issue_kind = 2'b00;
        if (!parameter_mode_q && t10_state_q == T10_STAGE1) begin
            arithmetic_active = 1'b1;
            issue_kind = 2'b01;
        end else if (!parameter_mode_q && t10_state_q == T10_STAGE2) begin
            arithmetic_active = 1'b1;
            issue_kind = 2'b10;
        end else if (parameter_mode_q && t2_push_result) begin
            arithmetic_active = 1'b1;
            issue_kind = 2'b11;
        end
    end
    assign multiplier_active_mask = arithmetic_active
        ? {MULTIPLIERS{1'b1}} : '0;
    assign result_fifo_occupancy = fifo_count_q;

    // Operand selection is deliberately above the sole multiplier instance.
    always_comb begin : select_unified_multiplier_operands
        for (int multiplier = 0; multiplier < MULTIPLIERS; multiplier++) begin
            multiplier_a[multiplier] = '0;
            multiplier_b[multiplier] = '0;
        end
        if (!parameter_mode_q && t10_state_q == T10_STAGE1) begin
            // Keep every array subscript elaboration-time bounded.  The
            // phase comparison is dynamic, but each selected data slice is
            // indexed only by the statically unrolled phase loop.
            for (int phase = 0; phase < T10_PHASES; phase++) begin
                if (phase_cycle_q == phase) begin
                    for (int rank_index = 0; rank_index < T10_RANK;
                         rank_index++) begin
                        for (int lane = 0; lane < T10_LANES; lane++) begin
                            for (int pair = 0; pair < 2; pair++) begin
                                multiplier_a[(((rank_index*T10_LANES)+lane)
                                    *2)+pair] = x_bank_q[active_bank_q]
                                    [(((phase*2)+pair)*T10_LANES)+lane];
                                multiplier_b[(((rank_index*T10_LANES)+lane)
                                    *2)+pair] = t10_right_q[
                                    (rank_index*T10_T)+(phase*2)+pair];
                            end
                        end
                    end
                end
            end
        end else if (!parameter_mode_q && t10_state_q == T10_STAGE2) begin
            for (int phase = 0; phase < T10_PHASES; phase++) begin
                if (phase_cycle_q == phase) begin
                    for (int row_in_beat = 0; row_in_beat < 2;
                         row_in_beat++) begin
                        for (int lane = 0; lane < T10_LANES; lane++) begin
                            for (int rank_index = 0; rank_index < T10_RANK;
                                 rank_index++) begin
                                multiplier_a[(((row_in_beat*T10_LANES)+lane)
                                    *T10_RANK)+rank_index]
                                    = t10_intermediate_q[
                                        (rank_index*T10_LANES)+lane];
                                multiplier_b[(((row_in_beat*T10_LANES)+lane)
                                    *T10_RANK)+rank_index] = t10_left_q[
                                    (((phase*2)+row_in_beat)*T10_RANK)
                                        +rank_index];
                            end
                        end
                    end
                end
            end
        end else if (parameter_mode_q && t2_push_result) begin
            for (int lane = 0; lane < T2_LANES; lane++) begin
                multiplier_a[(lane*4)+0]
                    = $signed(input_port0_values[(lane*IN_W) +: IN_W]);
                multiplier_b[(lane*4)+0] = t2_weight_q[0];
                multiplier_a[(lane*4)+1]
                    = $signed(input_port1_values[(lane*IN_W) +: IN_W]);
                multiplier_b[(lane*4)+1] = t2_weight_q[1];
                multiplier_a[(lane*4)+2]
                    = $signed(input_port0_values[(lane*IN_W) +: IN_W]);
                multiplier_b[(lane*4)+2] = t2_weight_q[2];
                multiplier_a[(lane*4)+3]
                    = $signed(input_port1_values[(lane*IN_W) +: IN_W]);
                multiplier_b[(lane*4)+3] = t2_weight_q[3];
            end
        end
    end

    qfit_signed_int8_mul96_pool #(
        .MULTIPLIERS(MULTIPLIERS), .IN_W(IN_W)
    ) u_mul_pool (
        .operand_a(multiplier_a),
        .operand_b(multiplier_b),
        .product(multiplier_product)
    );

    always_ff @(posedge clk_core) begin : unified_stream_state
        logic signed [ACC_W:0] t10_stage1_sum;
        logic signed [ACC_W+1:0] t10_stage2_sum;
        logic signed [ACC_W:0] t2_sum0;
        logic signed [ACC_W:0] t2_sum1;
        logic signed [ACC_W-1:0] saturated_output;
        logic [47:0] packed_result;
        logic next_bank;
        if (rst_core) begin
            t10_state_q <= T10_IDLE;
            bank_state_q[0] <= BANK_EMPTY;
            bank_state_q[1] <= BANK_EMPTY;
            active_bank_q <= 1'b0;
            fill_active_q <= 1'b0;
            fill_bank_q <= 1'b0;
            expected_input_beat_q <= '0;
            bank_tag_q[0] <= '0;
            bank_tag_q[1] <= '0;
            parameter_loaded_q <= 1'b0;
            parameter_mode_q <= 1'b0;
            t10_threshold_q <= '0;
            t10_requant_shift_q <= '0;
            t2_threshold_q <= '0;
            phase_cycle_q <= '0;
            fifo_write_pointer_q <= '0;
            fifo_read_pointer_q <= '0;
            fifo_count_q <= '0;
            done_q <= 1'b0;
            done_mode_q <= 1'b0;
            done_tag_q <= '0;
            protocol_error_q <= 1'b0;
            for (int index = 0; index < T10_RANK*T10_LANES; index++) begin
                t10_stage1_acc_q[index] <= '0;
                t10_intermediate_q[index] <= '0;
            end
        end else begin
            done_q <= 1'b0;

            if (parameter_release_valid && parameter_release_ready)
                parameter_loaded_q <= 1'b0;

            if (parameter_valid && parameter_ready) begin
                if (!parameter_mode && parameter_t10_requant_shift > 23) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    parameter_mode_q <= parameter_mode;
                    parameter_loaded_q <= 1'b1;
                    if (!parameter_mode) begin
                        for (int index = 0; index < T10_RANK*T10_T; index++)
                            t10_right_q[index] <= $signed(
                                parameter_t10_right_factor[(index*IN_W)+:IN_W]);
                        for (int index = 0; index < T10_T*T10_RANK; index++)
                            t10_left_q[index] <= $signed(
                                parameter_t10_left_factor[(index*IN_W)+:IN_W]);
                        for (int row = 0; row < T10_T; row++)
                            t10_bias_q[row] <= $signed(
                                parameter_t10_bias[(row*ACC_W)+:ACC_W]);
                        t10_threshold_q <= parameter_t10_threshold;
                        t10_requant_shift_q <= parameter_t10_requant_shift;
                    end else begin
                        for (int index = 0; index < 4; index++)
                            t2_weight_q[index] <= $signed(
                                parameter_t2_weight[(index*IN_W)+:IN_W]);
                        for (int index = 0; index < 2; index++)
                            t2_bias_q[index] <= $signed(
                                parameter_t2_bias[(index*ACC_W)+:ACC_W]);
                        t2_threshold_q <= parameter_t2_threshold;
                    end
                end
            end

            if (input_fire && !input_protocol_ok)
                protocol_error_q <= 1'b1;

            if (input_fire && input_protocol_ok && !parameter_mode_q) begin
                for (int index = 0; index < 32; index++)
                    x_bank_q[input_bank][(input_beat*32)+index]
                        <= $signed(input_port0_values[(index*IN_W)+:IN_W]);
                if (!fill_active_q) begin
                    bank_tag_q[input_bank] <= input_tag;
                    fill_bank_q <= input_bank;
                    bank_state_q[input_bank] <= BANK_FILL;
                    fill_active_q <= 1'b1;
                    expected_input_beat_q <= 1;
                end else if (input_beat == 4) begin
                    bank_state_q[fill_bank_q] <= BANK_READY;
                    fill_active_q <= 1'b0;
                    expected_input_beat_q <= '0;
                end else begin
                    expected_input_beat_q <= expected_input_beat_q + 1'b1;
                end
            end

            if (t2_push_result) begin
                packed_result = '0;
                for (int lane = 0; lane < T2_LANES; lane++) begin
                    t2_sum0 = {{1{t2_bias_q[0][ACC_W-1]}}, t2_bias_q[0]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+0][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+0]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+1][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+1]};
                    t2_sum1 = {{1{t2_bias_q[1][ACC_W-1]}}, t2_bias_q[1]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+2][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+2]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+3][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+3]};
                    packed_result[lane]
                        = sat_q25_to_q24(t2_sum0) >= t2_threshold_q;
                    packed_result[T2_LANES+lane]
                        = sat_q25_to_q24(t2_sum1) >= t2_threshold_q;
                end
                fifo_mode_q[fifo_write_pointer_q] <= 1'b1;
                fifo_tag_q[fifo_write_pointer_q] <= input_tag;
                fifo_beat_q[fifo_write_pointer_q] <= '0;
                fifo_valid_bits_q[fifo_write_pointer_q] <= {48{1'b1}};
                fifo_bits_q[fifo_write_pointer_q] <= packed_result;
                fifo_write_pointer_q <= fifo_write_pointer_q + 1'b1;
                done_q <= 1'b1;
                done_mode_q <= 1'b1;
                done_tag_q <= input_tag;
            end

            if (result_fire)
                fifo_read_pointer_q <= fifo_read_pointer_q + 1'b1;
            case ({push_result, result_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase

            if (!parameter_mode_q) begin
                case (t10_state_q)
                    T10_IDLE: begin
                        phase_cycle_q <= '0;
                        if (bank_state_q[0] == BANK_READY) begin
                            active_bank_q <= 1'b0;
                            bank_state_q[0] <= BANK_ACTIVE;
                            t10_state_q <= T10_STAGE1;
                        end else if (bank_state_q[1] == BANK_READY) begin
                            active_bank_q <= 1'b1;
                            bank_state_q[1] <= BANK_ACTIVE;
                            t10_state_q <= T10_STAGE1;
                        end
                    end

                    T10_STAGE1: begin
                        for (int accumulator = 0;
                             accumulator < T10_RANK*T10_LANES;
                             accumulator++) begin
                            t10_stage1_sum = {
                                (phase_cycle_q == 0 ? 1'b0
                                    : t10_stage1_acc_q[accumulator][ACC_W-1]),
                                (phase_cycle_q == 0 ? {ACC_W{1'b0}}
                                    : t10_stage1_acc_q[accumulator])
                            } + {
                                {(ACC_W+1-(2*IN_W)){
                                    multiplier_product[accumulator*2]
                                        [(2*IN_W)-1]}},
                                multiplier_product[accumulator*2]
                            } + {
                                {(ACC_W+1-(2*IN_W)){
                                    multiplier_product[(accumulator*2)+1]
                                        [(2*IN_W)-1]}},
                                multiplier_product[(accumulator*2)+1]
                            };
                            if (phase_cycle_q == 4)
                                t10_intermediate_q[accumulator]
                                    <= rne_sat_q24_to_q8(
                                        t10_stage1_sum[ACC_W-1:0],
                                        t10_requant_shift_q);
                            else
                                t10_stage1_acc_q[accumulator]
                                    <= t10_stage1_sum[ACC_W-1:0];
                        end
                        if (phase_cycle_q == 4) begin
                            phase_cycle_q <= '0;
                            if (fifo_free_slots >= 5)
                                t10_state_q <= T10_STAGE2;
                            else
                                t10_state_q <= T10_WAIT_STAGE2;
                        end else begin
                            phase_cycle_q <= phase_cycle_q + 1'b1;
                        end
                    end

                    T10_WAIT_STAGE2: begin
                        phase_cycle_q <= '0;
                        if (fifo_free_slots >= 5)
                            t10_state_q <= T10_STAGE2;
                    end

                    T10_STAGE2: begin
                        if (!t10_push_result) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            packed_result = '0;
                            for (int phase = 0; phase < T10_PHASES;
                                 phase++) begin
                                if (phase_cycle_q == phase) begin
                                    for (int output_index = 0;
                                         output_index < 32;
                                         output_index++) begin
                                        t10_stage2_sum = {{2{t10_bias_q[
                                            (phase*2)
                                                +(output_index/T10_LANES)]
                                                [ACC_W-1]}},
                                            t10_bias_q[(phase*2)
                                                +(output_index/T10_LANES)]}
                                            + {{(ACC_W+2-(2*IN_W)){
                                                multiplier_product[
                                                    output_index*T10_RANK]
                                                    [(2*IN_W)-1]}},
                                               multiplier_product[
                                                    output_index*T10_RANK]}
                                            + {{(ACC_W+2-(2*IN_W)){
                                                multiplier_product[
                                                    (output_index*T10_RANK)+1]
                                                    [(2*IN_W)-1]}},
                                               multiplier_product[
                                                    (output_index*T10_RANK)+1]}
                                            + {{(ACC_W+2-(2*IN_W)){
                                                multiplier_product[
                                                    (output_index*T10_RANK)+2]
                                                    [(2*IN_W)-1]}},
                                               multiplier_product[
                                                    (output_index*T10_RANK)+2]};
                                        saturated_output
                                            = sat_q26_to_q24(t10_stage2_sum);
                                        packed_result[output_index]
                                            = saturated_output
                                                >= t10_threshold_q;
                                    end
                                end
                            end
                            fifo_mode_q[fifo_write_pointer_q] <= 1'b0;
                            fifo_tag_q[fifo_write_pointer_q]
                                <= bank_tag_q[active_bank_q];
                            fifo_beat_q[fifo_write_pointer_q]
                                <= phase_cycle_q;
                            fifo_valid_bits_q[fifo_write_pointer_q]
                                <= {{16{1'b0}}, {32{1'b1}}};
                            fifo_bits_q[fifo_write_pointer_q] <= packed_result;
                            fifo_write_pointer_q
                                <= fifo_write_pointer_q + 1'b1;
                            if (phase_cycle_q == 4) begin
                                done_q <= 1'b1;
                                done_mode_q <= 1'b0;
                                done_tag_q <= bank_tag_q[active_bank_q];
                                bank_state_q[active_bank_q] <= BANK_EMPTY;
                                next_bank = ~active_bank_q;
                                phase_cycle_q <= '0;
                                if (bank_state_q[next_bank] == BANK_READY) begin
                                    active_bank_q <= next_bank;
                                    bank_state_q[next_bank] <= BANK_ACTIVE;
                                    t10_state_q <= T10_STAGE1;
                                end else begin
                                    t10_state_q <= T10_IDLE;
                                end
                            end else begin
                                phase_cycle_q <= phase_cycle_q + 1'b1;
                            end
                        end
                    end

                    default: begin
                        protocol_error_q <= 1'b1;
                        t10_state_q <= T10_IDLE;
                    end
                endcase
            end else begin
                t10_state_q <= T10_IDLE;
                phase_cycle_q <= '0;
            end
        end
    end
endmodule

`default_nettype wire
