`timescale 1ns/1ps
`default_nettype none

// M30 T10/rank-3 ATLIF operator-cohort streaming core.
//
// One factor/bias/threshold context is loaded once per operator cohort.  Each
// 16-lane T10 input tile then arrives as five 256-bit beats.  Two input banks
// overlap the next tile's five-beat load with the current tile's ten exact-96
// arithmetic cycles.  A 16-entry result FIFO absorbs bounded downstream
// stalls, and the Q24 threshold result is emitted as 32 packed binary values
// per beat rather than materializing 32 Q24 values.
//
// This module proves an unstalled steady-state II=10 arithmetic schedule.  It
// does not by itself prove trained INT8 accuracy, SRAM macro timing, system
// speedup, or energy.
module qfit_atlif_rank3_resident_stream_core #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T = 10,
    localparam int RANK = 3,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int INPUT_BEAT_BITS = 256,
    localparam int INPUTS_PER_BEAT = 32,
    localparam int OUTPUTS_PER_BEAT = 32,
    localparam int MULTIPLIERS = 96,
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  logic                                clk_core,
    input  logic                                rst_core,

    input  logic                                parameter_valid,
    output logic                                parameter_ready,
    input  logic [(RANK*T*IN_W)-1:0]            parameter_right_factor,
    input  logic [(T*RANK*IN_W)-1:0]            parameter_left_factor,
    input  logic [(T*ACC_W)-1:0]                parameter_bias_by_row,
    input  logic signed [ACC_W-1:0]             parameter_threshold,
    input  logic [4:0]                          parameter_requant_shift,
    output logic                                parameter_loaded,
    input  logic                                parameter_release_valid,
    output logic                                parameter_release_ready,

    input  logic                                input_valid,
    output logic                                input_ready,
    input  logic [TAG_W-1:0]                    input_tag,
    input  logic [2:0]                          input_beat,
    input  logic [INPUT_BEAT_BITS-1:0]          input_values,

    output logic                                result_valid,
    input  logic                                result_ready,
    output logic [TAG_W-1:0]                    result_tag,
    output logic [2:0]                          result_beat,
    output logic [OUTPUTS_PER_BEAT-1:0]         result_bits,
    output logic                                done,
    output logic [TAG_W-1:0]                    done_tag,
    output logic                                protocol_error,
    output logic                                busy,

    output logic                                arithmetic_active,
    output logic                                stage_select,
    output logic [2:0]                          phase_cycle,
    output logic [MULTIPLIERS-1:0]              multiplier_active_mask,
    output logic [FIFO_COUNT_W-1:0]             result_fifo_occupancy
);
    typedef enum logic [1:0] {IDLE, STAGE1, WAIT_STAGE2, STAGE2} state_t;
    typedef enum logic [1:0] {
        BANK_EMPTY, BANK_FILL, BANK_READY, BANK_ACTIVE
    } bank_state_t;

    state_t state_q;
    bank_state_t bank_state_q [0:1];
    logic active_bank_q;
    logic fill_active_q;
    logic fill_bank_q;
    logic [2:0] expected_input_beat_q;
    logic [TAG_W-1:0] bank_tag_q [0:1];
    logic signed [IN_W-1:0] x_bank_q [0:1][0:(T*LANES)-1];

    logic parameter_loaded_q;
    logic signed [IN_W-1:0] right_q [0:(RANK*T)-1];
    logic signed [IN_W-1:0] left_q [0:(T*RANK)-1];
    logic signed [ACC_W-1:0] bias_q [0:T-1];
    logic signed [ACC_W-1:0] threshold_q;
    logic [4:0] requant_shift_q;

    logic signed [ACC_W-1:0] stage1_acc_q [0:(RANK*LANES)-1];
    logic signed [IN_W-1:0] intermediate_q [0:(RANK*LANES)-1];
    logic signed [IN_W-1:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [IN_W-1:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [(2*IN_W)-1:0] multiplier_product [0:MULTIPLIERS-1];

    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [OUTPUTS_PER_BEAT-1:0] fifo_bits_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_write_pointer_q;
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;

    logic done_q;
    logic [TAG_W-1:0] done_tag_q;
    logic protocol_error_q;
    logic [2:0] phase_cycle_q;
    logic input_fire;
    logic input_bank;
    logic input_protocol_ok;
    logic result_fire;
    logic push_result;
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
            $fatal(1, "M30 FIFO_DEPTH must be a power of two and at least ten");
        if (INPUT_BEAT_BITS != 256 || INPUTS_PER_BEAT != 32)
            $fatal(1, "M30 input port must remain one 256-bit beat");
        if (T != 10 || RANK != 3 || LANES != 16 || MULTIPLIERS != 96)
            $fatal(1, "M30 shape/resource contract drift");
    end
`endif

    assign safe_parameter_boundary = state_q == IDLE
        && bank_state_q[0] == BANK_EMPTY
        && bank_state_q[1] == BANK_EMPTY
        && fifo_count_q == 0
        && !fill_active_q;
    assign parameter_ready = !rst_core && !parameter_loaded_q
        && safe_parameter_boundary
        && !protocol_error_q;
    assign parameter_release_ready = !rst_core && parameter_loaded_q
        && safe_parameter_boundary && !protocol_error_q;
    assign parameter_loaded = parameter_loaded_q;
    assign input_bank = fill_active_q ? fill_bank_q
        : (bank_state_q[0] == BANK_EMPTY ? 1'b0 : 1'b1);
    assign input_ready = !rst_core && parameter_loaded_q && !protocol_error_q
        && (fill_active_q || !parameter_release_valid)
        && (fill_active_q
            || bank_state_q[0] == BANK_EMPTY
            || bank_state_q[1] == BANK_EMPTY);
    assign input_fire = input_valid && input_ready;
    assign input_protocol_ok = fill_active_q
        ? (input_beat == expected_input_beat_q
            && input_tag == bank_tag_q[fill_bank_q])
        : (input_beat == 0);

    assign result_valid = fifo_count_q != 0;
    assign result_tag = fifo_tag_q[fifo_read_pointer_q];
    assign result_beat = fifo_beat_q[fifo_read_pointer_q];
    assign result_bits = fifo_bits_q[fifo_read_pointer_q];
    assign result_fire = result_valid && result_ready;
    assign fifo_free_slots = FIFO_DEPTH - fifo_count_q;
    assign push_result = state_q == STAGE2 && fifo_count_q < FIFO_DEPTH;
    assign result_fifo_occupancy = fifo_count_q;

    assign done = done_q;
    assign done_tag = done_tag_q;
    assign protocol_error = protocol_error_q;
    assign busy = state_q != IDLE
        || bank_state_q[0] != BANK_EMPTY
        || bank_state_q[1] != BANK_EMPTY
        || fifo_count_q != 0;
    assign stage_select = state_q == STAGE2;
    assign phase_cycle = phase_cycle_q;
    assign arithmetic_active = state_q == STAGE1 || state_q == STAGE2;
    assign multiplier_active_mask = arithmetic_active
        ? {MULTIPLIERS{1'b1}} : '0;

    for (genvar multiplier = 0; multiplier < MULTIPLIERS; multiplier++) begin
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
                            = x_bank_q[active_bank_q]
                                [(((phase_cycle_q*2)+pair)*LANES)+lane];
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

    always_ff @(posedge clk_core) begin : stream_state
        logic signed [ACC_W:0] stage1_sum;
        logic signed [ACC_W+1:0] stage2_sum;
        logic signed [ACC_W-1:0] saturated_output;
        logic [OUTPUTS_PER_BEAT-1:0] packed_result;
        logic next_bank;
        if (rst_core) begin
            state_q <= IDLE;
            bank_state_q[0] <= BANK_EMPTY;
            bank_state_q[1] <= BANK_EMPTY;
            active_bank_q <= 1'b0;
            fill_active_q <= 1'b0;
            fill_bank_q <= 1'b0;
            expected_input_beat_q <= '0;
            bank_tag_q[0] <= '0;
            bank_tag_q[1] <= '0;
            parameter_loaded_q <= 1'b0;
            threshold_q <= '0;
            requant_shift_q <= '0;
            phase_cycle_q <= '0;
            fifo_write_pointer_q <= '0;
            fifo_read_pointer_q <= '0;
            fifo_count_q <= '0;
            done_q <= 1'b0;
            done_tag_q <= '0;
            protocol_error_q <= 1'b0;
            for (int index = 0; index < RANK*LANES; index++) begin
                stage1_acc_q[index] <= '0;
                intermediate_q[index] <= '0;
            end
        end else begin
            done_q <= 1'b0;

            if (parameter_release_valid && parameter_release_ready)
                parameter_loaded_q <= 1'b0;

            if (parameter_valid && parameter_ready) begin
                if (parameter_requant_shift > 23) begin
                    protocol_error_q <= 1'b1;
                end else begin
                for (int index = 0; index < RANK*T; index++)
                    right_q[index] <= $signed(
                        parameter_right_factor[(index*IN_W) +: IN_W]
                    );
                for (int index = 0; index < T*RANK; index++)
                    left_q[index] <= $signed(
                        parameter_left_factor[(index*IN_W) +: IN_W]
                    );
                for (int row = 0; row < T; row++)
                    bias_q[row] <= $signed(
                        parameter_bias_by_row[(row*ACC_W) +: ACC_W]
                    );
                threshold_q <= parameter_threshold;
                requant_shift_q <= parameter_requant_shift;
                parameter_loaded_q <= 1'b1;
                end
            end

            if (input_fire) begin
                if (!input_protocol_ok) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    for (int index = 0; index < INPUTS_PER_BEAT; index++)
                        x_bank_q[input_bank]
                            [(input_beat*INPUTS_PER_BEAT)+index]
                            <= $signed(input_values[(index*IN_W) +: IN_W]);
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
            end

            if (result_fire)
                fifo_read_pointer_q <= fifo_read_pointer_q + 1'b1;

            case ({push_result, result_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase

            case (state_q)
                IDLE: begin
                    phase_cycle_q <= '0;
                    if (bank_state_q[0] == BANK_READY) begin
                        active_bank_q <= 1'b0;
                        bank_state_q[0] <= BANK_ACTIVE;
                        state_q <= STAGE1;
                    end else if (bank_state_q[1] == BANK_READY) begin
                        active_bank_q <= 1'b1;
                        bank_state_q[1] <= BANK_ACTIVE;
                        state_q <= STAGE1;
                    end
                end

                STAGE1: begin
                    for (int accumulator = 0; accumulator < RANK*LANES;
                         accumulator++) begin
                        stage1_sum = {
                            (phase_cycle_q == 0 ? 1'b0
                                : stage1_acc_q[accumulator][ACC_W-1]),
                            (phase_cycle_q == 0 ? {ACC_W{1'b0}}
                                : stage1_acc_q[accumulator])
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
                                <= rne_sat_q24_to_q8(
                                    stage1_sum[ACC_W-1:0], requant_shift_q
                                );
                        else
                            stage1_acc_q[accumulator]
                                <= stage1_sum[ACC_W-1:0];
                    end
                    if (phase_cycle_q == 4) begin
                        phase_cycle_q <= '0;
                        if (fifo_free_slots >= 5)
                            state_q <= STAGE2;
                        else
                            state_q <= WAIT_STAGE2;
                    end else begin
                        phase_cycle_q <= phase_cycle_q + 1'b1;
                    end
                end

                WAIT_STAGE2: begin
                    phase_cycle_q <= '0;
                    if (fifo_free_slots >= 5)
                        state_q <= STAGE2;
                end

                STAGE2: begin
                    if (!push_result) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        packed_result = '0;
                        for (int output_index = 0;
                             output_index < OUTPUTS_PER_BEAT; output_index++) begin
                            stage2_sum = {{2{bias_q[
                                (phase_cycle_q*2)+(output_index/LANES)][ACC_W-1]}},
                                bias_q[(phase_cycle_q*2)+(output_index/LANES)]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)+1][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)+1]}
                                + {{(ACC_W+2-(2*IN_W)){
                                    multiplier_product[(output_index*RANK)+2][(2*IN_W)-1]}},
                                   multiplier_product[(output_index*RANK)+2]};
                            saturated_output = sat_q26_to_q24(stage2_sum);
                            packed_result[output_index]
                                = saturated_output >= threshold_q;
                        end
                        fifo_tag_q[fifo_write_pointer_q]
                            <= bank_tag_q[active_bank_q];
                        fifo_beat_q[fifo_write_pointer_q] <= phase_cycle_q;
                        fifo_bits_q[fifo_write_pointer_q] <= packed_result;
                        fifo_write_pointer_q <= fifo_write_pointer_q + 1'b1;
                        if (phase_cycle_q == 4) begin
                            done_q <= 1'b1;
                            done_tag_q <= bank_tag_q[active_bank_q];
                            bank_state_q[active_bank_q] <= BANK_EMPTY;
                            next_bank = ~active_bank_q;
                            phase_cycle_q <= '0;
                            if (bank_state_q[next_bank] == BANK_READY) begin
                                active_bank_q <= next_bank;
                                bank_state_q[next_bank] <= BANK_ACTIVE;
                                state_q <= STAGE1;
                            end else begin
                                state_q <= IDLE;
                            end
                        end else begin
                            phase_cycle_q <= phase_cycle_q + 1'b1;
                        end
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
