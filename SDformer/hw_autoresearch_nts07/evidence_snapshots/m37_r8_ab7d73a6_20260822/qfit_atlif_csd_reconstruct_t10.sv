`timescale 1ns/1ps
`default_nettype none

// M37 phase-decoupled T10 CSD reconstruction engine.
//
// Forty-eight signed-INT8 rank/lane intermediates are reconstructed into two
// temporal rows per issue cycle.  Every one of the 30 layer-static signed
// INT8 left coefficients is represented by a canonical, at-most-four-term
// non-adjacent signed-power descriptor.  The data path contains shifts,
// negation, and addition only; the redundant coefficient value is used only
// by the fail-closed configuration checker.
module qfit_atlif_csd_reconstruct_t10 #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T = 10,
    localparam int RANK = 3,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int TERMS = 4,
    localparam int COEFFICIENTS = T*RANK,
    localparam int INTERMEDIATES = RANK*LANES,
    localparam int PRODUCTS_PER_BEAT = 2*LANES*RANK,
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      config_valid,
    output logic                                      config_ready,
    input  logic [(COEFFICIENTS*IN_W)-1:0]            config_left_factor,
    input  logic [(COEFFICIENTS*TERMS)-1:0]           config_term_valid,
    input  logic [(COEFFICIENTS*TERMS)-1:0]           config_term_negative,
    input  logic [(COEFFICIENTS*TERMS*3)-1:0]         config_term_shift,
    input  logic [(T*ACC_W)-1:0]                      config_bias,
    input  logic signed [ACC_W-1:0]                   config_threshold,
    output logic                                      descriptor_legal,
    output logic                                      config_loaded,
    input  logic                                      config_release_valid,
    output logic                                      config_release_ready,

    input  logic                                      input_valid,
    output logic                                      input_ready,
    input  logic [TAG_W-1:0]                          input_tag,
    input  logic [(INTERMEDIATES*IN_W)-1:0]           input_intermediate,

    output logic                                      result_valid,
    input  logic                                      result_ready,
    output logic [TAG_W-1:0]                          result_tag,
    output logic [2:0]                                result_beat,
    output logic [47:0]                               result_valid_bits,
    output logic [47:0]                               result_bits,
    output logic                                      done,
    output logic [TAG_W-1:0]                          done_tag,
    output logic                                      protocol_error,
    output logic                                      busy,

    output logic                                      arithmetic_active,
    output logic [2:0]                                phase_cycle,
    output logic                                      phase4_chain_accept,
    output logic [1:0]                                input_buffer_occupancy,
    output logic [FIFO_COUNT_W-1:0]                   result_fifo_occupancy,
    output logic                                      result_fifo_push,
    output logic                                      result_fifo_pop,
    output logic [2:0]                                result_fifo_push_beat,
    output logic [TAG_W-1:0]                          result_fifo_push_tag,
    output logic                                      input_accept,
    output logic                                      input_accept_bank,
    output logic                                      active_compute_bank,
    output logic [TAG_W-1:0]                          arithmetic_tag,
    output logic                                      uses_integer_multiplier
);
    logic config_loaded_q;
    logic protocol_error_q;
    logic term_valid_q [0:COEFFICIENTS-1][0:TERMS-1];
    logic term_negative_q [0:COEFFICIENTS-1][0:TERMS-1];
    logic [2:0] term_shift_q [0:COEFFICIENTS-1][0:TERMS-1];
    logic signed [ACC_W-1:0] bias_q [0:T-1];
    logic signed [ACC_W-1:0] threshold_q;

    logic signed [10:0] descriptor_sum_comb [0:COEFFICIENTS-1];
    logic descriptor_seen_invalid_comb [0:COEFFICIENTS-1];
    logic descriptor_previous_valid_comb [0:COEFFICIENTS-1];
    logic [2:0] descriptor_previous_shift_comb [0:COEFFICIENTS-1];

    logic [1:0] bank_valid_q;
    logic [TAG_W-1:0] bank_tag_q [0:1];
    logic signed [IN_W-1:0] intermediate_bank_q
        [0:1][0:INTERMEDIATES-1];
    logic compute_active_q;
    logic active_bank_q;
    logic [2:0] phase_cycle_q;
    logic input_bank;
    logic input_fire;

    logic product_valid_q;
    logic [TAG_W-1:0] product_tag_q;
    logic [2:0] product_beat_q;
    logic signed [17:0] product_q [0:PRODUCTS_PER_BEAT-1];
    logic product_stage_ready;
    logic issue_fire;

    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_valid_bits_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_bits_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_write_pointer_q;
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;
    logic result_fire;
    logic fifo_credit;
    logic push_result;
    logic done_q;
    logic [TAG_W-1:0] done_tag_q;
    logic safe_config_boundary;

    function automatic logic signed [17:0] signed_power_term(
        input logic signed [IN_W-1:0] value,
        input logic term_valid,
        input logic term_negative,
        input logic [2:0] term_shift
    );
        logic signed [17:0] value_wide;
        logic signed [17:0] shifted;
        begin
            value_wide = {{(18-IN_W){value[IN_W-1]}}, value};
            case (term_shift)
                3'd0: shifted = value_wide;
                3'd1: shifted = value_wide <<< 1;
                3'd2: shifted = value_wide <<< 2;
                3'd3: shifted = value_wide <<< 3;
                3'd4: shifted = value_wide <<< 4;
                3'd5: shifted = value_wide <<< 5;
                3'd6: shifted = value_wide <<< 6;
                default: shifted = value_wide <<< 7;
            endcase
            if (!term_valid)
                signed_power_term = 18'sd0;
            else if (term_negative)
                signed_power_term = -$signed(shifted);
            else
                signed_power_term = $signed(shifted);
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
        if (T != 10 || RANK != 3 || LANES != 16 || IN_W != 8
                || ACC_W != 24 || TERMS != 4 || PRODUCTS_PER_BEAT != 96)
            $fatal(1, "M37 shape/arithmetic contract drift");
        if (FIFO_DEPTH < 5 || (1 << FIFO_PTR_W) != FIFO_DEPTH)
            $fatal(1, "M37 FIFO_DEPTH must be a power-of-two and at least five");
    end
`endif

    // Canonical NAF: valid terms are compact, shifts strictly increase with
    // at least one zero between them, invalid fields are zero, and the signed
    // reconstruction equals the redundant signed-INT8 coefficient.
    always_comb begin : validate_all_descriptors
        descriptor_legal = 1'b1;
        for (int coefficient = 0; coefficient < COEFFICIENTS;
             coefficient++) begin
            descriptor_sum_comb[coefficient] = 11'sd0;
            descriptor_seen_invalid_comb[coefficient] = 1'b0;
            descriptor_previous_valid_comb[coefficient] = 1'b0;
            descriptor_previous_shift_comb[coefficient] = 3'd0;
            for (int term = 0; term < TERMS; term++) begin
                if (config_term_valid[(coefficient*TERMS)+term]) begin
                    if (descriptor_seen_invalid_comb[coefficient])
                        descriptor_legal = 1'b0;
                    if (descriptor_previous_valid_comb[coefficient]
                            && ({1'b0, config_term_shift[
                                    (((coefficient*TERMS)+term)*3) +: 3]}
                                <= ({1'b0,
                                    descriptor_previous_shift_comb[coefficient]}
                                    + 4'd1)))
                        descriptor_legal = 1'b0;
                    if (config_term_negative[(coefficient*TERMS)+term])
                        descriptor_sum_comb[coefficient]
                            = descriptor_sum_comb[coefficient]
                                - (11'sd1 <<< config_term_shift[
                                    (((coefficient*TERMS)+term)*3) +: 3]);
                    else
                        descriptor_sum_comb[coefficient]
                            = descriptor_sum_comb[coefficient]
                                + (11'sd1 <<< config_term_shift[
                                    (((coefficient*TERMS)+term)*3) +: 3]);
                    descriptor_previous_valid_comb[coefficient] = 1'b1;
                    descriptor_previous_shift_comb[coefficient]
                        = config_term_shift[
                            (((coefficient*TERMS)+term)*3) +: 3];
                end else begin
                    descriptor_seen_invalid_comb[coefficient] = 1'b1;
                    if (config_term_negative[(coefficient*TERMS)+term]
                            || config_term_shift[
                                (((coefficient*TERMS)+term)*3) +: 3] != 0)
                        descriptor_legal = 1'b0;
                end
            end
            if (descriptor_sum_comb[coefficient]
                    != $signed({{3{config_left_factor[
                            (coefficient*IN_W)+IN_W-1]}},
                        config_left_factor[(coefficient*IN_W) +: IN_W]}))
                descriptor_legal = 1'b0;
        end
    end

    assign result_valid = fifo_count_q != 0;
    assign result_tag = fifo_tag_q[fifo_read_pointer_q];
    assign result_beat = fifo_beat_q[fifo_read_pointer_q];
    assign result_valid_bits = fifo_valid_bits_q[fifo_read_pointer_q];
    assign result_bits = fifo_bits_q[fifo_read_pointer_q];
    assign result_fire = result_valid && result_ready;
    assign fifo_credit = fifo_count_q < FIFO_DEPTH || result_fire;
    assign push_result = product_valid_q && fifo_credit;
    assign product_stage_ready = !product_valid_q || fifo_credit;
    assign issue_fire = compute_active_q && product_stage_ready;

    always_comb begin
        if (!bank_valid_q[0])
            input_bank = 1'b0;
        else if (!bank_valid_q[1])
            input_bank = 1'b1;
        else
            input_bank = active_bank_q;
        input_ready = !rst_core && config_loaded_q && !protocol_error_q
            && (!bank_valid_q[0] || !bank_valid_q[1]
                || (issue_fire && phase_cycle_q == 4));
    end
    assign input_fire = input_valid && input_ready;

    assign safe_config_boundary = !compute_active_q && bank_valid_q == 0
        && !product_valid_q && fifo_count_q == 0;
    assign config_ready = !rst_core && !config_loaded_q
        && safe_config_boundary && !protocol_error_q;
    assign config_release_ready = !rst_core && config_loaded_q
        && safe_config_boundary && !protocol_error_q && !input_valid;
    assign config_loaded = config_loaded_q;
    assign protocol_error = protocol_error_q;
    assign busy = compute_active_q || bank_valid_q != 0
        || product_valid_q || fifo_count_q != 0;
    assign arithmetic_active = issue_fire;
    assign phase_cycle = phase_cycle_q;
    assign phase4_chain_accept = issue_fire && phase_cycle_q == 4
        && input_fire;
    assign input_buffer_occupancy = bank_valid_q[0] + bank_valid_q[1];
    assign result_fifo_occupancy = fifo_count_q;
    assign result_fifo_push = push_result;
    assign result_fifo_pop = result_fire;
    assign result_fifo_push_beat = product_beat_q;
    assign result_fifo_push_tag = product_tag_q;
    assign input_accept = input_fire;
    assign input_accept_bank = input_bank;
    assign active_compute_bank = active_bank_q;
    assign arithmetic_tag = bank_tag_q[active_bank_q];
    assign done = done_q;
    assign done_tag = done_tag_q;
    assign uses_integer_multiplier = 1'b0;

    always_ff @(posedge clk_core) begin : m37_state
        logic signed [17:0] product_sum;
        logic signed [ACC_W+1:0] output_sum;
        logic signed [ACC_W-1:0] saturated_output;
        logic [47:0] packed_result;
        logic next_bank;
        integer selected_row;
        integer selected_lane;
        integer selected_coefficient;
        integer selected_intermediate;
        if (rst_core) begin
            config_loaded_q <= 1'b0;
            protocol_error_q <= 1'b0;
            threshold_q <= '0;
            bank_valid_q <= '0;
            compute_active_q <= 1'b0;
            active_bank_q <= 1'b0;
            phase_cycle_q <= '0;
            product_valid_q <= 1'b0;
            product_tag_q <= '0;
            product_beat_q <= '0;
            fifo_write_pointer_q <= '0;
            fifo_read_pointer_q <= '0;
            fifo_count_q <= '0;
            done_q <= 1'b0;
            done_tag_q <= '0;
            for (int coefficient = 0; coefficient < COEFFICIENTS;
                 coefficient++) begin
                for (int term = 0; term < TERMS; term++) begin
                    term_valid_q[coefficient][term] <= 1'b0;
                    term_negative_q[coefficient][term] <= 1'b0;
                    term_shift_q[coefficient][term] <= '0;
                end
            end
            for (int row = 0; row < T; row++)
                bias_q[row] <= '0;
        end else begin
            done_q <= 1'b0;

            if (config_valid && config_ready) begin
                if (!descriptor_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    config_loaded_q <= 1'b1;
                    threshold_q <= config_threshold;
                    for (int coefficient = 0; coefficient < COEFFICIENTS;
                         coefficient++) begin
                        for (int term = 0; term < TERMS; term++) begin
                            term_valid_q[coefficient][term]
                                <= config_term_valid[
                                    (coefficient*TERMS)+term];
                            term_negative_q[coefficient][term]
                                <= config_term_negative[
                                    (coefficient*TERMS)+term];
                            term_shift_q[coefficient][term]
                                <= config_term_shift[
                                    (((coefficient*TERMS)+term)*3) +: 3];
                        end
                    end
                    for (int row = 0; row < T; row++)
                        bias_q[row] <= $signed(
                            config_bias[(row*ACC_W) +: ACC_W]);
                end
            end
            if (config_release_valid && config_release_ready)
                config_loaded_q <= 1'b0;

            if (result_fire)
                fifo_read_pointer_q <= fifo_read_pointer_q + 1'b1;

            if (push_result) begin
                packed_result = '0;
                for (int output_index = 0; output_index < 2*LANES;
                     output_index++) begin
                    selected_row = (product_beat_q*2)
                        + (output_index/LANES);
                    output_sum = {{2{bias_q[selected_row][ACC_W-1]}},
                        bias_q[selected_row]}
                        + {{(ACC_W+2-18){product_q[
                            output_index*RANK][17]}},
                            product_q[output_index*RANK]}
                        + {{(ACC_W+2-18){product_q[
                            (output_index*RANK)+1][17]}},
                            product_q[(output_index*RANK)+1]}
                        + {{(ACC_W+2-18){product_q[
                            (output_index*RANK)+2][17]}},
                            product_q[(output_index*RANK)+2]};
                    saturated_output = sat_q26_to_q24(output_sum);
                    packed_result[output_index]
                        = saturated_output >= threshold_q;
                end
                fifo_tag_q[fifo_write_pointer_q] <= product_tag_q;
                fifo_beat_q[fifo_write_pointer_q] <= product_beat_q;
                fifo_valid_bits_q[fifo_write_pointer_q]
                    <= {{16{1'b0}}, {32{1'b1}}};
                fifo_bits_q[fifo_write_pointer_q] <= packed_result;
                fifo_write_pointer_q <= fifo_write_pointer_q + 1'b1;
                if (product_beat_q == 4) begin
                    done_q <= 1'b1;
                    done_tag_q <= product_tag_q;
                end
            end
            case ({push_result, result_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase

            if (product_stage_ready) begin
                product_valid_q <= issue_fire;
                if (issue_fire) begin
                    product_tag_q <= bank_tag_q[active_bank_q];
                    product_beat_q <= phase_cycle_q;
                    for (int output_index = 0; output_index < 2*LANES;
                         output_index++) begin
                        selected_row = (phase_cycle_q*2)
                            + (output_index/LANES);
                        selected_lane = output_index % LANES;
                        for (int rank_index = 0; rank_index < RANK;
                             rank_index++) begin
                            // RANK=3: spell out x3 so DC cannot infer a
                            // control-index DW02_mult for this address math.
                            selected_coefficient = (selected_row << 1)
                                + selected_row + rank_index;
                            selected_intermediate
                                = (rank_index*LANES)+selected_lane;
                            product_sum = 18'sd0;
                            for (int term = 0; term < TERMS; term++)
                                product_sum = product_sum
                                    + signed_power_term(
                                        intermediate_bank_q[active_bank_q]
                                            [selected_intermediate],
                                        term_valid_q[selected_coefficient][term],
                                        term_negative_q[
                                            selected_coefficient][term],
                                        term_shift_q[selected_coefficient][term]);
                            product_q[(output_index*RANK)+rank_index]
                                <= product_sum;
                        end
                    end
                end
            end

            if (!compute_active_q) begin
                phase_cycle_q <= '0;
                if (bank_valid_q[0]) begin
                    compute_active_q <= 1'b1;
                    active_bank_q <= 1'b0;
                end else if (bank_valid_q[1]) begin
                    compute_active_q <= 1'b1;
                    active_bank_q <= 1'b1;
                end else if (input_fire) begin
                    compute_active_q <= 1'b1;
                    active_bank_q <= input_bank;
                end
            end else if (issue_fire) begin
                if (phase_cycle_q == 4) begin
                    next_bank = ~active_bank_q;
                    bank_valid_q[active_bank_q] <= 1'b0;
                    phase_cycle_q <= '0;
                    if (bank_valid_q[next_bank]) begin
                        compute_active_q <= 1'b1;
                        active_bank_q <= next_bank;
                    end else if (input_fire) begin
                        compute_active_q <= 1'b1;
                        active_bank_q <= input_bank;
                    end else begin
                        compute_active_q <= 1'b0;
                    end
                end else begin
                    phase_cycle_q <= phase_cycle_q + 1'b1;
                end
            end

            // This assignment intentionally follows phase-4 retirement so a
            // same-cycle replacement of the retiring bank remains occupied.
            if (input_fire) begin
                bank_valid_q[input_bank] <= 1'b1;
                bank_tag_q[input_bank] <= input_tag;
                for (int index = 0; index < INTERMEDIATES; index++)
                    intermediate_bank_q[input_bank][index] <= $signed(
                        input_intermediate[(index*IN_W) +: IN_W]);
            end
        end
    end
endmodule

`default_nettype wire
