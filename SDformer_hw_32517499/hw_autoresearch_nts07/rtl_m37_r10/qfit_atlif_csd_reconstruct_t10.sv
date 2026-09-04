`timescale 1ns/1ps
`default_nettype none

// M37-r10 area-recovery candidate.
//
// The layer-static coefficient descriptors and biases are stored as one packed
// five-entry phase table.  Exactly one shared 168-bit 5:1 mux selects the six
// coefficient descriptors and two biases required by an issue.  All arithmetic
// consumers then use constant packed-vector slices.  This avoids both the r8
// dynamic-unpacked-index Formality boundary and the r9 30-way equality-expanded
// selector.  The coefficient data path remains shift/negate/add only.
module qfit_atlif_csd_reconstruct_t10 #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T = 10,
    localparam int RANK = 3,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int TERMS = 4,
    localparam int PHASES = T/2,
    localparam int COEFFICIENTS = T*RANK,
    localparam int INTERMEDIATES = RANK*LANES,
    localparam int PRODUCTS_PER_BEAT = 2*LANES*RANK,
    localparam int PHASE_COEFFICIENTS = 2*RANK,
    localparam int PHASE_VALID_W = PHASE_COEFFICIENTS*TERMS,
    localparam int PHASE_SHIFT_W = PHASE_VALID_W*3,
    localparam int PHASE_BIAS_W = 2*ACC_W,
    localparam int PHASE_BUNDLE_W = PHASE_VALID_W+PHASE_VALID_W
        +PHASE_SHIFT_W+PHASE_BIAS_W,
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
    logic [(PHASES*PHASE_BUNDLE_W)-1:0] phase_table_q;
    logic [PHASE_BUNDLE_W-1:0] phase_bundle_comb;
    logic [PHASE_VALID_W-1:0] phase_term_valid_comb;
    logic [PHASE_VALID_W-1:0] phase_term_negative_comb;
    logic [PHASE_SHIFT_W-1:0] phase_term_shift_comb;
    logic [PHASE_BIAS_W-1:0] phase_bias_pair_comb;
    logic signed [ACC_W-1:0] threshold_q;

    logic signed [10:0] descriptor_sum_comb [0:COEFFICIENTS-1];
    logic descriptor_seen_invalid_comb [0:COEFFICIENTS-1];
    logic descriptor_previous_valid_comb [0:COEFFICIENTS-1];
    logic [2:0] descriptor_previous_shift_comb [0:COEFFICIENTS-1];

    logic [1:0] bank_valid_q;
    logic [TAG_W-1:0] bank0_tag_q;
    logic [TAG_W-1:0] bank1_tag_q;
    logic [(INTERMEDIATES*IN_W)-1:0] intermediate_bank0_q;
    logic [(INTERMEDIATES*IN_W)-1:0] intermediate_bank1_q;
    logic [(INTERMEDIATES*IN_W)-1:0] active_intermediate_comb;
    logic [TAG_W-1:0] active_tag_comb;
    logic compute_active_q;
    logic active_bank_q;
    logic [2:0] phase_cycle_q;
    logic input_bank;
    logic input_fire;

    logic product_valid_q;
    logic [TAG_W-1:0] product_tag_q;
    logic [2:0] product_beat_q;
    logic signed [17:0] product_q [0:PRODUCTS_PER_BEAT-1];
    logic signed [17:0] issue_product_comb [0:PRODUCTS_PER_BEAT-1];
    logic [PHASE_BIAS_W-1:0] product_bias_pair_q;
    logic [31:0] product_result_comb;
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

    function automatic logic output_event(
        input logic signed [ACC_W-1:0] bias,
        input logic signed [17:0] product0,
        input logic signed [17:0] product1,
        input logic signed [17:0] product2,
        input logic signed [ACC_W-1:0] threshold
    );
        logic signed [ACC_W+1:0] output_sum;
        logic signed [ACC_W-1:0] saturated_output;
        begin
            output_sum = {{2{bias[ACC_W-1]}}, bias}
                + {{(ACC_W+2-18){product0[17]}}, product0}
                + {{(ACC_W+2-18){product1[17]}}, product1}
                + {{(ACC_W+2-18){product2[17]}}, product2};
            saturated_output = sat_q26_to_q24(output_sum);
            output_event = saturated_output >= threshold;
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (T != 10 || RANK != 3 || LANES != 16 || IN_W != 8
                || ACC_W != 24 || TERMS != 4 || PHASES != 5
                || PRODUCTS_PER_BEAT != 96 || PHASE_BUNDLE_W != 168)
            $fatal(1, "M37-r10 shape/arithmetic contract drift");
        if (FIFO_DEPTH < 5 || (1 << FIFO_PTR_W) != FIFO_DEPTH)
            $fatal(1, "M37-r10 FIFO_DEPTH must be power-of-two and >= five");
    end
`endif

    // Canonical NAF validation remains on the live packed configuration pins.
    // Loop indices have compile-time bounds; no stored unpacked array is read.
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

    // One shared 168-bit 5:1 phase-bundle mux.  Every source slice is fixed.
    always_comb begin : select_phase_bundle
        case (phase_cycle_q)
            3'd0: phase_bundle_comb = phase_table_q[0 +: PHASE_BUNDLE_W];
            3'd1: phase_bundle_comb = phase_table_q[168 +: PHASE_BUNDLE_W];
            3'd2: phase_bundle_comb = phase_table_q[336 +: PHASE_BUNDLE_W];
            3'd3: phase_bundle_comb = phase_table_q[504 +: PHASE_BUNDLE_W];
            3'd4: phase_bundle_comb = phase_table_q[672 +: PHASE_BUNDLE_W];
            default: phase_bundle_comb = '0;
        endcase
    end
    assign phase_term_valid_comb = phase_bundle_comb[0 +: PHASE_VALID_W];
    assign phase_term_negative_comb = phase_bundle_comb[
        PHASE_VALID_W +: PHASE_VALID_W];
    assign phase_term_shift_comb = phase_bundle_comb[
        (2*PHASE_VALID_W) +: PHASE_SHIFT_W];
    assign phase_bias_pair_comb = phase_bundle_comb[
        (2*PHASE_VALID_W+PHASE_SHIFT_W) +: PHASE_BIAS_W];

    // Both input banks are packed vectors.  The only bank select is this
    // explicit 2:1 mux; every downstream rank/lane slice is constant.
    always_comb begin : select_active_input
        if (active_bank_q) begin
            active_intermediate_comb = intermediate_bank1_q;
            active_tag_comb = bank1_tag_q;
        end else begin
            active_intermediate_comb = intermediate_bank0_q;
            active_tag_comb = bank0_tag_q;
        end
    end

    genvar output_row_group;
    genvar lane;
    genvar rank;
    generate
        for (output_row_group = 0; output_row_group < 2;
             output_row_group = output_row_group + 1) begin : g_issue_row
            for (lane = 0; lane < LANES; lane = lane + 1) begin : g_issue_lane
                for (rank = 0; rank < RANK; rank = rank + 1) begin : g_issue_rank
                    localparam int DESC = (output_row_group*RANK)+rank;
                    localparam int PRODUCT = ((output_row_group*LANES+lane)*RANK)
                        +rank;
                    localparam int INPUT = (rank*LANES)+lane;
                    always_comb begin
                        issue_product_comb[PRODUCT] = 18'sd0;
                        issue_product_comb[PRODUCT]
                            = issue_product_comb[PRODUCT] + signed_power_term(
                                $signed(active_intermediate_comb[
                                    (INPUT*IN_W) +: IN_W]),
                                phase_term_valid_comb[(DESC*TERMS)+0],
                                phase_term_negative_comb[(DESC*TERMS)+0],
                                phase_term_shift_comb[(DESC*TERMS*3)+0 +: 3]);
                        issue_product_comb[PRODUCT]
                            = issue_product_comb[PRODUCT] + signed_power_term(
                                $signed(active_intermediate_comb[
                                    (INPUT*IN_W) +: IN_W]),
                                phase_term_valid_comb[(DESC*TERMS)+1],
                                phase_term_negative_comb[(DESC*TERMS)+1],
                                phase_term_shift_comb[(DESC*TERMS*3)+3 +: 3]);
                        issue_product_comb[PRODUCT]
                            = issue_product_comb[PRODUCT] + signed_power_term(
                                $signed(active_intermediate_comb[
                                    (INPUT*IN_W) +: IN_W]),
                                phase_term_valid_comb[(DESC*TERMS)+2],
                                phase_term_negative_comb[(DESC*TERMS)+2],
                                phase_term_shift_comb[(DESC*TERMS*3)+6 +: 3]);
                        issue_product_comb[PRODUCT]
                            = issue_product_comb[PRODUCT] + signed_power_term(
                                $signed(active_intermediate_comb[
                                    (INPUT*IN_W) +: IN_W]),
                                phase_term_valid_comb[(DESC*TERMS)+3],
                                phase_term_negative_comb[(DESC*TERMS)+3],
                                phase_term_shift_comb[(DESC*TERMS*3)+9 +: 3]);
                    end
                end
            end
        end
    endgenerate

    genvar result_row_group;
    genvar result_lane;
    generate
        for (result_row_group = 0; result_row_group < 2;
             result_row_group = result_row_group + 1) begin : g_result_row
            for (result_lane = 0; result_lane < LANES;
                 result_lane = result_lane + 1) begin : g_result_lane
                localparam int OUTPUT = (result_row_group*LANES)+result_lane;
                localparam int PRODUCT0 = OUTPUT*RANK;
                assign product_result_comb[OUTPUT] = output_event(
                    $signed(product_bias_pair_q[
                        (result_row_group*ACC_W) +: ACC_W]),
                    product_q[PRODUCT0],
                    product_q[PRODUCT0+1],
                    product_q[PRODUCT0+2],
                    threshold_q);
            end
        end
    endgenerate

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
    assign arithmetic_tag = active_tag_comb;
    assign done = done_q;
    assign done_tag = done_tag_q;
    assign uses_integer_multiplier = 1'b0;

    always_ff @(posedge clk_core) begin : m37_r10_state
        logic next_bank;
        if (rst_core) begin
            config_loaded_q <= 1'b0;
            protocol_error_q <= 1'b0;
            phase_table_q <= '0;
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
        end else begin
            done_q <= 1'b0;

            if (config_valid && config_ready) begin
                if (!descriptor_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    config_loaded_q <= 1'b1;
                    threshold_q <= config_threshold;
                    for (int config_phase = 0; config_phase < PHASES;
                         config_phase++) begin
                        phase_table_q[(config_phase*PHASE_BUNDLE_W)
                                +: PHASE_BUNDLE_W]
                            <= {config_bias[(config_phase*PHASE_BIAS_W)
                                    +: PHASE_BIAS_W],
                                config_term_shift[(config_phase*PHASE_SHIFT_W)
                                    +: PHASE_SHIFT_W],
                                config_term_negative[
                                    (config_phase*PHASE_VALID_W)
                                    +: PHASE_VALID_W],
                                config_term_valid[(config_phase*PHASE_VALID_W)
                                    +: PHASE_VALID_W]};
                    end
                end
            end
            if (config_release_valid && config_release_ready)
                config_loaded_q <= 1'b0;

            if (result_fire)
                fifo_read_pointer_q <= fifo_read_pointer_q + 1'b1;

            if (push_result) begin
                fifo_tag_q[fifo_write_pointer_q] <= product_tag_q;
                fifo_beat_q[fifo_write_pointer_q] <= product_beat_q;
                fifo_valid_bits_q[fifo_write_pointer_q]
                    <= {{16{1'b0}}, {32{1'b1}}};
                fifo_bits_q[fifo_write_pointer_q]
                    <= {{16{1'b0}}, product_result_comb};
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
                    product_tag_q <= active_tag_comb;
                    product_beat_q <= phase_cycle_q;
                    product_bias_pair_q <= phase_bias_pair_comb;
                    for (int product_index = 0;
                         product_index < PRODUCTS_PER_BEAT; product_index++)
                        product_q[product_index]
                            <= issue_product_comb[product_index];
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

            // Preserve r8/r9 phase-4 same-cycle bank replacement semantics.
            if (input_fire) begin
                bank_valid_q[input_bank] <= 1'b1;
                if (input_bank) begin
                    bank1_tag_q <= input_tag;
                    intermediate_bank1_q <= input_intermediate;
                end else begin
                    bank0_tag_q <= input_tag;
                    intermediate_bank0_q <= input_intermediate;
                end
            end
        end
    end
endmodule

`default_nettype wire
