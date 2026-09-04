`timescale 1ns/1ps
`default_nettype none

// M35 checkpoint-exact complement/CSD late-scale engine.
//
// A layer-static descriptor encodes delta as at most four signed powers of
// two, where threshold_uq0p24_raw = 2^24-delta.  Eight signed Acc32 inputs are
// scaled exactly as (Acc<<24)-Acc*delta without an integer multiplier.  The
// two-stage elastic pipeline is independent of the M31 INT8 multiplier pool.
// RNE, bias, saturation, system scheduling, and accuracy remain outside.
module qfit_complement_csd8_late_scale #(
    parameter int TAG_W = 48,
    parameter int EPOCH_W = 16,
    localparam int OUTPUTS = 8,
    localparam int TERMS = 4,
    localparam int RESULT_W = 56
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              config_valid,
    output logic                              config_ready,
    input  logic [EPOCH_W-1:0]                config_epoch,
    input  logic [9:0]                        config_delta,
    input  logic [TERMS-1:0]                  config_term_valid,
    input  logic [TERMS-1:0]                  config_term_negative,
    input  logic [3:0]                        config_term_shift [0:TERMS-1],
    output logic                              config_loaded,
    output logic [EPOCH_W-1:0]                loaded_epoch,
    input  logic                              config_release_valid,
    output logic                              config_release_ready,

    input  logic                              input_valid,
    output logic                              input_ready,
    input  logic [TAG_W-1:0]                  input_tag,
    input  logic [OUTPUTS-1:0]                input_valid_bits,
    input  logic signed [31:0]                input_accumulator [0:OUTPUTS-1],

    output logic                              output_valid,
    input  logic                              output_ready,
    output logic [TAG_W-1:0]                  output_tag,
    output logic [EPOCH_W-1:0]                output_epoch,
    output logic [OUTPUTS-1:0]                output_valid_bits,
    output logic signed [RESULT_W-1:0]        output_product [0:OUTPUTS-1],

    output logic                              descriptor_legal,
    output logic                              uses_integer_multiplier,
    output logic                              busy,
    output logic                              protocol_error
);
    logic config_loaded_q;
    logic [EPOCH_W-1:0] config_epoch_q;
    logic [9:0] config_delta_q;
    logic [TERMS-1:0] config_term_valid_q;
    logic [TERMS-1:0] config_term_negative_q;
    logic [3:0] config_term_shift_q [0:TERMS-1];

    logic stage1_valid_q;
    logic [TAG_W-1:0] stage1_tag_q;
    logic [EPOCH_W-1:0] stage1_epoch_q;
    logic [OUTPUTS-1:0] stage1_valid_bits_q;
    logic signed [RESULT_W-1:0] stage1_base_q [0:OUTPUTS-1];
    logic signed [RESULT_W-1:0] stage1_pair01_q [0:OUTPUTS-1];
    logic signed [RESULT_W-1:0] stage1_pair23_q [0:OUTPUTS-1];

    logic stage2_valid_q;
    logic [TAG_W-1:0] stage2_tag_q;
    logic [EPOCH_W-1:0] stage2_epoch_q;
    logic [OUTPUTS-1:0] stage2_valid_bits_q;
    logic signed [RESULT_W-1:0] stage2_product_q [0:OUTPUTS-1];

    logic signed [12:0] descriptor_sum;
    logic descriptor_shift_legal;
    logic stage1_ready;
    logic stage2_ready;
    logic input_fire;
    logic safe_config_boundary;

    function automatic logic signed [RESULT_W-1:0] shifted_term(
        input logic signed [31:0] accumulator,
        input logic term_valid,
        input logic term_negative,
        input logic [3:0] term_shift
    );
        logic signed [RESULT_W-1:0] accumulator_wide;
        logic signed [RESULT_W-1:0] magnitude;
        begin
            accumulator_wide = $signed({{24{accumulator[31]}}, accumulator});
            magnitude = accumulator_wide <<< term_shift;
            if (!term_valid)
                shifted_term = 56'sd0;
            else if (term_negative)
                shifted_term = -$signed(magnitude);
            else
                shifted_term = $signed(magnitude);
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (OUTPUTS != 8 || TERMS != 4 || RESULT_W != 56)
            $fatal(1, "M35 complement/CSD shape contract drift");
    end
`endif

    always_comb begin : validate_descriptor
        descriptor_sum = 13'sd0;
        descriptor_shift_legal = 1'b1;
        for (int term = 0; term < TERMS; term++) begin
            if (config_term_valid[term]) begin
                if (config_term_shift[term] > 9)
                    descriptor_shift_legal = 1'b0;
                if (config_term_negative[term])
                    descriptor_sum -= 13'sd1 <<< config_term_shift[term];
                else
                    descriptor_sum += 13'sd1 <<< config_term_shift[term];
            end
        end
        descriptor_legal = descriptor_shift_legal
            && descriptor_sum >= 0
            && descriptor_sum == $signed({3'b000, config_delta});
    end

    assign stage2_ready = !stage2_valid_q || output_ready;
    assign stage1_ready = !stage1_valid_q || stage2_ready;
    assign safe_config_boundary = !stage1_valid_q && !stage2_valid_q;
    assign config_ready = !rst_core && !config_loaded_q
        && safe_config_boundary && !protocol_error;
    assign config_release_ready = !rst_core && config_loaded_q
        && safe_config_boundary && !protocol_error && !input_valid;
    assign input_ready = !rst_core && config_loaded_q
        && !protocol_error && stage1_ready;
    assign input_fire = input_valid && input_ready;
    assign config_loaded = config_loaded_q;
    assign loaded_epoch = config_epoch_q;
    assign output_valid = stage2_valid_q;
    assign output_tag = stage2_tag_q;
    assign output_epoch = stage2_epoch_q;
    assign output_valid_bits = stage2_valid_bits_q;
    assign busy = stage1_valid_q || stage2_valid_q;
    assign uses_integer_multiplier = 1'b0;
    for (genvar output_index = 0; output_index < OUTPUTS; output_index++) begin
        assign output_product[output_index] = stage2_product_q[output_index];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            config_loaded_q <= 1'b0;
            config_epoch_q <= '0;
            config_delta_q <= '0;
            config_term_valid_q <= '0;
            config_term_negative_q <= '0;
            for (int term = 0; term < TERMS; term++)
                config_term_shift_q[term] <= '0;
            stage1_valid_q <= 1'b0;
            stage2_valid_q <= 1'b0;
            stage1_tag_q <= '0;
            stage2_tag_q <= '0;
            stage1_epoch_q <= '0;
            stage2_epoch_q <= '0;
            stage1_valid_bits_q <= '0;
            stage2_valid_bits_q <= '0;
            protocol_error <= 1'b0;
            for (int output_index = 0; output_index < OUTPUTS; output_index++) begin
                stage1_base_q[output_index] <= '0;
                stage1_pair01_q[output_index] <= '0;
                stage1_pair23_q[output_index] <= '0;
                stage2_product_q[output_index] <= '0;
            end
        end else begin
            if (config_valid && config_ready) begin
                if (descriptor_legal) begin
                    config_loaded_q <= 1'b1;
                    config_epoch_q <= config_epoch;
                    config_delta_q <= config_delta;
                    config_term_valid_q <= config_term_valid;
                    config_term_negative_q <= config_term_negative;
                    for (int term = 0; term < TERMS; term++)
                        config_term_shift_q[term] <= config_term_shift[term];
                end else begin
                    protocol_error <= 1'b1;
                end
            end
            if (config_release_valid && config_release_ready)
                config_loaded_q <= 1'b0;

            if (stage2_ready) begin
                stage2_valid_q <= stage1_valid_q;
                if (stage1_valid_q) begin
                    stage2_tag_q <= stage1_tag_q;
                    stage2_epoch_q <= stage1_epoch_q;
                    stage2_valid_bits_q <= stage1_valid_bits_q;
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++)
                        stage2_product_q[output_index] <=
                            $signed(stage1_base_q[output_index])
                            - ($signed(stage1_pair01_q[output_index])
                                + $signed(stage1_pair23_q[output_index]));
                end
            end
            if (stage1_ready) begin
                stage1_valid_q <= input_valid && config_loaded_q
                    && !protocol_error;
                if (input_valid && config_loaded_q && !protocol_error) begin
                    stage1_tag_q <= input_tag;
                    stage1_epoch_q <= config_epoch_q;
                    stage1_valid_bits_q <= input_valid_bits;
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++) begin
                        stage1_base_q[output_index] <= $signed({
                            input_accumulator[output_index], 24'b0
                        });
                        stage1_pair01_q[output_index] <=
                            shifted_term(input_accumulator[output_index],
                                config_term_valid_q[0],
                                config_term_negative_q[0],
                                config_term_shift_q[0])
                            + shifted_term(input_accumulator[output_index],
                                config_term_valid_q[1],
                                config_term_negative_q[1],
                                config_term_shift_q[1]);
                        stage1_pair23_q[output_index] <=
                            shifted_term(input_accumulator[output_index],
                                config_term_valid_q[2],
                                config_term_negative_q[2],
                                config_term_shift_q[2])
                            + shifted_term(input_accumulator[output_index],
                                config_term_valid_q[3],
                                config_term_negative_q[3],
                                config_term_shift_q[3]);
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
