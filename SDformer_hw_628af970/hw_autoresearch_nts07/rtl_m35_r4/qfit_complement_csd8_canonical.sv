`timescale 1ns/1ps
`default_nettype none

// M35-r4 H67-ep35 checkpoint-specific canonical complement/CSD engine.
//
// The runtime configuration port carries only descriptor_id.  IDs 0..9 map
// to the immutable canonical table whose canonical JSON SHA256 is
// 209d34c4df8d3babf2ad701ee6c1305b2be17eea8ac7cf2bb62d703c5d9caff7.
// IDs 10..15 are rejected.  In particular, no valid/sign/shift payload is
// exposed at this boundary, so alternate tuples reconstructing the same
// delta are not representable.  A new checkpoint requires a new ROM image,
// contract, source hash, and verification run.
module qfit_complement_csd8_canonical #(
    parameter int TAG_W = 48,
    parameter int EPOCH_W = 16,
    localparam int OUTPUTS = 8,
    localparam int TERMS = 4,
    localparam int RESULT_W = 56,
    localparam logic [63:0] CONFIG_FINGERPRINT64 = 64'h209d34c4df8d3bab
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              config_valid,
    output logic                              config_ready,
    input  logic [EPOCH_W-1:0]                config_epoch,
    input  logic [3:0]                        config_descriptor_id,
    output logic                              config_loaded,
    output logic [EPOCH_W-1:0]                loaded_epoch,
    output logic [3:0]                        loaded_descriptor_id,
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
    logic [3:0] config_descriptor_id_q;
    logic [TERMS-1:0] config_term_valid_q;
    logic [TERMS-1:0] config_term_negative_q;
    logic [3:0] config_term_shift_q [0:TERMS-1];

    logic rom_legal;
    logic [TERMS-1:0] rom_term_valid;
    logic [TERMS-1:0] rom_term_negative;
    logic [3:0] rom_term_shift [0:TERMS-1];

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

    logic stage1_ready;
    logic stage2_ready;
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
            $fatal(1, "M35-r4 canonical engine shape contract drift");
        if (CONFIG_FINGERPRINT64 != 64'h209d34c4df8d3bab)
            $fatal(1, "M35-r4 descriptor ROM fingerprint drift");
    end
`endif

    // Immutable deployment ROM.  Invalid slots have zero metadata.
    always_comb begin : decode_canonical_descriptor
        rom_legal = 1'b1;
        rom_term_valid = 4'b0000;
        rom_term_negative = 4'b0000;
        for (int term = 0; term < TERMS; term++)
            rom_term_shift[term] = 4'd0;
        unique case (config_descriptor_id)
            4'd0: begin
                rom_term_valid = 4'b0001;
                rom_term_shift[0] = 4'd1;
            end
            4'd1: begin
                rom_term_valid = 4'b0011;
                rom_term_negative = 4'b0001;
                rom_term_shift[0] = 4'd0;
                rom_term_shift[1] = 4'd4;
            end
            4'd2: begin
                rom_term_valid = 4'b0001;
                rom_term_shift[0] = 4'd0;
            end
            4'd3: begin
                rom_term_valid = 4'b0111;
                rom_term_shift[0] = 4'd0;
                rom_term_shift[1] = 4'd2;
                rom_term_shift[2] = 4'd4;
            end
            4'd4: begin
                rom_term_valid = 4'b0111;
                rom_term_negative = 4'b0011;
                rom_term_shift[0] = 4'd1;
                rom_term_shift[1] = 4'd4;
                rom_term_shift[2] = 4'd7;
            end
            4'd5: begin
                rom_term_valid = 4'b0011;
                rom_term_shift[0] = 4'd1;
                rom_term_shift[1] = 4'd4;
            end
            4'd6: begin
                rom_term_valid = 4'b0111;
                rom_term_negative = 4'b0010;
                rom_term_shift[0] = 4'd0;
                rom_term_shift[1] = 4'd3;
                rom_term_shift[2] = 4'd7;
            end
            4'd7: begin
                rom_term_valid = 4'b0011;
                rom_term_shift[0] = 4'd4;
                rom_term_shift[1] = 4'd7;
            end
            4'd8: begin
                rom_term_valid = 4'b0111;
                rom_term_negative = 4'b0010;
                rom_term_shift[0] = 4'd0;
                rom_term_shift[1] = 4'd5;
                rom_term_shift[2] = 4'd7;
            end
            4'd9: begin
                rom_term_valid = 4'b1111;
                rom_term_negative = 4'b0001;
                rom_term_shift[0] = 4'd2;
                rom_term_shift[1] = 4'd4;
                rom_term_shift[2] = 4'd6;
                rom_term_shift[3] = 4'd9;
            end
            default: begin
                rom_legal = 1'b0;
            end
        endcase
    end

    assign descriptor_legal = rom_legal;
    assign stage2_ready = !stage2_valid_q || output_ready;
    assign stage1_ready = !stage1_valid_q || stage2_ready;
    assign safe_config_boundary = !stage1_valid_q && !stage2_valid_q;
    assign config_ready = !rst_core && !config_loaded_q
        && safe_config_boundary && !protocol_error;
    assign config_release_ready = !rst_core && config_loaded_q
        && safe_config_boundary && !protocol_error && !input_valid;
    assign input_ready = !rst_core && config_loaded_q
        && !protocol_error && stage1_ready;
    assign config_loaded = config_loaded_q;
    assign loaded_epoch = config_epoch_q;
    assign loaded_descriptor_id = config_descriptor_id_q;
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
            config_descriptor_id_q <= '0;
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
                if (rom_legal) begin
                    config_loaded_q <= 1'b1;
                    config_epoch_q <= config_epoch;
                    config_descriptor_id_q <= config_descriptor_id;
                    config_term_valid_q <= rom_term_valid;
                    config_term_negative_q <= rom_term_negative;
                    for (int term = 0; term < TERMS; term++)
                        config_term_shift_q[term] <= rom_term_shift[term];
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
