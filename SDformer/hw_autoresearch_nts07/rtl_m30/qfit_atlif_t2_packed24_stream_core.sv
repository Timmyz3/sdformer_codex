`timescale 1ns/1ps
`default_nettype none

// Dense T=2 ATLIF packet engine for the M30B executable packing point.
// Each accepted packet contains 24 independent two-element temporal vectors.
// One shared 96-slot INT8 multiplier issue computes all 24 2x2 matvecs:
// four products per lane, followed by Q24 bias/saturation/threshold packing.
// The module proves an operator-level II=1 packet schedule.  It does not prove
// a unified T10/T2 multiplier hierarchy, SRAM ports, amplitude restoration, or
// any end-to-end speedup.
module qfit_atlif_t2_packed24_stream_core #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int LANES = 24,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int MULTIPLIERS = 96,
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         parameter_valid,
    output logic                         parameter_ready,
    input  logic [(4*IN_W)-1:0]          parameter_weight,
    input  logic [(2*ACC_W)-1:0]         parameter_bias,
    input  logic signed [ACC_W-1:0]      parameter_threshold,
    output logic                         parameter_loaded,
    input  logic                         parameter_release_valid,
    output logic                         parameter_release_ready,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [TAG_W-1:0]             input_tag,
    input  logic [LANES-1:0]             input_lane_valid,
    input  logic [255:0]                 input_t0_values,
    input  logic [255:0]                 input_t1_values,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_W-1:0]             result_tag,
    output logic [LANES-1:0]             result_lane_valid,
    output logic [LANES-1:0]             result_t0_bits,
    output logic [LANES-1:0]             result_t1_bits,
    output logic                         done,
    output logic [TAG_W-1:0]             done_tag,
    output logic                         protocol_error,
    output logic                         busy,
    output logic                         arithmetic_active,
    output logic [MULTIPLIERS-1:0]       multiplier_active_mask,
    output logic [FIFO_COUNT_W-1:0]      result_fifo_occupancy
);
    logic parameter_loaded_q;
    logic signed [IN_W-1:0] weight_q [0:3];
    logic signed [ACC_W-1:0] bias_q [0:1];
    logic signed [ACC_W-1:0] threshold_q;

    logic signed [IN_W-1:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [IN_W-1:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [(2*IN_W)-1:0] multiplier_product [0:MULTIPLIERS-1];

    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [LANES-1:0] fifo_lane_valid_q [0:FIFO_DEPTH-1];
    logic [LANES-1:0] fifo_t0_bits_q [0:FIFO_DEPTH-1];
    logic [LANES-1:0] fifo_t1_bits_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_write_pointer_q;
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;

    logic done_q;
    logic [TAG_W-1:0] done_tag_q;
    logic protocol_error_q;
    logic input_fire;
    logic input_protocol_ok;
    logic result_fire;
    logic push_result;
    logic fifo_credit_available;

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

`ifndef SYNTHESIS
    initial begin
        if (FIFO_DEPTH < 2 || (1 << FIFO_PTR_W) != FIFO_DEPTH)
            $fatal(1, "M30B FIFO_DEPTH must be a power of two and at least two");
        if (LANES != 24 || MULTIPLIERS != 96)
            $fatal(1, "M30B packed resource contract drift");
    end
`endif

    assign parameter_ready = !rst_core && !parameter_loaded_q
        && fifo_count_q == 0 && !protocol_error_q;
    // A held input belongs to the currently loaded context.  Give it priority
    // over release so that it can never be stranded and later consumed by a
    // newly loaded context.
    assign parameter_release_ready = !rst_core && parameter_loaded_q
        && fifo_count_q == 0 && !protocol_error_q && !input_valid;
    assign parameter_loaded = parameter_loaded_q;

    assign result_valid = fifo_count_q != 0;
    assign result_tag = fifo_tag_q[fifo_read_pointer_q];
    assign result_lane_valid = fifo_lane_valid_q[fifo_read_pointer_q];
    assign result_t0_bits = fifo_t0_bits_q[fifo_read_pointer_q];
    assign result_t1_bits = fifo_t1_bits_q[fifo_read_pointer_q];
    assign result_fire = result_valid && result_ready;
    assign fifo_credit_available = fifo_count_q < FIFO_DEPTH || result_fire;

    assign input_protocol_ok = input_lane_valid == {LANES{1'b1}}
        && input_t0_values[255:192] == '0
        && input_t1_values[255:192] == '0;
    assign input_ready = !rst_core && parameter_loaded_q && !protocol_error_q
        && fifo_credit_available;
    assign input_fire = input_valid && input_ready;
    assign push_result = input_fire && input_protocol_ok;

    assign done = done_q;
    assign done_tag = done_tag_q;
    assign protocol_error = protocol_error_q;
    assign busy = fifo_count_q != 0;
    assign arithmetic_active = push_result;
    assign multiplier_active_mask = arithmetic_active
        ? {MULTIPLIERS{1'b1}} : '0;
    assign result_fifo_occupancy = fifo_count_q;

    for (genvar multiplier = 0; multiplier < MULTIPLIERS; multiplier++) begin
        assign multiplier_product[multiplier]
            = multiplier_a[multiplier] * multiplier_b[multiplier];
    end

    always_comb begin : select_t2_multiplier_operands
        for (int lane = 0; lane < LANES; lane++) begin
            multiplier_a[(lane*4)+0]
                = $signed(input_t0_values[(lane*IN_W) +: IN_W]);
            multiplier_b[(lane*4)+0] = weight_q[0];
            multiplier_a[(lane*4)+1]
                = $signed(input_t1_values[(lane*IN_W) +: IN_W]);
            multiplier_b[(lane*4)+1] = weight_q[1];
            multiplier_a[(lane*4)+2]
                = $signed(input_t0_values[(lane*IN_W) +: IN_W]);
            multiplier_b[(lane*4)+2] = weight_q[2];
            multiplier_a[(lane*4)+3]
                = $signed(input_t1_values[(lane*IN_W) +: IN_W]);
            multiplier_b[(lane*4)+3] = weight_q[3];
        end
    end

    always_ff @(posedge clk_core) begin : t2_stream
        logic signed [ACC_W:0] sum_t0;
        logic signed [ACC_W:0] sum_t1;
        logic signed [ACC_W-1:0] saturated_t0;
        logic signed [ACC_W-1:0] saturated_t1;
        logic [LANES-1:0] packed_t0;
        logic [LANES-1:0] packed_t1;
        if (rst_core) begin
            parameter_loaded_q <= 1'b0;
            threshold_q <= '0;
            fifo_write_pointer_q <= '0;
            fifo_read_pointer_q <= '0;
            fifo_count_q <= '0;
            done_q <= 1'b0;
            done_tag_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            done_q <= 1'b0;

            if (parameter_release_valid && parameter_release_ready)
                parameter_loaded_q <= 1'b0;
            if (parameter_valid && parameter_ready) begin
                for (int index = 0; index < 4; index++)
                    weight_q[index] <= $signed(
                        parameter_weight[(index*IN_W) +: IN_W]
                    );
                for (int index = 0; index < 2; index++)
                    bias_q[index] <= $signed(
                        parameter_bias[(index*ACC_W) +: ACC_W]
                    );
                threshold_q <= parameter_threshold;
                parameter_loaded_q <= 1'b1;
            end

            if (input_fire && !input_protocol_ok)
                protocol_error_q <= 1'b1;

            if (push_result) begin
                packed_t0 = '0;
                packed_t1 = '0;
                for (int lane = 0; lane < LANES; lane++) begin
                    sum_t0 = {{1{bias_q[0][ACC_W-1]}}, bias_q[0]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+0][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+0]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+1][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+1]};
                    sum_t1 = {{1{bias_q[1][ACC_W-1]}}, bias_q[1]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+2][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+2]}
                        + {{(ACC_W+1-(2*IN_W)){
                            multiplier_product[(lane*4)+3][(2*IN_W)-1]}},
                           multiplier_product[(lane*4)+3]};
                    saturated_t0 = sat_q25_to_q24(sum_t0);
                    saturated_t1 = sat_q25_to_q24(sum_t1);
                    packed_t0[lane] = saturated_t0 >= threshold_q;
                    packed_t1[lane] = saturated_t1 >= threshold_q;
                end
                fifo_tag_q[fifo_write_pointer_q] <= input_tag;
                fifo_lane_valid_q[fifo_write_pointer_q] <= input_lane_valid;
                fifo_t0_bits_q[fifo_write_pointer_q] <= packed_t0;
                fifo_t1_bits_q[fifo_write_pointer_q] <= packed_t1;
                fifo_write_pointer_q <= fifo_write_pointer_q + 1'b1;
                done_q <= 1'b1;
                done_tag_q <= input_tag;
            end

            if (result_fire)
                fifo_read_pointer_q <= fifo_read_pointer_q + 1'b1;
            case ({push_result, result_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
