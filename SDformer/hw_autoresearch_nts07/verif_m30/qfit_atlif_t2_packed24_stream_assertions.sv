`timescale 1ns/1ps
`default_nettype none

module qfit_atlif_t2_packed24_stream_assertions #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic parameter_valid,
    input logic parameter_ready,
    input logic [31:0] parameter_weight,
    input logic [47:0] parameter_bias,
    input logic signed [23:0] parameter_threshold,
    input logic parameter_loaded,
    input logic parameter_release_valid,
    input logic parameter_release_ready,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [23:0] input_lane_valid,
    input logic [255:0] input_t0_values,
    input logic [255:0] input_t1_values,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [23:0] result_lane_valid,
    input logic [23:0] result_t0_bits,
    input logic [23:0] result_t1_bits,
    input logic done,
    input logic [TAG_W-1:0] done_tag,
    input logic protocol_error,
    input logic arithmetic_active,
    input logic [95:0] multiplier_active_mask,
    input logic [FIFO_COUNT_W-1:0] result_fifo_occupancy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M30B_SVA_BOUND=1");

    assert property (parameter_valid && !parameter_ready |=>
        $stable({parameter_valid, parameter_weight, parameter_bias,
                 parameter_threshold}));
    assert property (parameter_valid && parameter_ready |=> parameter_loaded);
    assert property (parameter_release_valid && parameter_release_ready
        |=> !parameter_loaded);
    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_lane_valid,
                 input_t0_values, input_t1_values}));
    assert property (!(parameter_release_valid && parameter_release_ready
                       && input_valid && input_ready));
    assert property (parameter_release_valid && input_valid
        |-> !parameter_release_ready);
    assert property (input_valid && input_ready |-> parameter_loaded);
    assert property (result_valid && !result_ready |=>
        $stable({result_valid, result_tag, result_lane_valid,
                 result_t0_bits, result_t1_bits}));
    assert property (arithmetic_active
        |-> multiplier_active_mask == {96{1'b1}});
    assert property (!arithmetic_active |-> multiplier_active_mask == '0);
    assert property (done |-> $past(input_valid && input_ready));
    assert property (done |-> done_tag == $past(input_tag));
    assert property (done |-> !$isunknown(done_tag));
    assert property (result_fifo_occupancy <= FIFO_DEPTH);
    assert property (!protocol_error);
    cover property (input_valid && input_ready
        ##1 input_valid && input_ready);
    cover property (result_fifo_occupancy == FIFO_DEPTH);
    cover property (parameter_release_valid && !parameter_release_ready
        && input_valid && input_ready);
endmodule

bind qfit_atlif_t2_packed24_stream_core
    qfit_atlif_t2_packed24_stream_assertions #(
        .TAG_W(TAG_W), .FIFO_DEPTH(FIFO_DEPTH)
    ) m30b_assertions (.*);

`default_nettype wire
