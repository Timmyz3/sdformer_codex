`timescale 1ns/1ps
`default_nettype none

module qfit_atlif_rank3_resident_stream_assertions #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic parameter_valid,
    input logic parameter_ready,
    input logic [(3*10*8)-1:0] parameter_right_factor,
    input logic [(10*3*8)-1:0] parameter_left_factor,
    input logic [(10*24)-1:0] parameter_bias_by_row,
    input logic signed [23:0] parameter_threshold,
    input logic [4:0] parameter_requant_shift,
    input logic parameter_loaded,
    input logic parameter_release_valid,
    input logic parameter_release_ready,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [2:0] input_beat,
    input logic [255:0] input_values,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [2:0] result_beat,
    input logic [31:0] result_bits,
    input logic done,
    input logic [TAG_W-1:0] done_tag,
    input logic protocol_error,
    input logic busy,
    input logic arithmetic_active,
    input logic stage_select,
    input logic [2:0] phase_cycle,
    input logic [95:0] multiplier_active_mask,
    input logic [FIFO_COUNT_W-1:0] result_fifo_occupancy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M30_SVA_BOUND=1");

    assert property (input_valid && !input_ready
        && !(parameter_release_valid && parameter_release_ready) |=>
        $stable({input_valid, input_tag, input_beat, input_values}));
    assert property (parameter_valid && !parameter_ready |=>
        $stable({parameter_valid, parameter_right_factor,
                 parameter_left_factor, parameter_bias_by_row,
                 parameter_threshold, parameter_requant_shift}));
    assert property (parameter_release_valid && !parameter_release_ready |=>
        $stable(parameter_release_valid));
    assert property (parameter_valid && parameter_ready |->
        parameter_requant_shift <= 23);
    assert property (parameter_valid && parameter_ready |=> parameter_loaded);
    assert property (parameter_release_valid && parameter_release_ready
        |=> !parameter_loaded);
    assert property (parameter_ready |-> !busy);
    assert property (parameter_release_ready |-> !busy);
    assert property (input_valid && input_ready |-> parameter_loaded);
    assert property (!(parameter_release_valid && parameter_release_ready
                       && input_valid && input_ready));
    assert property (result_valid && !result_ready |=>
        $stable({result_valid, result_tag, result_beat, result_bits}));
    assert property (result_valid |-> result_beat <= 4);
    assert property (arithmetic_active |-> multiplier_active_mask == {96{1'b1}});
    assert property (!arithmetic_active |-> multiplier_active_mask == '0);
    assert property (arithmetic_active |-> phase_cycle <= 4);
    assert property (done |-> $past(arithmetic_active && stage_select
                                     && phase_cycle == 4));
    assert property (done |-> !$isunknown(done_tag));
    assert property (result_fifo_occupancy <= FIFO_DEPTH);
    assert property (!protocol_error);
    cover property (arithmetic_active && !stage_select && phase_cycle == 0
        ##10 arithmetic_active && !stage_select && phase_cycle == 0);
    cover property (parameter_release_valid && parameter_release_ready
        && input_valid && !input_ready);
endmodule

bind qfit_atlif_rank3_resident_stream_core
    qfit_atlif_rank3_resident_stream_assertions #(
        .TAG_W(TAG_W), .FIFO_DEPTH(FIFO_DEPTH)
    ) m30_assertions (.*);

`default_nettype wire
