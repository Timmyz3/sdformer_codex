`timescale 1ns/1ps
`default_nettype none

module qfit_atlif_unified_t10_t2_stream_assertions #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic parameter_valid,
    input logic parameter_ready,
    input logic parameter_mode,
    input logic [(3*10*8)-1:0] parameter_t10_right_factor,
    input logic [(10*3*8)-1:0] parameter_t10_left_factor,
    input logic [(10*24)-1:0] parameter_t10_bias,
    input logic signed [23:0] parameter_t10_threshold,
    input logic [4:0] parameter_t10_requant_shift,
    input logic [31:0] parameter_t2_weight,
    input logic [47:0] parameter_t2_bias,
    input logic signed [23:0] parameter_t2_threshold,
    input logic parameter_loaded,
    input logic loaded_mode,
    input logic parameter_release_valid,
    input logic parameter_release_ready,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [2:0] input_beat,
    input logic [23:0] input_lane_valid,
    input logic [255:0] input_port0_values,
    input logic [255:0] input_port1_values,
    input logic result_valid,
    input logic result_ready,
    input logic result_mode,
    input logic [TAG_W-1:0] result_tag,
    input logic [2:0] result_beat,
    input logic [47:0] result_valid_bits,
    input logic [47:0] result_bits,
    input logic done,
    input logic done_mode,
    input logic [TAG_W-1:0] done_tag,
    input logic protocol_error,
    input logic busy,
    input logic arithmetic_active,
    input logic [1:0] issue_kind,
    input logic [2:0] phase_cycle,
    input logic [95:0] multiplier_active_mask,
    input logic [FIFO_COUNT_W-1:0] result_fifo_occupancy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M31_SVA_BOUND=1");

    assert property (parameter_valid && !parameter_ready |=>
        $stable({parameter_valid, parameter_mode,
                 parameter_t10_right_factor, parameter_t10_left_factor,
                 parameter_t10_bias, parameter_t10_threshold,
                 parameter_t10_requant_shift, parameter_t2_weight,
                 parameter_t2_bias, parameter_t2_threshold}));
    assert property (parameter_release_valid && !parameter_release_ready |=>
        $stable(parameter_release_valid));
    assert property (parameter_valid && parameter_ready && !parameter_mode
        |-> parameter_t10_requant_shift <= 23);
    assert property (parameter_valid && parameter_ready |=> parameter_loaded);
    assert property (parameter_valid && parameter_ready
        |=> loaded_mode == $past(parameter_mode));
    assert property (parameter_release_valid && parameter_release_ready
        |=> !parameter_loaded);
    assert property (parameter_loaded && $past(parameter_loaded)
        |-> $stable(loaded_mode));

    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_beat, input_lane_valid,
                 input_port0_values, input_port1_values}));
    assert property (input_valid && input_ready |-> parameter_loaded);
    assert property (parameter_release_valid && input_valid
        |-> !parameter_release_ready);
    assert property (!(parameter_release_valid && parameter_release_ready
                       && input_valid && input_ready));

    assert property (result_valid && !result_ready |=>
        $stable({result_valid, result_mode, result_tag, result_beat,
                 result_valid_bits, result_bits}));
    assert property (result_valid && !result_mode
        |-> result_beat <= 4
            && result_valid_bits == {{16{1'b0}}, {32{1'b1}}});
    assert property (result_valid && result_mode
        |-> result_beat == 0 && result_valid_bits == {48{1'b1}});

    assert property (arithmetic_active |-> issue_kind != 0
        && multiplier_active_mask == {96{1'b1}});
    assert property (!arithmetic_active |-> issue_kind == 0
        && multiplier_active_mask == '0);
    assert property (issue_kind == 1 |-> !loaded_mode && phase_cycle <= 4);
    assert property (issue_kind == 2 |-> !loaded_mode && phase_cycle <= 4);
    assert property (issue_kind == 3 |-> loaded_mode
        && input_valid && input_ready);
    assert property (done && done_mode
        |-> $past(issue_kind == 3) && done_tag == $past(input_tag));
    assert property (done && !done_mode
        |-> $past(issue_kind == 2 && phase_cycle == 4));
    assert property (done |-> !$isunknown(done_tag));
    assert property (result_fifo_occupancy <= FIFO_DEPTH);
    assert property (!protocol_error);

    cover property (issue_kind == 1 && phase_cycle == 0
        ##10 issue_kind == 1 && phase_cycle == 0);
    cover property (issue_kind == 3 ##1 issue_kind == 3);
    cover property (result_fifo_occupancy == FIFO_DEPTH);
    cover property (parameter_release_valid && !parameter_release_ready
        && input_valid && input_ready);
endmodule

bind qfit_atlif_unified_t10_t2_stream_core
    qfit_atlif_unified_t10_t2_stream_assertions #(
        .TAG_W(TAG_W), .FIFO_DEPTH(FIFO_DEPTH)
    ) m31_assertions (.*);

`default_nettype wire
