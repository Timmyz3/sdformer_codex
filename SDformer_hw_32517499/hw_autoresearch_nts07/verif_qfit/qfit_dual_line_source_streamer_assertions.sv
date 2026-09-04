`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_source_streamer_assertions #(
    parameter int TAG_W = 24,
    parameter int INDEX_W = 8,
    parameter int COUNT_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_ready,
    input logic source_valid,
    input logic source_ready,
    input logic [TAG_W-1:0] source_tag,
    input logic [INDEX_W-1:0] source_index,
    input logic source_negative,
    input logic source_use_motion,
    input logic source_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag,
    input logic done_use_motion,
    input logic [COUNT_W-1:0] done_source_count
);
    property p_source_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        source_valid && !source_ready |=> source_valid
            && $stable({source_tag, source_index, source_negative,
                        source_use_motion, source_last});
    endproperty
    assert property (p_source_stable_under_stall);

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid
            && $stable({done_tag, done_use_motion, done_source_count});
    endproperty
    assert property (p_done_stable_under_stall);

    property p_local_source_never_negative;
        @(posedge clk_core) disable iff (rst_core)
        source_valid && !source_use_motion |-> !source_negative;
    endproperty
    assert property (p_local_source_never_negative);

    property p_source_and_done_exclusive;
        @(posedge clk_core) disable iff (rst_core)
        !(source_valid && done_valid);
    endproperty
    assert property (p_source_and_done_exclusive);

    property p_ready_has_no_pending_output;
        @(posedge clk_core) disable iff (rst_core)
        command_ready |-> !source_valid && !done_valid;
    endproperty
    assert property (p_ready_has_no_pending_output);

    cover property (@(posedge clk_core) disable iff (rst_core)
        source_valid && source_ready && source_use_motion && source_negative);
    cover property (@(posedge clk_core) disable iff (rst_core)
        done_valid && done_ready && done_source_count == '0);
endmodule

`default_nettype wire
