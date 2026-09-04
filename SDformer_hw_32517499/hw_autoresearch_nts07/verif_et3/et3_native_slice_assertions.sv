`timescale 1ns/1ps
`default_nettype none

module et3_native_slice_assertions #(
    parameter int TAG_W = 8,
    parameter int GATE_W = 9,
    parameter int LANE_W = 2,
    parameter int MULT_W = 3,
    parameter int DEST_W = 4,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic run_active,
    input logic group_done,
    input logic source_valid,
    input logic source_ready,
    input logic [TAG_W-1:0] source_group_tag,
    input logic source_mode_multiset,
    input logic [GATE_W-1:0] source_gate_code,
    input logic [LANE_W-1:0] source_lane_id,
    input logic [MULT_W-1:0] source_multiplicity,
    input logic [DEST_W-1:0] source_destination,
    input logic source_head_last,
    input logic group_close_valid,
    input logic group_close_ready,
    input logic [TAG_W-1:0] group_close_tag,
    input logic trace_cmd_valid,
    input logic trace_cmd_ready,
    input logic [TAG_W-1:0] trace_cmd_group_tag,
    input logic trace_cmd_mode_multiset,
    input logic [GATE_W-1:0] trace_cmd_gate_code,
    input logic [LANE_W-1:0] trace_cmd_lane_id,
    input logic [MULT_W-1:0] trace_cmd_multiplicity,
    input logic [DEST_W-1:0] trace_cmd_destination,
    input logic trace_cmd_term_first,
    input logic trace_cmd_term_last,
    input logic trace_cmd_head_last,
    input logic trace_cmd_fallback,
    input logic [COUNTER_W-1:0] count_source_items,
    input logic [COUNTER_W-1:0] count_destination_beats
);
    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        source_valid && !source_ready
        |=> source_valid &&
            $stable({
                source_group_tag,
                source_mode_multiset,
                source_gate_code,
                source_lane_id,
                source_multiplicity,
                source_destination,
                source_head_last
            })
    ) else $error("ET3 source changed while stalled");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        trace_cmd_valid && !trace_cmd_ready
        |=> trace_cmd_valid &&
            $stable({
                trace_cmd_group_tag,
                trace_cmd_mode_multiset,
                trace_cmd_gate_code,
                trace_cmd_lane_id,
                trace_cmd_multiplicity,
                trace_cmd_destination,
                trace_cmd_term_first,
                trace_cmd_term_last,
                trace_cmd_head_last,
                trace_cmd_fallback
            })
    ) else $error("ET3 command changed while stalled");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        source_valid && source_ready && !source_mode_multiset
        |-> source_multiplicity == MULT_W'(1)
    ) else $error("Motion SET source used multiplicity other than one");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        trace_cmd_valid && trace_cmd_ready && !trace_cmd_mode_multiset
        |-> trace_cmd_multiplicity == MULT_W'(1)
    ) else $error("Motion SET command used multiplicity other than one");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        trace_cmd_valid && trace_cmd_head_last
        |-> trace_cmd_term_last
    ) else $error("head_last appeared before term_last");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        group_done |-> !run_active
    ) else $error("group_done asserted while epoch remained active");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        !run_active |-> !source_ready && !group_close_ready
    ) else $error("inactive epoch accepted source or close");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        count_destination_beats <= count_source_items
    ) else $error("directory emitted more destinations than accepted");

    assert property (@(posedge clk_core) disable iff (rst_core || flush)
        group_close_valid && source_valid && source_head_last &&
        (group_close_tag == source_group_tag) && source_ready
        |-> group_close_ready
    ) else $error("matching final source and close did not co-retire");

endmodule

`default_nettype wire
