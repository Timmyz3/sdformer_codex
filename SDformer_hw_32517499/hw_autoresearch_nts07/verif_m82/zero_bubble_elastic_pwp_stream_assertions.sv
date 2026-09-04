`timescale 1ns/1ps
`default_nettype none

module zero_bubble_elastic_pwp_stream_assertions #(
    parameter int LANES = 96,
    parameter int OUT_W = 12,
    parameter int TAG_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic beat_valid,
    input logic beat_ready,
    input logic beat_start,
    input logic beat_last,
    input logic [3:0] beat_width,
    input logic [TAG_W-1:0] beat_tag,
    input logic [255:0] beat_data,
    input logic beat_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [LANES*OUT_W-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic collecting,
    input logic busy
);
    ap_beat_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core) beat_accept == (beat_valid && beat_ready));
    ap_output_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_width,
                                     output_escape, output_values}));
    ap_escape_has_no_payload: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_escape
        |-> output_width == 12 && output_values == '0);
    ap_regular_output_width: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_escape
        |-> output_width inside {8, 9, 10, 11});
    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error |=> protocol_error);
    ap_fault_blocks_accepts: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error
        |-> !beat_accept && !output_accept);

    cp_width8: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_width == 8);
    cp_width9: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_width == 9);
    cp_width10: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_width == 10);
    cp_width11: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_width == 11);
    cp_escape12: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_escape);
    cp_zero_bubble_boundary: cover property (@(posedge clk_core)
        disable iff (rst_core) output_accept && beat_accept && beat_start);
    cp_output_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready);
    cp_protocol_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
endmodule

`default_nettype wire
