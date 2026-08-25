`timescale 1ns/1ps
`default_nettype none

module m133_dualrow512_elastic_pwp_stream_assertions #(
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
    input logic beat_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [LANES*OUT_W-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic collecting
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_beat_accept_definition:
        assert property (beat_accept == (beat_valid && beat_ready));
    ap_output_accept_definition:
        assert property (output_accept == (output_valid && output_ready));
    ap_fault_quarantines_all_accepts:
        assert property (protocol_error
                         |-> !(beat_accept || output_valid || output_accept));
    ap_output_stable_under_stall:
        assert property (output_valid && !output_ready
                         |=> output_valid
                             && $stable({output_tag, output_width,
                                         output_escape, output_values}));
    ap_start_has_metadata:
        assert property (beat_accept && beat_start
                         |-> beat_width inside
                             {4'd8, 4'd9, 4'd10, 4'd11, 4'd12});
    ap_continuation_metadata_zero:
        assert property (beat_accept && !beat_start
                         |-> beat_width == 0 && beat_tag == 0);
    ap_escape_zero_output:
        assert property (output_valid && output_escape
                         |-> output_width == 12 && output_values == 0);

    cp_width8:
        cover property (beat_accept && beat_start && beat_width == 8
                        ##1 beat_accept && beat_last);
    cp_width9:
        cover property (beat_accept && beat_start && beat_width == 9
                        ##1 beat_accept && beat_last);
    cp_width10:
        cover property (beat_accept && beat_start && beat_width == 10
                        ##1 beat_accept && beat_last);
    cp_width11:
        cover property (beat_accept && beat_start && beat_width == 11
                        ##1 beat_accept && !beat_last
                        ##1 beat_accept && beat_last);
    cp_escape:
        cover property (beat_accept && beat_start && beat_width == 12
                        && beat_last);
    cp_output_stall_release:
        cover property (output_valid && !output_ready
                        ##1 output_valid && output_ready);
    cp_last_to_next_start:
        cover property (beat_accept && beat_last
                        ##1 beat_accept && beat_start);
    cp_same_cycle_fault_quarantine:
        cover property (protocol_error && !beat_accept && !output_valid);
    cp_reset_quiesce:
        cover property (@(posedge clk_core) disable iff (1'b0)
                        rst_core && !beat_accept && !output_valid
                        && !protocol_error && !collecting);
endmodule

`default_nettype wire
