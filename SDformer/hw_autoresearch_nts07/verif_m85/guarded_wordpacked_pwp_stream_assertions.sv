`timescale 1ns/1ps
`default_nettype none

module guarded_wordpacked_pwp_stream_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic phase_load_valid,
    input logic phase_load_ready,
    input logic phase_loaded,
    input logic metadata_error,
    input logic lookup_valid,
    input logic lookup_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic busy
);
    ap_bad_metadata_blocks_lookup: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> !lookup_ready);
    ap_unloaded_phase_blocks_lookup: assert property (@(posedge clk_core)
        disable iff (rst_core) !phase_loaded |-> !lookup_ready);
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_width,
                                     output_escape, output_values}));
    ap_escape_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_escape
        |-> output_width == 12 && output_values == '0);
    ap_protocol_reflects_metadata: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> protocol_error);

    cp_phase_load: cover property (@(posedge clk_core)
        phase_load_valid && phase_load_ready);
    cp_lookup_stall: cover property (@(posedge clk_core)
        lookup_valid && !lookup_ready && busy);
    cp_escape: cover property (@(posedge clk_core)
        output_valid && output_escape);
    cp_width9: cover property (@(posedge clk_core)
        output_valid && output_width == 9);
    cp_width10: cover property (@(posedge clk_core)
        output_valid && output_width == 10);
    cp_width11: cover property (@(posedge clk_core)
        output_valid && output_width == 11);
    cp_metadata_error: cover property (@(posedge clk_core)
        metadata_error && protocol_error && !lookup_ready);
endmodule

`default_nettype wire
