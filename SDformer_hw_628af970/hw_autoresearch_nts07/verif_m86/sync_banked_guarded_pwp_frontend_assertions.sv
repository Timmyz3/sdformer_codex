`timescale 1ns/1ps
`default_nettype none

module sync_banked_guarded_pwp_frontend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic payload_load_valid,
    input logic payload_load_ready,
    input logic payload_load_accept,
    input logic phase_load_valid,
    input logic phase_load_ready,
    input logic phase_loaded,
    input logic metadata_error,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic descriptor_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic bank_read_issue,
    input logic bank_response_enqueue,
    input logic [2:0] response_fifo_level
);
    ap_payload_accept: assert property (@(posedge clk_core)
        disable iff (rst_core)
        payload_load_accept == (payload_load_valid && payload_load_ready));
    ap_descriptor_accept: assert property (@(posedge clk_core)
        disable iff (rst_core)
        descriptor_accept == (descriptor_valid && descriptor_ready));
    ap_unloaded_blocks_descriptor: assert property (@(posedge clk_core)
        disable iff (rst_core) !phase_loaded |-> !descriptor_ready);
    ap_bad_metadata_blocks_descriptor: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> !descriptor_ready);
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_width,
                                     output_escape, output_values}));
    ap_fifo_bound: assert property (@(posedge clk_core)
        disable iff (rst_core) response_fifo_level <= 4);
    ap_sync_response_follows_issue: assert property (@(posedge clk_core)
        disable iff (rst_core) bank_read_issue |=> bank_response_enqueue);
    ap_no_spurious_sync_response: assert property (@(posedge clk_core)
        disable iff (rst_core) bank_response_enqueue |-> $past(bank_read_issue));
    ap_escape_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_escape
        |-> output_width == 12 && output_values == '0);

    cp_phase_load: cover property (@(posedge clk_core)
        phase_load_valid && phase_load_ready);
    cp_fifo_backpressure: cover property (@(posedge clk_core)
        response_fifo_level >= 2 && output_valid && !output_ready);
    cp_fifo_full: cover property (@(posedge clk_core)
        response_fifo_level == 4);
    cp_width8: cover property (@(posedge clk_core)
        output_valid && output_width == 8);
    cp_width9: cover property (@(posedge clk_core)
        output_valid && output_width == 9);
    cp_width10: cover property (@(posedge clk_core)
        output_valid && output_width == 10);
    cp_width11: cover property (@(posedge clk_core)
        output_valid && output_width == 11);
    cp_escape: cover property (@(posedge clk_core)
        output_valid && output_escape);
    cp_protocol_attack: cover property (@(posedge clk_core) protocol_error);
endmodule

`default_nettype wire
