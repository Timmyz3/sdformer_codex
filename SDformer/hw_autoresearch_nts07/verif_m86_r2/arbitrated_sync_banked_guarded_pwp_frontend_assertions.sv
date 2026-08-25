`timescale 1ns/1ps
`default_nettype none

module arbitrated_sync_banked_guarded_pwp_frontend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic payload_load_valid,
    input logic payload_load_ready,
    input logic payload_load_accept,
    input logic phase_loaded,
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
    input logic busy,
    input logic payload_selected,
    input logic descriptor_selected
);
    ap_onehot_selection: assert property (@(posedge clk_core)
        disable iff (rst_core) !(payload_selected && descriptor_selected));
    ap_unloaded_contention_loader_wins: assert property (@(posedge clk_core)
        disable iff (rst_core)
        payload_load_valid && descriptor_valid && !phase_loaded && !busy
        |-> payload_selected && !descriptor_selected && payload_load_ready);
    ap_loaded_contention_descriptor_wins: assert property (@(posedge clk_core)
        disable iff (rst_core)
        payload_load_valid && descriptor_valid && phase_loaded && !busy
        |-> descriptor_selected && !payload_selected && descriptor_ready);
    ap_accepts_are_exclusive: assert property (@(posedge clk_core)
        disable iff (rst_core)
        !(payload_load_accept && descriptor_accept));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_width,
                                     output_escape, output_values}));
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));

    cp_unloaded_contention: cover property (@(posedge clk_core)
        payload_load_valid && descriptor_valid && !phase_loaded
        && payload_load_accept && !descriptor_accept);
    cp_loaded_contention: cover property (@(posedge clk_core)
        payload_load_valid && descriptor_valid && phase_loaded
        && descriptor_accept && !payload_load_accept);
    cp_legal_output: cover property (@(posedge clk_core)
        output_valid && output_width == 8 && !output_escape);
endmodule

`default_nettype wire
