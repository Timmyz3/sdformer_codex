`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_slot_flow_real_weight_projection2;
    tb_h67_temporal_slot_flow_real_trace_2s u_base ();

    h67_real_weight_projection2_monitor u_projection_monitor (
        .clk(u_base.clk),
        .rst_core(u_base.rst_core),
        .row_tag(u_base.row_tag),
        .stage_tag(u_base.stage_tag),
        .block_tag(u_base.block_tag),
        .head_tag(u_base.head_tag),
        .fixed_start(u_base.fixed_start),
        .fixed_done(u_base.fixed_done),
        .fixed_out_valid(u_base.fixed_out_valid),
        .fixed_out_last(u_base.fixed_out_last),
        .fixed_out_k(u_base.fixed_out_k),
        .fixed_out_gate(u_base.fixed_out_gate),
        .rqtb_start(u_base.rqtb_start),
        .rqtb_done(u_base.rqtb_done),
        .rqtb_out_valid(u_base.rqtb_out_valid),
        .rqtb_out_last(u_base.rqtb_out_last),
        .rqtb_out_k(u_base.rqtb_out_k),
        .rqtb_out_gate(u_base.rqtb_out_gate),
        .common_out_ready(u_base.common_out_ready)
    );
endmodule

`default_nettype wire
