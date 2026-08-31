`timescale 1ns/1ps
`default_nettype none

module m286_m273r2_independent_assertions #(
    parameter int FIFO_DEPTH=16
)(
    input logic clk_core,input logic rst_core,
    input logic config_valid,input logic config_ready,input logic config_accept,
    input logic raw_valid,input logic raw_ready,input logic raw_accept,
    input logic result_valid,input logic result_ready,input logic result_accept,
    input logic [47:0] result_tag,input logic [2:0] result_beat,
    input logic [47:0] result_valid_bits,input logic [47:0] result_data,
    input logic release_valid,input logic release_ready,input logic release_accept,
    input logic protocol_error,input logic fault_event,
    input logic product_push,input logic fifo_push,input logic fifo_pop,
    input logic stage1_issue,input logic stage2_issue,
    input logic [$clog2(FIFO_DEPTH+1)-1:0] result_fifo_occupancy,
    input logic [31:0] debug_tiles_loaded
);
    ap_m286_config_handshake:assert property(@(posedge clk_core)disable iff(rst_core)
        config_accept==(config_valid&&config_ready));
    ap_m286_raw_handshake:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_accept==(raw_valid&&raw_ready));
    ap_m286_result_handshake:assert property(@(posedge clk_core)disable iff(rst_core)
        result_accept==(result_valid&&result_ready));
    ap_m286_release_handshake:assert property(@(posedge clk_core)disable iff(rst_core)
        release_accept==(release_valid&&release_ready));
    ap_m286_fifo_alias:assert property(@(posedge clk_core)disable iff(rst_core)
        fifo_push==product_push&&fifo_pop==result_accept);
    ap_m286_result_stable_under_stall:assert property(
        @(posedge clk_core)disable iff(rst_core)
        result_valid&&!result_ready|=>protocol_error||
            (result_valid&&$stable({result_tag,result_beat,
                result_valid_bits,result_data})));
    ap_m286_fault_registered:assert property(@(posedge clk_core)disable iff(rst_core)
        fault_event|=>protocol_error);
    ap_m286_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        protocol_error|=>protocol_error);
    ap_m286_quarantine:assert property(@(posedge clk_core)disable iff(rst_core)
        protocol_error|->!(config_accept||raw_accept||result_valid||result_accept
            ||release_accept||stage1_issue||stage2_issue||product_push
            ||fifo_push||fifo_pop));
    ap_m286_n0_no_release:assert property(@(posedge clk_core)disable iff(rst_core)
        release_valid&&debug_tiles_loaded==0|->!release_ready&&!release_accept);
    ap_m286_fifo_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        result_fifo_occupancy<=FIFO_DEPTH);

    cp_m286_fault_with_fifo_pop_push:cover property(@(posedge clk_core)
        fault_event&&result_accept&&fifo_pop&&fifo_push);
    cp_m286_n0_held_after_fault:cover property(@(posedge clk_core)
        protocol_error&&release_valid&&debug_tiles_loaded==0);
    cp_m286_sticky_quarantine:cover property(@(posedge clk_core)
        protocol_error[*4]);
    cp_m286_full_pop_push:cover property(@(posedge clk_core)
        result_fifo_occupancy==FIFO_DEPTH&&fifo_pop&&fifo_push);
endmodule

`default_nettype wire
