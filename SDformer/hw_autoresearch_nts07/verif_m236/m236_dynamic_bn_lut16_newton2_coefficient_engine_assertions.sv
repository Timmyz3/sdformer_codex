`timescale 1ns/1ps
`default_nettype none
module m236_dynamic_bn_lut16_newton2_coefficient_engine_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core,input logic rst_core,
    input logic request_valid,input logic request_ready,input logic request_accept,
    input logic result_valid,input logic result_ready,input logic result_accept,
    input logic[TAG_BITS-1:0]result_tag,input logic[19:0]invstd_uq4p16,
    input logic signed[19:0]alpha_sq3p16,input logic signed[19:0]offset_sq3p16,
    input logic protocol_error,input logic busy,input logic[3:0]debug_state,
    input logic[31:0]debug_request_count,input logic[31:0]debug_result_count
);
    ap_request_accept: assert property(@(posedge clk_core) disable iff(rst_core)
        request_accept == (request_valid && request_ready));
    ap_result_accept: assert property(@(posedge clk_core) disable iff(rst_core)
        result_accept == (result_valid && result_ready));
    ap_result_stable: assert property(@(posedge clk_core) disable iff(rst_core)
        result_valid && !result_ready |=> protocol_error || (result_valid &&
        $stable({result_tag,invstd_uq4p16,alpha_sq3p16,offset_sq3p16})));
    ap_fault_atomic: assert property(@(posedge clk_core) disable iff(rst_core)
        protocol_error |-> !request_accept && !result_accept
        && !request_ready && !result_valid);
    ap_fault_sticky: assert property(@(posedge clk_core) disable iff(rst_core)
        $past(protocol_error) |-> protocol_error);
    ap_conservation: assert property(@(posedge clk_core) disable iff(rst_core)
        debug_result_count <= debug_request_count);
    ap_state_legal: assert property(@(posedge clk_core) disable iff(rst_core)
        debug_state <= 4'd11);
    ap_idle_not_busy: assert property(@(posedge clk_core) disable iff(rst_core)
        debug_state == 0 && !result_valid |-> !busy);
    cp_first_newton: cover property(@(posedge clk_core) debug_state == 4'd4);
    cp_second_newton: cover property(@(posedge clk_core) debug_state == 4'd7);
    cp_result: cover property(@(posedge clk_core) result_accept);
    cp_result_stall: cover property(@(posedge clk_core) result_valid&&!result_ready);
    cp_fault_with_pending_result: cover property(@(posedge clk_core)
        protocol_error && result_ready);
endmodule
`default_nettype wire
