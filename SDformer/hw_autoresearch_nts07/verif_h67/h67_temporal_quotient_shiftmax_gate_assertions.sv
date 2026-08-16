`timescale 1ns/1ps
`default_nettype none

module h67_temporal_quotient_shiftmax_gate_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int TOKEN_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic out_last,
    input logic [TOKEN_W-1:0] out_token_id,
    input logic [HEAD_DIM-1:0] out_k_bits,
    input logic [GATE_W-1:0] out_gate_q17,
    input logic [THRESHOLD_W-1:0] out_threshold_q8,
    input logic protocol_error,
    input logic window_done
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready
        |=> $stable({out_last, out_token_id, out_k_bits,
                     out_gate_q17, out_threshold_q8})
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_k_bits != 0
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        window_done |-> !out_valid
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error
    );
endmodule

bind h67_temporal_quotient_shiftmax_gate_top
    h67_temporal_quotient_shiftmax_gate_assertions #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .THRESHOLD_W(THRESHOLD_W),
        .TOKEN_W(TOKEN_W)
    ) u_h67_temporal_quotient_shiftmax_gate_assertions (.*);

`default_nettype wire
