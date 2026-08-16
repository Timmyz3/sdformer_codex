`timescale 1ns/1ps
`default_nettype none

module hitflow_nmf_g1_assertions #(
    parameter int TOKENS = 162,
    parameter int LANES = 32,
    parameter int GATE_W = 9,
    parameter int TAG_W = 32,
    parameter int COUNTER_W = 32
) (
    input logic                      clk_core,
    input logic                      rst_core,
    input logic                      token_valid,
    input logic                      token_ready,
    input logic                      protocol_error,
    input logic                      term_valid,
    input logic                      term_ready,
    input logic [TAG_W-1:0]          term_tag,
    input logic [GATE_W-1:0]         term_gate_code,
    input logic [$clog2(LANES)-1:0]  term_lane,
    input logic [TOKENS-1:0]         term_destination_bitmap,
    input logic                      fallback_valid,
    input logic                      fallback_ready,
    input logic [TAG_W-1:0]          fallback_tag,
    input logic [$clog2(TOKENS)-1:0] fallback_token_id,
    input logic [GATE_W-1:0]         fallback_gate_code,
    input logic [LANES-1:0]          fallback_k_bits,
    input logic                      group_done_valid,
    input logic                      group_done_ready,
    input logic [TAG_W-1:0]          group_done_tag,
    input logic [COUNTER_W-1:0]      count_tokens
);

    property p_term_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            term_valid && !term_ready |=> term_valid &&
            $stable(term_tag) && $stable(term_gate_code) &&
            $stable(term_lane) && $stable(term_destination_bitmap);
    endproperty

    property p_fallback_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            fallback_valid && !fallback_ready |=> fallback_valid &&
            $stable(fallback_tag) && $stable(fallback_token_id) &&
            $stable(fallback_gate_code) && $stable(fallback_k_bits);
    endproperty

    property p_done_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            group_done_valid && !group_done_ready |=>
            group_done_valid && $stable(group_done_tag);
    endproperty

    property p_protocol_error_rejects_token;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error && token_valid |-> !token_ready;
    endproperty

    property p_term_is_nonzero;
        @(posedge clk_core) disable iff (rst_core)
            term_valid |-> (term_gate_code != '0) &&
                           (term_destination_bitmap != '0);
    endproperty

    property p_done_has_complete_token_count;
        @(posedge clk_core) disable iff (rst_core)
            group_done_valid |-> (count_tokens == TOKENS) &&
                                  !term_valid && !fallback_valid;
    endproperty

    assert property (p_term_stable_under_backpressure);
    assert property (p_fallback_stable_under_backpressure);
    assert property (p_done_stable_under_backpressure);
    assert property (p_protocol_error_rejects_token);
    assert property (p_term_is_nonzero);
    assert property (p_done_has_complete_token_count);

endmodule

`default_nettype wire
