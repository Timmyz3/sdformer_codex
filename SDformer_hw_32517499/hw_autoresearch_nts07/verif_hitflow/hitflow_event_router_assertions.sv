`timescale 1ns/1ps
`default_nettype none

module hitflow_event_router_assertions #(
    parameter int DATA_W = 32,
    parameter int TAG_W  = 48
) (
    input logic                  clk_core,
    input logic                  rst_core,
    input logic                  single_valid,
    input logic                  single_ready,
    input logic [DATA_W-1:0]     single_data,
    input logic [TAG_W-1:0]      single_tag,
    input logic                  fanout_q_valid,
    input logic                  fanout_q_ready,
    input logic [DATA_W-1:0]     fanout_q_data,
    input logic [TAG_W-1:0]      fanout_q_tag,
    input logic                  fanout_k_valid,
    input logic                  fanout_k_ready,
    input logic [DATA_W-1:0]     fanout_k_data,
    input logic [TAG_W-1:0]      fanout_k_tag,
    input logic                  pair_valid,
    input logic                  pair_ready,
    input logic [(4*DATA_W)-1:0] pair_data,
    input logic [TAG_W-1:0]      pair_tag,
    input logic                  pair_tag_mismatch,
    input logic                  pair_duplicate_slot,
    input logic                  route_unsupported,
    input logic                  in_ready
);

    property p_single_stable;
        @(posedge clk_core) disable iff (rst_core)
            single_valid && !single_ready |=> single_valid && $stable(single_data) && $stable(single_tag);
    endproperty

    property p_fanout_q_stable;
        @(posedge clk_core) disable iff (rst_core)
            fanout_q_valid && !fanout_q_ready |=> fanout_q_valid && $stable(fanout_q_data) && $stable(fanout_q_tag);
    endproperty

    property p_fanout_k_stable;
        @(posedge clk_core) disable iff (rst_core)
            fanout_k_valid && !fanout_k_ready |=> fanout_k_valid && $stable(fanout_k_data) && $stable(fanout_k_tag);
    endproperty

    property p_pair_stable;
        @(posedge clk_core) disable iff (rst_core)
            pair_valid && !pair_ready |=> pair_valid && $stable(pair_data) && $stable(pair_tag);
    endproperty

    property p_rejected_pair_error;
        @(posedge clk_core) disable iff (rst_core)
            pair_tag_mismatch || pair_duplicate_slot |-> !in_ready;
    endproperty

    property p_rejected_route_error;
        @(posedge clk_core) disable iff (rst_core)
            route_unsupported |-> !in_ready;
    endproperty

    assert property (p_single_stable);
    assert property (p_fanout_q_stable);
    assert property (p_fanout_k_stable);
    assert property (p_pair_stable);
    assert property (p_rejected_pair_error);
    assert property (p_rejected_route_error);

endmodule

`default_nettype wire
