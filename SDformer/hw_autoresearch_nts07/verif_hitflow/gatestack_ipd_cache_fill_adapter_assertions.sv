`timescale 1ns/1ps
`default_nettype none

module gatestack_ipd_cache_fill_adapter_assertions #(
    parameter int TAG_W = 32,
    parameter int HEAD_ID_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic begin_valid,
    input logic begin_cache_allowed,
    input logic cache_begin_valid,
    input logic cache_begin_ready,
    input logic [HEAD_ID_W-1:0] cache_begin_head_id,
    input logic [TAG_W-1:0] cache_begin_tag,
    input logic [7:0] cache_begin_term_count,
    input logic cache_entry_valid,
    input logic cache_entry_ready,
    input logic [8:0] cache_entry_gate_code,
    input logic [4:0] cache_entry_lane_id,
    input logic [7:0] cache_entry_destination_count,
    input logic cache_entry_last,
    input logic protocol_error
);
    property p_begin_stable;
        @(posedge clk_core) disable iff (rst_core)
        cache_begin_valid && !cache_begin_ready |=> cache_begin_valid &&
            $stable({cache_begin_head_id, cache_begin_tag,
                     cache_begin_term_count});
    endproperty
    assert property (p_begin_stable);

    property p_non_ipd_never_starts_cache;
        @(posedge clk_core) disable iff (rst_core)
        begin_valid && !begin_cache_allowed |-> !cache_begin_valid;
    endproperty
    assert property (p_non_ipd_never_starts_cache);

    property p_entry_stable;
        @(posedge clk_core) disable iff (rst_core)
        cache_entry_valid && !cache_entry_ready |=> cache_entry_valid &&
            $stable({cache_entry_gate_code, cache_entry_lane_id,
                     cache_entry_destination_count, cache_entry_last});
    endproperty
    assert property (p_entry_stable);

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    assert property (p_protocol_error_sticky);
endmodule

`default_nettype wire
