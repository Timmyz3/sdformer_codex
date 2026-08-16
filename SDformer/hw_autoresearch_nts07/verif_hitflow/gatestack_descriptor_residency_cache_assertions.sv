`timescale 1ns/1ps
`default_nettype none

module gatestack_descriptor_residency_cache_assertions #(
    parameter int TAG_W = 32,
    parameter int TERM_INDEX_W = 7
) (
    input logic clk_core,
    input logic rst_core,
    input logic lookup_meta_valid,
    input logic lookup_meta_ready,
    input logic lookup_hit,
    input logic [TAG_W-1:0] lookup_tag,
    input logic [7:0] lookup_term_count,
    input logic lookup_entry_valid,
    input logic lookup_entry_ready,
    input logic [8:0] lookup_gate_code,
    input logic [4:0] lookup_lane_id,
    input logic [7:0] lookup_destination_count,
    input logic [TERM_INDEX_W-1:0] lookup_term_index,
    input logic lookup_entry_last,
    input logic release_valid,
    input logic release_ready,
    input logic [TAG_W-1:0] release_expected_tag,
    input logic release_slot_valid,
    input logic release_tag_matches,
    input logic protocol_error
);

    property p_meta_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        lookup_meta_valid && !lookup_meta_ready |=> lookup_meta_valid &&
            $stable({lookup_hit, lookup_tag, lookup_term_count});
    endproperty

    property p_entry_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        lookup_entry_valid && !lookup_entry_ready |=> lookup_entry_valid &&
            $stable({lookup_gate_code, lookup_lane_id,
                     lookup_destination_count, lookup_term_index,
                     lookup_entry_last});
    endproperty

    property p_entry_count_nonzero;
        @(posedge clk_core) disable iff (rst_core)
        lookup_entry_valid |-> lookup_destination_count != 0;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty

    assert property (p_meta_stable_under_stall);
    assert property (p_entry_stable_under_stall);
    assert property (p_entry_count_nonzero);

    property p_release_tag_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        release_valid && !release_ready |=> release_valid &&
            $stable(release_expected_tag);
    endproperty
    assert property (p_release_tag_stable_under_stall);

    property p_stale_release_reports_error;
        @(posedge clk_core) disable iff (rst_core)
        release_valid && release_ready && release_slot_valid &&
            !release_tag_matches |=> protocol_error;
    endproperty
    assert property (p_stale_release_reports_error);

    assert property (p_protocol_error_sticky);

endmodule

`default_nettype wire
