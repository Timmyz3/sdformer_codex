`timescale 1ns/1ps
`default_nettype none

module gatestack_obi_iterator_assertions #(
    parameter int SLOTS = 4,
    parameter int LANES = 32,
    parameter int TAG_W = 32,
    parameter int SLOT_ID_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS),
    parameter int LANE_ID_W = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input logic clk_core,
    input logic rst_core,
    input logic load_valid,
    input logic load_ready,
    input logic entry_valid,
    input logic entry_ready,
    input logic [TAG_W-1:0] entry_tag,
    input logic [SLOT_ID_W-1:0] entry_slot_id,
    input logic [LANE_ID_W-1:0] entry_lane_id,
    input logic entry_last,
    input logic done_valid,
    input logic done_ready,
    input logic [TAG_W-1:0] done_tag
);

    property p_entry_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        entry_valid && !entry_ready |=> entry_valid &&
            $stable({entry_tag, entry_slot_id, entry_lane_id, entry_last});
    endproperty

    property p_done_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        done_valid && !done_ready |=> done_valid && $stable(done_tag);
    endproperty

    property p_entry_in_range;
        @(posedge clk_core) disable iff (rst_core)
        entry_valid |-> (32'(entry_slot_id) < SLOTS) &&
                         (32'(entry_lane_id) < LANES);
    endproperty

    property p_entry_and_done_exclusive;
        @(posedge clk_core) disable iff (rst_core)
        !(entry_valid && done_valid);
    endproperty

    property p_load_starts_progress;
        @(posedge clk_core) disable iff (rst_core)
        load_valid && load_ready |=> entry_valid || done_valid;
    endproperty

    assert property (p_entry_stable_under_stall);
    assert property (p_done_stable_under_stall);
    assert property (p_entry_in_range);
    assert property (p_entry_and_done_exclusive);
    assert property (p_load_starts_progress);

    cover property (@(posedge clk_core) disable iff (rst_core)
        entry_valid && entry_ready && entry_last);

endmodule

`default_nettype wire
