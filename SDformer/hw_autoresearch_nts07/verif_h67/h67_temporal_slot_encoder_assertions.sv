`timescale 1ns/1ps
`default_nettype none

// Handshake-qualified merge contract for TESC encoder.
// Icarus ignores this file; Verilator --assert is the DATE path.
// QUOTIENT_ENABLE=0 (Fixed2S) still counts equal pairs but always emits 2 slots.
module h67_temporal_slot_encoder_assertions #(
    parameter bit QUOTIENT_ENABLE = 1'b1
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic pair_commit,
    input logic score_equal,
    input logic [1:0] packet_slot_count,
    input logic [31:0] perf_pairs,
    input logic [31:0] perf_slots,
    input logic [31:0] perf_equal_pairs
);
    property p_equal_merges_to_one_slot;
        @(posedge clk_core) disable iff (rst_core || window_start)
        pair_commit && score_equal && QUOTIENT_ENABLE
            |-> packet_slot_count == 2'd1;
    endproperty
    assert property (p_equal_merges_to_one_slot);

    property p_unequal_or_fixed_two_slots;
        @(posedge clk_core) disable iff (rst_core || window_start)
        pair_commit && (!score_equal || !QUOTIENT_ENABLE)
            |-> packet_slot_count == 2'd2;
    endproperty
    assert property (p_unequal_or_fixed_two_slots);

    property p_rqtb_slot_identity_after_commit;
        @(posedge clk_core) disable iff (rst_core || window_start)
        pair_commit && QUOTIENT_ENABLE
            |=> (perf_slots + perf_equal_pairs) == (perf_pairs << 1);
    endproperty
    assert property (p_rqtb_slot_identity_after_commit);

    property p_fixed_slots_are_double;
        @(posedge clk_core) disable iff (rst_core || window_start)
        pair_commit && !QUOTIENT_ENABLE
            |=> perf_slots == (perf_pairs << 1);
    endproperty
    assert property (p_fixed_slots_are_double);
endmodule

bind h67_temporal_slot_encoder h67_temporal_slot_encoder_assertions #(
    .QUOTIENT_ENABLE(QUOTIENT_ENABLE)
) u_h67_temporal_slot_encoder_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .pair_commit(pair_commit),
        .score_equal(score_equal),
        .packet_slot_count(packet_slot_count),
        .perf_pairs(perf_pairs),
        .perf_slots(perf_slots),
        .perf_equal_pairs(perf_equal_pairs)
    );

`default_nettype wire
