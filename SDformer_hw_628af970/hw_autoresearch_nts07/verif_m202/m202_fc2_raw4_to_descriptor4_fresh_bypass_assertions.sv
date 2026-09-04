`timescale 1ns/1ps
`default_nettype none

module m202_fc2_raw4_to_descriptor4_fresh_bypass_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic header_accept,
    input logic raw_valid,
    input logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [95:0] raw_bitmap [0:3],
    input logic raw_accept,
    input logic descriptor_ready,
    input logic descriptor_accept,
    input logic protocol_error,
    input logic fresh_mode
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    // A ready sink must consume a nonzero fresh packet in the very cycle in
    // which the empty-reservoir bypass is selected.  Observe the bound DUT's
    // internal mode directly; do not assume a particular driver gap after the
    // header handshake.
    ap_fresh_bypass_same_cycle:
        assert property (fresh_mode && descriptor_ready
            |-> raw_accept && descriptor_accept);
    ap_fresh_descriptor_implies_raw_accept:
        assert property (fresh_mode && descriptor_accept
            && !protocol_error |-> raw_accept);

    cp_first_packet_fresh_bypass:
        cover property (fresh_mode && raw_accept && descriptor_accept);
    cp_first_packet_four_fresh:
        cover property (fresh_mode && raw_accept && descriptor_accept
            && raw_lane_valid == 4'hf
            && raw_bitmap[0] != 0 && raw_bitmap[1] != 0
            && raw_bitmap[2] != 0 && raw_bitmap[3] != 0);
`endif
endmodule

bind m202_fc2_raw4_to_descriptor4_fresh_bypass_compactor
    m202_fc2_raw4_to_descriptor4_fresh_bypass_assertions fresh_sva (.*);

`default_nettype wire
