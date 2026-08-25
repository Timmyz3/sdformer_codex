`timescale 1ns/1ps
`default_nettype none

module qfit_adaptive_parent_selector_p256_assertions #(
    parameter int TAG_W = 48,
    parameter int TILE_BITS = 256,
    parameter int COUNT_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic [1:0] out_parent_id,
    input logic [TILE_BITS-1:0] out_add_bits,
    input logic [TILE_BITS-1:0] out_subtract_bits,
    input logic [COUNT_W-1:0] out_source_count
);
    property p_output_stable_on_stall;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=> out_valid
            && $stable({out_tag, out_parent_id, out_add_bits,
                        out_subtract_bits, out_source_count});
    endproperty
    assert property (p_output_stable_on_stall);

    property p_signed_masks_disjoint;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> !(|(out_add_bits & out_subtract_bits));
    endproperty
    assert property (p_signed_masks_disjoint);

    property p_source_count_matches_masks;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_source_count
            == $countones(out_add_bits | out_subtract_bits);
    endproperty
    assert property (p_source_count_matches_masks);

    cover property (@(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready ##[1:8] out_valid && out_ready
        && out_parent_id == 2'd0);
    cover property (@(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready ##[1:8] out_valid && out_ready
        && out_parent_id == 2'd1);
    cover property (@(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready ##[1:8] out_valid && out_ready
        && out_parent_id == 2'd2);
    cover property (@(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready ##[1:8] out_valid && out_ready
        && out_parent_id == 2'd3);
endmodule

`default_nettype wire
