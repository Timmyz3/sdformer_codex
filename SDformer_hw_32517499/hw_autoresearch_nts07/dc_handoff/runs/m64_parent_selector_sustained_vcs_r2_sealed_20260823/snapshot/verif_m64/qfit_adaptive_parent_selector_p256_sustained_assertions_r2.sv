`timescale 1ns/1ps
`default_nettype none

module qfit_adaptive_parent_selector_p256_sustained_assertions_r2 #(
    parameter int TAG_W = 48,
    parameter int TILE_BITS = 256,
    parameter int COUNT_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic [TAG_W-1:0] in_tag,
    input logic [TILE_BITS-1:0] in_target_bits,
    input logic [TILE_BITS-1:0] in_left_bits,
    input logic [TILE_BITS-1:0] in_up_bits,
    input logic [TILE_BITS-1:0] in_previous_bits,
    input logic in_left_valid,
    input logic in_up_valid,
    input logic in_previous_valid,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic [1:0] out_parent_id,
    input logic [TILE_BITS-1:0] out_add_bits,
    input logic [TILE_BITS-1:0] out_subtract_bits,
    input logic [COUNT_W-1:0] out_source_count,
    input logic s0_valid,
    input logic s1_valid,
    input logic throughput_phase,
    input logic random_backpressure_phase,
    input logic forced_tie_input
);
    initial $display("M64_R2_SUSTAINED_ASSERTION_MODULE_ACTIVE=1");

    ap_input_stable_on_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        in_valid && !in_ready |=> in_valid
            && $stable({in_tag, in_target_bits, in_left_bits, in_up_bits,
                        in_previous_bits, in_left_valid, in_up_valid,
                        in_previous_valid}));

    ap_output_stable_on_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && !out_ready |=> out_valid
            && $stable({out_tag, out_parent_id, out_add_bits,
                        out_subtract_bits, out_source_count}));

    ap_signed_masks_disjoint: assert property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid |-> !(|(out_add_bits & out_subtract_bits)));

    ap_source_count_matches_masks: assert property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid |-> out_source_count
            == $countones(out_add_bits | out_subtract_bits));

    ap_output_valid_is_stage1_valid: assert property (@(posedge clk_core)
        disable iff (rst_core) out_valid == s1_valid);

    cp_back_to_back_input_accept: cover property (@(posedge clk_core)
        disable iff (rst_core)
        in_valid && in_ready ##1 in_valid && in_ready);

    cp_sustained_accept_8: cover property (@(posedge clk_core)
        disable iff (rst_core)
        (in_valid && in_ready)[*8]);

    cp_source_count_256: cover property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && out_ready && out_source_count == 9'd256);

    cp_parent_zero: cover property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && out_ready && out_parent_id == 2'd0);
    cp_parent_left: cover property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && out_ready && out_parent_id == 2'd1);
    cp_parent_up: cover property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && out_ready && out_parent_id == 2'd2);
    cp_parent_previous: cover property (@(posedge clk_core)
        disable iff (rst_core)
        out_valid && out_ready && out_parent_id == 2'd3);

    cp_forced_tie_accept: cover property (@(posedge clk_core)
        disable iff (rst_core)
        in_valid && in_ready && forced_tie_input);

    cp_random_output_backpressure: cover property (@(posedge clk_core)
        disable iff (rst_core)
        random_backpressure_phase && out_valid && !out_ready);

    cp_pipeline_full_push_pop_same_cycle: cover property (
        @(posedge clk_core) disable iff (rst_core)
        s0_valid && s1_valid && in_valid && in_ready
            && out_valid && out_ready);

    cp_full_throughput_8: cover property (@(posedge clk_core)
        disable iff (rst_core)
        (throughput_phase && s0_valid && s1_valid
         && in_valid && in_ready && out_valid && out_ready)[*8]);
endmodule

`default_nettype wire
