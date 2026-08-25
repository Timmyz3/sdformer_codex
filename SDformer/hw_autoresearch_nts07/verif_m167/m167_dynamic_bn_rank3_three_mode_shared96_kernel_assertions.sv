`timescale 1ns/1ps
`default_nettype none

module m167_dynamic_bn_rank3_three_mode_shared96_kernel_assertions #(
    parameter int TAG_BITS = 16
) (
    input logic                    clk_core,
    input logic                    rst_core,
    input logic                    issue_valid,
    input logic                    issue_ready,
    input logic [1:0]              issue_mode,
    input logic                    issue_accept,
    input logic                    result_valid,
    input logic                    result_ready,
    input logic [1:0]              result_mode,
    input logic [TAG_BITS-1:0]     result_tag,
    input logic signed [16:0]      front_projection_delta [0:2][0:15],
    input logic signed [8:0]       front_moment_sum_delta [0:15],
    input logic [16:0]             front_moment_sumsq_delta [0:15],
    input logic [31:0]             back_event_bits,
    input logic signed [23:0]      back_event_amplitude,
    input logic signed [15:0]      prefold_product [0:95],
    input logic                    result_accept,
    input logic [95:0]             main_product_active_mask,
    input logic [31:0]             square_product_active_mask,
    input logic                    protocol_error,
    input logic                    busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_issue_accept_definition:
        assert property (issue_accept == (issue_valid && issue_ready));
    ap_result_accept_definition:
        assert property (result_accept == (result_valid && result_ready));
    ap_fault_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_closes_issue:
        assert property (protocol_error |=> !issue_ready);
    ap_result_busy:
        assert property (result_valid |-> busy);
    ap_result_header_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid && $stable({result_mode, result_tag,
                back_event_bits, back_event_amplitude}));
    ap_legal_issue_activates_main96:
        assert property (issue_valid && issue_mode != 3
            |-> &main_product_active_mask);
    ap_front_activates_square32:
        assert property (issue_valid && issue_mode == 0
            |-> &square_product_active_mask);
    ap_nonfront_disables_square32:
        assert property (issue_valid && issue_mode inside {1, 2}
            |-> square_product_active_mask == 0);
    ap_back_preserves_nonzero_amplitude:
        assert property (result_valid && result_mode == 1
            |-> !$isunknown(back_event_amplitude));

    generate
        for (genvar rank = 0; rank < 3; rank++) begin : g_rank
            for (genvar lane = 0; lane < 16; lane++) begin : g_lane
                ap_front_projection_stable_under_stall:
                    assert property (result_valid && !result_ready
                        && result_mode == 0
                        |=> $stable(front_projection_delta[rank][lane]));
            end
        end
        for (genvar lane = 0; lane < 16; lane++) begin : g_moment
            ap_front_moment_stable_under_stall:
                assert property (result_valid && !result_ready
                    && result_mode == 0
                    |=> $stable({front_moment_sum_delta[lane],
                        front_moment_sumsq_delta[lane]}));
        end
        for (genvar slot = 0; slot < 96; slot++) begin : g_prefold
            ap_prefold_stable_under_stall:
                assert property (result_valid && !result_ready
                    && result_mode == 2
                    |=> $stable(prefold_product[slot]));
        end
    endgenerate

    cp_front_issue:
        cover property (issue_accept && issue_mode == 0);
    cp_back_issue_with_amplitude:
        cover property (issue_accept && issue_mode == 1
            ##1 result_valid && result_mode == 1
                && back_event_amplitude != 0);
    cp_prefold_issue:
        cover property (issue_accept && issue_mode == 2);
    cp_same_cycle_result_replace:
        cover property (result_accept && issue_accept);
    cp_stall_then_accept:
        cover property (result_valid && !result_ready
            ##1 result_valid && result_ready);
    cp_fault_preserves_pending_result:
        cover property (protocol_error && result_valid);
endmodule

`default_nettype wire
