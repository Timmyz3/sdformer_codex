`timescale 1ns/1ps
`default_nettype none

module m144_sequence_fenced_raw128_overlap_wrapper_assertions #(
    parameter int TAG_BITS = 16,
    parameter int SEQUENCE_BITS = 32,
    parameter int BANKS = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic pwp_valid,
    input logic pwp_ready,
    input logic [1:0] pwp_bank,
    input logic [TAG_BITS-1:0] pwp_window_tag,
    input logic [SEQUENCE_BITS-1:0] pwp_sequence,
    input logic pwp_accept,
    input logic pwp_done_valid,
    input logic correction_valid,
    input logic correction_ready,
    input logic [1:0] correction_bank,
    input logic [TAG_BITS-1:0] correction_window_tag,
    input logic [SEQUENCE_BITS-1:0] correction_sequence,
    input logic correction_accept,
    input logic correction_done_valid,
    input logic outer_barrier_valid,
    input logic outer_barrier_ready,
    input logic outer_barrier_accept,
    input logic outer_commit_valid,
    input logic outer_commit_done_valid,
    input logic outer_commit_done_accept,
    input logic [TAG_BITS-1:0] outer_commit_tag,
    input logic [SEQUENCE_BITS-1:0] outer_commit_fence_sequence,
    input logic [BANKS-1:0] observed_bank_free,
    input logic observed_pwp_busy,
    input logic observed_correction_busy,
    input logic observed_barrier_active,
    input logic [SEQUENCE_BITS-1:0] observed_next_sequence,
    input logic [SEQUENCE_BITS-1:0] observed_next_completion_sequence,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_row_accept_definition:
        assert property (row_accept == (row_valid && row_ready));
    ap_pwp_accept_definition:
        assert property (pwp_accept == (pwp_valid && pwp_ready));
    ap_correction_accept_definition:
        assert property (correction_accept
                         == (correction_valid && correction_ready));
    ap_barrier_accept_definition:
        assert property (outer_barrier_accept
                         == (outer_barrier_valid
                             && outer_barrier_ready));
    ap_commit_done_accept_requires_offer:
        assert property (outer_commit_done_accept
                         |-> outer_commit_done_valid
                             && outer_commit_valid);

    ap_pwp_stable_under_stall:
        assert property (pwp_valid && !pwp_ready
            |=> pwp_valid
                && $stable({pwp_bank, pwp_window_tag,
                            pwp_sequence}));
    ap_correction_stable_under_stall:
        assert property (correction_valid && !correction_ready
            |=> correction_valid
                && $stable({correction_bank, correction_window_tag,
                            correction_sequence}));
    ap_commit_stable_until_ack:
        assert property (outer_commit_valid
                         && !outer_commit_done_accept
            |=> outer_commit_valid
                && $stable({outer_commit_tag,
                            outer_commit_fence_sequence}));
    ap_barrier_fence_stable:
        assert property (observed_barrier_active
            ##1 observed_barrier_active
            |-> $stable(outer_commit_fence_sequence));

    ap_pwp_cannot_cross_active_fence:
        assert property (observed_barrier_active && pwp_valid
                         |-> pwp_sequence
                             <= outer_commit_fence_sequence);
    ap_correction_cannot_cross_active_fence:
        assert property (observed_barrier_active && correction_valid
                         |-> correction_sequence
                             <= outer_commit_fence_sequence);
    ap_commit_requires_drained_fence:
        assert property (outer_commit_valid
                         |-> observed_barrier_active
                             && observed_next_completion_sequence
                                > outer_commit_fence_sequence);
    ap_sequence_monotonic:
        assert property (observed_next_sequence
                         >= $past(observed_next_sequence));
    ap_completion_sequence_monotonic:
        assert property (observed_next_completion_sequence
                         >= $past(observed_next_completion_sequence));
    ap_protocol_error_sticky:
        assert property (protocol_error |=> protocol_error);

    cp_four_bank_lookahead:
        cover property (observed_barrier_active
                        && observed_bank_free == 0);
    cp_fence_blocks_post_sequence:
        cover property (observed_barrier_active
                        && observed_next_sequence
                           > outer_commit_fence_sequence + 1'b1
                        ##[1:50] outer_commit_valid);
    cp_commit_then_post_fence_pwp:
        cover property (outer_commit_done_accept
                        ##[1:50] pwp_accept
                        && pwp_sequence
                           > outer_commit_fence_sequence);
    cp_pwp_correction_overlap:
        cover property (observed_pwp_busy
                        && observed_correction_busy);
    cp_minimum_one_cycle_endpoint:
        cover property (pwp_accept ##1 !pwp_done_valid
                        ##[1:20] pwp_done_valid);
endmodule

`default_nettype wire
