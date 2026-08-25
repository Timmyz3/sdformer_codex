`timescale 1ns/1ps
`default_nettype none

module m146_four_bank_age_queue_scheduler_assertions #(
    parameter int TAG_BITS = 16,
    parameter int SEQUENCE_BITS = 32,
    parameter int BANKS = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic fill_valid,
    input logic fill_ready,
    input logic fill_accept,
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
    input logic release_valid,
    input logic [BANKS-1:0] observed_bank_free,
    input logic [2:0] observed_pwp_queue_count,
    input logic [2:0] observed_correction_queue_count,
    input logic observed_pwp_busy,
    input logic observed_correction_busy,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_fill_accept_definition:
        assert property (fill_accept == (fill_valid && fill_ready));
    ap_pwp_accept_definition:
        assert property (pwp_accept == (pwp_valid && pwp_ready));
    ap_correction_accept_definition:
        assert property (correction_accept
                         == (correction_valid && correction_ready));
    ap_pwp_stable_under_stall:
        assert property (pwp_valid && !pwp_ready
            |=> pwp_valid
                && $stable({pwp_bank, pwp_window_tag, pwp_sequence}));
    ap_correction_stable_under_stall:
        assert property (correction_valid && !correction_ready
            |=> correction_valid
                && $stable({correction_bank, correction_window_tag,
                            correction_sequence}));
    ap_pwp_issues_owned_bank:
        assert property (pwp_valid |-> !observed_bank_free[pwp_bank]);
    ap_correction_issues_owned_bank:
        assert property (correction_valid
                         |-> !observed_bank_free[correction_bank]);
    ap_release_requires_done:
        assert property (release_valid |-> correction_done_valid);
    ap_pwp_queue_bound:
        assert property (observed_pwp_queue_count <= BANKS);
    ap_correction_queue_bound:
        assert property (observed_correction_queue_count <= BANKS);
    ap_protocol_error_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_no_release_during_reset:
        assert property (disable iff (1'b0)
                         rst_core |-> !release_valid);

    cp_all_banks_live:
        cover property (observed_bank_free == 0);
    cp_engines_overlap:
        cover property (observed_pwp_busy
                        && observed_correction_busy);
    cp_pwp_queue_full:
        cover property (observed_pwp_queue_count == BANKS);
    cp_pwp_to_correction_handoff:
        cover property (pwp_done_valid
                        ##[1:10] correction_accept);
    cp_correction_release:
        cover property (correction_done_valid ##0 release_valid
                        ##1 (|observed_bank_free));
endmodule

`default_nettype wire
