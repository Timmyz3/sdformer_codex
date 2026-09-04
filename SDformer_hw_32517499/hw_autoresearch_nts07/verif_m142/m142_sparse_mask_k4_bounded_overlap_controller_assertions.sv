`timescale 1ns/1ps
`default_nettype none

module m142_sparse_mask_k4_bounded_overlap_controller_assertions #(
    parameter int TAG_BITS = 16,
    parameter int ROW_BITS = 9,
    parameter int BANKS = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [1:0] descriptor_bank,
    input logic [TAG_BITS-1:0] descriptor_window_tag,
    input logic [ROW_BITS-1:0] descriptor_row,
    input logic [2:0] descriptor_block,
    input logic [1:0] descriptor_source_count_m1,
    input logic [3:0] descriptor_source [0:3],
    input logic [3:0] descriptor_negate,
    input logic descriptor_row_last,
    input logic descriptor_window_last,
    input logic descriptor_accept,
    input logic pwp_valid,
    input logic pwp_ready,
    input logic [1:0] pwp_bank,
    input logic [TAG_BITS-1:0] pwp_window_tag,
    input logic pwp_accept,
    input logic pwp_done_valid,
    input logic correction_valid,
    input logic correction_ready,
    input logic [1:0] correction_bank,
    input logic [TAG_BITS-1:0] correction_window_tag,
    input logic correction_accept,
    input logic correction_done_valid,
    input logic [1:0] correction_done_bank,
    input logic [BANKS-1:0] observed_bank_free,
    input logic [BANKS-1:0] observed_bank_fill,
    input logic [BANKS-1:0] observed_bank_filled,
    input logic [BANKS-1:0] observed_bank_pwp,
    input logic [BANKS-1:0] observed_bank_wait_correction,
    input logic [BANKS-1:0] observed_bank_correction,
    input logic observed_pwp_busy,
    input logic observed_correction_busy,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_row_accept_definition:
        assert property (row_accept == (row_valid && row_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_pwp_accept_definition:
        assert property (pwp_accept == (pwp_valid && pwp_ready));
    ap_correction_accept_definition:
        assert property (correction_accept
                         == (correction_valid && correction_ready));

    ap_descriptor_stable_under_stall:
        assert property (descriptor_valid && !descriptor_ready
            |=> descriptor_valid
                && $stable({descriptor_bank, descriptor_window_tag,
                            descriptor_row, descriptor_block,
                            descriptor_source_count_m1,
                            descriptor_source[0], descriptor_source[1],
                            descriptor_source[2], descriptor_source[3],
                            descriptor_negate, descriptor_row_last,
                            descriptor_window_last}));
    ap_pwp_stable_under_stall:
        assert property (pwp_valid && !pwp_ready
            |=> pwp_valid
                && $stable({pwp_bank, pwp_window_tag}));
    ap_correction_stable_under_stall:
        assert property (correction_valid && !correction_ready
            |=> correction_valid
                && $stable({correction_bank, correction_window_tag}));

    ap_descriptor_count_legal:
        assert property (descriptor_valid
            |-> descriptor_source_count_m1 inside {[2'd0:2'd3]});
    ap_descriptor_sources_strictly_ordered:
        assert property (descriptor_valid
            |-> (descriptor_source_count_m1 < 1
                 || descriptor_source[1] > descriptor_source[0])
                && (descriptor_source_count_m1 < 2
                    || descriptor_source[2] > descriptor_source[1])
                && (descriptor_source_count_m1 < 3
                    || descriptor_source[3] > descriptor_source[2]));
    ap_descriptor_padding_zero:
        assert property (descriptor_valid
            |-> (descriptor_source_count_m1 >= 1
                 || (descriptor_source[1] == 0
                     && descriptor_negate[1] == 0))
                && (descriptor_source_count_m1 >= 2
                    || (descriptor_source[2] == 0
                        && descriptor_negate[2] == 0))
                && (descriptor_source_count_m1 >= 3
                    || (descriptor_source[3] == 0
                        && descriptor_negate[3] == 0)));

    ap_pwp_launch_only_from_materialized_bank:
        assert property (pwp_accept |-> observed_bank_filled[pwp_bank]);
    ap_correction_launch_only_from_wait_bank:
        assert property (correction_accept
                         |-> observed_bank_wait_correction[correction_bank]);
    ap_engine_ownership_disjoint:
        assert property ((observed_bank_pwp
                          & observed_bank_correction) == 0);
    ap_protocol_error_sticky:
        assert property (protocol_error |=> protocol_error);

    generate
        for (genvar bank = 0; bank < BANKS; bank++) begin : bank_contract
            ap_exactly_one_bank_state:
                assert property ($onehot({observed_bank_free[bank],
                    observed_bank_fill[bank], observed_bank_filled[bank],
                    observed_bank_pwp[bank],
                    observed_bank_wait_correction[bank],
                    observed_bank_correction[bank]}));
            ap_release_only_after_matching_correction:
                assert property ($rose(observed_bank_free[bank])
                    |-> $past(correction_done_valid
                              && correction_done_bank == bank[1:0]));
        end
    endgenerate

    cp_k4_descriptor:
        cover property (descriptor_accept
                        && descriptor_source_count_m1 == 3);
    cp_descriptor_stall:
        cover property (descriptor_valid && !descriptor_ready ##1
                        descriptor_accept);
    cp_pwp_correction_overlap:
        cover property (observed_pwp_busy && observed_correction_busy);
    cp_all_banks_owned:
        cover property (observed_bank_free == 0);
    cp_materialized_before_pwp:
        cover property (pwp_accept && observed_bank_filled[pwp_bank]);
    cp_correction_release:
        cover property (correction_done_valid ##1
                        observed_bank_free[correction_done_bank]);
endmodule

`default_nettype wire
