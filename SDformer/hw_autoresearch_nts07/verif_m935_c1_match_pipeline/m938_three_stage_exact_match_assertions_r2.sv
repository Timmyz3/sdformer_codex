`timescale 1ns/1ps
`default_nettype none

// M938 repaired supplemental SVA candidate for the additive M935 parent-match
// pipeline.  It qualifies overlap and READY by the exact bank identity.
// The inherited M919/M912 execution assertions remain bound unchanged; this
// module covers only the new F/G/R preprocess control and exact tuple format.
module m938_three_stage_exact_match_assertions_r2 (
    input logic clk_core,
    input logic reset_n,
    input logic protocol_error,
    input logic match_active,
    input logic match_issue_done,
    input logic match_bank,
    input logic [5:0] match_row,
    input logic match_f_valid,
    input logic match_f_bank,
    input logic [5:0] match_f_row,
    input logic [15:0] match_f_mask,
    input logic [4:0] match_f_pop,
    input logic match_g_valid,
    input logic match_g_bank,
    input logic [5:0] match_g_row,
    input logic [15:0] match_g_mask,
    input logic [4:0] match_g_pop,
    input logic match_r_parent_valid,
    input logic [4:0] match_r_parent_pop,
    input logic [5:0] match_r_parent_id,
    input logic [15:0] match_r_parent_mask,
    input logic [31:0] match_r_directory,
    input logic [2:0] match_bank_state,
    input logic exec_active,
    input logic exec_bank
);
    localparam logic [2:0] BANK_MATCH = 3'd2;
    localparam logic [2:0] BANK_READY = 3'd3;

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (!reset_n || protocol_error);

    // One row enters F on every active issue cycle.  Row 63 closes issue but
    // does not make the bank ready; F and G must drain first.
    ap_f_accepts_each_issue: assert property (
        match_active && !match_issue_done
        |=> match_f_valid && match_f_bank == $past(match_bank)
            && match_f_row == $past(match_row));
    ap_f_ii1: assert property (
        match_f_valid && match_f_row < 6'd63
        |=> match_f_valid && match_f_bank == $past(match_f_bank)
            && match_f_row == $past(match_f_row) + 1'b1);
    ap_f_row63_closes_issue: assert property (
        match_f_valid && match_f_row == 6'd63
        |=> !match_f_valid && match_issue_done);

    // G is an exact one-cycle metadata delay of F.  Only compact metadata is
    // observable here; the wide execution payload is outside this pipeline.
    ap_g_follows_f: assert property (
        match_f_valid |=> match_g_valid
            && match_g_bank == $past(match_f_bank)
            && match_g_row == $past(match_f_row)
            && match_g_mask == $past(match_f_mask)
            && match_g_pop == $past(match_f_pop));

    // R tuple must use the frozen directory layout and exact subset/equality
    // legality.  Maximum-pop/lowest-id optimality is checked per row by the
    // independent software oracle in the M935 TB.
    ap_r_directory_format: assert property (
        match_g_valid |-> match_r_directory[31:28] == 4'b0
            && match_r_directory[27:23] == match_g_pop
            && match_r_directory[22] == match_r_parent_valid
            && match_r_directory[21:16] == match_r_parent_id
            && match_r_directory[15:0]
                == (match_r_parent_valid
                    ? (match_g_mask ^ match_r_parent_mask)
                    : match_g_mask));
    ap_r_parent_legal: assert property (
        match_g_valid && match_r_parent_valid
        |-> match_g_pop >= 2 && match_r_parent_pop >= 1
            && (match_r_parent_mask & match_g_mask) == match_r_parent_mask
            && !(match_r_parent_mask == match_g_mask
                && match_r_parent_id >= match_g_row));
    ap_no_parent_below_two: assert property (
        match_g_valid && match_g_pop < 2 |-> !match_r_parent_valid);

    // BANK_READY is a drain event, not a row-63 issue event.  At the sampled
    // row-63 R result the bank is still MATCH; only the next sample may expose
    // READY (or EXEC if ownership transfers immediately afterward).
    ap_match_state_held_through_r63: assert property (
        match_g_valid && match_g_row == 6'd63
        |-> match_bank_state == BANK_MATCH && match_active
            && match_issue_done);
    ap_ready_after_r63_commit: assert property (
        match_g_valid && match_g_row == 6'd63
        |=> !match_active
            && match_g_bank == $past(match_g_bank)
            && match_bank_state == BANK_READY);
    ap_overlap_is_bank_distinct: assert property (
        match_g_valid && exec_active |-> match_g_bank != exec_bank);

    cp_full_64_row_ii1: cover property (
        match_f_valid && match_f_row == 6'd0
        ##1 match_f_valid && match_f_row == 6'd1
        ##62 match_f_valid && match_f_row == 6'd63
        ##1 match_g_valid && match_g_row == 6'd63
        ##1 !match_active);
    cp_bank_distinct_overlap: cover property (
        match_g_valid && exec_active && match_g_bank != exec_bank);
    cp_same_pop_lowest_id_witness: cover property (
        match_g_valid && match_g_row == 6'd4
            && match_r_parent_valid && match_r_parent_id == 6'd1
            && match_r_parent_pop == match_g_pop);
endmodule

bind m935_m912_three_stage_exact_parent_match_product_capture_island
    m938_three_stage_exact_match_assertions_r2 u_m938_match_assertions_r2 (
        .clk_core(clk_core),
        .reset_n(reset_n),
        .protocol_error(protocol_error),
        .match_active(match_active_q),
        .match_issue_done(match_issue_done_q),
        .match_bank(match_bank_q),
        .match_row(match_row_q),
        .match_f_valid(match_f_valid_q),
        .match_f_bank(match_f_bank_q),
        .match_f_row(match_f_row_q),
        .match_f_mask(match_f_mask_q),
        .match_f_pop(match_f_pop_q),
        .match_g_valid(match_g_valid_q),
        .match_g_bank(match_g_bank_q),
        .match_g_row(match_g_row_q),
        .match_g_mask(match_g_mask_q),
        .match_g_pop(match_g_pop_q),
        .match_r_parent_valid(match_r_winner_w[27]),
        .match_r_parent_pop(match_r_winner_w[26:22]),
        .match_r_parent_id(match_r_winner_w[21:16]),
        .match_r_parent_mask(match_r_winner_w[15:0]),
        .match_r_directory(match_r_directory_w),
        .match_bank_state(bank_state_q[match_g_bank_q]),
        .exec_active(exec_active_q),
        .exec_bank(exec_bank_q)
    );

`default_nettype wire
