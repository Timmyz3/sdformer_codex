`timescale 1ns/1ps
`default_nettype none

module qfit_linear5_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic term_ready,
    input logic term_fire,
    input logic live_contract_valid,
    input logic replay_valid_q,
    input logic [4:0] replay_remaining_mask_q,
    input logic [4:0] replay_issue_mask,
    input logic [4:0] replay_next_remaining,
    input logic window_close,
    input logic window_close_ready,
    input logic close_pending_q,
    input logic [2:0] state_q,
    input logic protocol_error_q
);
    property p_replay_blocks_new_term;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid_q |-> !term_ready;
    endproperty

    property p_replay_issues_only_pending_roles;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid_q
            |-> (replay_issue_mask & ~replay_remaining_mask_q) == '0;
    endproperty

    property p_replay_makes_progress;
        @(posedge clk_core) disable iff (rst_core)
            replay_valid_q
            |-> replay_issue_mask != '0
                && $countones(replay_next_remaining)
                   < $countones(replay_remaining_mask_q);
    endproperty

    property p_illegal_term_sets_error;
        @(posedge clk_core) disable iff (rst_core)
            term_fire && !live_contract_valid |=> protocol_error_q;
    endproperty

    property p_close_is_not_lost;
        @(posedge clk_core) disable iff (rst_core)
            window_close && window_close_ready
            |=> close_pending_q || state_q == 3'd3;
    endproperty

    assert property (p_replay_blocks_new_term);
    assert property (p_replay_issues_only_pending_roles);
    assert property (p_replay_makes_progress);
    assert property (p_illegal_term_sets_error);
    assert property (p_close_is_not_lost);
endmodule

bind qfit_linear5_projection_top qfit_linear5_assertions
    u_qfit_linear5_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_ready(term_ready),
        .term_fire(term_fire),
        .live_contract_valid(live_contract_valid),
        .replay_valid_q(replay_valid_q),
        .replay_remaining_mask_q(replay_remaining_mask_q),
        .replay_issue_mask(replay_issue_mask),
        .replay_next_remaining(replay_next_remaining),
        .window_close(window_close),
        .window_close_ready(window_close_ready),
        .close_pending_q(close_pending_q),
        .state_q(state_q),
        .protocol_error_q(protocol_error_q)
    );

`default_nettype wire
