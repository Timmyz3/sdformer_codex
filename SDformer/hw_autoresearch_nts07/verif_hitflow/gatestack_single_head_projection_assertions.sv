`timescale 1ns/1ps
`default_nettype none

module gatestack_single_head_projection_assertions #(
    parameter int TOKENS = 162,
    parameter int BANKS = 2,
    parameter int TOKEN_ID_W = 8,
    parameter int OUT_TILE = 8,
    parameter int ACC_W = 32,
    parameter int OUTPUT_TILE_W = 8,
    parameter int TAG_W = 32,
    parameter int COUNTER_W = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter bit IMPLICIT_BIAS_FINALIZE_ENABLE = 1'b0
) (
    input logic clk_core,
    input logic rst_core,
    input logic bias_req_valid,
    input logic bias_req_ready,
    input logic [TAG_W-1:0] bias_req_tag,
    input logic [OUTPUT_TILE_W-1:0] bias_req_output_tile,
    input logic [TOKEN_ID_W-1:0] bias_req_token_id,
    input logic bias_rsp_valid,
    input logic bias_rsp_ready,
    input logic [TAG_W-1:0] bias_rsp_tag,
    input logic [TOKEN_ID_W-1:0] bias_rsp_token_id,
    input logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values,
    input logic [BANKS-1:0] final_valid,
    input logic [BANKS-1:0] final_ready,
    input logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids,
    input logic [TAG_W-1:0] final_tag,
    input logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,
    input logic group_done_valid,
    input logic group_done_ready,
    input logic [TAG_W-1:0] group_done_tag,
    input logic [COUNTER_W-1:0] count_bias_commits
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        bias_req_valid && !bias_req_ready |=>
        bias_req_valid && $stable(bias_req_tag) &&
        $stable(bias_req_output_tile) && $stable(bias_req_token_id));
    assert property (@(posedge clk_core) disable iff (rst_core)
        bias_rsp_valid && !bias_rsp_ready |=>
        bias_rsp_valid && $stable(bias_rsp_tag) &&
        $stable(bias_rsp_token_id) && $stable(bias_rsp_values));
    assert property (@(posedge clk_core) disable iff (rst_core)
        bias_req_valid && bias_req_ready |-> !bias_rsp_ready);
    assert property (@(posedge clk_core) disable iff (rst_core)
        (BIAS_STATIONARY_ENABLE || IMPLICIT_BIAS_FINALIZE_ENABLE) &&
        bias_req_valid |-> bias_req_token_id == '0);
    assert property (@(posedge clk_core) disable iff (rst_core)
        group_done_valid |-> count_bias_commits >= TOKENS);
    assert property (@(posedge clk_core) disable iff (rst_core)
        group_done_valid && !group_done_ready |=>
        group_done_valid && $stable(group_done_tag));
    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_final_stall
        assert property (@(posedge clk_core) disable iff (rst_core)
            final_valid[bank] && !final_ready[bank] |=>
            final_valid[bank] && $stable(final_tag) &&
            $stable(final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]) &&
            $stable(final_values[(bank*OUT_TILE*ACC_W) +:
                                 (OUT_TILE*ACC_W)]));
    end

    logic outstanding_q;
    logic [TAG_W-1:0] expected_tag_q;
    logic [TOKEN_ID_W-1:0] expected_token_q;
    logic [COUNTER_W-1:0] previous_bias_commits_q;
    logic [COUNTER_W:0] commit_credits_q;
    logic matching_rsp_fire;
    logic [COUNTER_W:0] bias_commit_delta;
    logic [COUNTER_W:0] matching_rsp_credit;
    assign matching_rsp_fire = bias_rsp_valid && bias_rsp_ready &&
                               outstanding_q &&
                               bias_rsp_tag == expected_tag_q &&
                               bias_rsp_token_id == expected_token_q;
    assign bias_commit_delta = {1'b0, count_bias_commits} -
                               {1'b0, previous_bias_commits_q};
    assign matching_rsp_credit = matching_rsp_fire ?
        ((BIAS_STATIONARY_ENABLE || IMPLICIT_BIAS_FINALIZE_ENABLE) ?
         (COUNTER_W+1)'(TOKENS) : (COUNTER_W+1)'(1)) : '0;
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            outstanding_q <= 1'b0;
            expected_tag_q <= '0;
            expected_token_q <= '0;
            previous_bias_commits_q <= '0;
            commit_credits_q <= '0;
        end else begin
            if (bias_req_valid && bias_req_ready) begin
                assert (!outstanding_q);
                outstanding_q <= 1'b1;
                expected_tag_q <= bias_req_tag;
                expected_token_q <= bias_req_token_id;
            end else if (bias_rsp_valid && bias_rsp_ready) begin
                outstanding_q <= 1'b0;
            end
            assert (bias_commit_delta <=
                    commit_credits_q + matching_rsp_credit);
            commit_credits_q <= commit_credits_q + matching_rsp_credit -
                                bias_commit_delta;
            previous_bias_commits_q <= count_bias_commits;
        end
    end
endmodule

`default_nettype wire
