`timescale 1ns/1ps
`default_nettype none

module gatestack_multihead_tile_projection_assertions #(
    parameter int TOKENS = 162,
    parameter int TAG_W = 32,
    parameter int OUTPUT_TILE_W = 8,
    parameter int HEAD_COUNT_W = 6,
    parameter int INPUT_CH_W = 10,
    parameter int TOKEN_ID_W = 8,
    parameter int OUT_TILE = 8,
    parameter int ACC_W = 32,
    parameter int COUNTER_W = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0
) (
    input logic clk_core,
    input logic rst_core,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [TAG_W-1:0] tile_start_tag,
    input logic [OUTPUT_TILE_W-1:0] tile_start_output_tile,
    input logic [HEAD_COUNT_W-1:0] tile_start_head_count,
    input logic head_start_valid,
    input logic head_start_ready,
    input logic [TAG_W-1:0] head_start_tag,
    input logic [HEAD_COUNT_W-1:0] head_start_index,
    input logic [INPUT_CH_W-1:0] head_start_input_channel_base,
    input logic head_start_last,
    input logic head_done_valid,
    input logic head_done_ready,
    input logic [TAG_W-1:0] head_done_tag,
    input logic [HEAD_COUNT_W-1:0] head_done_index,
    input logic head_done_last,
    input logic head_done_error,
    input logic tile_done_valid,
    input logic tile_done_ready,
    input logic [TAG_W-1:0] tile_done_tag,
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
    input logic [COUNTER_W-1:0] count_bias_commits
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        tile_start_valid && !tile_start_ready |=>
        tile_start_valid && $stable(tile_start_tag) &&
        $stable(tile_start_output_tile) && $stable(tile_start_head_count));
    assert property (@(posedge clk_core) disable iff (rst_core)
        head_start_valid && !head_start_ready |=>
        head_start_valid && $stable(head_start_tag) &&
        $stable(head_start_index) && $stable(head_start_input_channel_base) &&
        $stable(head_start_last));
    assert property (@(posedge clk_core) disable iff (rst_core)
        head_done_valid && !head_done_ready |=>
        head_done_valid && $stable(head_done_tag) &&
        $stable(head_done_index) && $stable(head_done_last) &&
        $stable(head_done_error));
    assert property (@(posedge clk_core) disable iff (rst_core)
        tile_done_valid && !tile_done_ready |=>
        tile_done_valid && $stable(tile_done_tag));
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
        BIAS_STATIONARY_ENABLE && bias_req_valid |-> bias_req_token_id == '0);

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
        (BIAS_STATIONARY_ENABLE ? (COUNTER_W+1)'(TOKENS) :
                                   (COUNTER_W+1)'(1)) : '0;
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
