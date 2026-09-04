`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf32_bank_executor_assertions #(
    parameter int OUT_TILE = 32,
    parameter int PRODUCT_W = 17,
    parameter int GROUP_TAG_W = 32,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int TOKEN_ID_W = 8,
    parameter int OUTPUT_TILE_W = 8,
    parameter int EPOCH_W = 4,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic clear_error,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [GROUP_TAG_W-1:0] cmd_group_tag,
    input logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq,
    input logic cmd_term_last,
    input logic cmd_head_last,
    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic [GROUP_TAG_W-1:0] weight_req_tag,
    input logic [INPUT_CH_W-1:0] weight_req_input_channel,
    input logic [OUTPUT_TILE_W-1:0] weight_req_output_tile,
    input logic [EPOCH_W-1:0] weight_req_epoch,
    input logic weight_rsp_valid,
    input logic weight_rsp_ready,
    input logic [EPOCH_W-1:0] weight_rsp_epoch,
    input logic [1:0] acc_update_valid,
    input logic [1:0] acc_update_ready,
    input logic [(2*TOKEN_ID_W)-1:0] acc_update_token_ids,
    input logic [GROUP_TAG_W-1:0] acc_update_tag,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values,
    input logic term_done,
    input logic [GROUP_TAG_W-1:0] term_done_group_tag,
    input logic [ISSUE_SEQ_W-1:0] term_done_issue_seq,
    input logic term_done_head_last,
    input logic protocol_error,
    input logic [COUNTER_W-1:0] count_stale_weight_responses,
    input logic [EPOCH_W-1:0] epoch_q,
    input logic engine_weight_rsp_valid
);
    property p_weight_request_stable;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_req_valid && !weight_req_ready |=> weight_req_valid &&
            $stable({weight_req_tag, weight_req_input_channel,
                     weight_req_output_tile, weight_req_epoch});
    endproperty
    property p_acc_update_stable;
        @(posedge clk_core) disable iff (rst_core || flush)
        (acc_update_valid != 2'b00) &&
        !(|(acc_update_valid & acc_update_ready)) |=>
            (acc_update_valid != 2'b00) &&
            $stable({acc_update_valid, acc_update_token_ids,
                     acc_update_tag, acc_update_values});
    endproperty
    property p_command_consumption_matches_update;
        @(posedge clk_core) disable iff (rst_core || flush)
        (cmd_valid && cmd_ready) ==
        (|(acc_update_valid & acc_update_ready));
    endproperty
    property p_update_onehot0;
        @(posedge clk_core) disable iff (rst_core || flush)
        $onehot0(acc_update_valid);
    endproperty
    property p_even_bank_token;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[0] |-> !acc_update_token_ids[0];
    endproperty
    property p_odd_bank_token;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[1] |-> acc_update_token_ids[TOKEN_ID_W];
    endproperty
    property p_term_done_exact;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_done == (cmd_valid && cmd_ready && cmd_term_last);
    endproperty
    property p_term_done_metadata;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_done |-> (term_done_group_tag == cmd_group_tag) &&
                      (term_done_issue_seq == cmd_term_issue_seq);
    endproperty
    property p_head_last_exact;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_done_head_last == (term_done && cmd_head_last);
    endproperty
    property p_flush_masks_outputs;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> !cmd_ready && !weight_req_valid && !weight_rsp_ready &&
                  (acc_update_valid == 2'b00) && !term_done &&
                  !term_done_head_last;
    endproperty
    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error |=> clear_error || protocol_error;
    endproperty
    property p_clear_error_clears_sticky;
        @(posedge clk_core) disable iff (rst_core)
        clear_error |=> !protocol_error;
    endproperty
    property p_stale_response_dropped;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_rsp_valid && weight_rsp_ready &&
        (weight_rsp_epoch != epoch_q) |->
            !engine_weight_rsp_valid && (acc_update_valid == 2'b00) &&
            !term_done && !term_done_head_last;
    endproperty
    property p_stale_response_counted;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_rsp_valid && weight_rsp_ready &&
        (weight_rsp_epoch != epoch_q) |=>
            count_stale_weight_responses ==
            $past(count_stale_weight_responses) + 1'b1;
    endproperty

    assert property (p_weight_request_stable);
    assert property (p_acc_update_stable);
    assert property (p_command_consumption_matches_update);
    assert property (p_update_onehot0);
    assert property (p_even_bank_token);
    assert property (p_odd_bank_token);
    assert property (p_term_done_exact);
    assert property (p_term_done_metadata);
    assert property (p_head_last_exact);
    assert property (p_flush_masks_outputs);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_error_clears_sticky);
    assert property (p_stale_response_dropped);
    assert property (p_stale_response_counted);
endmodule

`default_nettype wire
