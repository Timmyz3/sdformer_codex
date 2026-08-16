`timescale 1ns/1ps
`default_nettype none

module gatestack_ppdi_dctf32_bank_executor_assertions #(
    parameter int OUT_TILE = 32,
    parameter int PRODUCT_W = 17,
    parameter int GROUP_TAG_W = 32,
    parameter int CMD_SEQUENCE_W = 16,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int GATE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = 8,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic clear_error,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [GROUP_TAG_W-1:0] cmd_group_tag,
    input logic [CMD_SEQUENCE_W-1:0] cmd_sequence,
    input logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq,
    input logic cmd_term_first,
    input logic cmd_term_last,
    input logic cmd_head_last,
    input logic [INPUT_CH_W-1:0] cmd_input_channel,
    input logic [GATE_W-1:0] cmd_gate_code,
    input logic [LANE_ID_W-1:0] cmd_lane_id,
    input logic [1:0] cmd_destination_valid,
    input logic [(2*TOKEN_ID_W)-1:0] cmd_destination_tokens,
    input logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile,
    input logic [1:0] acc_update_valid,
    input logic [1:0] acc_update_ready,
    input logic [(2*TOKEN_ID_W)-1:0] acc_update_token_ids,
    input logic [GROUP_TAG_W-1:0] acc_update_tag,
    input logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values,
    input logic term_done,
    input logic [GROUP_TAG_W-1:0] term_done_group_tag,
    input logic [ISSUE_SEQ_W-1:0] term_done_issue_seq,
    input logic term_done_head_last,
    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic weight_rsp_valid,
    input logic weight_rsp_ready,
    input logic protocol_error,
    input logic engine_protocol_error,
    input logic command_protocol_bad,
    input logic engine_product_valid,
    input logic engine_product_identity_ok,
    input logic unknown_stale_response_fire,
    input logic [COUNTER_W-1:0] count_stale_weight_responses,
    input logic term_active_q,
    input logic zero_gate_active,
    input logic zero_gate_retire_enable,
    input logic active_command_ok,
    input logic engine_term_valid,
    input logic start_fire,
    input logic update_fire,
    input logic [1:0] acc_port_fire,
    input logic [1:0] destination_done_q,
    input logic weight_response_is_stale,
    input logic engine_weight_rsp_valid
);
    property p_command_stable_until_retire;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_active_q && cmd_valid && !cmd_ready |=> cmd_valid &&
            $stable({cmd_group_tag, cmd_sequence, cmd_term_issue_seq,
                     cmd_term_first, cmd_term_last, cmd_head_last,
                     cmd_input_channel, cmd_gate_code, cmd_lane_id,
                     cmd_destination_valid, cmd_destination_tokens,
                     logical_supertile});
    endproperty
    property p_first_prefetch_holds_command;
        @(posedge clk_core) disable iff (rst_core || flush)
        start_fire && !cmd_ready |=> cmd_valid &&
            $stable({cmd_group_tag, cmd_sequence, cmd_term_issue_seq,
                     cmd_term_first, cmd_term_last, cmd_head_last,
                     cmd_input_channel, cmd_gate_code, cmd_lane_id,
                     cmd_destination_valid, cmd_destination_tokens,
                     logical_supertile});
    endproperty
    property p_weight_request_held;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_req_valid && !weight_req_ready |=> weight_req_valid;
    endproperty
    property p_even_update_stable;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[0] && !acc_update_ready[0] |=>
            acc_update_valid[0] &&
            $stable({acc_update_token_ids[0 +: TOKEN_ID_W],
                     acc_update_tag, acc_update_values});
    endproperty
    property p_odd_update_stable;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[1] && !acc_update_ready[1] |=>
            acc_update_valid[1] &&
            $stable({acc_update_token_ids[TOKEN_ID_W +: TOKEN_ID_W],
                     acc_update_tag, acc_update_values});
    endproperty
    property p_even_parity;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[0] |-> !acc_update_token_ids[0];
    endproperty
    property p_odd_parity;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid[1] |-> acc_update_token_ids[TOKEN_ID_W];
    endproperty
    property p_completed_port_not_reissued;
        @(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q[0] |-> !acc_update_valid[0];
    endproperty
    property p_completed_odd_not_reissued;
        @(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q[1] |-> !acc_update_valid[1];
    endproperty
    property p_even_fire_sets_done;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_port_fire[0] && !update_fire |=> destination_done_q[0];
    endproperty
    property p_odd_fire_sets_done;
        @(posedge clk_core) disable iff (rst_core || flush)
        acc_port_fire[1] && !update_fire |=> destination_done_q[1];
    endproperty
    property p_even_done_has_cause;
        @(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q[0] |->
            $past(destination_done_q[0] || acc_port_fire[0]);
    endproperty
    property p_odd_done_has_cause;
        @(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q[1] |->
            $past(destination_done_q[1] || acc_port_fire[1]);
    endproperty
    property p_command_ready_has_all_commits;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_ready |-> cmd_valid && (zero_gate_retire_enable ||
            (&((~cmd_destination_valid) | destination_done_q |
               (acc_update_valid & acc_update_ready))));
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
    property p_clear_old_child_error;
        @(posedge clk_core) disable iff (rst_core)
        clear_error && engine_protocol_error && !command_protocol_bad &&
        !(engine_product_valid && !engine_product_identity_ok) &&
        !unknown_stale_response_fire |=>
            !engine_protocol_error && !protocol_error;
    endproperty
    property p_stale_response_dropped;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_rsp_valid && weight_rsp_ready &&
        weight_response_is_stale |-> !engine_weight_rsp_valid;
    endproperty
    property p_stale_response_counted;
        @(posedge clk_core) disable iff (rst_core || flush)
        weight_rsp_valid && weight_rsp_ready &&
        weight_response_is_stale |=>
            count_stale_weight_responses ==
            $past(count_stale_weight_responses) + 1'b1;
    endproperty
    property p_zero_gate_avoids_weight_and_acc;
        @(posedge clk_core) disable iff (rst_core || flush)
        zero_gate_active |-> !engine_term_valid && !weight_req_valid &&
                            (acc_update_valid == 2'b00);
    endproperty
    property p_zero_gate_active_command_retires;
        @(posedge clk_core) disable iff (rst_core || flush)
        zero_gate_active && cmd_valid && active_command_ok |->
            zero_gate_retire_enable && cmd_ready;
    endproperty
    property p_zero_gate_term_done_exact;
        @(posedge clk_core) disable iff (rst_core || flush)
        zero_gate_active && cmd_valid && active_command_ok |->
            term_done == (cmd_ready && cmd_term_last);
    endproperty

    assert property (p_command_stable_until_retire);
    assert property (p_first_prefetch_holds_command);
    assert property (p_weight_request_held);
    assert property (p_even_update_stable);
    assert property (p_odd_update_stable);
    assert property (p_even_parity);
    assert property (p_odd_parity);
    assert property (p_completed_port_not_reissued);
    assert property (p_completed_odd_not_reissued);
    assert property (p_even_fire_sets_done);
    assert property (p_odd_fire_sets_done);
    assert property (p_even_done_has_cause);
    assert property (p_odd_done_has_cause);
    assert property (p_command_ready_has_all_commits);
    assert property (p_term_done_exact);
    assert property (p_term_done_metadata);
    assert property (p_head_last_exact);
    assert property (p_flush_masks_outputs);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_old_child_error);
    assert property (p_stale_response_dropped);
    assert property (p_stale_response_counted);
    assert property (p_zero_gate_avoids_weight_and_acc);
    assert property (p_zero_gate_active_command_retires);
    assert property (p_zero_gate_term_done_exact);

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        acc_update_valid == 2'b11 && acc_update_ready == 2'b01);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q == 2'b01 && acc_update_valid == 2'b10);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        destination_done_q == 2'b01 && acc_update_valid == 2'b10 &&
        weight_rsp_valid && weight_rsp_ready &&
        weight_response_is_stale);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_ready && cmd_destination_valid == 2'b11);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_ready && cmd_destination_valid == 2'b01);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_ready && cmd_destination_valid == 2'b10);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        zero_gate_active && cmd_valid && cmd_ready && cmd_term_last);
endmodule

`default_nettype wire
