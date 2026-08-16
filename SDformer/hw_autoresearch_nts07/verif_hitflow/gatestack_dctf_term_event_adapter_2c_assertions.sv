`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf_term_event_adapter_2c_assertions #(
    parameter int TAG_W = 32,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int TOKEN_ID_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int CMD_SEQUENCE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic clear_error,
    input logic term_valid,
    input logic term_ready,
    input logic event_valid,
    input logic event_ready,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [TAG_W-1:0] cmd_group_tag,
    input logic [CMD_SEQUENCE_W-1:0] cmd_sequence,
    input logic [GATE_CODE_W-1:0] cmd_gate_code,
    input logic [LANE_ID_W-1:0] cmd_lane_id,
    input logic [TOKEN_ID_W-1:0] cmd_destination_token,
    input logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq,
    input logic cmd_term_first,
    input logic cmd_term_last,
    input logic cmd_head_last,
    input logic [INPUT_CH_W-1:0] cmd_input_channel_base,
    input logic [LOGICAL_SUPERTILE_W-1:0] cmd_logical_supertile,
    input logic idle,
    input logic protocol_error,
    input logic [1:0] context_valid_q,
    input logic [1:0] context_complete_q,
    input logic fill_active_q,
    input logic fill_drop_q,
    input logic fill_context_q,
    input logic head_context_q,
    input logic tail_context_q
);
    logic command_seen_q;
    logic [CMD_SEQUENCE_W-1:0] last_command_sequence_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            command_seen_q <= 1'b0;
            last_command_sequence_q <= '0;
        end else if (cmd_valid && cmd_ready) begin
            command_seen_q <= 1'b1;
            last_command_sequence_q <= cmd_sequence;
        end
    end

    property p_cmd_only_from_validated_head;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid |-> context_valid_q[head_context_q] &&
                     context_complete_q[head_context_q];
    endproperty

    property p_no_context_overwrite;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_valid && term_ready |-> !context_valid_q[tail_context_q];
    endproperty

    property p_live_fill_owns_context;
        @(posedge clk_core) disable iff (rst_core || flush)
        fill_active_q && !fill_drop_q |-> context_valid_q[fill_context_q];
    endproperty

    property p_idle_exact;
        @(posedge clk_core) disable iff (rst_core || flush)
        idle |-> !fill_active_q && context_valid_q == 2'b00 && !cmd_valid;
    endproperty

    property p_flush_clears_contexts;
        @(posedge clk_core) disable iff (rst_core)
        flush |=> idle && context_valid_q == 2'b00 &&
                  context_complete_q == 2'b00 && !cmd_valid;
    endproperty

    property p_cmd_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && !cmd_ready |=> flush ||
            (cmd_valid && $stable({cmd_group_tag, cmd_sequence,
             cmd_gate_code, cmd_lane_id, cmd_destination_token,
             cmd_term_issue_seq, cmd_term_first, cmd_term_last,
             cmd_head_last, cmd_input_channel_base,
             cmd_logical_supertile}));
    endproperty

    property p_head_last_is_term_last;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_head_last |-> cmd_term_last;
    endproperty

    property p_command_sequence_order;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_ready && command_seen_q |->
            cmd_sequence == last_command_sequence_q + 1'b1;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error && !flush |=>
            clear_error || flush || protocol_error;
    endproperty

    assert property (p_cmd_only_from_validated_head);
    assert property (p_no_context_overwrite);
    assert property (p_live_fill_owns_context);
    assert property (p_idle_exact);
    assert property (p_flush_clears_contexts);
    assert property (p_cmd_stable_under_backpressure);
    assert property (p_head_last_is_term_last);
    assert property (p_command_sequence_order);
    assert property (p_protocol_error_sticky);

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    event_valid && event_ready && cmd_valid && cmd_ready);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    context_valid_q == 2'b11);
endmodule

`default_nettype wire
