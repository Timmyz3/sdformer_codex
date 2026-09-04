`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf_term_event_adapter_assertions #(
    parameter int TAG_W = 32,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int CMD_SEQUENCE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic clear_error,
    input logic term_ready,
    input logic event_ready,
    input logic event_fire,
    input logic event_contract_ok,
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
    input logic protocol_error
);
    logic cmd_seen_q;
    logic [CMD_SEQUENCE_W-1:0] last_cmd_sequence_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cmd_seen_q <= 1'b0;
            last_cmd_sequence_q <= '0;
        end else if (cmd_valid && cmd_ready) begin
            cmd_seen_q <= 1'b1;
            last_cmd_sequence_q <= cmd_sequence;
        end
    end

    property p_flush_blocks_interfaces;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> !term_ready && !event_ready && !cmd_valid;
    endproperty

    property p_ready_roles_are_exclusive;
        @(posedge clk_core) disable iff (rst_core)
        !(term_ready && event_ready);
    endproperty

    property p_cmd_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
        cmd_valid && !cmd_ready && !flush |=>
            flush || (cmd_valid &&
            $stable({cmd_group_tag, cmd_sequence, cmd_gate_code,
                     cmd_lane_id, cmd_destination_token,
                     cmd_term_issue_seq, cmd_term_first,
                     cmd_term_last, cmd_head_last}));
    endproperty

    property p_cmd_sequence_order;
        @(posedge clk_core) disable iff (rst_core)
        cmd_valid && cmd_ready && cmd_seen_q |->
            cmd_sequence == last_cmd_sequence_q + 1'b1;
    endproperty

    property p_head_last_is_term_last;
        @(posedge clk_core) disable iff (rst_core)
        cmd_valid && cmd_head_last |-> cmd_term_last;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error |=> clear_error || protocol_error;
    endproperty

    property p_clear_error_clears_sticky;
        @(posedge clk_core) disable iff (rst_core)
        clear_error |=> !protocol_error;
    endproperty

    property p_bad_event_does_not_emit_next_cycle;
        @(posedge clk_core) disable iff (rst_core || flush)
        event_fire && !event_contract_ok |=> !cmd_valid;
    endproperty

    assert property (p_flush_blocks_interfaces);
    assert property (p_ready_roles_are_exclusive);
    assert property (p_cmd_stable_under_backpressure);
    assert property (p_cmd_sequence_order);
    assert property (p_head_last_is_term_last);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_error_clears_sticky);
    assert property (p_bad_event_does_not_emit_next_cycle);

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    cmd_valid && cmd_ready && cmd_term_first);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    cmd_valid && cmd_ready && cmd_term_last && cmd_head_last);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    event_fire && !event_contract_ok);
endmodule

`default_nettype wire
