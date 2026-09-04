`timescale 1ns/1ps
`default_nettype none

module gatestack_ppdi_term_event_adapter_2c_assertions #(
    parameter int TOKENS = 162,
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
    input logic [GATE_CODE_W-1:0] term_gate_code,
    input logic event_valid,
    input logic event_ready,
    input logic event_term_last,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [TAG_W-1:0] cmd_group_tag,
    input logic [CMD_SEQUENCE_W-1:0] cmd_sequence,
    input logic [GATE_CODE_W-1:0] cmd_gate_code,
    input logic [LANE_ID_W-1:0] cmd_lane_id,
    input logic [1:0] cmd_destination_valid,
    input logic [(2*TOKEN_ID_W)-1:0] cmd_destination_tokens,
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
    input logic [7:0] context_destination_count_q [0:1],
    input logic [7:0] context_even_count_q [0:1],
    input logic [7:0] context_odd_count_q [0:1],
    input logic fill_active_q,
    input logic fill_drop_q,
    input logic fill_context_q,
    input logic head_context_q,
    input logic tail_context_q,
    input logic term_count_in_range,
    input logic term_contract_ok,
    input logic event_contract_ok,
    input logic event_fire,
    input logic cmd_fire,
    input logic [7:0] emit_index_q
);
    logic command_seen_q;
    logic [CMD_SEQUENCE_W-1:0] last_command_sequence_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            command_seen_q <= 1'b0;
            last_command_sequence_q <= '0;
        end else if (cmd_fire) begin
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

    property p_malformed_context_not_emitted;
        @(posedge clk_core) disable iff (rst_core || flush)
        fill_active_q && fill_drop_q &&
        (fill_context_q == head_context_q) |-> !cmd_valid;
    endproperty

    property p_drop_context_never_completes;
        @(posedge clk_core) disable iff (rst_core || flush)
        fill_active_q && fill_drop_q |->
            !context_valid_q[fill_context_q] &&
            !context_complete_q[fill_context_q];
    endproperty

    property p_bad_term_does_not_allocate;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_valid && term_ready && !term_contract_ok |=>
            !context_valid_q[$past(tail_context_q)];
    endproperty

    property p_zero_gate_term_allocates_normally;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_valid && term_ready && term_gate_code == '0 &&
        term_count_in_range |=>
            fill_active_q && !fill_drop_q &&
            context_valid_q[$past(tail_context_q)];
    endproperty

    property p_term_contract_matches_scalar_count_check;
        @(posedge clk_core) disable iff (rst_core || flush)
        term_contract_ok == term_count_in_range;
    endproperty

    property p_even_context_complete_has_legal_last;
        @(posedge clk_core) disable iff (rst_core || flush)
        $rose(context_complete_q[0]) |->
            $past(event_fire && event_contract_ok && event_term_last &&
                  !fill_context_q);
    endproperty

    property p_odd_context_complete_has_legal_last;
        @(posedge clk_core) disable iff (rst_core || flush)
        $rose(context_complete_q[1]) |->
            $past(event_fire && event_contract_ok && event_term_last &&
                  fill_context_q);
    endproperty

    property p_head_advances_after_last_commit;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_fire && cmd_term_last |=>
            head_context_q != $past(head_context_q);
    endproperty

    property p_tail_advances_after_validated_commit;
        @(posedge clk_core) disable iff (rst_core || flush)
        event_fire && event_contract_ok && event_term_last |=>
            tail_context_q != $past(tail_context_q);
    endproperty

    property p_cmd_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && !cmd_ready |=> flush ||
            (cmd_valid && $stable({cmd_group_tag, cmd_sequence,
             cmd_gate_code, cmd_lane_id, cmd_destination_valid,
             cmd_destination_tokens, cmd_term_issue_seq, cmd_term_first,
             cmd_term_last, cmd_head_last, cmd_input_channel_base,
             cmd_logical_supertile}));
    endproperty

    property p_destination_valid_nonzero;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid |-> cmd_destination_valid != 2'b00;
    endproperty

    property p_even_destination_parity;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_destination_valid[0] |->
            !cmd_destination_tokens[0] &&
            32'(cmd_destination_tokens[0 +: TOKEN_ID_W]) < TOKENS;
    endproperty

    property p_odd_destination_parity;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_destination_valid[1] |->
            cmd_destination_tokens[TOKEN_ID_W] &&
            32'(cmd_destination_tokens[TOKEN_ID_W +: TOKEN_ID_W]) < TOKENS;
    endproperty

    property p_complete_partition_count;
        @(posedge clk_core) disable iff (rst_core || flush)
        context_complete_q[head_context_q] |->
            context_even_count_q[head_context_q] +
            context_odd_count_q[head_context_q] ==
            context_destination_count_q[head_context_q];
    endproperty

    property p_destination_mask_matches_counts;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid |->
            (cmd_destination_valid[0] ==
             (emit_index_q < context_even_count_q[head_context_q])) &&
            (cmd_destination_valid[1] ==
             (emit_index_q < context_odd_count_q[head_context_q]));
    endproperty

    property p_term_first_boundary;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid |-> (cmd_term_first == (emit_index_q == 0));
    endproperty

    property p_term_last_boundary;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid |-> (cmd_term_last ==
            ((emit_index_q + 1'b1 >=
              context_even_count_q[head_context_q]) &&
             (emit_index_q + 1'b1 >=
              context_odd_count_q[head_context_q])));
    endproperty

    property p_head_last_is_term_last;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_head_last |-> cmd_term_last;
    endproperty

    property p_command_sequence_order;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_fire && command_seen_q |->
            cmd_sequence == last_command_sequence_q + 1'b1;
    endproperty

    property p_flush_clears_contexts;
        @(posedge clk_core) disable iff (rst_core)
        flush |=> idle && context_valid_q == 2'b00 &&
                  context_complete_q == 2'b00 && !fill_active_q &&
                  !cmd_valid;
    endproperty

    property p_flush_masks_interfaces;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> !term_ready && !event_ready && !cmd_valid;
    endproperty

    property p_idle_exact;
        @(posedge clk_core) disable iff (rst_core || flush)
        idle |-> !fill_active_q && context_valid_q == 2'b00 && !cmd_valid;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error && !flush |=>
            clear_error || flush || protocol_error;
    endproperty

    property p_clear_error_clears_sticky;
        @(posedge clk_core) disable iff (rst_core)
        clear_error &&
        !(term_valid && term_ready && !term_contract_ok) &&
        !(event_fire && !fill_drop_q && !event_contract_ok) |=>
            !protocol_error;
    endproperty

    property p_new_error_wins_over_clear;
        @(posedge clk_core) disable iff (rst_core || flush)
        clear_error &&
        ((term_valid && term_ready && !term_contract_ok) ||
         (event_fire && !fill_drop_q && !event_contract_ok)) |=>
            protocol_error;
    endproperty

    assert property (p_cmd_only_from_validated_head);
    assert property (p_no_context_overwrite);
    assert property (p_live_fill_owns_context);
    assert property (p_malformed_context_not_emitted);
    assert property (p_drop_context_never_completes);
    assert property (p_bad_term_does_not_allocate);
    assert property (p_zero_gate_term_allocates_normally);
    assert property (p_term_contract_matches_scalar_count_check);
    assert property (p_even_context_complete_has_legal_last);
    assert property (p_odd_context_complete_has_legal_last);
    assert property (p_head_advances_after_last_commit);
    assert property (p_tail_advances_after_validated_commit);
    assert property (p_cmd_stable_under_backpressure);
    assert property (p_destination_valid_nonzero);
    assert property (p_even_destination_parity);
    assert property (p_odd_destination_parity);
    assert property (p_complete_partition_count);
    assert property (p_destination_mask_matches_counts);
    assert property (p_term_first_boundary);
    assert property (p_term_last_boundary);
    assert property (p_head_last_is_term_last);
    assert property (p_command_sequence_order);
    assert property (p_flush_masks_interfaces);
    assert property (p_flush_clears_contexts);
    assert property (p_idle_exact);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_error_clears_sticky);
    assert property (p_new_error_wins_over_clear);

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        event_valid && event_ready && cmd_valid);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        context_valid_q == 2'b11);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_fire && cmd_destination_valid == 2'b11);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_fire && cmd_destination_valid == 2'b01);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
        cmd_fire && cmd_destination_valid == 2'b10);
endmodule

`default_nettype wire
