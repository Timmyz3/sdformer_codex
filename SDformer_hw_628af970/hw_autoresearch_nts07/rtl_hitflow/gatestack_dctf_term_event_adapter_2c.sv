`timescale 1ns/1ps
`default_nettype none

// Two ordered contexts overlap whole-term collection/validation with command
// emission. A context becomes visible to cmd_valid only after its final event
// has passed every contract check, so malformed terms never partially execute.
module gatestack_dctf_term_event_adapter_2c #(
    parameter int TOKENS = 162,
    parameter int EVENT_WAYS = 4,
    parameter int TAG_W = 32,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int INPUT_CH_W = 10,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int ISSUE_SEQ_W = 13,
    parameter int CMD_SEQUENCE_W = 16,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      flush,
    input  logic                                      clear_error,

    input  logic                                      term_valid,
    output logic                                      term_ready,
    input  logic [TAG_W-1:0]                          term_tag,
    input  logic [GATE_CODE_W-1:0]                    term_gate_code,
    input  logic [LANE_ID_W-1:0]                      term_lane_id,
    input  logic [7:0]                                term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]                    term_issue_seq,
    input  logic                                      term_head_last,
    input  logic [INPUT_CH_W-1:0]                     term_input_channel_base,
    input  logic [LOGICAL_SUPERTILE_W-1:0]            term_logical_supertile,

    input  logic                                      event_valid,
    output logic                                      event_ready,
    input  logic [GATE_CODE_W-1:0]                    event_gate_code,
    input  logic [LANE_ID_W-1:0]                      event_lane_id,
    input  logic [EVENT_WAYS-1:0]                     event_token_valid,
    input  logic [(EVENT_WAYS*TOKEN_ID_W)-1:0]        event_token_ids,
    input  logic [WAY_COUNT_W-1:0]                    event_count,
    input  logic [ISSUE_SEQ_W-1:0]                    event_issue_seq,
    input  logic                                      event_term_first,
    input  logic                                      event_term_last,
    input  logic                                      event_head_last,

    output logic                                      cmd_valid,
    input  logic                                      cmd_ready,
    output logic [TAG_W-1:0]                          cmd_group_tag,
    output logic [CMD_SEQUENCE_W-1:0]                 cmd_sequence,
    output logic [GATE_CODE_W-1:0]                    cmd_gate_code,
    output logic [LANE_ID_W-1:0]                      cmd_lane_id,
    output logic [TOKEN_ID_W-1:0]                     cmd_destination_token,
    output logic [ISSUE_SEQ_W-1:0]                    cmd_term_issue_seq,
    output logic                                      cmd_term_first,
    output logic                                      cmd_term_last,
    output logic                                      cmd_head_last,
    output logic [INPUT_CH_W-1:0]                     cmd_input_channel_base,
    output logic [LOGICAL_SUPERTILE_W-1:0]            cmd_logical_supertile,

    output logic                                      idle,
    output logic                                      protocol_error
);
    localparam int CONTEXTS = 2;
    localparam int CONTEXT_LIMIT = CONTEXTS;
    localparam int TOKEN_INDEX_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS);
    localparam int EVENT_WAY_LIMIT = EVENT_WAYS;

    logic [CONTEXTS-1:0] context_valid_q;
    logic [CONTEXTS-1:0] context_complete_q;
    logic [TAG_W-1:0] context_tag_q [0:CONTEXTS-1];
    logic [GATE_CODE_W-1:0] context_gate_q [0:CONTEXTS-1];
    logic [LANE_ID_W-1:0] context_lane_q [0:CONTEXTS-1];
    logic [7:0] context_destination_count_q [0:CONTEXTS-1];
    logic [ISSUE_SEQ_W-1:0] context_issue_seq_q [0:CONTEXTS-1];
    logic context_head_last_q [0:CONTEXTS-1];
    logic [INPUT_CH_W-1:0] context_input_channel_base_q
        [0:CONTEXTS-1];
    logic [LOGICAL_SUPERTILE_W-1:0] context_logical_supertile_q
        [0:CONTEXTS-1];
    logic [7:0] context_received_q [0:CONTEXTS-1];
    logic [TOKENS-1:0] context_seen_q [0:CONTEXTS-1];
    logic [TOKEN_ID_W-1:0] token_mem_q [0:CONTEXTS-1][0:TOKENS-1];

    logic fill_active_q;
    logic fill_drop_q;
    logic fill_context_q;
    logic head_context_q;
    logic tail_context_q;
    logic [7:0] emit_index_q;
    logic [CMD_SEQUENCE_W-1:0] next_cmd_sequence_q;

    logic term_fire;
    logic event_fire;
    logic cmd_fire;
    logic term_count_in_range;
    logic event_metadata_matches;
    logic event_first_matches;
    logic event_last_matches;
    logic event_head_last_matches;
    logic event_tokens_in_range;
    logic event_valid_mask_matches_count;
    logic event_has_duplicate;
    logic event_contract_ok;
    logic [7:0] received_after_event;
    logic [WAY_COUNT_W-1:0] valid_count_comb;
    logic [WAY_COUNT_W-1:0] prior_valid_count [0:EVENT_WAYS-1];
    logic [TOKENS-1:0] event_seen_comb;

    assign term_ready = !flush && !fill_active_q &&
                        !context_valid_q[tail_context_q];
    assign event_ready = !flush && fill_active_q;
    assign cmd_valid = !flush && context_valid_q[head_context_q] &&
                       context_complete_q[head_context_q];
    assign idle = !fill_active_q && (context_valid_q == '0);
    assign term_fire = term_valid && term_ready;
    assign event_fire = event_valid && event_ready;
    assign cmd_fire = cmd_valid && cmd_ready;

    assign cmd_group_tag = context_tag_q[head_context_q];
    assign cmd_sequence = next_cmd_sequence_q;
    assign cmd_gate_code = context_gate_q[head_context_q];
    assign cmd_lane_id = context_lane_q[head_context_q];
    assign cmd_destination_token =
        token_mem_q[head_context_q][TOKEN_INDEX_W'(emit_index_q)];
    assign cmd_term_issue_seq = context_issue_seq_q[head_context_q];
    assign cmd_term_first = emit_index_q == '0;
    assign cmd_term_last = emit_index_q + 1'b1 ==
                           context_destination_count_q[head_context_q];
    assign cmd_head_last = cmd_term_last &&
                           context_head_last_q[head_context_q];
    assign cmd_input_channel_base =
        context_input_channel_base_q[head_context_q];
    assign cmd_logical_supertile =
        context_logical_supertile_q[head_context_q];

    assign term_count_in_range = (term_destination_count != 0) &&
                                 (32'(term_destination_count) <= TOKENS);
    assign event_metadata_matches =
        (event_gate_code == context_gate_q[fill_context_q]) &&
        (event_lane_id == context_lane_q[fill_context_q]) &&
        (event_issue_seq == context_issue_seq_q[fill_context_q]);
    assign event_first_matches = event_term_first ==
        (context_received_q[fill_context_q] == 0);
    assign received_after_event =
        context_received_q[fill_context_q] + 8'(event_count);
    assign event_last_matches = event_term_last ==
        (received_after_event ==
         context_destination_count_q[fill_context_q]);
    assign event_head_last_matches = event_head_last ==
        (event_term_last && context_head_last_q[fill_context_q]);
    assign event_valid_mask_matches_count = valid_count_comb == event_count;
    assign event_contract_ok = event_metadata_matches &&
        event_first_matches && event_last_matches &&
        event_head_last_matches && event_tokens_in_range &&
        event_valid_mask_matches_count && !event_has_duplicate &&
        (event_count != 0) &&
        (received_after_event <=
         context_destination_count_q[fill_context_q]);

    always_comb begin
        valid_count_comb = '0;
        event_seen_comb = '0;
        event_tokens_in_range = 1'b1;
        event_has_duplicate = 1'b0;
        for (int way = 32'd0; way < EVENT_WAY_LIMIT;
             way = way + 32'd1) begin
            prior_valid_count[way] = valid_count_comb;
            if (event_token_valid[way]) begin
                valid_count_comb = valid_count_comb + 1'b1;
                if (32'(event_token_ids[(way*TOKEN_ID_W) +: TOKEN_ID_W]) >=
                    TOKENS) begin
                    event_tokens_in_range = 1'b0;
                end else if (!fill_drop_q) begin
                    if (context_seen_q[fill_context_q][TOKEN_INDEX_W'(
                            event_token_ids[(way*TOKEN_ID_W) +:
                                            TOKEN_ID_W])] ||
                        event_seen_comb[TOKEN_INDEX_W'(
                            event_token_ids[(way*TOKEN_ID_W) +:
                                            TOKEN_ID_W])]) begin
                        event_has_duplicate = 1'b1;
                    end
                    event_seen_comb[TOKEN_INDEX_W'(
                        event_token_ids[(way*TOKEN_ID_W) +:
                                        TOKEN_ID_W])] = 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            context_valid_q <= '0;
            context_complete_q <= '0;
            fill_active_q <= 1'b0;
            fill_drop_q <= 1'b0;
            fill_context_q <= 1'b0;
            head_context_q <= 1'b0;
            tail_context_q <= 1'b0;
            emit_index_q <= '0;
            next_cmd_sequence_q <= '0;
            protocol_error <= 1'b0;
            for (int ctx = 32'd0; ctx < CONTEXT_LIMIT;
                 ctx = ctx + 32'd1) begin
                context_tag_q[ctx] <= '0;
                context_gate_q[ctx] <= '0;
                context_lane_q[ctx] <= '0;
                context_destination_count_q[ctx] <= '0;
                context_issue_seq_q[ctx] <= '0;
                context_head_last_q[ctx] <= 1'b0;
                context_input_channel_base_q[ctx] <= '0;
                context_logical_supertile_q[ctx] <= '0;
                context_received_q[ctx] <= '0;
                context_seen_q[ctx] <= '0;
            end
        end else if (flush) begin
            context_valid_q <= '0;
            context_complete_q <= '0;
            fill_active_q <= 1'b0;
            fill_drop_q <= 1'b0;
            fill_context_q <= 1'b0;
            head_context_q <= 1'b0;
            tail_context_q <= 1'b0;
            emit_index_q <= '0;
            if (clear_error)
                protocol_error <= 1'b0;
        end else begin
            if (clear_error)
                protocol_error <= 1'b0;

            if (term_fire) begin
                if (term_count_in_range) begin
                    context_valid_q[tail_context_q] <= 1'b1;
                    context_complete_q[tail_context_q] <= 1'b0;
                    context_tag_q[tail_context_q] <= term_tag;
                    context_gate_q[tail_context_q] <= term_gate_code;
                    context_lane_q[tail_context_q] <= term_lane_id;
                    context_destination_count_q[tail_context_q] <=
                        term_destination_count;
                    context_issue_seq_q[tail_context_q] <= term_issue_seq;
                    context_head_last_q[tail_context_q] <= term_head_last;
                    context_input_channel_base_q[tail_context_q] <=
                        term_input_channel_base;
                    context_logical_supertile_q[tail_context_q] <=
                        term_logical_supertile;
                    context_received_q[tail_context_q] <= '0;
                    context_seen_q[tail_context_q] <= '0;
                    fill_context_q <= tail_context_q;
                    fill_active_q <= 1'b1;
                    fill_drop_q <= 1'b0;
                end else begin
                    if (!clear_error)
                        protocol_error <= 1'b1;
                    if (term_destination_count != 0) begin
                        fill_context_q <= tail_context_q;
                        fill_active_q <= 1'b1;
                        fill_drop_q <= 1'b1;
                    end
                end
            end

            if (event_fire) begin
                if (fill_drop_q) begin
                    if (event_term_last) begin
                        fill_active_q <= 1'b0;
                        fill_drop_q <= 1'b0;
                    end
                end else if (!event_contract_ok) begin
                    if (!clear_error)
                        protocol_error <= 1'b1;
                    context_valid_q[fill_context_q] <= 1'b0;
                    context_complete_q[fill_context_q] <= 1'b0;
                    context_received_q[fill_context_q] <= '0;
                    context_seen_q[fill_context_q] <= '0;
                    if (event_term_last) begin
                        fill_active_q <= 1'b0;
                    end else begin
                        fill_drop_q <= 1'b1;
                    end
                end else begin
                    context_received_q[fill_context_q] <=
                        received_after_event;
                    context_seen_q[fill_context_q] <=
                        context_seen_q[fill_context_q] | event_seen_comb;
                    for (int way = 32'd0; way < EVENT_WAY_LIMIT;
                         way = way + 32'd1) begin
                        if (event_token_valid[way]) begin
                            token_mem_q[fill_context_q][TOKEN_INDEX_W'(
                                context_received_q[fill_context_q] +
                                8'(prior_valid_count[way]))] <=
                                event_token_ids[(way*TOKEN_ID_W) +:
                                                TOKEN_ID_W];
                        end
                    end
                    if (event_term_last) begin
                        context_complete_q[fill_context_q] <= 1'b1;
                        fill_active_q <= 1'b0;
                        fill_drop_q <= 1'b0;
                        tail_context_q <= ~tail_context_q;
                    end
                end
            end

            if (cmd_fire) begin
                next_cmd_sequence_q <= next_cmd_sequence_q + 1'b1;
                if (cmd_term_last) begin
                    context_valid_q[head_context_q] <= 1'b0;
                    context_complete_q[head_context_q] <= 1'b0;
                    context_received_q[head_context_q] <= '0;
                    context_seen_q[head_context_q] <= '0;
                    head_context_q <= ~head_context_q;
                    emit_index_q <= '0;
                end else begin
                    emit_index_q <= emit_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
