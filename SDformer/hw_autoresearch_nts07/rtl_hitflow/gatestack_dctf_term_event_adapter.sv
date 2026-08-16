`timescale 1ns/1ps
`default_nettype none

// Validates one multi-beat backend term before serializing its destinations.
module gatestack_dctf_term_event_adapter #(
    parameter int TOKENS = 162,
    parameter int EVENT_WAYS = 4,
    parameter int TAG_W = 32,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
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

    output logic                                      protocol_error
);
    localparam int TOKEN_INDEX_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS);
    localparam int EVENT_WAY_LIMIT = EVENT_WAYS;
    localparam logic [1:0] ST_IDLE = 2'd0;
    localparam logic [1:0] ST_COLLECT = 2'd1;
    localparam logic [1:0] ST_DRAIN = 2'd2;
    localparam logic [1:0] ST_EMIT = 2'd3;

    logic [1:0] state_q;
    logic [TAG_W-1:0] current_tag_q;
    logic [GATE_CODE_W-1:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_destination_count_q;
    logic [ISSUE_SEQ_W-1:0] current_issue_seq_q;
    logic current_head_last_q;
    logic [7:0] current_received_q;
    logic [TOKENS-1:0] current_seen_q;
    logic [TOKEN_ID_W-1:0] token_mem_q [0:TOKENS-1];
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

    assign term_ready = !flush && (state_q == ST_IDLE);
    assign event_ready = !flush &&
                         ((state_q == ST_COLLECT) || (state_q == ST_DRAIN));
    assign cmd_valid = !flush && (state_q == ST_EMIT);
    assign term_fire = term_valid && term_ready;
    assign event_fire = event_valid && event_ready;
    assign cmd_fire = cmd_valid && cmd_ready;

    assign cmd_group_tag = current_tag_q;
    assign cmd_sequence = next_cmd_sequence_q;
    assign cmd_gate_code = current_gate_q;
    assign cmd_lane_id = current_lane_q;
    assign cmd_destination_token =
        token_mem_q[TOKEN_INDEX_W'(emit_index_q)];
    assign cmd_term_issue_seq = current_issue_seq_q;
    assign cmd_term_first = emit_index_q == '0;
    assign cmd_term_last = emit_index_q + 1'b1 ==
                           current_destination_count_q;
    assign cmd_head_last = cmd_term_last && current_head_last_q;

    assign term_count_in_range = (term_destination_count != 0) &&
                                 (32'(term_destination_count) <= TOKENS);
    assign event_metadata_matches =
        (event_gate_code == current_gate_q) &&
        (event_lane_id == current_lane_q) &&
        (event_issue_seq == current_issue_seq_q);
    assign event_first_matches = event_term_first ==
                                 (current_received_q == 0);
    assign received_after_event = current_received_q + 8'(event_count);
    assign event_last_matches = event_term_last ==
        (received_after_event == current_destination_count_q);
    assign event_head_last_matches = event_head_last ==
        (event_term_last && current_head_last_q);
    assign event_valid_mask_matches_count = valid_count_comb == event_count;
    assign event_contract_ok = event_metadata_matches &&
        event_first_matches && event_last_matches &&
        event_head_last_matches && event_tokens_in_range &&
        event_valid_mask_matches_count && !event_has_duplicate &&
        (event_count != 0) &&
        (received_after_event <= current_destination_count_q);

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
                end else begin
                    if (current_seen_q[TOKEN_INDEX_W'(
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
            state_q <= ST_IDLE;
            current_tag_q <= '0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_destination_count_q <= '0;
            current_issue_seq_q <= '0;
            current_head_last_q <= 1'b0;
            current_received_q <= '0;
            current_seen_q <= '0;
            emit_index_q <= '0;
            next_cmd_sequence_q <= '0;
            protocol_error <= 1'b0;
        end else if (flush) begin
            state_q <= ST_IDLE;
            current_received_q <= '0;
            current_seen_q <= '0;
            emit_index_q <= '0;
            if (clear_error)
                protocol_error <= 1'b0;
        end else begin
            if (clear_error)
                protocol_error <= 1'b0;
            if (term_fire) begin
                current_tag_q <= term_tag;
                current_gate_q <= term_gate_code;
                current_lane_q <= term_lane_id;
                current_destination_count_q <= term_destination_count;
                current_issue_seq_q <= term_issue_seq;
                current_head_last_q <= term_head_last;
                current_received_q <= '0;
                current_seen_q <= '0;
                emit_index_q <= '0;
                if (term_count_in_range) begin
                    state_q <= ST_COLLECT;
                end else begin
                    if (!clear_error)
                        protocol_error <= 1'b1;
                    if (term_destination_count == 0)
                        state_q <= ST_IDLE;
                    else
                        state_q <= ST_DRAIN;
                end
            end

            if (event_fire && (state_q == ST_COLLECT)) begin
                if (!event_contract_ok) begin
                    if (!clear_error)
                        protocol_error <= 1'b1;
                    current_received_q <= '0;
                    current_seen_q <= '0;
                    if (event_term_last)
                        state_q <= ST_IDLE;
                    else
                        state_q <= ST_DRAIN;
                end else begin
                    current_received_q <= received_after_event;
                    current_seen_q <= current_seen_q | event_seen_comb;
                    for (int way = 32'd0; way < EVENT_WAY_LIMIT;
                         way = way + 32'd1) begin
                        if (event_token_valid[way]) begin
                            token_mem_q[TOKEN_INDEX_W'(
                                current_received_q +
                                8'(prior_valid_count[way]))] <=
                                event_token_ids[(way*TOKEN_ID_W) +:
                                                TOKEN_ID_W];
                        end
                    end
                    if (event_term_last) begin
                        emit_index_q <= '0;
                        state_q <= ST_EMIT;
                    end
                end
            end

            if (event_fire && (state_q == ST_DRAIN) && event_term_last) begin
                state_q <= ST_IDLE;
                current_received_q <= '0;
                current_seen_q <= '0;
            end

            if (cmd_fire) begin
                next_cmd_sequence_q <= next_cmd_sequence_q + 1'b1;
                if (cmd_term_last) begin
                    state_q <= ST_IDLE;
                    current_received_q <= '0;
                    current_seen_q <= '0;
                    emit_index_q <= '0;
                end else begin
                    emit_index_q <= emit_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
