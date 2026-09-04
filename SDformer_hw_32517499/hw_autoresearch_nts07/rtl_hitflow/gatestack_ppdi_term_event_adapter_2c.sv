`timescale 1ns/1ps
`default_nettype none

// Two ordered contexts collect and validate complete terms. Tokens are
// partitioned by parity as each event arrives, before the context commits.
module gatestack_ppdi_term_event_adapter_2c #(
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
    output logic [1:0]                                cmd_destination_valid,
    output logic [(2*TOKEN_ID_W)-1:0]                 cmd_destination_tokens,
    output logic [ISSUE_SEQ_W-1:0]                    cmd_term_issue_seq,
    output logic                                      cmd_term_first,
    output logic                                      cmd_term_last,
    output logic                                      cmd_head_last,
    output logic [INPUT_CH_W-1:0]                     cmd_input_channel_base,
    output logic [LOGICAL_SUPERTILE_W-1:0]            cmd_logical_supertile,

    output logic                                      idle,
    output logic                                      protocol_error
);
    localparam int CONTEXTS = 32'd2;
    localparam int CONTEXT_LIMIT = CONTEXTS;
    localparam int EVENT_WAY_LIMIT = EVENT_WAYS;
    localparam int TOKEN_INDEX_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS);
    localparam int EVEN_CAP = (TOKENS + 1) / 2;
    localparam int ODD_CAP_RAW = TOKENS / 2;
    localparam int PARITY_BANKS = 32'd4;
    localparam int PARITY_BANK_W = (PARITY_BANKS <= 1) ? 1 :
                                   $clog2(PARITY_BANKS);
    localparam int EVEN_ROWS = (EVEN_CAP + PARITY_BANKS - 1) / PARITY_BANKS;
    localparam int ODD_ROWS_RAW = (ODD_CAP_RAW + PARITY_BANKS - 1) /
                                  PARITY_BANKS;
    localparam int ODD_ROWS = (ODD_ROWS_RAW < 1) ? 1 : ODD_ROWS_RAW;
    localparam int EVEN_ROW_W = (EVEN_ROWS <= 1) ? 1 : $clog2(EVEN_ROWS);
    localparam int ODD_ROW_W = (ODD_ROWS <= 1) ? 1 : $clog2(ODD_ROWS);

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
    logic [7:0] context_even_count_q [0:CONTEXTS-1];
    logic [7:0] context_odd_count_q [0:CONTEXTS-1];
    logic [TOKENS-1:0] context_seen_q [0:CONTEXTS-1];
    logic [TOKEN_ID_W-1:0] even_bank_read_data
        [0:CONTEXTS-1][0:PARITY_BANKS-1];
    logic [TOKEN_ID_W-1:0] odd_bank_read_data
        [0:CONTEXTS-1][0:PARITY_BANKS-1];

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
    logic term_contract_ok;
    logic event_metadata_matches;
    logic event_first_matches;
    logic event_last_matches;
    logic event_head_last_matches;
    logic event_tokens_in_range;
    logic event_valid_mask_matches_count;
    logic event_has_duplicate;
    logic event_parity_capacity_ok;
    logic event_contract_ok;
    logic [7:0] received_after_event;
    logic [7:0] even_count_after_event;
    logic [7:0] odd_count_after_event;
    logic [7:0] head_command_count;
    logic [WAY_COUNT_W-1:0] valid_count_comb;
    logic [WAY_COUNT_W-1:0] even_count_comb;
    logic [WAY_COUNT_W-1:0] odd_count_comb;
    logic [WAY_COUNT_W-1:0] prior_even_count [0:EVENT_WAYS-1];
    logic [WAY_COUNT_W-1:0] prior_odd_count [0:EVENT_WAYS-1];
    logic [TOKENS-1:0] event_seen_comb;
    logic [PARITY_BANKS-1:0] even_bank_write_valid;
    logic [PARITY_BANKS-1:0] odd_bank_write_valid;
    logic [EVEN_ROW_W-1:0] even_bank_write_row [0:PARITY_BANKS-1];
    logic [ODD_ROW_W-1:0] odd_bank_write_row [0:PARITY_BANKS-1];
    logic [TOKEN_ID_W-1:0] even_bank_write_data [0:PARITY_BANKS-1];
    logic [TOKEN_ID_W-1:0] odd_bank_write_data [0:PARITY_BANKS-1];
    logic event_bank_conflict;
    logic [PARITY_BANK_W-1:0] emit_bank;
    logic [EVEN_ROW_W-1:0] emit_even_row;
    logic [ODD_ROW_W-1:0] emit_odd_row;

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
    assign cmd_destination_valid[0] =
        emit_index_q < context_even_count_q[head_context_q];
    assign cmd_destination_valid[1] =
        emit_index_q < context_odd_count_q[head_context_q];
    assign emit_bank = PARITY_BANK_W'(32'(emit_index_q) % PARITY_BANKS);
    assign emit_even_row = EVEN_ROW_W'(32'(emit_index_q) / PARITY_BANKS);
    assign emit_odd_row = ODD_ROW_W'(32'(emit_index_q) / PARITY_BANKS);
    assign cmd_destination_tokens[0 +: TOKEN_ID_W] =
        cmd_destination_valid[0] ?
        even_bank_read_data[head_context_q][emit_bank] : '0;
    assign cmd_destination_tokens[TOKEN_ID_W +: TOKEN_ID_W] =
        cmd_destination_valid[1] ?
        odd_bank_read_data[head_context_q][emit_bank] : '0;
    assign cmd_term_issue_seq = context_issue_seq_q[head_context_q];
    assign cmd_term_first = emit_index_q == '0;
    assign head_command_count =
        (context_even_count_q[head_context_q] >=
         context_odd_count_q[head_context_q]) ?
        context_even_count_q[head_context_q] :
        context_odd_count_q[head_context_q];
    assign cmd_term_last = emit_index_q + 1'b1 == head_command_count;
    assign cmd_head_last = cmd_term_last &&
                           context_head_last_q[head_context_q];
    assign cmd_input_channel_base =
        context_input_channel_base_q[head_context_q];
    assign cmd_logical_supertile =
        context_logical_supertile_q[head_context_q];

    assign term_count_in_range = (term_destination_count != 0) &&
                                 (32'(term_destination_count) <= TOKENS);
    assign term_contract_ok = term_count_in_range;
    assign event_metadata_matches =
        (event_gate_code == context_gate_q[fill_context_q]) &&
        (event_lane_id == context_lane_q[fill_context_q]) &&
        (event_issue_seq == context_issue_seq_q[fill_context_q]);
    assign event_first_matches = event_term_first ==
        (context_received_q[fill_context_q] == 0);
    assign received_after_event =
        context_received_q[fill_context_q] + 8'(event_count);
    assign even_count_after_event =
        context_even_count_q[fill_context_q] + 8'(even_count_comb);
    assign odd_count_after_event =
        context_odd_count_q[fill_context_q] + 8'(odd_count_comb);
    assign event_last_matches = event_term_last ==
        (received_after_event ==
         context_destination_count_q[fill_context_q]);
    assign event_head_last_matches = event_head_last ==
        (event_term_last && context_head_last_q[fill_context_q]);
    assign event_valid_mask_matches_count = valid_count_comb == event_count;
    assign event_parity_capacity_ok =
        (32'(even_count_after_event) <= EVEN_CAP) &&
        (32'(odd_count_after_event) <= ODD_CAP_RAW);
    assign event_contract_ok = event_metadata_matches &&
        event_first_matches && event_last_matches &&
        event_head_last_matches && event_tokens_in_range &&
        event_valid_mask_matches_count && !event_has_duplicate &&
        event_parity_capacity_ok && !event_bank_conflict &&
        (event_count != 0) &&
        (received_after_event <=
         context_destination_count_q[fill_context_q]);

    always_comb begin
        valid_count_comb = '0;
        even_count_comb = '0;
        odd_count_comb = '0;
        event_seen_comb = '0;
        event_tokens_in_range = 1'b1;
        event_has_duplicate = 1'b0;
        event_bank_conflict = 1'b0;
        even_bank_write_valid = '0;
        odd_bank_write_valid = '0;
        for (int bank = 32'd0; bank < PARITY_BANKS;
             bank = bank + 32'd1) begin
            even_bank_write_row[bank] = '0;
            odd_bank_write_row[bank] = '0;
            even_bank_write_data[bank] = '0;
            odd_bank_write_data[bank] = '0;
        end
        for (int way = 32'd0; way < EVENT_WAY_LIMIT;
             way = way + 32'd1) begin
            prior_even_count[way] = even_count_comb;
            prior_odd_count[way] = odd_count_comb;
            if (event_token_valid[way]) begin
                valid_count_comb = valid_count_comb + 1'b1;
                if (event_token_ids[way*TOKEN_ID_W]) begin
                    if (odd_bank_write_valid[PARITY_BANK_W'(
                            (32'(context_odd_count_q[fill_context_q]) +
                             32'(prior_odd_count[way])) % PARITY_BANKS)]) begin
                        event_bank_conflict = 1'b1;
                    end
                    odd_bank_write_valid[PARITY_BANK_W'(
                        (32'(context_odd_count_q[fill_context_q]) +
                         32'(prior_odd_count[way])) % PARITY_BANKS)] = 1'b1;
                    odd_bank_write_row[PARITY_BANK_W'(
                        (32'(context_odd_count_q[fill_context_q]) +
                         32'(prior_odd_count[way])) % PARITY_BANKS)] =
                        ODD_ROW_W'((
                            32'(context_odd_count_q[fill_context_q]) +
                            32'(prior_odd_count[way])) / PARITY_BANKS);
                    odd_bank_write_data[PARITY_BANK_W'(
                        (32'(context_odd_count_q[fill_context_q]) +
                         32'(prior_odd_count[way])) % PARITY_BANKS)] =
                        event_token_ids[(way*TOKEN_ID_W) +: TOKEN_ID_W];
                    odd_count_comb = odd_count_comb + 1'b1;
                end else begin
                    if (even_bank_write_valid[PARITY_BANK_W'(
                            (32'(context_even_count_q[fill_context_q]) +
                             32'(prior_even_count[way])) % PARITY_BANKS)]) begin
                        event_bank_conflict = 1'b1;
                    end
                    even_bank_write_valid[PARITY_BANK_W'(
                        (32'(context_even_count_q[fill_context_q]) +
                         32'(prior_even_count[way])) % PARITY_BANKS)] = 1'b1;
                    even_bank_write_row[PARITY_BANK_W'(
                        (32'(context_even_count_q[fill_context_q]) +
                         32'(prior_even_count[way])) % PARITY_BANKS)] =
                        EVEN_ROW_W'((
                            32'(context_even_count_q[fill_context_q]) +
                            32'(prior_even_count[way])) / PARITY_BANKS);
                    even_bank_write_data[PARITY_BANK_W'(
                        (32'(context_even_count_q[fill_context_q]) +
                         32'(prior_even_count[way])) % PARITY_BANKS)] =
                        event_token_ids[(way*TOKEN_ID_W) +: TOKEN_ID_W];
                    even_count_comb = even_count_comb + 1'b1;
                end
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

    generate
        for (genvar ctx = 32'd0; ctx < CONTEXT_LIMIT;
             ctx = ctx + 32'd1) begin : g_ctx_mem
            for (genvar bank = 32'd0; bank < PARITY_BANKS;
                 bank = bank + 32'd1) begin : g_parity_bank
                gatestack_ppdi_token_bank #(
                    .TOKEN_ID_W(TOKEN_ID_W),
                    .DEPTH(EVEN_ROWS),
                    .ADDR_W(EVEN_ROW_W)
                ) u_even_token_bank (
                    .clk_core(clk_core),
                    .write_enable(!rst_core && !flush && event_fire &&
                        event_contract_ok &&
                        (fill_context_q == 1'(ctx)) &&
                        even_bank_write_valid[bank]),
                    .write_address(even_bank_write_row[bank]),
                    .write_data(even_bank_write_data[bank]),
                    .read_address(emit_even_row),
                    .read_data(even_bank_read_data[ctx][bank])
                );
                gatestack_ppdi_token_bank #(
                    .TOKEN_ID_W(TOKEN_ID_W),
                    .DEPTH(ODD_ROWS),
                    .ADDR_W(ODD_ROW_W)
                ) u_odd_token_bank (
                    .clk_core(clk_core),
                    .write_enable(!rst_core && !flush && event_fire &&
                        event_contract_ok &&
                        (fill_context_q == 1'(ctx)) &&
                        odd_bank_write_valid[bank]),
                    .write_address(odd_bank_write_row[bank]),
                    .write_data(odd_bank_write_data[bank]),
                    .read_address(emit_odd_row),
                    .read_data(odd_bank_read_data[ctx][bank])
                );
            end
        end
    endgenerate

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
                context_even_count_q[ctx] <= '0;
                context_odd_count_q[ctx] <= '0;
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
                if (term_contract_ok) begin
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
                    context_even_count_q[tail_context_q] <= '0;
                    context_odd_count_q[tail_context_q] <= '0;
                    context_seen_q[tail_context_q] <= '0;
                    fill_context_q <= tail_context_q;
                    fill_active_q <= 1'b1;
                    fill_drop_q <= 1'b0;
                end else begin
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
                    protocol_error <= 1'b1;
                    context_valid_q[fill_context_q] <= 1'b0;
                    context_complete_q[fill_context_q] <= 1'b0;
                    context_received_q[fill_context_q] <= '0;
                    context_even_count_q[fill_context_q] <= '0;
                    context_odd_count_q[fill_context_q] <= '0;
                    context_seen_q[fill_context_q] <= '0;
                    if (event_term_last) begin
                        fill_active_q <= 1'b0;
                    end else begin
                        fill_drop_q <= 1'b1;
                    end
                end else begin
                    context_received_q[fill_context_q] <=
                        received_after_event;
                    context_even_count_q[fill_context_q] <=
                        even_count_after_event;
                    context_odd_count_q[fill_context_q] <=
                        odd_count_after_event;
                    context_seen_q[fill_context_q] <=
                        context_seen_q[fill_context_q] | event_seen_comb;
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
                    context_even_count_q[head_context_q] <= '0;
                    context_odd_count_q[head_context_q] <= '0;
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
