`timescale 1ns/1ps
`default_nettype none

// Tile-lifetime projection controller. One accumulator group spans all input
// heads; the shared TDR backend is restarted for each head and bias is applied
// exactly once after the final head completes.
module gatestack_multihead_tile_projection_top #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int EVENT_WAYS      = 4,
    parameter int OUT_TILE        = 8,
    parameter int BANKS           = 2,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int ACC_W           = 32,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int HEAD_COUNT_W    = 6,
    parameter int COUNTER_W       = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int TOKEN_ID_W      = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int OUTSTANDING_W   = ISSUE_SEQ_W + 1
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      tile_start_valid,
    output logic                                      tile_start_ready,
    input  logic [TAG_W-1:0]                          tile_start_tag,
    input  logic [OUTPUT_TILE_W-1:0]                  tile_start_output_tile,
    input  logic [HEAD_COUNT_W-1:0]                   tile_start_head_count,

    input  logic                                      head_start_valid,
    output logic                                      head_start_ready,
    input  logic [TAG_W-1:0]                          head_start_tag,
    input  logic [HEAD_COUNT_W-1:0]                   head_start_index,
    input  logic [INPUT_CH_W-1:0]                     head_start_input_channel_base,
    input  logic                                      head_start_last,

    input  logic                                      term_valid,
    output logic                                      term_ready,
    input  logic [GATE_W-1:0]                         term_gate_code,
    input  logic [LANE_ID_W-1:0]                      term_lane_id,
    input  logic [7:0]                                term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]                    term_issue_seq,
    input  logic                                      term_head_last,
    input  logic                                      event_valid,
    output logic                                      event_ready,
    input  logic [GATE_W-1:0]                         event_gate_code,
    input  logic [LANE_ID_W-1:0]                      event_lane_id,
    input  logic [EVENT_WAYS-1:0]                     event_token_valid,
    input  logic [(EVENT_WAYS*TOKEN_ID_W)-1:0]        event_token_ids,
    input  logic [WAY_COUNT_W-1:0]                    event_count,
    input  logic [ISSUE_SEQ_W-1:0]                    event_issue_seq,
    input  logic                                      event_term_first,
    input  logic                                      event_term_last,
    input  logic                                      event_head_last,
    input  logic                                      source_done_valid,
    output logic                                      source_done_ready,
    input  logic [TAG_W-1:0]                          source_done_tag,
    input  logic                                      source_done_error,

    output logic                                      head_done_valid,
    input  logic                                      head_done_ready,
    output logic [TAG_W-1:0]                          head_done_tag,
    output logic [HEAD_COUNT_W-1:0]                   head_done_index,
    output logic                                      head_done_last,
    output logic                                      head_done_error,

    output logic                                      weight_req_valid,
    input  logic                                      weight_req_ready,
    output logic [TAG_W-1:0]                          weight_req_tag,
    output logic [INPUT_CH_W-1:0]                     weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]                  weight_req_output_tile,
    input  logic                                      weight_rsp_valid,
    output logic                                      weight_rsp_ready,
    input  logic [TAG_W-1:0]                          weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]                     weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]                  weight_rsp_output_tile,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]            weight_rsp_weights,

    output logic                                      bias_req_valid,
    input  logic                                      bias_req_ready,
    output logic [TAG_W-1:0]                          bias_req_tag,
    output logic [OUTPUT_TILE_W-1:0]                  bias_req_output_tile,
    output logic [TOKEN_ID_W-1:0]                     bias_req_token_id,
    input  logic                                      bias_rsp_valid,
    output logic                                      bias_rsp_ready,
    input  logic [TAG_W-1:0]                          bias_rsp_tag,
    input  logic [TOKEN_ID_W-1:0]                     bias_rsp_token_id,
    input  logic [(OUT_TILE*ACC_W)-1:0]               bias_rsp_values,
    output logic [BANKS-1:0]                          final_valid,
    input  logic [BANKS-1:0]                          final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]             final_token_ids,
    output logic [TAG_W-1:0]                          final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0]         final_values,
    output logic                                      tile_done_valid,
    input  logic                                      tile_done_ready,
    output logic [TAG_W-1:0]                          tile_done_tag,
    output logic                                      protocol_error,
    output logic                                      accumulator_overflow,
    output logic [COUNTER_W-1:0]                      count_heads,
    output logic [COUNTER_W-1:0]                      count_terms,
    output logic [COUNTER_W-1:0]                      count_completed_terms,
    output logic [COUNTER_W-1:0]                      count_bias_commits
);
    localparam int BANK_ID_W = (BANKS <= 1) ? 1 : $clog2(BANKS);
    typedef enum logic [2:0] {
        ST_IDLE, ST_WAIT_HEAD, ST_RUN_HEAD, ST_BIAS, ST_FINISH, ST_DONE
    } state_t;
    state_t state_q;
    logic [TAG_W-1:0] tile_tag_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic [HEAD_COUNT_W-1:0] expected_heads_q;
    logic [HEAD_COUNT_W-1:0] heads_completed_q;
    logic [HEAD_COUNT_W-1:0] active_head_index_q;
    logic active_head_last_q;
    logic [TOKEN_ID_W:0] bias_token_q;
    logic bias_outstanding_q;
    logic [TAG_W-1:0] bias_expected_tag_q;
    logic [TOKEN_ID_W-1:0] bias_expected_token_q;
    logic bias_resident_valid_q;
    logic [(OUT_TILE*ACC_W)-1:0] bias_resident_values_q;
    logic sticky_protocol_q;

    logic tile_start_fire;
    logic head_start_fire;
    logic backend_start_valid;
    logic backend_start_ready;
    logic backend_done_valid;
    logic backend_done_ready;
    logic [TAG_W-1:0] backend_done_tag;
    logic backend_done_error;
    logic backend_protocol_error;
    logic backend_decoder_done_valid;
    logic [TAG_W-1:0] backend_decoder_done_tag;
    logic backend_decoder_done_error;
    logic expected_last_head;
    logic head_metadata_legal;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [OUTSTANDING_W-1:0] backend_outstanding;
    logic [COUNTER_W-1:0] backend_count_sessions;
    logic [COUNTER_W-1:0] backend_count_empty;
    logic [COUNTER_W-1:0] acc_count_updates;
    logic [COUNTER_W-1:0] acc_count_writes;
    logic [COUNTER_W-1:0] acc_count_bank_stall;
    logic [COUNTER_W-1:0] acc_count_final_stall;
    /* verilator lint_on UNUSEDSIGNAL */

    logic [BANKS-1:0] backend_update_valid;
    logic [BANKS-1:0] backend_update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] backend_update_token_ids;
    logic [TAG_W-1:0] backend_update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] backend_update_values;
    logic acc_group_start_valid;
    logic acc_group_start_ready;
    logic [BANKS-1:0] acc_update_valid;
    logic [BANKS-1:0] acc_update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [TAG_W-1:0] acc_update_tag;
    logic acc_update_is_bias;
    logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic [(OUT_TILE*ACC_W)-1:0] acc_update_bias_values;
    logic acc_group_finish_valid;
    logic acc_group_finish_ready;
    logic [TAG_W-1:0] acc_group_finish_tag;
    logic acc_protocol_error;
    logic bias_req_fire;
    logic bias_rsp_fire;
    logic bias_rsp_match;
    logic bias_commit_fire;
    logic [BANKS-1:0] bias_bank_onehot;
    logic [BANK_ID_W-1:0] bias_bank_index;

    assign tile_start_ready = state_q == ST_IDLE && acc_group_start_ready &&
                              tile_start_head_count != '0;
    assign tile_start_fire = tile_start_valid && tile_start_ready;
    assign acc_group_start_valid = state_q == ST_IDLE && tile_start_valid &&
                                   tile_start_head_count != '0;

    assign head_metadata_legal = head_start_tag == tile_tag_q &&
                                 head_start_index == heads_completed_q &&
                                 head_start_last == expected_last_head;
    assign head_start_ready = state_q == ST_WAIT_HEAD && backend_start_ready &&
                              head_metadata_legal;
    assign head_start_fire = head_start_valid && head_start_ready;
    assign backend_start_valid = state_q == ST_WAIT_HEAD && head_start_valid &&
                                 head_metadata_legal;
    assign expected_last_head = heads_completed_q ==
                                (expected_heads_q - 1'b1);

    assign backend_done_ready = state_q == ST_RUN_HEAD && head_done_ready;
    assign head_done_valid = state_q == ST_RUN_HEAD && backend_done_valid;
    assign head_done_tag = tile_tag_q;
    assign head_done_index = active_head_index_q;
    assign head_done_last = active_head_last_q;
    assign head_done_error = backend_done_error ||
                             (backend_done_tag != tile_tag_q);

    assign tile_done_valid = state_q == ST_DONE;
    assign tile_done_tag = tile_tag_q;
    assign bias_bank_index = BANK_ID_W'(32'(bias_token_q) % BANKS);
    always_comb begin
        bias_bank_onehot = '0;
        bias_bank_onehot[bias_bank_index] = 1'b1;
    end
    assign bias_req_valid = state_q == ST_BIAS && !bias_outstanding_q &&
                            (!BIAS_STATIONARY_ENABLE ||
                             !bias_resident_valid_q);
    assign bias_req_tag = tile_tag_q;
    assign bias_req_output_tile = output_tile_q;
    assign bias_req_token_id = BIAS_STATIONARY_ENABLE ? '0 :
                               bias_token_q[TOKEN_ID_W-1:0];
    assign bias_req_fire = bias_req_valid && bias_req_ready;
    assign bias_rsp_match = bias_rsp_tag == bias_expected_tag_q &&
                            bias_rsp_token_id == bias_expected_token_q;
    assign bias_rsp_ready = state_q == ST_BIAS && bias_outstanding_q &&
                            (BIAS_STATIONARY_ENABLE ||
                             acc_update_ready[bias_bank_index]);
    assign bias_rsp_fire = bias_rsp_valid && bias_rsp_ready;
    assign bias_commit_fire = BIAS_STATIONARY_ENABLE ?
                              (state_q == ST_BIAS &&
                               bias_resident_valid_q &&
                               acc_update_ready[bias_bank_index]) :
                              (bias_rsp_fire && bias_rsp_match);

    always_comb begin
        acc_update_valid = '0;
        acc_update_token_ids = '0;
        acc_update_tag = tile_tag_q;
        acc_update_is_bias = 1'b0;
        acc_update_values = '0;
        acc_update_bias_values = '0;
        backend_update_ready = '0;
        if (state_q == ST_BIAS) begin
            acc_update_valid = bias_bank_onehot & {BANKS{bias_commit_fire}};
            acc_update_token_ids[
                (32'(bias_bank_index)*TOKEN_ID_W) +: TOKEN_ID_W] =
                bias_token_q[TOKEN_ID_W-1:0];
            acc_update_is_bias = 1'b1;
            acc_update_bias_values = BIAS_STATIONARY_ENABLE ?
                                     bias_resident_values_q : bias_rsp_values;
        end else if (state_q == ST_RUN_HEAD) begin
            acc_update_valid = backend_update_valid;
            acc_update_token_ids = backend_update_token_ids;
            acc_update_tag = backend_update_tag;
            acc_update_values = backend_update_values;
            backend_update_ready = acc_update_ready;
        end
    end

    assign acc_group_finish_valid = state_q == ST_FINISH;
    assign protocol_error = sticky_protocol_q || backend_protocol_error ||
                            acc_protocol_error ||
                            (tile_start_valid && tile_start_head_count == '0) ||
                            (state_q == ST_WAIT_HEAD && head_start_valid &&
                             !head_metadata_legal);

    gatestack_tdr_multicast_backend #(
        .TOKENS(TOKENS), .LANES(LANES), .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS), .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W), .TAG_W(TAG_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .WAY_COUNT_W(WAY_COUNT_W), .OUTSTANDING_W(OUTSTANDING_W)
    ) u_backend (
        .clk_core(clk_core), .rst_core(rst_core),
        .session_start_valid(backend_start_valid),
        .session_start_ready(backend_start_ready),
        .session_tag(head_start_tag),
        .session_input_channel_base(head_start_input_channel_base),
        .session_output_tile(output_tile_q),
        .term_valid(term_valid), .term_ready(term_ready),
        .term_gate_code(term_gate_code), .term_lane_id(term_lane_id),
        .term_destination_count(term_destination_count),
        .term_issue_seq(term_issue_seq), .term_head_last(term_head_last),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_gate_code(event_gate_code), .event_lane_id(event_lane_id),
        .event_token_valid(event_token_valid),
        .event_token_ids(event_token_ids), .event_count(event_count),
        .event_issue_seq(event_issue_seq),
        .event_term_first(event_term_first),
        .event_term_last(event_term_last), .event_head_last(event_head_last),
        .source_done_valid(source_done_valid),
        .source_done_ready(source_done_ready),
        .source_done_tag(source_done_tag),
        .source_done_error(source_done_error),
        .decoder_done_valid(backend_decoder_done_valid),
        .decoder_done_ready(1'b1),
        .decoder_done_tag(backend_decoder_done_tag),
        .decoder_done_error(backend_decoder_done_error),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready), .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .update_valid(backend_update_valid),
        .update_ready(backend_update_ready),
        .update_token_ids(backend_update_token_ids),
        .update_tag(backend_update_tag),
        .update_values(backend_update_values),
        .backend_done_valid(backend_done_valid),
        .backend_done_ready(backend_done_ready),
        .backend_done_tag(backend_done_tag),
        .backend_done_error(backend_done_error),
        .protocol_error(backend_protocol_error),
        .outstanding_terms(backend_outstanding),
        .count_sessions(backend_count_sessions), .count_terms(count_terms),
        .count_completed_terms(count_completed_terms),
        .count_empty_sessions(backend_count_empty)
    );

    hitflow_banked_accumulator #(
        .TOKENS(TOKENS), .BANKS(BANKS), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W)
    ) u_accumulator (
        .clk_core(clk_core), .rst_core(rst_core),
        .flush(1'b0),
        .group_start_valid(acc_group_start_valid),
        .group_start_ready(acc_group_start_ready),
        .group_start_tag(tile_start_tag), .update_valid(acc_update_valid),
        .update_ready(acc_update_ready),
        .update_token_ids(acc_update_token_ids), .update_tag(acc_update_tag),
        .update_is_bias(acc_update_is_bias),
        .update_values(acc_update_values),
        .update_bias_values(acc_update_bias_values), .final_valid(final_valid),
        .final_ready(final_ready), .final_token_ids(final_token_ids),
        .final_tag(final_tag), .final_values(final_values),
        .group_finish_valid(acc_group_finish_valid),
        .group_finish_ready(acc_group_finish_ready),
        .group_finish_tag(acc_group_finish_tag),
        .protocol_error(acc_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_updates(acc_count_updates), .count_writes(acc_count_writes),
        .count_bias_commits(count_bias_commits),
        .count_bank_stall_cycles(acc_count_bank_stall),
        .count_final_stall_cycles(acc_count_final_stall)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tile_tag_q <= '0;
            output_tile_q <= '0;
            expected_heads_q <= '0;
            heads_completed_q <= '0;
            active_head_index_q <= '0;
            active_head_last_q <= 1'b0;
            bias_token_q <= '0;
            bias_outstanding_q <= 1'b0;
            bias_expected_tag_q <= '0;
            bias_expected_token_q <= '0;
            bias_resident_valid_q <= 1'b0;
            bias_resident_values_q <= '0;
            sticky_protocol_q <= 1'b0;
            count_heads <= '0;
        end else begin
            if (backend_protocol_error || acc_protocol_error ||
                (backend_done_valid &&
                 (backend_done_tag != tile_tag_q || backend_done_error)) ||
                (backend_decoder_done_valid &&
                 (backend_decoder_done_tag != tile_tag_q ||
                  backend_decoder_done_error)) ||
                (bias_rsp_valid &&
                 (!bias_outstanding_q || !bias_rsp_match))) begin
                sticky_protocol_q <= 1'b1;
            end

            if (bias_req_fire) begin
                bias_outstanding_q <= 1'b1;
                bias_expected_tag_q <= bias_req_tag;
                bias_expected_token_q <= bias_req_token_id;
            end else if (bias_rsp_fire) begin
                bias_outstanding_q <= 1'b0;
            end
            if (BIAS_STATIONARY_ENABLE && bias_rsp_fire && bias_rsp_match) begin
                bias_resident_valid_q <= 1'b1;
                bias_resident_values_q <= bias_rsp_values;
            end

            if (state_q == ST_IDLE) begin
                if (tile_start_fire) begin
                    tile_tag_q <= tile_start_tag;
                    output_tile_q <= tile_start_output_tile;
                    expected_heads_q <= tile_start_head_count;
                    heads_completed_q <= '0;
                    bias_token_q <= '0;
                    bias_outstanding_q <= 1'b0;
                    bias_resident_valid_q <= 1'b0;
                    state_q <= ST_WAIT_HEAD;
                end
            end else if (state_q == ST_WAIT_HEAD) begin
                if (head_start_fire) begin
                    active_head_index_q <= head_start_index;
                    active_head_last_q <= head_start_last;
                    if (head_start_tag != tile_tag_q ||
                        head_start_index != heads_completed_q ||
                        head_start_last != expected_last_head)
                        sticky_protocol_q <= 1'b1;
                    state_q <= ST_RUN_HEAD;
                end
            end else if (state_q == ST_RUN_HEAD) begin
                if (backend_done_valid && backend_done_ready) begin
                    count_heads <= count_heads + 1'b1;
                    heads_completed_q <= heads_completed_q + 1'b1;
                    if (expected_last_head) begin
                        bias_token_q <= '0;
                        state_q <= ST_BIAS;
                    end else begin
                        state_q <= ST_WAIT_HEAD;
                    end
                end
            end else if (state_q == ST_BIAS) begin
                if (bias_commit_fire) begin
                    if (bias_token_q == (TOKEN_ID_W+1)'(TOKENS - 1)) begin
                        bias_resident_valid_q <= 1'b0;
                        state_q <= ST_FINISH;
                    end else begin
                        bias_token_q <= bias_token_q + 1'b1;
                    end
                end
            end else if (state_q == ST_FINISH) begin
                if (acc_group_finish_valid && acc_group_finish_ready) begin
                    if (acc_group_finish_tag != tile_tag_q)
                        sticky_protocol_q <= 1'b1;
                    state_q <= ST_DONE;
                end
            end else if (state_q == ST_DONE) begin
                if (tile_done_valid && tile_done_ready) begin
                    sticky_protocol_q <= 1'b0;
                    bias_resident_valid_q <= 1'b0;
                    state_q <= ST_IDLE;
                end
            end else begin
                state_q <= ST_IDLE;
                sticky_protocol_q <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
