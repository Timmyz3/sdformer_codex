`timescale 1ns/1ps
`default_nettype none

// Connects event adaptation, three-way command distribution, and three
// independent 32-lane term executors without joining their physical outputs.
module gatestack_dctf96_term_datapath_top #(
    parameter int Q = 2,
    parameter int TOKENS = 162,
    parameter int EVENT_WAYS = 4,
    parameter int OUT_TILE = 32,
    parameter int GATE_W = 9,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int GROUP_TAG_W = 32,
    parameter int CMD_SEQUENCE_W = 16,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int INPUT_CHANNELS = (1 << INPUT_CH_W),
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int OUTPUT_TILE_W = 8,
    parameter int LOGICAL_SUPERTILE_W = OUTPUT_TILE_W,
    parameter int EPOCH_W = 4,
    parameter int COUNTER_W = 32,
    parameter int ADAPTER_CONTEXTS = 1,
    parameter bit PPDI_ENABLE = 1'b0,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      flush,
    input  logic                                      clear_error,

    input  logic                                      term_valid,
    output logic                                      term_ready,
    input  logic [GROUP_TAG_W-1:0]                    term_tag,
    input  logic [GATE_W-1:0]                         term_gate_code,
    input  logic [LANE_ID_W-1:0]                      term_lane_id,
    input  logic [7:0]                                term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]                    term_issue_seq,
    input  logic                                      term_head_last,
    input  logic [LOGICAL_SUPERTILE_W-1:0]            logical_supertile,
    input  logic [INPUT_CH_W-1:0]                     head_input_channel_base,

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

    output logic [2:0]                                weight_req_valid,
    input  logic [2:0]                                weight_req_ready,
    output logic [(3*GROUP_TAG_W)-1:0]                weight_req_tags,
    output logic [(3*INPUT_CH_W)-1:0]                 weight_req_input_channels,
    output logic [(3*OUTPUT_TILE_W)-1:0]              weight_req_output_tiles,
    output logic [(3*EPOCH_W)-1:0]                    weight_req_epochs,
    input  logic [2:0]                                weight_rsp_valid,
    output logic [2:0]                                weight_rsp_ready,
    input  logic [(3*GROUP_TAG_W)-1:0]                weight_rsp_tags,
    input  logic [(3*INPUT_CH_W)-1:0]                 weight_rsp_input_channels,
    input  logic [(3*OUTPUT_TILE_W)-1:0]              weight_rsp_output_tiles,
    input  logic [(3*EPOCH_W)-1:0]                    weight_rsp_epochs,
    input  logic [(3*OUT_TILE*WEIGHT_W)-1:0]          weight_rsp_weights,

    output logic [5:0]                                acc_update_valid,
    input  logic [5:0]                                acc_update_ready,
    output logic [(6*TOKEN_ID_W)-1:0]                 acc_update_token_ids,
    output logic [(3*GROUP_TAG_W)-1:0]                acc_update_tags,
    output logic [(3*OUT_TILE*PRODUCT_W)-1:0]         acc_update_values,

    output logic [2:0]                                bank_term_done,
    output logic [(3*GROUP_TAG_W)-1:0]                bank_term_done_group_tags,
    output logic [(3*ISSUE_SEQ_W)-1:0]                bank_term_done_issue_seqs,
    output logic [2:0]                                bank_term_done_head_last,
    output logic                                      head_compute_done,
    output logic [GROUP_TAG_W-1:0]                    head_compute_done_group_tag,
    output logic [ISSUE_SEQ_W-1:0]                    head_compute_done_issue_seq,

    output logic                                      dispatch_retire_valid,
    output logic [GROUP_TAG_W-1:0]                    dispatch_retire_group_tag,
    output logic [CMD_SEQUENCE_W-1:0]                 dispatch_retire_sequence,
    output logic [ISSUE_SEQ_W-1:0]                    dispatch_retire_issue_seq,
    output logic                                      dispatch_retire_term_first,
    output logic                                      dispatch_retire_term_last,
    output logic                                      dispatch_retire_head_last,

    output logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0]   fabric_occupancy,
    output logic [COUNTER_W-1:0]                      fabric_count_accepted,
    output logic [(3*COUNTER_W)-1:0]                  fabric_count_bank_consumed,
    output logic [COUNTER_W-1:0]                      fabric_count_retired,
    output logic [COUNTER_W-1:0]                      fabric_count_input_stall,
    output logic [(3*COUNTER_W)-1:0]                  fabric_count_bank_stall,
    output logic [COUNTER_W-1:0]                      fabric_max_occupancy,
    output logic [COUNTER_W-1:0]                      fabric_count_skew_cycles,
    output logic [COUNTER_W-1:0]                      issued_terms,
    output logic [(3*COUNTER_W)-1:0]                  completed_terms,
    output logic [(3*COUNTER_W)-1:0]                  count_stale_weight_responses,
    output logic                                      datapath_idle,
    output logic                                      protocol_error
);
    localparam int BANK_COUNT = 32'd3;
    localparam int BANK_LIMIT = BANK_COUNT;
    localparam int TRACK_DEPTH = Q + 32'd4;
    localparam int TRACK_LIMIT = TRACK_DEPTH;
    localparam int TRACK_PTR_W = (TRACK_DEPTH < 32'd2) ?
                                 32'd1 : $clog2(TRACK_DEPTH);

    logic adapter_cmd_valid;
    logic adapter_cmd_ready;
    logic adapter_term_valid;
    logic adapter_term_ready;
    logic adapter_event_valid;
    logic adapter_event_ready;
    logic [GROUP_TAG_W-1:0] adapter_cmd_group_tag;
    logic [CMD_SEQUENCE_W-1:0] adapter_cmd_sequence;
    logic [GATE_W-1:0] adapter_cmd_gate_code;
    logic [LANE_ID_W-1:0] adapter_cmd_lane_id;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [TOKEN_ID_W-1:0] adapter_cmd_destination_token;
    logic [1:0] adapter_cmd_destination_valid;
    logic [(2*TOKEN_ID_W)-1:0] adapter_cmd_destination_tokens;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [ISSUE_SEQ_W-1:0] adapter_cmd_term_issue_seq;
    logic adapter_cmd_term_first;
    logic adapter_cmd_term_last;
    logic adapter_cmd_head_last;
    logic [INPUT_CH_W-1:0] adapter_cmd_input_channel_base;
    logic [LOGICAL_SUPERTILE_W-1:0] adapter_cmd_logical_supertile;
    logic adapter_protocol_error;
    logic adapter_idle;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [LOGICAL_SUPERTILE_W-1:0] latched_logical_supertile_q;
    logic [INPUT_CH_W-1:0] latched_input_channel_base_q;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [INPUT_CH_W:0] term_input_channel_sum;
    logic [INPUT_CH_W:0] cmd_input_channel_sum;
    logic [INPUT_CH_W-1:0] fabric_cmd_input_channel;
    logic term_input_channel_in_range;
    logic term_supertile_in_range;
    logic term_metadata_legal;
    logic term_fire;
    logic legal_term_fire;
    logic illegal_term_fire;
    logic illegal_event_fire;
    logic illegal_drop_active_q;
    logic fabric_cmd_valid;
    logic fabric_cmd_ready;
    logic fabric_cmd_fire;

    logic [2:0] fabric_bank_valid;
    logic [2:0] fabric_bank_ready;
    logic [(3*GROUP_TAG_W)-1:0] fabric_bank_group_tags;
    logic [(3*CMD_SEQUENCE_W)-1:0] fabric_bank_sequences;
    logic [(3*ISSUE_SEQ_W)-1:0] fabric_bank_term_issue_seqs;
    logic [2:0] fabric_bank_term_first;
    logic [2:0] fabric_bank_term_last;
    logic [2:0] fabric_bank_head_last;
    logic [(3*INPUT_CH_W)-1:0] fabric_bank_input_channels;
    logic [(3*LOGICAL_SUPERTILE_W)-1:0] fabric_bank_logical_supertiles;
    logic [(3*GATE_W)-1:0] fabric_bank_gate_codes;
    logic [(3*LANE_ID_W)-1:0] fabric_bank_lane_ids;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [(3*TOKEN_ID_W)-1:0] fabric_bank_destination_tokens;
    logic [5:0] fabric_bank_ppdi_destination_valid;
    logic [(6*TOKEN_ID_W)-1:0] fabric_bank_ppdi_destination_tokens;
    /* verilator lint_on UNUSEDSIGNAL */

    logic [2:0] executor_protocol_error;
    logic [COUNTER_W-1:0] executor_completed_q [0:BANK_COUNT-1];
    logic [COUNTER_W-1:0] executor_stale_count [0:BANK_COUNT-1];

    logic [TRACK_DEPTH-1:0] track_valid_q;
    logic [2:0] track_done_mask_q [0:TRACK_DEPTH-1];
    logic [GROUP_TAG_W-1:0] track_tag_q [0:TRACK_DEPTH-1];
    logic [ISSUE_SEQ_W-1:0] track_issue_seq_q [0:TRACK_DEPTH-1];
    logic track_head_last_q [0:TRACK_DEPTH-1];
    logic [TRACK_PTR_W-1:0] track_head_q;
    logic [TRACK_PTR_W-1:0] track_tail_q;
    logic [TRACK_PTR_W-1:0] bank_done_ptr_q [0:BANK_COUNT-1];
    logic [TRACK_PTR_W:0] track_count_q;
    logic [2:0] bank_done_match;
    logic [2:0] head_done_bits;
    logic [2:0] head_done_mask_after;
    logic track_full;
    logic track_pop;
    logic issue_term_fire;
    logic tracking_protocol_error;
    logic protocol_error_q;

    assign term_metadata_legal = term_input_channel_in_range &&
                                 term_supertile_in_range;
    assign adapter_term_valid = term_valid && term_metadata_legal &&
                                !illegal_drop_active_q && !flush;
    assign term_ready = adapter_term_ready && !illegal_drop_active_q &&
                        !flush;
    assign term_fire = term_valid && term_ready;
    assign legal_term_fire = term_fire && term_metadata_legal;
    assign illegal_term_fire = term_fire && !term_metadata_legal;
    assign adapter_event_valid = event_valid && !illegal_drop_active_q;
    assign event_ready = illegal_drop_active_q ? !flush : adapter_event_ready;
    assign illegal_event_fire = illegal_drop_active_q && event_valid &&
                                event_ready;
    assign term_input_channel_sum =
        (INPUT_CH_W+1)'(head_input_channel_base) +
        (INPUT_CH_W+1)'(term_lane_id);
    assign cmd_input_channel_sum =
        (INPUT_CH_W+1)'(adapter_cmd_input_channel_base) +
        (INPUT_CH_W+1)'(adapter_cmd_lane_id);
    assign fabric_cmd_input_channel = cmd_input_channel_sum[INPUT_CH_W-1:0];
    assign term_input_channel_in_range =
        !term_input_channel_sum[INPUT_CH_W] &&
        (32'(term_input_channel_sum) < INPUT_CHANNELS);
    assign term_supertile_in_range =
        ((32'(logical_supertile) * BANK_COUNT) + (BANK_COUNT - 32'd1)) <
        (32'd1 << OUTPUT_TILE_W);

    assign track_full = track_count_q == (TRACK_PTR_W+1)'(TRACK_DEPTH);
    assign fabric_cmd_valid = adapter_cmd_valid &&
                              (!adapter_cmd_term_last || !track_full);
    assign adapter_cmd_ready = fabric_cmd_ready &&
                               (!adapter_cmd_term_last || !track_full);
    assign fabric_cmd_fire = fabric_cmd_valid && fabric_cmd_ready;
    assign issue_term_fire = fabric_cmd_fire && adapter_cmd_term_last;
    assign datapath_idle = adapter_idle && !illegal_drop_active_q &&
                           (fabric_occupancy == '0) &&
                           (track_count_q == '0);

    generate
        if (PPDI_ENABLE) begin : g_ppdi_adapter_2c
            gatestack_ppdi_term_event_adapter_2c #(
                .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS),
                .TAG_W(GROUP_TAG_W), .GATE_CODE_W(GATE_W),
                .LANE_ID_W(LANE_ID_W), .INPUT_CH_W(INPUT_CH_W),
                .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                .TOKEN_ID_W(TOKEN_ID_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
                .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
                .WAY_COUNT_W(WAY_COUNT_W)
            ) u_ppdi_term_event_adapter_2c (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .clear_error(clear_error), .term_valid(adapter_term_valid),
                .term_ready(adapter_term_ready), .term_tag(term_tag),
                .term_gate_code(term_gate_code), .term_lane_id(term_lane_id),
                .term_destination_count(term_destination_count),
                .term_issue_seq(term_issue_seq),
                .term_head_last(term_head_last),
                .term_input_channel_base(head_input_channel_base),
                .term_logical_supertile(logical_supertile),
                .event_valid(adapter_event_valid),
                .event_ready(adapter_event_ready),
                .event_gate_code(event_gate_code),
                .event_lane_id(event_lane_id),
                .event_token_valid(event_token_valid),
                .event_token_ids(event_token_ids), .event_count(event_count),
                .event_issue_seq(event_issue_seq),
                .event_term_first(event_term_first),
                .event_term_last(event_term_last),
                .event_head_last(event_head_last),
                .cmd_valid(adapter_cmd_valid), .cmd_ready(adapter_cmd_ready),
                .cmd_group_tag(adapter_cmd_group_tag),
                .cmd_sequence(adapter_cmd_sequence),
                .cmd_gate_code(adapter_cmd_gate_code),
                .cmd_lane_id(adapter_cmd_lane_id),
                .cmd_destination_valid(adapter_cmd_destination_valid),
                .cmd_destination_tokens(adapter_cmd_destination_tokens),
                .cmd_term_issue_seq(adapter_cmd_term_issue_seq),
                .cmd_term_first(adapter_cmd_term_first),
                .cmd_term_last(adapter_cmd_term_last),
                .cmd_head_last(adapter_cmd_head_last),
                .cmd_input_channel_base(adapter_cmd_input_channel_base),
                .cmd_logical_supertile(adapter_cmd_logical_supertile),
                .idle(adapter_idle),
                .protocol_error(adapter_protocol_error)
            );
            assign adapter_cmd_destination_token =
                adapter_cmd_destination_tokens[0 +: TOKEN_ID_W];
        end else if (ADAPTER_CONTEXTS == 2) begin : g_adapter_2c
            gatestack_dctf_term_event_adapter_2c #(
                .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS),
                .TAG_W(GROUP_TAG_W), .GATE_CODE_W(GATE_W),
                .LANE_ID_W(LANE_ID_W), .INPUT_CH_W(INPUT_CH_W),
                .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                .TOKEN_ID_W(TOKEN_ID_W),
                .ISSUE_SEQ_W(ISSUE_SEQ_W),
                .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
                .WAY_COUNT_W(WAY_COUNT_W)
            ) u_term_event_adapter_2c (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .clear_error(clear_error), .term_valid(adapter_term_valid),
                .term_ready(adapter_term_ready), .term_tag(term_tag),
                .term_gate_code(term_gate_code), .term_lane_id(term_lane_id),
                .term_destination_count(term_destination_count),
                .term_issue_seq(term_issue_seq),
                .term_head_last(term_head_last),
                .term_input_channel_base(head_input_channel_base),
                .term_logical_supertile(logical_supertile),
                .event_valid(adapter_event_valid),
                .event_ready(adapter_event_ready),
                .event_gate_code(event_gate_code),
                .event_lane_id(event_lane_id),
                .event_token_valid(event_token_valid),
                .event_token_ids(event_token_ids), .event_count(event_count),
                .event_issue_seq(event_issue_seq),
                .event_term_first(event_term_first),
                .event_term_last(event_term_last),
                .event_head_last(event_head_last),
                .cmd_valid(adapter_cmd_valid), .cmd_ready(adapter_cmd_ready),
                .cmd_group_tag(adapter_cmd_group_tag),
                .cmd_sequence(adapter_cmd_sequence),
                .cmd_gate_code(adapter_cmd_gate_code),
                .cmd_lane_id(adapter_cmd_lane_id),
                .cmd_destination_token(adapter_cmd_destination_token),
                .cmd_term_issue_seq(adapter_cmd_term_issue_seq),
                .cmd_term_first(adapter_cmd_term_first),
                .cmd_term_last(adapter_cmd_term_last),
                .cmd_head_last(adapter_cmd_head_last),
                .cmd_input_channel_base(adapter_cmd_input_channel_base),
                .cmd_logical_supertile(adapter_cmd_logical_supertile),
                .idle(adapter_idle),
                .protocol_error(adapter_protocol_error)
            );
            assign adapter_cmd_destination_valid = 2'b01;
            assign adapter_cmd_destination_tokens = {
                adapter_cmd_destination_token,
                adapter_cmd_destination_token
            };
        end else begin : g_adapter_1c
            gatestack_dctf_term_event_adapter #(
                .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS),
                .TAG_W(GROUP_TAG_W), .GATE_CODE_W(GATE_W),
                .LANE_ID_W(LANE_ID_W), .TOKEN_ID_W(TOKEN_ID_W),
                .ISSUE_SEQ_W(ISSUE_SEQ_W),
                .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
                .WAY_COUNT_W(WAY_COUNT_W)
            ) u_term_event_adapter (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .clear_error(clear_error), .term_valid(adapter_term_valid),
                .term_ready(adapter_term_ready), .term_tag(term_tag),
                .term_gate_code(term_gate_code), .term_lane_id(term_lane_id),
                .term_destination_count(term_destination_count),
                .term_issue_seq(term_issue_seq),
                .term_head_last(term_head_last),
                .event_valid(adapter_event_valid),
                .event_ready(adapter_event_ready),
                .event_gate_code(event_gate_code),
                .event_lane_id(event_lane_id),
                .event_token_valid(event_token_valid),
                .event_token_ids(event_token_ids), .event_count(event_count),
                .event_issue_seq(event_issue_seq),
                .event_term_first(event_term_first),
                .event_term_last(event_term_last),
                .event_head_last(event_head_last),
                .cmd_valid(adapter_cmd_valid), .cmd_ready(adapter_cmd_ready),
                .cmd_group_tag(adapter_cmd_group_tag),
                .cmd_sequence(adapter_cmd_sequence),
                .cmd_gate_code(adapter_cmd_gate_code),
                .cmd_lane_id(adapter_cmd_lane_id),
                .cmd_destination_token(adapter_cmd_destination_token),
                .cmd_term_issue_seq(adapter_cmd_term_issue_seq),
                .cmd_term_first(adapter_cmd_term_first),
                .cmd_term_last(adapter_cmd_term_last),
                .cmd_head_last(adapter_cmd_head_last),
                .protocol_error(adapter_protocol_error)
            );
            assign adapter_idle = adapter_term_ready;
            assign adapter_cmd_input_channel_base =
                latched_input_channel_base_q;
            assign adapter_cmd_logical_supertile =
                latched_logical_supertile_q;
            assign adapter_cmd_destination_valid = 2'b01;
            assign adapter_cmd_destination_tokens = {
                adapter_cmd_destination_token,
                adapter_cmd_destination_token
            };
        end
    endgenerate

    generate
        if (PPDI_ENABLE) begin : g_ppdi_fabric
            gatestack_ppdi_dctf_term_fabric #(
                .Q(Q), .GROUP_TAG_W(GROUP_TAG_W),
                .SEQUENCE_W(CMD_SEQUENCE_W),
                .TERM_ISSUE_SEQ_W(ISSUE_SEQ_W),
                .INPUT_CH_W(INPUT_CH_W),
                .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                .GATE_CODE_W(GATE_W), .LANE_ID_W(LANE_ID_W),
                .DEST_TOKEN_W(TOKEN_ID_W), .COUNTER_W(COUNTER_W)
            ) u_ppdi_term_fabric (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .cmd_valid(fabric_cmd_valid), .cmd_ready(fabric_cmd_ready),
                .cmd_group_tag(adapter_cmd_group_tag),
                .cmd_sequence(adapter_cmd_sequence),
                .cmd_term_issue_seq(adapter_cmd_term_issue_seq),
                .cmd_term_first(adapter_cmd_term_first),
                .cmd_term_last(adapter_cmd_term_last),
                .cmd_head_last(adapter_cmd_head_last),
                .cmd_input_channel(fabric_cmd_input_channel),
                .cmd_logical_supertile(adapter_cmd_logical_supertile),
                .cmd_gate_code(adapter_cmd_gate_code),
                .cmd_lane_id(adapter_cmd_lane_id),
                .cmd_destination_valid(adapter_cmd_destination_valid),
                .cmd_destination_tokens(adapter_cmd_destination_tokens),
                .bank_valid(fabric_bank_valid),
                .bank_ready(fabric_bank_ready),
                .bank_group_tags(fabric_bank_group_tags),
                .bank_sequences(fabric_bank_sequences),
                .bank_term_issue_seqs(fabric_bank_term_issue_seqs),
                .bank_term_first(fabric_bank_term_first),
                .bank_term_last(fabric_bank_term_last),
                .bank_head_last(fabric_bank_head_last),
                .bank_input_channels(fabric_bank_input_channels),
                .bank_logical_supertiles(fabric_bank_logical_supertiles),
                .bank_gate_codes(fabric_bank_gate_codes),
                .bank_lane_ids(fabric_bank_lane_ids),
                .bank_destination_valid(fabric_bank_ppdi_destination_valid),
                .bank_destination_tokens(
                    fabric_bank_ppdi_destination_tokens),
                .retire_valid(dispatch_retire_valid),
                .retire_group_tag(dispatch_retire_group_tag),
                .retire_sequence(dispatch_retire_sequence),
                .retire_term_issue_seq(dispatch_retire_issue_seq),
                .retire_term_first(dispatch_retire_term_first),
                .retire_term_last(dispatch_retire_term_last),
                .retire_head_last(dispatch_retire_head_last),
                .occupancy(fabric_occupancy),
                .count_accepted(fabric_count_accepted),
                .count_bank_consumed(fabric_count_bank_consumed),
                .count_retired(fabric_count_retired),
                .count_input_stall(fabric_count_input_stall),
                .count_bank_stall(fabric_count_bank_stall),
                .max_occupancy(fabric_max_occupancy),
                .count_skew_cycles(fabric_count_skew_cycles)
            );
            for (genvar bank = 32'd0; bank < BANK_COUNT;
                 bank = bank + 32'd1) begin : g_ppdi_scalar_alias
                assign fabric_bank_destination_tokens[
                    (bank*TOKEN_ID_W) +: TOKEN_ID_W] =
                    fabric_bank_ppdi_destination_tokens[
                        (bank*2*TOKEN_ID_W) +: TOKEN_ID_W];
            end
        end else begin : g_scalar_fabric
            gatestack_dctf_term_fabric #(
                .Q(Q), .GROUP_TAG_W(GROUP_TAG_W),
                .SEQUENCE_W(CMD_SEQUENCE_W),
                .TERM_ISSUE_SEQ_W(ISSUE_SEQ_W),
                .INPUT_CH_W(INPUT_CH_W),
                .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                .GATE_CODE_W(GATE_W), .LANE_ID_W(LANE_ID_W),
                .DEST_TOKEN_W(TOKEN_ID_W), .COUNTER_W(COUNTER_W)
            ) u_term_fabric (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .cmd_valid(fabric_cmd_valid), .cmd_ready(fabric_cmd_ready),
                .cmd_group_tag(adapter_cmd_group_tag),
                .cmd_sequence(adapter_cmd_sequence),
                .cmd_term_issue_seq(adapter_cmd_term_issue_seq),
                .cmd_term_first(adapter_cmd_term_first),
                .cmd_term_last(adapter_cmd_term_last),
                .cmd_head_last(adapter_cmd_head_last),
                .cmd_input_channel(fabric_cmd_input_channel),
                .cmd_logical_supertile(adapter_cmd_logical_supertile),
                .cmd_gate_code(adapter_cmd_gate_code),
                .cmd_lane_id(adapter_cmd_lane_id),
                .cmd_destination_token(adapter_cmd_destination_token),
                .bank_valid(fabric_bank_valid),
                .bank_ready(fabric_bank_ready),
                .bank_group_tags(fabric_bank_group_tags),
                .bank_sequences(fabric_bank_sequences),
                .bank_term_issue_seqs(fabric_bank_term_issue_seqs),
                .bank_term_first(fabric_bank_term_first),
                .bank_term_last(fabric_bank_term_last),
                .bank_head_last(fabric_bank_head_last),
                .bank_input_channels(fabric_bank_input_channels),
                .bank_logical_supertiles(fabric_bank_logical_supertiles),
                .bank_gate_codes(fabric_bank_gate_codes),
                .bank_lane_ids(fabric_bank_lane_ids),
                .bank_destination_tokens(fabric_bank_destination_tokens),
                .retire_valid(dispatch_retire_valid),
                .retire_group_tag(dispatch_retire_group_tag),
                .retire_sequence(dispatch_retire_sequence),
                .retire_term_issue_seq(dispatch_retire_issue_seq),
                .retire_term_first(dispatch_retire_term_first),
                .retire_term_last(dispatch_retire_term_last),
                .retire_head_last(dispatch_retire_head_last),
                .occupancy(fabric_occupancy),
                .count_accepted(fabric_count_accepted),
                .count_bank_consumed(fabric_count_bank_consumed),
                .count_retired(fabric_count_retired),
                .count_input_stall(fabric_count_input_stall),
                .count_bank_stall(fabric_count_bank_stall),
                .max_occupancy(fabric_max_occupancy),
                .count_skew_cycles(fabric_count_skew_cycles)
            );
            for (genvar bank = 32'd0; bank < BANK_COUNT;
                 bank = bank + 32'd1) begin : g_scalar_ppdi_alias
                assign fabric_bank_ppdi_destination_valid[
                    (bank*2) +: 2] = 2'b01;
                assign fabric_bank_ppdi_destination_tokens[
                    (bank*2*TOKEN_ID_W) +: (2*TOKEN_ID_W)] = {
                    fabric_bank_destination_tokens[
                        (bank*TOKEN_ID_W) +: TOKEN_ID_W],
                    fabric_bank_destination_tokens[
                        (bank*TOKEN_ID_W) +: TOKEN_ID_W]
                };
            end
        end
    endgenerate

    generate
        for (genvar bank = 32'd0; bank < BANK_LIMIT;
             bank = bank + 32'd1) begin : g_executor
            if (PPDI_ENABLE) begin : g_ppdi_executor
                gatestack_ppdi_dctf32_bank_executor #(
                    .BANK_ID(bank), .BANK_COUNT(BANK_COUNT),
                    .TOKENS(TOKENS), .OUT_TILE(OUT_TILE), .GATE_W(GATE_W),
                    .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W),
                    .GROUP_TAG_W(GROUP_TAG_W),
                    .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
                    .ISSUE_SEQ_W(ISSUE_SEQ_W), .INPUT_CH_W(INPUT_CH_W),
                    .LANE_ID_W(LANE_ID_W), .TOKEN_ID_W(TOKEN_ID_W),
                    .OUTPUT_TILE_W(OUTPUT_TILE_W),
                    .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                    .EPOCH_W(EPOCH_W), .COUNTER_W(COUNTER_W)
                ) u_ppdi_executor (
                    .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                    .clear_error(clear_error),
                    .cmd_valid(fabric_bank_valid[bank]),
                    .cmd_ready(fabric_bank_ready[bank]),
                    .cmd_group_tag(fabric_bank_group_tags[
                        (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                    .cmd_sequence(fabric_bank_sequences[
                        (bank*CMD_SEQUENCE_W) +: CMD_SEQUENCE_W]),
                    .cmd_term_issue_seq(fabric_bank_term_issue_seqs[
                        (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
                    .cmd_term_first(fabric_bank_term_first[bank]),
                    .cmd_term_last(fabric_bank_term_last[bank]),
                    .cmd_head_last(fabric_bank_head_last[bank]),
                    .cmd_input_channel(fabric_bank_input_channels[
                        (bank*INPUT_CH_W) +: INPUT_CH_W]),
                    .cmd_gate_code(fabric_bank_gate_codes[
                        (bank*GATE_W) +: GATE_W]),
                    .cmd_lane_id(fabric_bank_lane_ids[
                        (bank*LANE_ID_W) +: LANE_ID_W]),
                    .cmd_destination_valid(
                        fabric_bank_ppdi_destination_valid[(bank*2) +: 2]),
                    .cmd_destination_tokens(
                        fabric_bank_ppdi_destination_tokens[
                            (bank*2*TOKEN_ID_W) +: (2*TOKEN_ID_W)]),
                    .logical_supertile(fabric_bank_logical_supertiles[
                        (bank*LOGICAL_SUPERTILE_W) +:
                        LOGICAL_SUPERTILE_W]),
                    .weight_req_valid(weight_req_valid[bank]),
                    .weight_req_ready(weight_req_ready[bank]),
                    .weight_req_tag(weight_req_tags[
                        (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                    .weight_req_input_channel(weight_req_input_channels[
                        (bank*INPUT_CH_W) +: INPUT_CH_W]),
                    .weight_req_output_tile(weight_req_output_tiles[
                        (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
                    .weight_req_epoch(weight_req_epochs[
                        (bank*EPOCH_W) +: EPOCH_W]),
                    .weight_rsp_valid(weight_rsp_valid[bank]),
                    .weight_rsp_ready(weight_rsp_ready[bank]),
                    .weight_rsp_tag(weight_rsp_tags[
                        (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                    .weight_rsp_input_channel(weight_rsp_input_channels[
                        (bank*INPUT_CH_W) +: INPUT_CH_W]),
                    .weight_rsp_output_tile(weight_rsp_output_tiles[
                        (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
                    .weight_rsp_epoch(weight_rsp_epochs[
                        (bank*EPOCH_W) +: EPOCH_W]),
                    .weight_rsp_weights(weight_rsp_weights[
                        (bank*OUT_TILE*WEIGHT_W) +:
                        (OUT_TILE*WEIGHT_W)]),
                    .acc_update_valid(acc_update_valid[(bank*2) +: 2]),
                    .acc_update_ready(acc_update_ready[(bank*2) +: 2]),
                    .acc_update_token_ids(acc_update_token_ids[
                        (bank*2*TOKEN_ID_W) +: (2*TOKEN_ID_W)]),
                    .acc_update_tag(acc_update_tags[
                        (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                    .acc_update_values(acc_update_values[
                        (bank*OUT_TILE*PRODUCT_W) +:
                        (OUT_TILE*PRODUCT_W)]),
                    .term_done(bank_term_done[bank]),
                    .term_done_group_tag(bank_term_done_group_tags[
                        (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                    .term_done_issue_seq(bank_term_done_issue_seqs[
                        (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
                    .term_done_head_last(bank_term_done_head_last[bank]),
                    .protocol_error(executor_protocol_error[bank]),
                    .count_stale_weight_responses(
                        executor_stale_count[bank])
                );
            end else begin : g_scalar_executor
            gatestack_dctf32_bank_executor #(
                .BANK_ID(bank),
                .BANK_COUNT(BANK_COUNT),
                .TOKENS(TOKENS),
                .OUT_TILE(OUT_TILE),
                .GATE_W(GATE_W),
                .WEIGHT_W(WEIGHT_W),
                .PRODUCT_W(PRODUCT_W),
                .GROUP_TAG_W(GROUP_TAG_W),
                .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
                .ISSUE_SEQ_W(ISSUE_SEQ_W),
                .INPUT_CH_W(INPUT_CH_W),
                .LANE_ID_W(LANE_ID_W),
                .TOKEN_ID_W(TOKEN_ID_W),
                .OUTPUT_TILE_W(OUTPUT_TILE_W),
                .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
                .EPOCH_W(EPOCH_W),
                .COUNTER_W(COUNTER_W)
            ) u_executor (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .flush(flush),
                .clear_error(clear_error),
                .cmd_valid(fabric_bank_valid[bank]),
                .cmd_ready(fabric_bank_ready[bank]),
                .cmd_group_tag(fabric_bank_group_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                .cmd_sequence(fabric_bank_sequences[
                    (bank*CMD_SEQUENCE_W) +: CMD_SEQUENCE_W]),
                .cmd_term_issue_seq(fabric_bank_term_issue_seqs[
                    (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
                .cmd_term_first(fabric_bank_term_first[bank]),
                .cmd_term_last(fabric_bank_term_last[bank]),
                .cmd_head_last(fabric_bank_head_last[bank]),
                .cmd_input_channel(fabric_bank_input_channels[
                    (bank*INPUT_CH_W) +: INPUT_CH_W]),
                .cmd_gate_code(fabric_bank_gate_codes[
                    (bank*GATE_W) +: GATE_W]),
                .cmd_lane_id(fabric_bank_lane_ids[
                    (bank*LANE_ID_W) +: LANE_ID_W]),
                .cmd_destination_token(fabric_bank_destination_tokens[
                    (bank*TOKEN_ID_W) +: TOKEN_ID_W]),
                .logical_supertile(fabric_bank_logical_supertiles[
                    (bank*LOGICAL_SUPERTILE_W) +:
                    LOGICAL_SUPERTILE_W]),
                .weight_req_valid(weight_req_valid[bank]),
                .weight_req_ready(weight_req_ready[bank]),
                .weight_req_tag(weight_req_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                .weight_req_input_channel(weight_req_input_channels[
                    (bank*INPUT_CH_W) +: INPUT_CH_W]),
                .weight_req_output_tile(weight_req_output_tiles[
                    (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
                .weight_req_epoch(weight_req_epochs[
                    (bank*EPOCH_W) +: EPOCH_W]),
                .weight_rsp_valid(weight_rsp_valid[bank]),
                .weight_rsp_ready(weight_rsp_ready[bank]),
                .weight_rsp_tag(weight_rsp_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                .weight_rsp_input_channel(weight_rsp_input_channels[
                    (bank*INPUT_CH_W) +: INPUT_CH_W]),
                .weight_rsp_output_tile(weight_rsp_output_tiles[
                    (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W]),
                .weight_rsp_epoch(weight_rsp_epochs[
                    (bank*EPOCH_W) +: EPOCH_W]),
                .weight_rsp_weights(weight_rsp_weights[
                    (bank*OUT_TILE*WEIGHT_W) +: (OUT_TILE*WEIGHT_W)]),
                .acc_update_valid(acc_update_valid[(bank*2) +: 2]),
                .acc_update_ready(acc_update_ready[(bank*2) +: 2]),
                .acc_update_token_ids(acc_update_token_ids[
                    (bank*2*TOKEN_ID_W) +: (2*TOKEN_ID_W)]),
                .acc_update_tag(acc_update_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                .acc_update_values(acc_update_values[
                    (bank*OUT_TILE*PRODUCT_W) +: (OUT_TILE*PRODUCT_W)]),
                .term_done(bank_term_done[bank]),
                .term_done_group_tag(bank_term_done_group_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W]),
                .term_done_issue_seq(bank_term_done_issue_seqs[
                    (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W]),
                .term_done_head_last(bank_term_done_head_last[bank]),
                .protocol_error(executor_protocol_error[bank]),
                .count_stale_weight_responses(executor_stale_count[bank])
            );
            end

            assign completed_terms[(bank*COUNTER_W) +: COUNTER_W] =
                executor_completed_q[bank];
            assign count_stale_weight_responses[
                (bank*COUNTER_W) +: COUNTER_W] = executor_stale_count[bank];
        end
    endgenerate

    always_comb begin
        bank_done_match = '0;
        head_done_bits = '0;
        for (int bank = 32'd0; bank < BANK_LIMIT;
             bank = bank + 32'd1) begin
            bank_done_match[bank] = track_valid_q[bank_done_ptr_q[bank]] &&
                (bank_term_done_group_tags[
                    (bank*GROUP_TAG_W) +: GROUP_TAG_W] ==
                 track_tag_q[bank_done_ptr_q[bank]]) &&
                (bank_term_done_issue_seqs[
                    (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W] ==
                 track_issue_seq_q[bank_done_ptr_q[bank]]) &&
                (bank_term_done_head_last[bank] ==
                 (bank_term_done[bank] &&
                  track_head_last_q[bank_done_ptr_q[bank]])) &&
                !track_done_mask_q[bank_done_ptr_q[bank]][bank];
            head_done_bits[bank] = bank_term_done[bank] &&
                bank_done_match[bank] &&
                (bank_done_ptr_q[bank] == track_head_q);
        end
        head_done_mask_after = track_done_mask_q[track_head_q] |
                               head_done_bits;
    end

    assign track_pop = !flush && track_valid_q[track_head_q] &&
                       (&head_done_mask_after);
    assign head_compute_done = track_pop &&
                               track_head_last_q[track_head_q];
    assign head_compute_done_group_tag = track_tag_q[track_head_q];
    assign head_compute_done_issue_seq = track_issue_seq_q[track_head_q];
    assign tracking_protocol_error =
        |(bank_term_done & ~bank_done_match);
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            latched_logical_supertile_q <= '0;
            latched_input_channel_base_q <= '0;
            track_valid_q <= '0;
            track_head_q <= '0;
            track_tail_q <= '0;
            track_count_q <= '0;
            issued_terms <= '0;
            protocol_error_q <= 1'b0;
            illegal_drop_active_q <= 1'b0;
            for (int entry = 32'd0; entry < TRACK_LIMIT;
                 entry = entry + 32'd1) begin
                track_done_mask_q[entry] <= '0;
                track_tag_q[entry] <= '0;
                track_issue_seq_q[entry] <= '0;
                track_head_last_q[entry] <= 1'b0;
            end
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                bank_done_ptr_q[bank] <= '0;
                executor_completed_q[bank] <= '0;
            end
        end else begin
            if (clear_error)
                protocol_error_q <= 1'b0;
            // A newly accepted illegal term wins over a concurrent clear.
            if (illegal_term_fire)
                protocol_error_q <= 1'b1;
            if (legal_term_fire) begin
                latched_logical_supertile_q <= logical_supertile;
                latched_input_channel_base_q <= head_input_channel_base;
            end
            if (illegal_term_fire && (term_destination_count != 0))
                illegal_drop_active_q <= 1'b1;
            if (illegal_event_fire && event_term_last)
                illegal_drop_active_q <= 1'b0;
            if (!clear_error &&
                (adapter_protocol_error || (|executor_protocol_error) ||
                 tracking_protocol_error ||
                 (fabric_cmd_fire && cmd_input_channel_sum[INPUT_CH_W])))
                protocol_error_q <= 1'b1;

            if (flush) begin
                illegal_drop_active_q <= 1'b0;
                track_valid_q <= '0;
                track_head_q <= '0;
                track_tail_q <= '0;
                track_count_q <= '0;
                for (int entry = 32'd0; entry < TRACK_LIMIT;
                     entry = entry + 32'd1)
                    track_done_mask_q[entry] <= '0;
                for (int bank = 32'd0; bank < BANK_LIMIT;
                     bank = bank + 32'd1)
                    bank_done_ptr_q[bank] <= '0;
            end else begin
                for (int bank = 32'd0; bank < BANK_LIMIT;
                     bank = bank + 32'd1) begin
                    if (bank_term_done[bank]) begin
                        executor_completed_q[bank] <=
                            executor_completed_q[bank] + 1'b1;
                        if (bank_done_match[bank]) begin
                            track_done_mask_q[bank_done_ptr_q[bank]][bank] <=
                                1'b1;
                            if (bank_done_ptr_q[bank] ==
                                TRACK_PTR_W'(TRACK_DEPTH - 1))
                                bank_done_ptr_q[bank] <= '0;
                            else
                                bank_done_ptr_q[bank] <=
                                    bank_done_ptr_q[bank] + 1'b1;
                        end
                    end
                end

                if (track_pop) begin
                    track_valid_q[track_head_q] <= 1'b0;
                    track_done_mask_q[track_head_q] <= '0;
                    if (track_head_q == TRACK_PTR_W'(TRACK_DEPTH - 1))
                        track_head_q <= '0;
                    else
                        track_head_q <= track_head_q + 1'b1;
                end

                if (issue_term_fire) begin
                    track_valid_q[track_tail_q] <= 1'b1;
                    track_done_mask_q[track_tail_q] <= '0;
                    track_tag_q[track_tail_q] <= adapter_cmd_group_tag;
                    track_issue_seq_q[track_tail_q] <=
                        adapter_cmd_term_issue_seq;
                    track_head_last_q[track_tail_q] <= adapter_cmd_head_last;
                    if (track_tail_q == TRACK_PTR_W'(TRACK_DEPTH - 1))
                        track_tail_q <= '0;
                    else
                        track_tail_q <= track_tail_q + 1'b1;
                    issued_terms <= issued_terms + 1'b1;
                end

                case ({issue_term_fire, track_pop})
                    2'b10: track_count_q <= track_count_q + 1'b1;
                    2'b01: track_count_q <= track_count_q - 1'b1;
                    default: track_count_q <= track_count_q;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
