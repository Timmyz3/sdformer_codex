`timescale 1ns/1ps
`default_nettype none

// Full 96-lane projection datapath with three physical 32-lane banks. Term
// products, bias traffic, accumulation, and final emission remain bank-local.
module gatestack_dctf96_banklocal_projection_top #(
    parameter int Q = 2,
    parameter int TOKENS = 162,
    parameter int EVENT_WAYS = 4,
    parameter int OUT_TILE = 32,
    parameter int GATE_W = 9,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int CMD_SEQUENCE_W = 16,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int INPUT_CHANNELS = (1 << INPUT_CH_W),
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int OUTPUT_TILE_W = 8,
    parameter int LOGICAL_SUPERTILE_W = OUTPUT_TILE_W,
    parameter int HEAD_COUNT_W = 6,
    parameter int EPOCH_W = 4,
    parameter int COUNTER_W = 32,
    parameter int ADAPTER_CONTEXTS = 1,
    parameter bit PPDI_ENABLE = 1'b0,
    parameter bit IMPLICIT_BIAS_FINALIZE_ENABLE = 1'b0,
    parameter int WAY_COUNT_W = $clog2(EVENT_WAYS + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      flush,

    input  logic                                      tile_start_valid,
    output logic                                      tile_start_ready,
    input  logic [TAG_W-1:0]                          tile_start_tag,
    input  logic [LOGICAL_SUPERTILE_W-1:0]            tile_start_logical_supertile,
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

    output logic [2:0]                                weight_req_valid,
    input  logic [2:0]                                weight_req_ready,
    output logic [(3*TAG_W)-1:0]                      weight_req_tags,
    output logic [(3*INPUT_CH_W)-1:0]                 weight_req_input_channels,
    output logic [(3*OUTPUT_TILE_W)-1:0]              weight_req_output_tiles,
    output logic [(3*EPOCH_W)-1:0]                    weight_req_epochs,
    input  logic [2:0]                                weight_rsp_valid,
    output logic [2:0]                                weight_rsp_ready,
    input  logic [(3*TAG_W)-1:0]                      weight_rsp_tags,
    input  logic [(3*INPUT_CH_W)-1:0]                 weight_rsp_input_channels,
    input  logic [(3*OUTPUT_TILE_W)-1:0]              weight_rsp_output_tiles,
    input  logic [(3*EPOCH_W)-1:0]                    weight_rsp_epochs,
    input  logic [(3*OUT_TILE*WEIGHT_W)-1:0]          weight_rsp_weights,

    output logic [2:0]                                bias_req_valid,
    input  logic [2:0]                                bias_req_ready,
    output logic [(3*TAG_W)-1:0]                      bias_req_tags,
    output logic [(3*OUTPUT_TILE_W)-1:0]              bias_req_output_tiles,
    output logic [(3*TOKEN_ID_W)-1:0]                 bias_req_token_ids,
    output logic [(3*EPOCH_W)-1:0]                    bias_req_epochs,
    input  logic [2:0]                                bias_rsp_valid,
    output logic [2:0]                                bias_rsp_ready,
    input  logic [(3*TAG_W)-1:0]                      bias_rsp_tags,
    input  logic [(3*OUTPUT_TILE_W)-1:0]              bias_rsp_output_tiles,
    input  logic [(3*TOKEN_ID_W)-1:0]                 bias_rsp_token_ids,
    input  logic [(3*EPOCH_W)-1:0]                    bias_rsp_epochs,
    input  logic [(3*OUT_TILE*ACC_W)-1:0]             bias_rsp_values,

    output logic [5:0]                                final_valid,
    input  logic [5:0]                                final_ready,
    output logic [(6*TOKEN_ID_W)-1:0]                 final_token_ids,
    output logic [(3*TAG_W)-1:0]                      final_tags,
    output logic [(6*OUT_TILE*ACC_W)-1:0]             final_values,

    output logic                                      tile_done_valid,
    input  logic                                      tile_done_ready,
    output logic [TAG_W-1:0]                          tile_done_tag,
    output logic                                      tile_done_error,
    output logic                                      protocol_error,
    output logic                                      accumulator_overflow,

    output logic [COUNTER_W-1:0]                      count_heads,
    output logic [COUNTER_W-1:0]                      count_issued_terms,
    output logic [(3*COUNTER_W)-1:0]                  count_completed_terms,
    output logic [(3*COUNTER_W)-1:0]                  count_bias_commits,
    output logic [(3*COUNTER_W)-1:0]                  count_stale_weight_responses,
    output logic [(3*COUNTER_W)-1:0]                  count_stale_bias_responses
);
    localparam int PROJECTION_BANKS = 32'd3;
    localparam int PROJECTION_BANKS_LOOP = PROJECTION_BANKS;
    localparam int BIAS_COUNT_W = TOKEN_ID_W + 32'd1;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_WAIT_HEAD,
        ST_RUN_HEAD,
        ST_HEAD_DONE,
        ST_BIAS,
        ST_FINISH,
        ST_TILE_DONE
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tile_tag_q;
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile_q;
    logic [HEAD_COUNT_W-1:0] expected_heads_q;
    logic [HEAD_COUNT_W-1:0] heads_completed_q;
    logic [HEAD_COUNT_W-1:0] active_head_index_q;
    logic [INPUT_CH_W-1:0] active_input_channel_base_q;
    logic active_head_last_q;
    logic active_head_error_q;
    logic head_done_error_q;
    logic tile_done_error_q;
    logic source_done_seen_q;
    logic sticky_protocol_q;
    logic [EPOCH_W-1:0] bias_epoch_q;

    logic tile_start_legal;
    logic tile_start_fire;
    logic expected_last_head;
    logic head_start_legal;
    logic head_start_fire;
    logic source_done_match;
    logic source_done_fire;
    logic term_input_fire;
    logic event_input_fire;
    logic flush_q;
    logic dctf_flush_pulse;

    logic dctf_term_ready;
    logic dctf_event_ready;
    logic [5:0] dctf_acc_update_valid;
    logic [5:0] dctf_acc_update_ready;
    logic [(6*TOKEN_ID_W)-1:0] dctf_acc_update_token_ids;
    logic [(3*TAG_W)-1:0] dctf_acc_update_tags;
    logic [(3*OUT_TILE*PRODUCT_W)-1:0] dctf_acc_update_values;
    logic dctf_datapath_idle;
    logic dctf_protocol_error;
    /* verilator lint_off UNUSEDSIGNAL */
    logic dctf_head_compute_done;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [2:0] dctf_weight_req_valid;
    logic [2:0] dctf_weight_rsp_ready;

    logic [2:0] acc_group_start_ready;
    logic [2:0] acc_group_start_valid;
    logic [5:0] acc_update_valid;
    logic [5:0] acc_update_ready;
    logic [(6*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [(3*OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [(3*OUT_TILE*ACC_W)-1:0] acc_update_bias_values;
    logic [2:0] acc_group_finish_ready;
    logic [2:0] acc_group_finish_valid;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [(3*TAG_W)-1:0] acc_group_finish_tags;
    logic [2:0] acc_protocol_error;
    logic [2:0] acc_overflow;

    logic [BIAS_COUNT_W-1:0] bias_token_q [0:PROJECTION_BANKS-1];
    logic [2:0] bias_outstanding_q;
    logic [(3*TAG_W)-1:0] bias_expected_tags_q;
    logic [(3*OUTPUT_TILE_W)-1:0] bias_expected_tiles_q;
    logic [(3*TOKEN_ID_W)-1:0] bias_expected_tokens_q;
    logic [(3*EPOCH_W)-1:0] bias_expected_epochs_q;
    logic [2:0] bias_rsp_stale;
    logic [2:0] bias_rsp_match;
    logic [2:0] bias_rsp_wrong_current;
    logic [2:0] bias_req_fire;
    logic [2:0] bias_rsp_fire;
    logic [2:0] bias_commit_fire;
    logic [2:0] bias_all_committed;
    logic [2:0] ibf_finalize_start_valid;
    logic [2:0] ibf_finalize_start_ready;
    logic [2:0] ibf_finalize_start_fire;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [2:0] ibf_finalize_done_valid;
    logic [2:0] ibf_finalize_done_ready;
    logic [(3*TAG_W)-1:0] ibf_finalize_done_tags;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [2:0] ibf_started_q;

    logic finish_atomic_fire;
    logic finish_tags_match;
    logic [2:0] physical_tile_in_range;

    logic [COUNTER_W-1:0] unused_fabric_count_accepted;
    logic [(3*COUNTER_W)-1:0] unused_fabric_count_bank_consumed;
    logic [COUNTER_W-1:0] unused_fabric_count_retired;
    logic [COUNTER_W-1:0] unused_fabric_count_input_stall;
    logic [(3*COUNTER_W)-1:0] unused_fabric_count_bank_stall;
    logic [COUNTER_W-1:0] unused_fabric_max_occupancy;
    logic [COUNTER_W-1:0] unused_fabric_count_skew;
    logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0] unused_fabric_occupancy;
    logic [2:0] unused_bank_term_done;
    logic [(3*TAG_W)-1:0] unused_bank_term_done_tags;
    logic [(3*ISSUE_SEQ_W)-1:0] unused_bank_term_done_sequences;
    logic [2:0] unused_bank_term_done_last;
    logic [TAG_W-1:0] unused_head_compute_tag;
    logic [ISSUE_SEQ_W-1:0] unused_head_compute_sequence;
    logic unused_dispatch_valid;
    logic [TAG_W-1:0] unused_dispatch_tag;
    logic [CMD_SEQUENCE_W-1:0] unused_dispatch_sequence;
    logic [ISSUE_SEQ_W-1:0] unused_dispatch_issue_sequence;
    logic unused_dispatch_first;
    logic unused_dispatch_last;
    logic unused_dispatch_head_last;

    assign physical_tile_in_range[0] =
        (32'(tile_start_logical_supertile) * 32'd3) <
        (32'd1 << OUTPUT_TILE_W);
    assign physical_tile_in_range[1] =
        ((32'(tile_start_logical_supertile) * 32'd3) + 32'd1) <
        (32'd1 << OUTPUT_TILE_W);
    assign physical_tile_in_range[2] =
        ((32'(tile_start_logical_supertile) * 32'd3) + 32'd2) <
        (32'd1 << OUTPUT_TILE_W);
    assign tile_start_legal = (tile_start_head_count != '0) &&
                              (&physical_tile_in_range);
    assign tile_start_ready = !flush && (state_q == ST_IDLE) &&
                              tile_start_legal && !accumulator_overflow &&
                              (&acc_group_start_ready);
    assign tile_start_fire = tile_start_valid && tile_start_ready;
    assign acc_group_start_valid = {PROJECTION_BANKS{
        tile_start_valid && tile_start_legal && (state_q == ST_IDLE) &&
        (&acc_group_start_ready) && !accumulator_overflow && !flush
    }};

    assign expected_last_head = heads_completed_q ==
                                (expected_heads_q - 1'b1);
    assign head_start_legal = (head_start_tag == tile_tag_q) &&
                              (head_start_index == heads_completed_q) &&
                              (head_start_last == expected_last_head) &&
                              (32'(head_start_input_channel_base) <
                               INPUT_CHANNELS);
    assign head_start_ready = !flush && (state_q == ST_WAIT_HEAD) &&
                              head_start_legal;
    assign head_start_fire = head_start_valid && head_start_ready;

    assign term_ready = (state_q == ST_RUN_HEAD) &&
                        !source_done_seen_q && !flush && dctf_term_ready;
    assign event_ready = (state_q == ST_RUN_HEAD) &&
                         !source_done_seen_q && !flush && dctf_event_ready;
    assign source_done_match = source_done_tag == tile_tag_q;
    assign source_done_ready = !flush && (state_q == ST_RUN_HEAD) &&
                               !source_done_seen_q && source_done_match;
    assign source_done_fire = source_done_valid && source_done_ready;
    assign term_input_fire = term_valid && term_ready;
    assign event_input_fire = event_valid && event_ready;
    assign dctf_flush_pulse = flush && !flush_q;

    assign head_done_valid = !flush && (state_q == ST_HEAD_DONE);
    assign head_done_tag = tile_tag_q;
    assign head_done_index = active_head_index_q;
    assign head_done_last = active_head_last_q;
    assign head_done_error = head_done_error_q;

    assign tile_done_valid = !flush && (state_q == ST_TILE_DONE);
    assign tile_done_tag = tile_tag_q;
    assign tile_done_error = tile_done_error_q;
    assign protocol_error = sticky_protocol_q || dctf_protocol_error ||
                            (|acc_protocol_error) ||
                            (tile_start_valid && (state_q == ST_IDLE) &&
                             !tile_start_legal) ||
                            (head_start_valid && (state_q == ST_WAIT_HEAD) &&
                             !head_start_legal) ||
                            (source_done_valid && (state_q == ST_RUN_HEAD) &&
                             !source_done_seen_q && !source_done_match) ||
                            (|bias_rsp_wrong_current);
    assign accumulator_overflow = |acc_overflow;
    assign weight_req_valid = flush ? 3'b000 : dctf_weight_req_valid;
    assign weight_rsp_ready = flush ? 3'b000 : dctf_weight_rsp_ready;

    always_comb begin
        acc_update_valid = '0;
        acc_update_token_ids = '0;
        acc_update_values = '0;
        acc_update_bias_values = '0;
        dctf_acc_update_ready = '0;

        if ((state_q == ST_RUN_HEAD) && !flush) begin
            acc_update_valid = dctf_acc_update_valid;
            acc_update_token_ids = dctf_acc_update_token_ids;
            acc_update_values = dctf_acc_update_values;
            dctf_acc_update_ready = acc_update_ready;
        end else if ((state_q == ST_BIAS) && !flush &&
                     !IMPLICIT_BIAS_FINALIZE_ENABLE) begin
            for (int bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
                 bank = bank + 32'd1) begin
                if (bias_rsp_valid[bank] && bias_rsp_match[bank]) begin
                    if (bias_rsp_token_ids[bank*TOKEN_ID_W]) begin
                        acc_update_valid[(bank*32'd2)+32'd1] = 1'b1;
                    end else begin
                        acc_update_valid[(bank*32'd2)] = 1'b1;
                    end
                    acc_update_token_ids[(bank*32'd2*TOKEN_ID_W) +:
                                         (32'd2*TOKEN_ID_W)] = {
                        bias_rsp_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W],
                        bias_rsp_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]
                    };
                    acc_update_bias_values[(bank*OUT_TILE*ACC_W) +:
                                           (OUT_TILE*ACC_W)] =
                        bias_rsp_values[(bank*OUT_TILE*ACC_W) +:
                                        (OUT_TILE*ACC_W)];
                end
            end
        end
    end

    generate
        for (genvar bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
             bank = bank + 32'd1) begin : g_bias_control
            localparam int EVEN_ACC_PORT = bank * 32'd2;
            localparam int ODD_ACC_PORT = (bank * 32'd2) + 32'd1;

            assign bias_req_valid[bank] = !flush && (state_q == ST_BIAS) &&
                                          !bias_outstanding_q[bank] &&
                                          !bias_all_committed[bank];
            assign bias_req_tags[(bank*TAG_W) +: TAG_W] = tile_tag_q;
            assign bias_req_output_tiles[(bank*OUTPUT_TILE_W) +:
                                         OUTPUT_TILE_W] =
                OUTPUT_TILE_W'((32'(logical_supertile_q) * 32'd3) + bank);
            assign bias_req_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] =
                IMPLICIT_BIAS_FINALIZE_ENABLE ? '0 :
                bias_token_q[bank][TOKEN_ID_W-1:0];
            assign bias_req_epochs[(bank*EPOCH_W) +: EPOCH_W] = bias_epoch_q;
            assign bias_req_fire[bank] = bias_req_valid[bank] &&
                                         bias_req_ready[bank];

            assign bias_rsp_stale[bank] =
                bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] != bias_epoch_q;
            assign bias_rsp_match[bank] = (state_q == ST_BIAS) &&
                bias_outstanding_q[bank] && !bias_rsp_stale[bank] &&
                (bias_rsp_tags[(bank*TAG_W) +: TAG_W] ==
                 bias_expected_tags_q[(bank*TAG_W) +: TAG_W]) &&
                (bias_rsp_output_tiles[(bank*OUTPUT_TILE_W) +:
                                       OUTPUT_TILE_W] ==
                 bias_expected_tiles_q[(bank*OUTPUT_TILE_W) +:
                                       OUTPUT_TILE_W]) &&
                (bias_rsp_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] ==
                 bias_expected_tokens_q[(bank*TOKEN_ID_W) +: TOKEN_ID_W]) &&
                (bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] ==
                 bias_expected_epochs_q[(bank*EPOCH_W) +: EPOCH_W]);
            assign bias_rsp_wrong_current[bank] = bias_rsp_valid[bank] &&
                                                  !bias_rsp_stale[bank] &&
                                                  !bias_rsp_match[bank];
            assign bias_rsp_ready[bank] = !flush &&
                (bias_rsp_stale[bank] || bias_rsp_wrong_current[bank] ||
                 (bias_rsp_match[bank] ?
                    (IMPLICIT_BIAS_FINALIZE_ENABLE ?
                        ibf_finalize_start_ready[bank] :
                     bias_rsp_token_ids[bank*TOKEN_ID_W] ?
                        acc_update_ready[ODD_ACC_PORT] :
                        acc_update_ready[EVEN_ACC_PORT]) : 1'b0));
            assign bias_rsp_fire[bank] = bias_rsp_valid[bank] &&
                                         bias_rsp_ready[bank];
            assign bias_commit_fire[bank] =
                                            !IMPLICIT_BIAS_FINALIZE_ENABLE &&
                                            bias_rsp_fire[bank] &&
                                            bias_rsp_match[bank];
            assign ibf_finalize_start_valid[bank] =
                IMPLICIT_BIAS_FINALIZE_ENABLE && (state_q == ST_BIAS) &&
                bias_outstanding_q[bank] && bias_rsp_valid[bank] &&
                bias_rsp_match[bank];
            assign ibf_finalize_start_fire[bank] =
                ibf_finalize_start_valid[bank] &&
                ibf_finalize_start_ready[bank];
            assign bias_all_committed[bank] =
                IMPLICIT_BIAS_FINALIZE_ENABLE ? ibf_started_q[bank] :
                (bias_token_q[bank] == BIAS_COUNT_W'(TOKENS));
        end
    endgenerate

    assign finish_atomic_fire = !flush && (state_q == ST_FINISH) &&
                                (&acc_group_finish_ready);
    assign acc_group_finish_valid = {PROJECTION_BANKS{finish_atomic_fire}};
    assign ibf_finalize_done_ready =
        {PROJECTION_BANKS{IMPLICIT_BIAS_FINALIZE_ENABLE &&
                          finish_atomic_fire}};
    assign finish_tags_match =
        (acc_group_finish_tags[(32'd0*TAG_W) +: TAG_W] == tile_tag_q) &&
        (acc_group_finish_tags[(32'd1*TAG_W) +: TAG_W] == tile_tag_q) &&
        (acc_group_finish_tags[(32'd2*TAG_W) +: TAG_W] == tile_tag_q);

    gatestack_dctf96_term_datapath_top #(
        .Q(Q), .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE), .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W), .GROUP_TAG_W(TAG_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W), .INPUT_CHANNELS(INPUT_CHANNELS),
        .LANE_ID_W(LANE_ID_W), .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W), .EPOCH_W(EPOCH_W),
        .COUNTER_W(COUNTER_W), .ADAPTER_CONTEXTS(ADAPTER_CONTEXTS),
        .PPDI_ENABLE(PPDI_ENABLE),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) u_term_datapath (
        .clk_core(clk_core), .rst_core(rst_core), .flush(dctf_flush_pulse),
        .clear_error(dctf_flush_pulse),
        .term_valid(term_valid && (state_q == ST_RUN_HEAD) &&
                    !source_done_seen_q && !flush),
        .term_ready(dctf_term_ready), .term_tag(tile_tag_q),
        .term_gate_code(term_gate_code), .term_lane_id(term_lane_id),
        .term_destination_count(term_destination_count),
        .term_issue_seq(term_issue_seq), .term_head_last(term_head_last),
        .logical_supertile(logical_supertile_q),
        .head_input_channel_base(active_input_channel_base_q),
        .event_valid(event_valid && (state_q == ST_RUN_HEAD) &&
                     !source_done_seen_q && !flush),
        .event_ready(dctf_event_ready), .event_gate_code(event_gate_code),
        .event_lane_id(event_lane_id), .event_token_valid(event_token_valid),
        .event_token_ids(event_token_ids), .event_count(event_count),
        .event_issue_seq(event_issue_seq),
        .event_term_first(event_term_first), .event_term_last(event_term_last),
        .event_head_last(event_head_last),
        .weight_req_valid(dctf_weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tags(weight_req_tags),
        .weight_req_input_channels(weight_req_input_channels),
        .weight_req_output_tiles(weight_req_output_tiles),
        .weight_req_epochs(weight_req_epochs),
        .weight_rsp_valid(weight_rsp_valid & {3{!flush}}),
        .weight_rsp_ready(dctf_weight_rsp_ready),
        .weight_rsp_tags(weight_rsp_tags),
        .weight_rsp_input_channels(weight_rsp_input_channels),
        .weight_rsp_output_tiles(weight_rsp_output_tiles),
        .weight_rsp_epochs(weight_rsp_epochs),
        .weight_rsp_weights(weight_rsp_weights),
        .acc_update_valid(dctf_acc_update_valid),
        .acc_update_ready(dctf_acc_update_ready),
        .acc_update_token_ids(dctf_acc_update_token_ids),
        .acc_update_tags(dctf_acc_update_tags),
        .acc_update_values(dctf_acc_update_values),
        .bank_term_done(unused_bank_term_done),
        .bank_term_done_group_tags(unused_bank_term_done_tags),
        .bank_term_done_issue_seqs(unused_bank_term_done_sequences),
        .bank_term_done_head_last(unused_bank_term_done_last),
        .head_compute_done(dctf_head_compute_done),
        .head_compute_done_group_tag(unused_head_compute_tag),
        .head_compute_done_issue_seq(unused_head_compute_sequence),
        .dispatch_retire_valid(unused_dispatch_valid),
        .dispatch_retire_group_tag(unused_dispatch_tag),
        .dispatch_retire_sequence(unused_dispatch_sequence),
        .dispatch_retire_issue_seq(unused_dispatch_issue_sequence),
        .dispatch_retire_term_first(unused_dispatch_first),
        .dispatch_retire_term_last(unused_dispatch_last),
        .dispatch_retire_head_last(unused_dispatch_head_last),
        .fabric_occupancy(unused_fabric_occupancy),
        .fabric_count_accepted(unused_fabric_count_accepted),
        .fabric_count_bank_consumed(unused_fabric_count_bank_consumed),
        .fabric_count_retired(unused_fabric_count_retired),
        .fabric_count_input_stall(unused_fabric_count_input_stall),
        .fabric_count_bank_stall(unused_fabric_count_bank_stall),
        .fabric_max_occupancy(unused_fabric_max_occupancy),
        .fabric_count_skew_cycles(unused_fabric_count_skew),
        .issued_terms(count_issued_terms),
        .completed_terms(count_completed_terms),
        .count_stale_weight_responses(count_stale_weight_responses),
        .datapath_idle(dctf_datapath_idle),
        .protocol_error(dctf_protocol_error)
    );

    generate
        for (genvar bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
             bank = bank + 32'd1) begin : g_accumulator
            logic [COUNTER_W-1:0] unused_count_updates;
            logic [COUNTER_W-1:0] unused_count_writes;
            logic [COUNTER_W-1:0] unused_count_bank_stall;
            logic [COUNTER_W-1:0] unused_count_final_stall;
            logic [COUNTER_W-1:0] unused_count_final_emits;

            if (IMPLICIT_BIAS_FINALIZE_ENABLE) begin : g_ibf
                hitflow_implicit_bias_finalizer_accumulator #(
                    .TOKENS(TOKENS), .BANKS(32'd2),
                    .PRODUCT_W(PRODUCT_W), .ACC_W(ACC_W),
                    .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
                    .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W)
                ) u_accumulator (
                    .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                    .group_start_valid(acc_group_start_valid[bank]),
                    .group_start_ready(acc_group_start_ready[bank]),
                    .group_start_tag(tile_start_tag),
                    .update_valid(acc_update_valid[
                        (bank*32'd2) +: 32'd2]),
                    .update_ready(acc_update_ready[
                        (bank*32'd2) +: 32'd2]),
                    .update_token_ids(acc_update_token_ids[
                        (bank*32'd2*TOKEN_ID_W) +:
                        (32'd2*TOKEN_ID_W)]),
                    .update_tag(dctf_acc_update_tags[
                        (bank*TAG_W) +: TAG_W]),
                    .update_values(acc_update_values[
                        (bank*OUT_TILE*PRODUCT_W) +:
                        (OUT_TILE*PRODUCT_W)]),
                    .finalize_start_valid(
                        ibf_finalize_start_valid[bank]),
                    .finalize_start_ready(
                        ibf_finalize_start_ready[bank]),
                    .finalize_start_tag(tile_tag_q),
                    .finalize_bias_values(bias_rsp_values[
                        (bank*OUT_TILE*ACC_W) +: (OUT_TILE*ACC_W)]),
                    .final_valid(final_valid[(bank*32'd2) +: 32'd2]),
                    .final_ready(final_ready[(bank*32'd2) +: 32'd2]),
                    .final_token_ids(final_token_ids[
                        (bank*32'd2*TOKEN_ID_W) +:
                        (32'd2*TOKEN_ID_W)]),
                    .final_tag(final_tags[(bank*TAG_W) +: TAG_W]),
                    .final_values(final_values[
                        (bank*32'd2*OUT_TILE*ACC_W) +:
                        (32'd2*OUT_TILE*ACC_W)]),
                    .finalize_done_valid(ibf_finalize_done_valid[bank]),
                    .finalize_done_ready(ibf_finalize_done_ready[bank]),
                    .finalize_done_tag(ibf_finalize_done_tags[
                        (bank*TAG_W) +: TAG_W]),
                    .protocol_error(acc_protocol_error[bank]),
                    .accumulator_overflow(acc_overflow[bank]),
                    .count_updates(unused_count_updates),
                    .count_product_writes(unused_count_writes),
                    .count_final_reads(count_bias_commits[
                        (bank*COUNTER_W) +: COUNTER_W]),
                    .count_final_emits(unused_count_final_emits),
                    .count_update_stall_cycles(unused_count_bank_stall),
                    .count_final_stall_cycles(unused_count_final_stall)
                );
                assign acc_group_finish_ready[bank] =
                    ibf_finalize_done_valid[bank];
                assign acc_group_finish_tags[(bank*TAG_W) +: TAG_W] =
                    ibf_finalize_done_tags[(bank*TAG_W) +: TAG_W];
            end else begin : g_rmw
                assign ibf_finalize_start_ready[bank] = 1'b0;
                assign ibf_finalize_done_valid[bank] = 1'b0;
                assign ibf_finalize_done_tags[(bank*TAG_W) +: TAG_W] = '0;
                assign unused_count_final_emits = '0;
                hitflow_banked_accumulator #(
                .TOKENS(TOKENS), .BANKS(32'd2), .PRODUCT_W(PRODUCT_W),
                .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
                .COUNTER_W(COUNTER_W), .TOKEN_ID_W(TOKEN_ID_W)
                ) u_accumulator (
                .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
                .group_start_valid(acc_group_start_valid[bank]),
                .group_start_ready(acc_group_start_ready[bank]),
                .group_start_tag(tile_start_tag),
                .update_valid(acc_update_valid[(bank*32'd2) +: 32'd2]),
                .update_ready(acc_update_ready[(bank*32'd2) +: 32'd2]),
                .update_token_ids(acc_update_token_ids[
                    (bank*32'd2*TOKEN_ID_W) +: (32'd2*TOKEN_ID_W)]),
                .update_tag((state_q == ST_RUN_HEAD) ?
                    dctf_acc_update_tags[(bank*TAG_W) +: TAG_W] : tile_tag_q),
                .update_is_bias(state_q == ST_BIAS),
                .update_values(acc_update_values[
                    (bank*OUT_TILE*PRODUCT_W) +: (OUT_TILE*PRODUCT_W)]),
                .update_bias_values(acc_update_bias_values[
                    (bank*OUT_TILE*ACC_W) +: (OUT_TILE*ACC_W)]),
                .final_valid(final_valid[(bank*32'd2) +: 32'd2]),
                .final_ready(final_ready[(bank*32'd2) +: 32'd2]),
                .final_token_ids(final_token_ids[
                    (bank*32'd2*TOKEN_ID_W) +: (32'd2*TOKEN_ID_W)]),
                .final_tag(final_tags[(bank*TAG_W) +: TAG_W]),
                .final_values(final_values[
                    (bank*32'd2*OUT_TILE*ACC_W) +:
                    (32'd2*OUT_TILE*ACC_W)]),
                .group_finish_valid(acc_group_finish_valid[bank]),
                .group_finish_ready(acc_group_finish_ready[bank]),
                .group_finish_tag(acc_group_finish_tags[
                    (bank*TAG_W) +: TAG_W]),
                .protocol_error(acc_protocol_error[bank]),
                .accumulator_overflow(acc_overflow[bank]),
                .count_updates(unused_count_updates),
                .count_writes(unused_count_writes),
                .count_bias_commits(count_bias_commits[
                    (bank*COUNTER_W) +: COUNTER_W]),
                .count_bank_stall_cycles(unused_count_bank_stall),
                .count_final_stall_cycles(unused_count_final_stall)
                );
            end
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tile_tag_q <= '0;
            logical_supertile_q <= '0;
            expected_heads_q <= '0;
            heads_completed_q <= '0;
            active_head_index_q <= '0;
            active_input_channel_base_q <= '0;
            active_head_last_q <= 1'b0;
            active_head_error_q <= 1'b0;
            head_done_error_q <= 1'b0;
            tile_done_error_q <= 1'b0;
            source_done_seen_q <= 1'b0;
            sticky_protocol_q <= 1'b0;
            flush_q <= 1'b0;
            bias_epoch_q <= '0;
            bias_outstanding_q <= '0;
            ibf_started_q <= '0;
            bias_expected_tags_q <= '0;
            bias_expected_tiles_q <= '0;
            bias_expected_tokens_q <= '0;
            bias_expected_epochs_q <= '0;
            count_heads <= '0;
            count_stale_bias_responses <= '0;
            for (int bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
                 bank = bank + 32'd1)
                bias_token_q[bank] <= '0;
        end else if (flush) begin
            flush_q <= 1'b1;
            state_q <= ST_IDLE;
            tile_tag_q <= '0;
            logical_supertile_q <= '0;
            expected_heads_q <= '0;
            heads_completed_q <= '0;
            active_head_index_q <= '0;
            active_input_channel_base_q <= '0;
            active_head_last_q <= 1'b0;
            active_head_error_q <= 1'b0;
            head_done_error_q <= 1'b0;
            tile_done_error_q <= 1'b0;
            source_done_seen_q <= 1'b0;
            sticky_protocol_q <= 1'b0;
            if (!flush_q)
                bias_epoch_q <= bias_epoch_q + 1'b1;
            bias_outstanding_q <= '0;
            ibf_started_q <= '0;
            bias_expected_tags_q <= '0;
            bias_expected_tiles_q <= '0;
            bias_expected_tokens_q <= '0;
            bias_expected_epochs_q <= '0;
            for (int bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
                 bank = bank + 32'd1)
                bias_token_q[bank] <= '0;
        end else begin
            flush_q <= 1'b0;
            if (dctf_protocol_error || (|acc_protocol_error) ||
                (tile_start_valid && (state_q == ST_IDLE) &&
                 !tile_start_legal) ||
                (head_start_valid && (state_q == ST_WAIT_HEAD) &&
                 !head_start_legal) ||
                (source_done_valid && (state_q == ST_RUN_HEAD) &&
                 !source_done_seen_q && !source_done_match) ||
                (|bias_rsp_wrong_current))
                sticky_protocol_q <= 1'b1;

            for (int bank = 32'd0; bank < PROJECTION_BANKS_LOOP;
                 bank = bank + 32'd1) begin
                if (bias_req_fire[bank]) begin
                    bias_outstanding_q[bank] <= 1'b1;
                    bias_expected_tags_q[(bank*TAG_W) +: TAG_W] <=
                        bias_req_tags[(bank*TAG_W) +: TAG_W];
                    bias_expected_tiles_q[(bank*OUTPUT_TILE_W) +:
                                          OUTPUT_TILE_W] <=
                        bias_req_output_tiles[(bank*OUTPUT_TILE_W) +:
                                              OUTPUT_TILE_W];
                    bias_expected_tokens_q[(bank*TOKEN_ID_W) +: TOKEN_ID_W] <=
                        bias_req_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W];
                    bias_expected_epochs_q[(bank*EPOCH_W) +: EPOCH_W] <=
                        bias_req_epochs[(bank*EPOCH_W) +: EPOCH_W];
                end
                if (bias_rsp_fire[bank] && bias_rsp_stale[bank]) begin
                    count_stale_bias_responses[(bank*COUNTER_W) +:
                                               COUNTER_W] <=
                        count_stale_bias_responses[(bank*COUNTER_W) +:
                                                   COUNTER_W] + 1'b1;
                end else if (ibf_finalize_start_fire[bank]) begin
                    bias_outstanding_q[bank] <= 1'b0;
                    ibf_started_q[bank] <= 1'b1;
                end else if (bias_commit_fire[bank]) begin
                    bias_outstanding_q[bank] <= 1'b0;
                    bias_token_q[bank] <= bias_token_q[bank] + 1'b1;
                end
            end

            case (state_q)
                ST_IDLE: begin
                    if (tile_start_fire) begin
                        tile_tag_q <= tile_start_tag;
                        logical_supertile_q <= tile_start_logical_supertile;
                        expected_heads_q <= tile_start_head_count;
                        heads_completed_q <= '0;
                        active_head_error_q <= 1'b0;
                        head_done_error_q <= 1'b0;
                        tile_done_error_q <= 1'b0;
                        source_done_seen_q <= 1'b0;
                        bias_outstanding_q <= '0;
                        ibf_started_q <= '0;
                        for (int bank = 32'd0;
                             bank < PROJECTION_BANKS_LOOP;
                             bank = bank + 32'd1)
                            bias_token_q[bank] <= '0;
                        state_q <= ST_WAIT_HEAD;
                    end
                end
                ST_WAIT_HEAD: begin
                    if (head_start_fire) begin
                        active_head_index_q <= head_start_index;
                        active_input_channel_base_q <=
                            head_start_input_channel_base;
                        active_head_last_q <= head_start_last;
                        active_head_error_q <= 1'b0;
                        head_done_error_q <= 1'b0;
                        source_done_seen_q <= 1'b0;
                        state_q <= ST_RUN_HEAD;
                    end
                end
                ST_RUN_HEAD: begin
                    if (source_done_fire) begin
                        source_done_seen_q <= 1'b1;
                        active_head_error_q <= source_done_error;
                        if (source_done_error)
                            sticky_protocol_q <= 1'b1;
                    end
                    if ((source_done_seen_q || source_done_fire) &&
                        !term_input_fire && !event_input_fire &&
                        dctf_datapath_idle) begin
                        head_done_error_q <=
                            (source_done_fire ? source_done_error :
                             active_head_error_q) || sticky_protocol_q ||
                            dctf_protocol_error || (|acc_protocol_error) ||
                            (|acc_overflow);
                        state_q <= ST_HEAD_DONE;
                    end
                end
                ST_HEAD_DONE: begin
                    if (head_done_valid && head_done_ready) begin
                        count_heads <= count_heads + 1'b1;
                        heads_completed_q <= heads_completed_q + 1'b1;
                        source_done_seen_q <= 1'b0;
                        if (active_head_last_q) begin
                            bias_outstanding_q <= '0;
                            ibf_started_q <= '0;
                            for (int bank = 32'd0;
                                 bank < PROJECTION_BANKS_LOOP;
                                 bank = bank + 32'd1)
                                bias_token_q[bank] <= '0;
                            state_q <= ST_BIAS;
                        end else begin
                            state_q <= ST_WAIT_HEAD;
                        end
                    end
                end
                ST_BIAS: begin
                    if (&bias_all_committed)
                        state_q <= ST_FINISH;
                end
                ST_FINISH: begin
                    if (finish_atomic_fire) begin
                        tile_done_error_q <= sticky_protocol_q ||
                            dctf_protocol_error || (|acc_protocol_error) ||
                            (|acc_overflow) || !finish_tags_match;
                        if (!finish_tags_match)
                            sticky_protocol_q <= 1'b1;
                        state_q <= ST_TILE_DONE;
                    end
                end
                ST_TILE_DONE: begin
                    if (tile_done_valid && tile_done_ready)
                        state_q <= ST_IDLE;
                end
                default: begin
                    state_q <= ST_IDLE;
                    sticky_protocol_q <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
