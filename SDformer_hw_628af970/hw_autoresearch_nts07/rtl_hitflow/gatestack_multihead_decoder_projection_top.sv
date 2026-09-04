`timescale 1ns/1ps
`default_nettype none

// Actual resident/IPD32W/RAW41 decoders sharing one tile-lifetime multihead
// projection backend. Slot/cache/launch/lifecycle remain outside this boundary.
module gatestack_multihead_decoder_projection_top #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int MAX_TERMS       = 128,
    parameter int RESIDENT_TERMS  = 80,
    // 0: IPD32W, 1: FADC24, 2: header-steered adaptive IPD32W/FADC24.
    parameter int CSR_FORMAT_FADC24 = 0,
    parameter int EVENT_WAYS      = 4,
    parameter int OUT_TILE        = 8,
    parameter int BANKS           = 2,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int ACC_W           = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int HEAD_COUNT_W    = 6,
    parameter int WORD_INDEX_W    = 7,
    parameter int EVENT_COUNT_W   = 13,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int ROUTE_W         = 2,
    parameter int FORMAT_W        = 2,
    parameter int RES_TERM_IDX_W  = (RESIDENT_TERMS <= 1) ?
                                      1 : $clog2(RESIDENT_TERMS),
    parameter int IPD_TERM_IDX_W  = (MAX_TERMS <= 1) ? 1 : $clog2(MAX_TERMS)
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
    input  logic [TAG_W-1:0]                          head_start_payload_tag,
    input  logic [HEAD_COUNT_W-1:0]                   head_start_index,
    input  logic [ROUTE_W-1:0]                        head_start_route_select,
    input  logic [FORMAT_W-1:0]                       head_start_csr_format,
    input  logic [INPUT_CH_W-1:0]                     head_start_input_channel_base,
    input  logic                                      head_start_last,
    input  logic [7:0]                                resident_term_count,
    input  logic [EVENT_COUNT_W-1:0]                  resident_event_count,

    input  logic                                      resident_descriptor_valid,
    output logic                                      resident_descriptor_ready,
    input  logic [GATE_W-1:0]                         resident_descriptor_gate_code,
    input  logic [LANE_ID_W-1:0]                      resident_descriptor_lane_id,
    input  logic [7:0]                                resident_descriptor_destination_count,
    input  logic [RES_TERM_IDX_W-1:0]                 resident_descriptor_term_index,
    input  logic                                      resident_descriptor_last,
    input  logic                                      resident_word_valid,
    output logic                                      resident_word_ready,
    input  logic [63:0]                               resident_word_data,
    input  logic [WORD_INDEX_W-1:0]                   resident_word_index,
    input  logic                                      resident_word_last,

    input  logic                                      ipd_word_valid,
    output logic                                      ipd_word_ready,
    input  logic [63:0]                               ipd_word_data,
    input  logic [WORD_INDEX_W-1:0]                   ipd_word_index,
    input  logic                                      ipd_word_last,
    input  logic                                      raw_word_valid,
    output logic                                      raw_word_ready,
    input  logic [63:0]                               raw_word_data,
    input  logic [WORD_INDEX_W-1:0]                   raw_word_index,
    input  logic                                      raw_word_last,

    output logic                                      ipd_fill_begin_valid,
    input  logic                                      ipd_fill_begin_ready,
    output logic [TAG_W-1:0]                          ipd_fill_begin_tag,
    output logic [7:0]                                ipd_fill_begin_term_count,
    output logic                                      ipd_fill_entry_valid,
    input  logic                                      ipd_fill_entry_ready,
    output logic [GATE_W-1:0]                         ipd_fill_gate_code,
    output logic [LANE_ID_W-1:0]                      ipd_fill_lane_id,
    output logic [7:0]                                ipd_fill_destination_count,
    output logic                                      ipd_fill_entry_last,
    output logic                                      ipd_fill_cache_allowed,

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
    output logic                                      decoder_done_valid,
    input  logic                                      decoder_done_ready,
    output logic [TAG_W-1:0]                          decoder_done_payload_tag,
    output logic                                      decoder_done_error,
    output logic                                      head_done_valid,
    input  logic                                      head_done_ready,
    output logic [TAG_W-1:0]                          head_done_tag,
    output logic [HEAD_COUNT_W-1:0]                   head_done_index,
    output logic                                      head_done_last,
    output logic                                      head_done_error,
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
    localparam logic [ROUTE_W-1:0] ROUTE_RESIDENT = ROUTE_W'(0);
    localparam logic [ROUTE_W-1:0] ROUTE_IPD = ROUTE_W'(1);
    localparam logic [ROUTE_W-1:0] ROUTE_RAW = ROUTE_W'(2);
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);
    logic routed_head_valid;
    logic routed_head_ready;
    logic selected_decoder_ready;
    logic resident_start_valid, resident_start_ready;
    logic ipd_start_valid, ipd_start_ready;
    logic raw_start_valid, raw_start_ready;

    logic resident_term_valid, resident_term_ready;
    logic [GATE_W-1:0] resident_term_gate;
    logic [LANE_ID_W-1:0] resident_term_lane;
    logic [7:0] resident_term_destinations;
    logic resident_term_head_last;
    logic resident_event_valid, resident_event_ready;
    logic [GATE_W-1:0] resident_event_gate;
    logic [LANE_ID_W-1:0] resident_event_lane;
    logic [EVENT_WAYS-1:0] resident_event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] resident_event_token_ids;
    logic [WAY_COUNT_W-1:0] resident_event_count_out;
    logic resident_event_first, resident_event_last;
    logic resident_event_head_last;
    logic resident_done_valid, resident_done_ready;
    logic [TAG_W-1:0] resident_done_tag;
    logic resident_done_error, resident_protocol_error;

    logic ipd_term_valid, ipd_term_ready;
    logic [GATE_W-1:0] ipd_term_gate;
    logic [LANE_ID_W-1:0] ipd_term_lane;
    logic [7:0] ipd_term_destinations;
    logic ipd_term_head_last;
    logic ipd_event_valid, ipd_event_ready;
    logic [GATE_W-1:0] ipd_event_gate;
    logic [LANE_ID_W-1:0] ipd_event_lane;
    logic [EVENT_WAYS-1:0] ipd_event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] ipd_event_token_ids;
    logic [WAY_COUNT_W-1:0] ipd_event_count_out;
    logic ipd_event_first, ipd_event_last, ipd_event_head_last;
    logic ipd_done_valid, ipd_done_ready;
    logic [TAG_W-1:0] ipd_done_tag;
    logic ipd_done_error, ipd_protocol_error;

    logic raw_direct_valid, raw_direct_ready;
    logic [GATE_W-1:0] raw_direct_gate;
    logic [LANE_ID_W-1:0] raw_direct_lane;
    logic [TOKEN_ID_W-1:0] raw_direct_token;
    /* verilator lint_off UNUSEDSIGNAL */
    logic raw_direct_head_last;
    /* verilator lint_on UNUSEDSIGNAL */
    logic raw_done_valid, raw_decoder_done_ready;
    logic [TAG_W-1:0] raw_done_tag;
    logic raw_done_error, raw_protocol_error;
    logic raw_tail_valid, raw_tail_ready;
    logic [GATE_W-1:0] raw_tail_gate;
    logic [LANE_ID_W-1:0] raw_tail_lane;
    logic [TOKEN_ID_W-1:0] raw_tail_token;
    logic raw_tail_head_last;
    logic raw_tail_done_valid, raw_tail_done_ready;
    logic [TAG_W-1:0] raw_tail_done_tag;
    logic raw_tail_done_error, raw_tail_protocol_error;
    logic raw_adapter_term_valid, raw_adapter_term_ready;
    logic [GATE_W-1:0] raw_adapter_term_gate;
    logic [LANE_ID_W-1:0] raw_adapter_term_lane;
    logic [7:0] raw_adapter_term_destinations;
    logic raw_adapter_term_head_last;
    logic raw_adapter_event_valid, raw_adapter_event_ready;
    logic [GATE_W-1:0] raw_adapter_event_gate;
    logic [LANE_ID_W-1:0] raw_adapter_event_lane;
    logic [EVENT_WAYS-1:0] raw_adapter_event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] raw_adapter_event_token_ids;
    logic [WAY_COUNT_W-1:0] raw_adapter_event_count;
    logic raw_adapter_event_first, raw_adapter_event_last;
    logic raw_adapter_event_head_last;
    logic raw_adapter_busy_q;
    logic raw_tail_fire, raw_adapter_event_fire;
    logic raw_completion_blocked;

    logic [2:0] source_term_valid, source_term_ready;
    logic [(3*GATE_W)-1:0] source_term_gate_code;
    logic [(3*LANE_ID_W)-1:0] source_term_lane_id;
    logic [(3*8)-1:0] source_term_destination_count;
    logic [2:0] source_term_head_last;
    logic [2:0] source_event_valid, source_event_ready;
    logic [(3*GATE_W)-1:0] source_event_gate_code;
    logic [(3*LANE_ID_W)-1:0] source_event_lane_id;
    logic [(3*EVENT_WAYS)-1:0] source_event_token_valid;
    logic [(3*EVENT_WAYS*TOKEN_ID_W)-1:0] source_event_token_ids;
    logic [(3*WAY_COUNT_W)-1:0] source_event_count;
    logic [2:0] source_event_term_first, source_event_term_last;
    logic [2:0] source_event_head_last;
    logic [2:0] source_done_valid, source_done_ready;
    logic [(3*TAG_W)-1:0] source_done_tag;
    logic [2:0] source_done_error;
    logic routed_protocol_error;
    /* verilator lint_off UNUSEDSIGNAL */
    logic route_active;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [ROUTE_W-1:0] route_active_select;
    logic [TAG_W-1:0] expected_payload_tag_q;
    logic [TAG_W-1:0] active_execution_tag_q;
    logic [FORMAT_W-1:0] active_format_q;
    logic decoder_done_valid_q;
    logic [TAG_W-1:0] decoder_done_payload_tag_q;
    logic decoder_done_error_q;
    logic [TAG_W-1:0] selected_done_payload_tag;
    logic selected_done_error;
    logic decoder_completion_slot_ready;
    logic head_start_fire;
    logic source_done_fire;
    logic head_start_format_legal;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [RES_TERM_IDX_W-1:0] resident_term_index;
    logic [IPD_TERM_IDX_W-1:0] ipd_term_index;
    logic [ISSUE_SEQ_W-1:0] raw_adapter_term_issue_seq;
    logic [ISSUE_SEQ_W-1:0] raw_adapter_event_issue_seq;
    logic [COUNTER_W-1:0] resident_count_heads, resident_count_terms;
    logic [COUNTER_W-1:0] resident_count_events;
    logic [COUNTER_W-1:0] resident_count_desc_stall;
    logic [COUNTER_W-1:0] resident_count_input_stall;
    logic [COUNTER_W-1:0] resident_count_term_stall;
    logic [COUNTER_W-1:0] resident_count_output_stall;
    logic [COUNTER_W-1:0] ipd_count_heads, ipd_count_terms;
    logic [COUNTER_W-1:0] ipd_count_events, ipd_count_input_stall;
    logic [COUNTER_W-1:0] ipd_count_term_stall, ipd_count_output_stall;
    logic [COUNTER_W-1:0] fadc_count_bitmap_terms;
    logic [COUNTER_W-1:0] raw_count_heads, raw_count_records;
    logic [COUNTER_W-1:0] raw_count_kzero, raw_count_events;
    logic [COUNTER_W-1:0] raw_count_input_stall, raw_count_output_stall;
    logic [COUNTER_W-1:0] raw_adapter_count_inputs;
    logic [COUNTER_W-1:0] raw_adapter_count_term_stall;
    logic [COUNTER_W-1:0] raw_adapter_count_event_stall;
    logic [COUNTER_W-1:0] raw_tail_count_inputs;
    logic [COUNTER_W-1:0] raw_tail_count_outputs;
    logic [COUNTER_W-1:0] raw_tail_count_empty;
    /* verilator lint_on UNUSEDSIGNAL */

    always_comb begin
        selected_decoder_ready = 1'b0;
        if (head_start_route_select == ROUTE_RESIDENT)
            selected_decoder_ready = resident_start_ready;
        else if (head_start_route_select == ROUTE_IPD)
            selected_decoder_ready = ipd_start_ready;
        else if (head_start_route_select == ROUTE_RAW)
            selected_decoder_ready = raw_start_ready;
    end
    assign decoder_completion_slot_ready = !decoder_done_valid_q ||
                                            decoder_done_ready;
    assign head_start_ready = routed_head_ready && selected_decoder_ready &&
                              decoder_completion_slot_ready &&
                              head_start_format_legal;
    assign head_start_fire = head_start_valid && head_start_ready;
    assign routed_head_valid = head_start_valid && selected_decoder_ready &&
                               decoder_completion_slot_ready &&
                               head_start_format_legal;
    assign resident_start_valid = head_start_valid && routed_head_ready &&
                                  decoder_completion_slot_ready &&
                                  head_start_format_legal &&
                                  head_start_route_select == ROUTE_RESIDENT;
    assign ipd_start_valid = head_start_valid && routed_head_ready &&
                             decoder_completion_slot_ready &&
                             head_start_format_legal &&
                             head_start_route_select == ROUTE_IPD;
    assign raw_start_valid = head_start_valid && routed_head_ready &&
                             decoder_completion_slot_ready &&
                             head_start_format_legal &&
                             head_start_route_select == ROUTE_RAW;

    always_comb begin
        head_start_format_legal = 1'b0;
        if (head_start_route_select == ROUTE_RESIDENT)
            head_start_format_legal =
                head_start_csr_format == FORMAT_IPD32W;
        else if (head_start_route_select == ROUTE_RAW)
            head_start_format_legal = head_start_csr_format == FORMAT_RAW;
        else if (head_start_route_select == ROUTE_IPD) begin
            if (CSR_FORMAT_FADC24 == 0)
                head_start_format_legal =
                    head_start_csr_format == FORMAT_IPD32W;
            else if (CSR_FORMAT_FADC24 == 1)
                head_start_format_legal =
                    head_start_csr_format == FORMAT_FADC24;
            else
                head_start_format_legal =
                    head_start_csr_format == FORMAT_IPD32W ||
                    head_start_csr_format == FORMAT_FADC24;
        end
    end

    assign resident_term_ready = source_term_ready[0];
    assign ipd_term_ready = source_term_ready[1] && ipd_fill_entry_ready;
    assign raw_adapter_term_ready = source_term_ready[2];
    assign resident_event_ready = source_event_ready[0];
    assign ipd_event_ready = source_event_ready[1];
    assign raw_adapter_event_ready = source_event_ready[2];
    assign resident_done_ready = source_done_ready[0];
    assign ipd_done_ready = source_done_ready[1];
    assign raw_completion_blocked = raw_adapter_busy_q ||
                                    raw_adapter_event_valid;
    assign raw_tail_done_ready = source_done_ready[2] &&
                                 !raw_completion_blocked;
    assign raw_tail_fire = raw_tail_valid && raw_tail_ready;
    assign raw_adapter_event_fire = raw_adapter_event_valid &&
                                    raw_adapter_event_ready;

    always_comb begin
        source_term_valid = {raw_adapter_term_valid,
                             ipd_term_valid && ipd_fill_entry_ready,
                             resident_term_valid};
        source_term_gate_code = {raw_adapter_term_gate,
                                 ipd_term_gate, resident_term_gate};
        source_term_lane_id = {raw_adapter_term_lane,
                               ipd_term_lane, resident_term_lane};
        source_term_destination_count = {raw_adapter_term_destinations,
                                         ipd_term_destinations,
                                         resident_term_destinations};
        source_term_head_last = {raw_adapter_term_head_last,
                                 ipd_term_head_last,
                                 resident_term_head_last};
        source_event_valid = {raw_adapter_event_valid,
                              ipd_event_valid, resident_event_valid};
        source_event_gate_code = {raw_adapter_event_gate,
                                  ipd_event_gate, resident_event_gate};
        source_event_lane_id = {raw_adapter_event_lane,
                                ipd_event_lane, resident_event_lane};
        source_event_token_valid = {raw_adapter_event_token_valid,
                                    ipd_event_token_valid,
                                    resident_event_token_valid};
        source_event_token_ids = {raw_adapter_event_token_ids,
                                  ipd_event_token_ids,
                                  resident_event_token_ids};
        source_event_count = {raw_adapter_event_count,
                              ipd_event_count_out,
                              resident_event_count_out};
        source_event_term_first = {raw_adapter_event_first,
                                   ipd_event_first,
                                   resident_event_first};
        source_event_term_last = {raw_adapter_event_last,
                                  ipd_event_last,
                                  resident_event_last};
        source_event_head_last = {raw_adapter_event_head_last,
                                  ipd_event_head_last,
                                  resident_event_head_last};
        source_done_valid = {raw_tail_done_valid && !raw_completion_blocked,
                             ipd_done_valid, resident_done_valid};
        source_done_tag = {active_execution_tag_q,
                           active_execution_tag_q,
                           active_execution_tag_q};
        source_done_error = {
            raw_tail_done_error ||
                raw_tail_done_tag != expected_payload_tag_q,
            ipd_done_error || ipd_done_tag != expected_payload_tag_q,
            resident_done_error ||
                resident_done_tag != expected_payload_tag_q
        };
    end
    assign ipd_fill_entry_valid = ipd_term_valid && source_term_ready[1];
    assign ipd_fill_gate_code = ipd_term_gate;
    assign ipd_fill_lane_id = ipd_term_lane;
    assign ipd_fill_destination_count = ipd_term_destinations;
    assign ipd_fill_entry_last = ipd_term_head_last;
    assign ipd_fill_cache_allowed = active_format_q == FORMAT_IPD32W;

    always_comb begin
        selected_done_payload_tag = '0;
        selected_done_error = 1'b1;
        if (route_active_select == ROUTE_RESIDENT) begin
            selected_done_payload_tag = resident_done_tag;
            selected_done_error = source_done_error[0];
        end else if (route_active_select == ROUTE_IPD) begin
            selected_done_payload_tag = ipd_done_tag;
            selected_done_error = source_done_error[1];
        end else if (route_active_select == ROUTE_RAW) begin
            selected_done_payload_tag = raw_tail_done_tag;
            selected_done_error = source_done_error[2];
        end
    end
    assign source_done_fire = |(source_done_valid & source_done_ready);
    assign decoder_done_valid = decoder_done_valid_q;
    assign decoder_done_payload_tag = decoder_done_payload_tag_q;
    assign decoder_done_error = decoder_done_error_q;

    gatestack_resident_replay_joiner #(
        .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(RESIDENT_TERMS),
        .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W),
        .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .TERM_INDEX_W(RES_TERM_IDX_W), .EVENT_COUNT_W(EVENT_COUNT_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) u_resident_decoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .start_valid(resident_start_valid), .start_ready(resident_start_ready),
        .start_tag(head_start_payload_tag),
        .start_term_count(resident_term_count),
        .start_event_count(resident_event_count),
        .descriptor_valid(resident_descriptor_valid),
        .descriptor_ready(resident_descriptor_ready),
        .descriptor_gate_code(resident_descriptor_gate_code),
        .descriptor_lane_id(resident_descriptor_lane_id),
        .descriptor_destination_count(resident_descriptor_destination_count),
        .descriptor_term_index(resident_descriptor_term_index),
        .descriptor_last(resident_descriptor_last),
        .word_valid(resident_word_valid), .word_ready(resident_word_ready),
        .word_data(resident_word_data), .word_index(resident_word_index),
        .word_last(resident_word_last), .term_valid(resident_term_valid),
        .term_ready(resident_term_ready),
        .term_gate_code(resident_term_gate),
        .term_lane_id(resident_term_lane),
        .term_destination_count(resident_term_destinations),
        .term_index(resident_term_index),
        .term_head_last(resident_term_head_last),
        .event_valid(resident_event_valid),
        .event_ready(resident_event_ready),
        .event_gate_code(resident_event_gate),
        .event_lane_id(resident_event_lane),
        .event_token_valid(resident_event_token_valid),
        .event_token_ids(resident_event_token_ids),
        .event_count(resident_event_count_out),
        .event_term_first(resident_event_first),
        .event_term_last(resident_event_last),
        .event_head_last(resident_event_head_last),
        .done_valid(resident_done_valid), .done_ready(resident_done_ready),
        .done_tag(resident_done_tag), .done_error(resident_done_error),
        .protocol_error(resident_protocol_error),
        .count_heads(resident_count_heads),
        .count_terms(resident_count_terms),
        .count_events(resident_count_events),
        .count_descriptor_stall_cycles(resident_count_desc_stall),
        .count_input_stall_cycles(resident_count_input_stall),
        .count_term_stall_cycles(resident_count_term_stall),
        .count_output_stall_cycles(resident_count_output_stall)
    );

    generate
        if (CSR_FORMAT_FADC24 == 2) begin : g_adaptive_csr_decoder
            gatestack_adaptive_csr_replay_decoder #(
                .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(MAX_TERMS),
                .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W),
                .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
                .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
                .TERM_INDEX_W(IPD_TERM_IDX_W),
                .EVENT_COUNT_W(EVENT_COUNT_W), .WAY_COUNT_W(WAY_COUNT_W)
            ) u_adaptive_csr_decoder (
                .clk_core(clk_core), .rst_core(rst_core),
                .start_valid(ipd_start_valid), .start_ready(ipd_start_ready),
                .start_format_valid(1'b1),
                .start_select_fadc(head_start_csr_format == FORMAT_FADC24),
                .word_valid(ipd_word_valid), .word_ready(ipd_word_ready),
                .word_data(ipd_word_data), .word_index(ipd_word_index),
                .word_last(ipd_word_last),
                .descriptor_begin_valid(ipd_fill_begin_valid),
                .descriptor_begin_ready(ipd_fill_begin_ready),
                .descriptor_begin_tag(ipd_fill_begin_tag),
                .descriptor_begin_term_count(ipd_fill_begin_term_count),
                .term_valid(ipd_term_valid), .term_ready(ipd_term_ready),
                .term_gate_code(ipd_term_gate), .term_lane_id(ipd_term_lane),
                .term_destination_count(ipd_term_destinations),
                .term_index(ipd_term_index),
                .term_head_last(ipd_term_head_last),
                .event_valid(ipd_event_valid), .event_ready(ipd_event_ready),
                .event_gate_code(ipd_event_gate),
                .event_lane_id(ipd_event_lane),
                .event_token_valid(ipd_event_token_valid),
                .event_token_ids(ipd_event_token_ids),
                .event_count(ipd_event_count_out),
                .event_term_first(ipd_event_first),
                .event_term_last(ipd_event_last),
                .event_head_last(ipd_event_head_last),
                .done_valid(ipd_done_valid), .done_ready(ipd_done_ready),
                .done_tag(ipd_done_tag), .done_error(ipd_done_error),
                .protocol_error(ipd_protocol_error),
                .count_heads(ipd_count_heads), .count_terms(ipd_count_terms),
                .count_events(ipd_count_events),
                .count_input_stall_cycles(ipd_count_input_stall),
                .count_term_stall_cycles(ipd_count_term_stall),
                .count_output_stall_cycles(ipd_count_output_stall)
            );
        end else if (CSR_FORMAT_FADC24 == 1) begin : g_fadc24_decoder
            gatestack_fadc24_streaming_replay_decoder #(
                .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(MAX_TERMS),
                .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W),
                .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
                .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
                .TERM_INDEX_W(IPD_TERM_IDX_W),
                .EVENT_COUNT_W(EVENT_COUNT_W), .WAY_COUNT_W(WAY_COUNT_W)
            ) u_fadc24_decoder (
                .clk_core(clk_core), .rst_core(rst_core),
                .start_valid(ipd_start_valid), .start_ready(ipd_start_ready),
                .word_valid(ipd_word_valid), .word_ready(ipd_word_ready),
                .word_data(ipd_word_data), .word_index(ipd_word_index),
                .word_last(ipd_word_last),
                .descriptor_begin_valid(ipd_fill_begin_valid),
                .descriptor_begin_ready(ipd_fill_begin_ready),
                .descriptor_begin_tag(ipd_fill_begin_tag),
                .descriptor_begin_term_count(ipd_fill_begin_term_count),
                .term_valid(ipd_term_valid), .term_ready(ipd_term_ready),
                .term_gate_code(ipd_term_gate), .term_lane_id(ipd_term_lane),
                .term_destination_count(ipd_term_destinations),
                .term_index(ipd_term_index),
                .term_head_last(ipd_term_head_last),
                .event_valid(ipd_event_valid), .event_ready(ipd_event_ready),
                .event_gate_code(ipd_event_gate),
                .event_lane_id(ipd_event_lane),
                .event_token_valid(ipd_event_token_valid),
                .event_token_ids(ipd_event_token_ids),
                .event_count(ipd_event_count_out),
                .event_term_first(ipd_event_first),
                .event_term_last(ipd_event_last),
                .event_head_last(ipd_event_head_last),
                .done_valid(ipd_done_valid), .done_ready(ipd_done_ready),
                .done_tag(ipd_done_tag), .done_error(ipd_done_error),
                .protocol_error(ipd_protocol_error),
                .count_heads(ipd_count_heads), .count_terms(ipd_count_terms),
                .count_events(ipd_count_events),
                .count_bitmap_terms(fadc_count_bitmap_terms),
                .count_input_stall_cycles(ipd_count_input_stall),
                .count_term_stall_cycles(ipd_count_term_stall),
                .count_output_stall_cycles(ipd_count_output_stall)
            );
        end else begin : g_ipd32w_decoder
            gatestack_ipd32w_replay_decoder #(
                .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(MAX_TERMS),
                .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W),
                .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
                .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
                .TERM_INDEX_W(IPD_TERM_IDX_W),
                .EVENT_COUNT_W(EVENT_COUNT_W), .WAY_COUNT_W(WAY_COUNT_W)
            ) u_ipd32w_decoder (
                .clk_core(clk_core), .rst_core(rst_core),
                .start_valid(ipd_start_valid), .start_ready(ipd_start_ready),
                .word_valid(ipd_word_valid), .word_ready(ipd_word_ready),
                .word_data(ipd_word_data), .word_index(ipd_word_index),
                .word_last(ipd_word_last),
                .descriptor_begin_valid(ipd_fill_begin_valid),
                .descriptor_begin_ready(ipd_fill_begin_ready),
                .descriptor_begin_tag(ipd_fill_begin_tag),
                .descriptor_begin_term_count(ipd_fill_begin_term_count),
                .term_valid(ipd_term_valid), .term_ready(ipd_term_ready),
                .term_gate_code(ipd_term_gate), .term_lane_id(ipd_term_lane),
                .term_destination_count(ipd_term_destinations),
                .term_index(ipd_term_index),
                .term_head_last(ipd_term_head_last),
                .event_valid(ipd_event_valid), .event_ready(ipd_event_ready),
                .event_gate_code(ipd_event_gate),
                .event_lane_id(ipd_event_lane),
                .event_token_valid(ipd_event_token_valid),
                .event_token_ids(ipd_event_token_ids),
                .event_count(ipd_event_count_out),
                .event_term_first(ipd_event_first),
                .event_term_last(ipd_event_last),
                .event_head_last(ipd_event_head_last),
                .done_valid(ipd_done_valid), .done_ready(ipd_done_ready),
                .done_tag(ipd_done_tag), .done_error(ipd_done_error),
                .protocol_error(ipd_protocol_error),
                .count_heads(ipd_count_heads), .count_terms(ipd_count_terms),
                .count_events(ipd_count_events),
                .count_input_stall_cycles(ipd_count_input_stall),
                .count_term_stall_cycles(ipd_count_term_stall),
                .count_output_stall_cycles(ipd_count_output_stall)
            );
        end
    endgenerate

    gatestack_raw41_replay_decoder #(
        .TOKENS(TOKENS), .LANES(LANES), .TAG_W(TAG_W),
        .WORD_INDEX_W(WORD_INDEX_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W)
    ) u_raw_decoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .start_valid(raw_start_valid), .start_ready(raw_start_ready),
        .start_tag(head_start_payload_tag), .word_valid(raw_word_valid),
        .word_ready(raw_word_ready), .word_data(raw_word_data),
        .word_index(raw_word_index), .word_last(raw_word_last),
        .direct_valid(raw_direct_valid), .direct_ready(raw_direct_ready),
        .direct_gate_code(raw_direct_gate),
        .direct_lane_id(raw_direct_lane),
        .direct_token_id(raw_direct_token),
        .direct_head_last(raw_direct_head_last),
        .done_valid(raw_done_valid), .done_ready(raw_decoder_done_ready),
        .done_tag(raw_done_tag), .done_error(raw_done_error),
        .protocol_error(raw_protocol_error), .count_heads(raw_count_heads),
        .count_records(raw_count_records),
        .count_kzero_records(raw_count_kzero),
        .count_direct_events(raw_count_events),
        .count_input_stall_cycles(raw_count_input_stall),
        .count_output_stall_cycles(raw_count_output_stall)
    );

    gatestack_raw_tail_retimer #(
        .TAG_W(TAG_W), .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W), .COUNTER_W(COUNTER_W)
    ) u_raw_tail_retimer (
        .clk_core(clk_core), .rst_core(rst_core),
        .input_valid(raw_direct_valid), .input_ready(raw_direct_ready),
        .input_gate_code(raw_direct_gate), .input_lane_id(raw_direct_lane),
        .input_token_id(raw_direct_token),
        .input_done_valid(raw_done_valid),
        .input_done_ready(raw_decoder_done_ready),
        .input_done_tag(raw_done_tag), .input_done_error(raw_done_error),
        .output_valid(raw_tail_valid), .output_ready(raw_tail_ready),
        .output_gate_code(raw_tail_gate), .output_lane_id(raw_tail_lane),
        .output_token_id(raw_tail_token),
        .output_head_last(raw_tail_head_last),
        .output_done_valid(raw_tail_done_valid),
        .output_done_ready(raw_tail_done_ready),
        .output_done_tag(raw_tail_done_tag),
        .output_done_error(raw_tail_done_error),
        .protocol_error(raw_tail_protocol_error),
        .count_inputs(raw_tail_count_inputs),
        .count_outputs(raw_tail_count_outputs),
        .count_empty_sessions(raw_tail_count_empty)
    );

    gatestack_raw_issue_adapter #(
        .EVENT_WAYS(EVENT_WAYS), .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .WAY_COUNT_W(WAY_COUNT_W), .COUNTER_W(COUNTER_W)
    ) u_raw_adapter (
        .clk_core(clk_core), .rst_core(rst_core),
        .direct_valid(raw_tail_valid), .direct_ready(raw_tail_ready),
        .direct_gate_code(raw_tail_gate),
        .direct_lane_id(raw_tail_lane),
        .direct_token_id(raw_tail_token),
        .direct_head_last(raw_tail_head_last),
        .term_valid(raw_adapter_term_valid),
        .term_ready(raw_adapter_term_ready),
        .term_gate_code(raw_adapter_term_gate),
        .term_lane_id(raw_adapter_term_lane),
        .term_destination_count(raw_adapter_term_destinations),
        .term_issue_seq(raw_adapter_term_issue_seq),
        .term_head_last(raw_adapter_term_head_last),
        .event_valid(raw_adapter_event_valid),
        .event_ready(raw_adapter_event_ready),
        .event_gate_code(raw_adapter_event_gate),
        .event_lane_id(raw_adapter_event_lane),
        .event_token_valid(raw_adapter_event_token_valid),
        .event_token_ids(raw_adapter_event_token_ids),
        .event_count(raw_adapter_event_count),
        .event_issue_seq(raw_adapter_event_issue_seq),
        .event_term_first(raw_adapter_event_first),
        .event_term_last(raw_adapter_event_last),
        .event_head_last(raw_adapter_event_head_last),
        .count_direct_inputs(raw_adapter_count_inputs),
        .count_term_stall_cycles(raw_adapter_count_term_stall),
        .count_event_stall_cycles(raw_adapter_count_event_stall)
    );

    gatestack_routed_multihead_tile_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .SOURCES(3),
        .EVENT_WAYS(EVENT_WAYS), .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS), .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W), .ACC_W(ACC_W),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .TAG_W(TAG_W), .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .HEAD_COUNT_W(HEAD_COUNT_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W), .WAY_COUNT_W(WAY_COUNT_W),
        .ROUTE_W(ROUTE_W)
    ) u_routed_projection (
        .clk_core(clk_core), .rst_core(rst_core),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_start_valid(routed_head_valid),
        .head_start_ready(routed_head_ready),
        .head_start_tag(head_start_tag), .head_start_index(head_start_index),
        .head_start_route_select(head_start_route_select),
        .head_start_input_channel_base(head_start_input_channel_base),
        .head_start_last(head_start_last),
        .source_term_valid(source_term_valid),
        .source_term_ready(source_term_ready),
        .source_term_gate_code(source_term_gate_code),
        .source_term_lane_id(source_term_lane_id),
        .source_term_destination_count(source_term_destination_count),
        .source_term_head_last(source_term_head_last),
        .source_event_valid(source_event_valid),
        .source_event_ready(source_event_ready),
        .source_event_gate_code(source_event_gate_code),
        .source_event_lane_id(source_event_lane_id),
        .source_event_token_valid(source_event_token_valid),
        .source_event_token_ids(source_event_token_ids),
        .source_event_count(source_event_count),
        .source_event_term_first(source_event_term_first),
        .source_event_term_last(source_event_term_last),
        .source_event_head_last(source_event_head_last),
        .source_done_valid(source_done_valid),
        .source_done_ready(source_done_ready),
        .source_done_tag(source_done_tag),
        .source_done_error(source_done_error),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready), .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .bias_req_valid(bias_req_valid), .bias_req_ready(bias_req_ready),
        .bias_req_tag(bias_req_tag),
        .bias_req_output_tile(bias_req_output_tile),
        .bias_req_token_id(bias_req_token_id),
        .bias_rsp_valid(bias_rsp_valid), .bias_rsp_ready(bias_rsp_ready),
        .bias_rsp_tag(bias_rsp_tag),
        .bias_rsp_token_id(bias_rsp_token_id),
        .bias_rsp_values(bias_rsp_values), .final_valid(final_valid),
        .final_ready(final_ready), .final_token_ids(final_token_ids),
        .final_tag(final_tag), .final_values(final_values),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag), .head_done_index(head_done_index),
        .head_done_last(head_done_last), .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag),
        .route_active(route_active),
        .route_active_select(route_active_select),
        .protocol_error(routed_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_heads(count_heads), .count_terms(count_terms),
        .count_completed_terms(count_completed_terms),
        .count_bias_commits(count_bias_commits)
    );

    assign protocol_error = routed_protocol_error || resident_protocol_error ||
        ipd_protocol_error || raw_protocol_error || raw_tail_protocol_error ||
        (head_start_valid &&
         (32'(head_start_route_select) >= 3 || !head_start_format_legal));

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            raw_adapter_busy_q <= 1'b0;
            expected_payload_tag_q <= '0;
            active_execution_tag_q <= '0;
            active_format_q <= FORMAT_RAW;
            decoder_done_valid_q <= 1'b0;
            decoder_done_payload_tag_q <= '0;
            decoder_done_error_q <= 1'b0;
        end else begin
            if (decoder_done_valid_q && decoder_done_ready)
                decoder_done_valid_q <= 1'b0;
            if (head_start_fire) begin
                expected_payload_tag_q <= head_start_payload_tag;
                active_execution_tag_q <= head_start_tag;
                active_format_q <= head_start_csr_format;
            end
            if (source_done_fire) begin
                decoder_done_valid_q <= 1'b1;
                decoder_done_payload_tag_q <= selected_done_payload_tag;
                decoder_done_error_q <= selected_done_error;
            end
            if (raw_start_valid && raw_start_ready)
                raw_adapter_busy_q <= 1'b0;
            if (raw_tail_fire)
                raw_adapter_busy_q <= 1'b1;
            if (raw_adapter_event_fire)
                raw_adapter_busy_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
