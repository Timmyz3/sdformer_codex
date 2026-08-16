`timescale 1ns/1ps
`default_nettype none

// Side-effect-free metadata planner. It inspects slot/cache state and emits an
// immutable replay plan; decoder, slot replay and backend start belong to the
// downstream atomic commit barrier.
module gatestack_replay_plan_builder #(
    parameter int CONTEXTS        = 2,
    parameter int HEADS           = 24,
    parameter int HEAD_BITS       = 6642,
    parameter int SLOT_CAPACITY_BITS = ((HEAD_BITS + 63) / 64) * 64,
    parameter int RESIDENT_TERMS  = 80,
    parameter int ENABLE_RESIDENCY = 1,
    // 0: IPD32W-only, 1: FADC24-only, 2: runtime typed IPD32W/FADC24.
    parameter int CSR_FORMAT_FADC24 = 0,
    parameter int TAG_W           = 32,
    parameter int SIZE_W          = 16,
    parameter int FORMAT_W        = 2,
    parameter int EVENT_COUNT_W   = 13,
    parameter int WORD_INDEX_W    = 7,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int HEAD_COUNT_W    = 6,
    parameter int COUNTER_W       = 32,
    parameter int CONTEXT_ID_W    = (CONTEXTS <= 1) ?
                                     1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W       = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int ROUTE_W         = 2
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [CONTEXT_ID_W-1:0]      request_context_id,
    input  logic [HEAD_ID_W-1:0]         request_head_id,
    input  logic [TAG_W-1:0]             request_execution_tag,
    input  logic [HEAD_COUNT_W-1:0]      request_head_index,
    input  logic [INPUT_CH_W-1:0]        request_input_channel_base,
    input  logic [OUTPUT_TILE_W-1:0]     request_output_tile,
    input  logic                         request_last_head,
    input  logic                         request_last_output_tile,

    output logic                         slot_inspect_valid,
    input  logic                         slot_inspect_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_inspect_context_id,
    output logic [HEAD_ID_W-1:0]         slot_inspect_head_id,
    input  logic                         slot_meta_valid,
    output logic                         slot_meta_ready,
    input  logic                         slot_meta_exists,
    input  logic [TAG_W-1:0]             slot_meta_tag,
    input  logic                         slot_meta_mode_is_csr,
    input  logic [FORMAT_W-1:0]          slot_meta_format,
    input  logic [SIZE_W-1:0]            slot_meta_payload_bits,
    input  logic [SIZE_W-1:0]            slot_meta_word_count,

    output logic                         cache_lookup_valid,
    input  logic                         cache_lookup_ready,
    output logic [CONTEXT_ID_W-1:0]      cache_lookup_context_id,
    output logic [HEAD_ID_W-1:0]         cache_lookup_head_id,
    output logic [TAG_W-1:0]             cache_lookup_expected_tag,
    input  logic                         cache_meta_valid,
    output logic                         cache_meta_ready,
    input  logic                         cache_meta_hit,
    input  logic [TAG_W-1:0]             cache_meta_tag,
    input  logic [7:0]                   cache_meta_term_count,

    output logic                         plan_valid,
    input  logic                         plan_ready,
    output logic [CONTEXT_ID_W-1:0]      plan_context_id,
    output logic [HEAD_ID_W-1:0]         plan_head_id,
    output logic [TAG_W-1:0]             plan_payload_tag,
    output logic [TAG_W-1:0]             plan_execution_tag,
    output logic [ROUTE_W-1:0]           plan_route,
    output logic [FORMAT_W-1:0]          plan_format,
    output logic [HEAD_COUNT_W-1:0]      plan_head_index,
    output logic [INPUT_CH_W-1:0]        plan_input_channel_base,
    output logic [OUTPUT_TILE_W-1:0]     plan_output_tile,
    output logic                         plan_last_head,
    output logic                         plan_last_output_tile,
    output logic                         plan_cache_owned,
    output logic                         plan_slot_replay_required,
    output logic [WORD_INDEX_W-1:0]      plan_replay_start_word,
    output logic [7:0]                   plan_resident_term_count,
    output logic [EVENT_COUNT_W-1:0]     plan_resident_event_count,

    output logic                         reject_valid,
    input  logic                         reject_ready,
    output logic [TAG_W-1:0]             reject_payload_tag,
    output logic [TAG_W-1:0]             reject_execution_tag,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_requests,
    output logic [COUNTER_W-1:0]         count_resident_plans,
    output logic [COUNTER_W-1:0]         count_ipd_plans,
    output logic [COUNTER_W-1:0]         count_fadc_plans,
    output logic [COUNTER_W-1:0]         count_raw_plans,
    output logic [COUNTER_W-1:0]         count_rejects
);
    localparam int WORDS_PER_HEAD = (HEAD_BITS + 63) / 64;
    localparam logic [ROUTE_W-1:0] ROUTE_RESIDENT = ROUTE_W'(0);
    localparam logic [ROUTE_W-1:0] ROUTE_IPD = ROUTE_W'(1);
    localparam logic [ROUTE_W-1:0] ROUTE_RAW = ROUTE_W'(2);
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_SLOT_REQUEST,
        ST_SLOT_META,
        ST_CACHE_REQUEST,
        ST_CACHE_META,
        ST_PLAN,
        ST_REJECT
    } state_t;

    state_t state_q;
    logic [CONTEXT_ID_W-1:0] context_q;
    logic [HEAD_ID_W-1:0] head_q;
    logic [TAG_W-1:0] execution_tag_q;
    logic [HEAD_COUNT_W-1:0] head_index_q;
    logic [INPUT_CH_W-1:0] input_channel_base_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic last_head_q, last_output_tile_q;
    logic [TAG_W-1:0] payload_tag_q;
    logic [SIZE_W-1:0] payload_bits_q;
    logic [SIZE_W-1:0] word_count_q;
    logic [ROUTE_W-1:0] route_q;
    logic [FORMAT_W-1:0] format_q;
    logic cache_owned_q, slot_required_q;
    logic [WORD_INDEX_W-1:0] replay_start_word_q;
    logic [7:0] resident_term_count_q;
    logic [EVENT_COUNT_W-1:0] resident_event_count_q;

    logic request_context_legal, request_head_legal;
    logic request_fire, slot_inspect_fire, slot_meta_fire;
    logic cache_lookup_fire, cache_meta_fire, plan_fire, reject_fire;
    logic format_contract_ok, decoder_format_supported;
    logic raw_contract_ok, resident_contract_ok;
    logic [31:0] token_start_word_comb;
    logic [31:0] token_offset_bytes_comb;
    logic [31:0] payload_bytes_comb;
    logic [31:0] event_count_comb;

    assign request_context_legal =
        32'(request_context_id) < 32'(CONTEXTS);
    assign request_head_legal = 32'(request_head_id) < 32'(HEADS) &&
                                32'(request_head_index) < 32'(HEADS);
    assign request_ready = state_q == ST_IDLE &&
                           request_context_legal && request_head_legal;
    assign request_fire = request_valid && request_ready;

    assign slot_inspect_valid = state_q == ST_SLOT_REQUEST;
    assign slot_inspect_context_id = context_q;
    assign slot_inspect_head_id = head_q;
    assign slot_inspect_fire = slot_inspect_valid && slot_inspect_ready;
    assign slot_meta_ready = state_q == ST_SLOT_META;
    assign slot_meta_fire = slot_meta_valid && slot_meta_ready;

    assign cache_lookup_valid = state_q == ST_CACHE_REQUEST;
    assign cache_lookup_context_id = context_q;
    assign cache_lookup_head_id = head_q;
    assign cache_lookup_expected_tag = payload_tag_q;
    assign cache_lookup_fire = cache_lookup_valid && cache_lookup_ready;
    assign cache_meta_ready = state_q == ST_CACHE_META;
    assign cache_meta_fire = cache_meta_valid && cache_meta_ready;

    assign token_start_word_comb = 32'd2 +
        ((32'(cache_meta_term_count) + 32'd1) >> 1);
    assign token_offset_bytes_comb = token_start_word_comb << 3;
    assign payload_bytes_comb = 32'(payload_bits_q) >> 3;
    assign event_count_comb = payload_bytes_comb - token_offset_bytes_comb;
    assign raw_contract_ok = 32'(slot_meta_payload_bits) ==
                             32'(HEAD_BITS) &&
                             32'(slot_meta_word_count) ==
                             32'(WORDS_PER_HEAD);
    assign format_contract_ok =
        (slot_meta_format == FORMAT_RAW && !slot_meta_mode_is_csr) ||
        ((slot_meta_format == FORMAT_IPD32W ||
          slot_meta_format == FORMAT_FADC24) && slot_meta_mode_is_csr);
    assign decoder_format_supported =
        slot_meta_format == FORMAT_RAW ||
        (CSR_FORMAT_FADC24 == 0 && slot_meta_format == FORMAT_IPD32W) ||
        (CSR_FORMAT_FADC24 == 1 && slot_meta_format == FORMAT_FADC24) ||
        (CSR_FORMAT_FADC24 == 2 &&
         (slot_meta_format == FORMAT_IPD32W ||
          slot_meta_format == FORMAT_FADC24));
    assign resident_contract_ok = cache_meta_hit &&
        cache_meta_tag == payload_tag_q &&
        32'(cache_meta_term_count) <= 32'(RESIDENT_TERMS) &&
        payload_bits_q[2:0] == 3'b000 &&
        32'(payload_bits_q) <= 32'(SLOT_CAPACITY_BITS) &&
        payload_bytes_comb >= token_offset_bytes_comb &&
        event_count_comb < (32'(1) << EVENT_COUNT_W) &&
        ((cache_meta_term_count == '0) == (event_count_comb == '0)) &&
        token_start_word_comb <= 32'(word_count_q) &&
        (event_count_comb == '0 ||
         token_start_word_comb < 32'(word_count_q));

    assign plan_valid = state_q == ST_PLAN;
    assign plan_fire = plan_valid && plan_ready;
    assign plan_context_id = context_q;
    assign plan_head_id = head_q;
    assign plan_payload_tag = payload_tag_q;
    assign plan_execution_tag = execution_tag_q;
    assign plan_route = route_q;
    assign plan_format = format_q;
    assign plan_head_index = head_index_q;
    assign plan_input_channel_base = input_channel_base_q;
    assign plan_output_tile = output_tile_q;
    assign plan_last_head = last_head_q;
    assign plan_last_output_tile = last_output_tile_q;
    assign plan_cache_owned = cache_owned_q;
    assign plan_slot_replay_required = slot_required_q;
    assign plan_replay_start_word = replay_start_word_q;
    assign plan_resident_term_count = resident_term_count_q;
    assign plan_resident_event_count = resident_event_count_q;

    assign reject_valid = state_q == ST_REJECT;
    assign reject_payload_tag = payload_tag_q;
    assign reject_execution_tag = execution_tag_q;
    assign reject_fire = reject_valid && reject_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            head_q <= '0;
            execution_tag_q <= '0;
            head_index_q <= '0;
            input_channel_base_q <= '0;
            output_tile_q <= '0;
            last_head_q <= 1'b0;
            last_output_tile_q <= 1'b0;
            payload_tag_q <= '0;
            payload_bits_q <= '0;
            word_count_q <= '0;
            route_q <= ROUTE_RESIDENT;
            format_q <= FORMAT_RAW;
            cache_owned_q <= 1'b0;
            slot_required_q <= 1'b0;
            replay_start_word_q <= '0;
            resident_term_count_q <= '0;
            resident_event_count_q <= '0;
            protocol_error <= 1'b0;
            count_requests <= '0;
            count_resident_plans <= '0;
            count_ipd_plans <= '0;
            count_fadc_plans <= '0;
            count_raw_plans <= '0;
            count_rejects <= '0;
        end else begin
            if (request_valid && state_q == ST_IDLE &&
                (!request_context_legal || !request_head_legal))
                protocol_error <= 1'b1;

            if (request_fire) begin
                context_q <= request_context_id;
                head_q <= request_head_id;
                execution_tag_q <= request_execution_tag;
                head_index_q <= request_head_index;
                input_channel_base_q <= request_input_channel_base;
                output_tile_q <= request_output_tile;
                last_head_q <= request_last_head;
                last_output_tile_q <= request_last_output_tile;
                count_requests <= count_requests + 1'b1;
                state_q <= ST_SLOT_REQUEST;
            end
            if (slot_inspect_fire)
                state_q <= ST_SLOT_META;

            if (slot_meta_fire) begin
                payload_tag_q <= slot_meta_tag;
                payload_bits_q <= slot_meta_payload_bits;
                word_count_q <= slot_meta_word_count;
                format_q <= slot_meta_format;
                if (!slot_meta_exists) begin
                    protocol_error <= 1'b1;
                    state_q <= ST_REJECT;
                end else if (!format_contract_ok) begin
                    protocol_error <= 1'b1;
                    state_q <= ST_REJECT;
                end else if (!decoder_format_supported) begin
                    protocol_error <= 1'b1;
                    state_q <= ST_REJECT;
                end else if (slot_meta_format == FORMAT_IPD32W &&
                             ENABLE_RESIDENCY != 0) begin
                    state_q <= ST_CACHE_REQUEST;
                end else if (slot_meta_format == FORMAT_IPD32W) begin
                    route_q <= ROUTE_IPD;
                    cache_owned_q <= 1'b0;
                    slot_required_q <= 1'b1;
                    replay_start_word_q <= '0;
                    resident_term_count_q <= '0;
                    resident_event_count_q <= '0;
                    count_ipd_plans <= count_ipd_plans + 1'b1;
                    state_q <= ST_PLAN;
                end else if (slot_meta_format == FORMAT_FADC24) begin
                    route_q <= ROUTE_IPD;
                    cache_owned_q <= 1'b0;
                    slot_required_q <= 1'b1;
                    replay_start_word_q <= '0;
                    resident_term_count_q <= '0;
                    resident_event_count_q <= '0;
                    count_fadc_plans <= count_fadc_plans + 1'b1;
                    state_q <= ST_PLAN;
                end else if (!raw_contract_ok) begin
                    protocol_error <= 1'b1;
                    state_q <= ST_REJECT;
                end else begin
                    route_q <= ROUTE_RAW;
                    cache_owned_q <= 1'b0;
                    slot_required_q <= 1'b1;
                    replay_start_word_q <= '0;
                    resident_term_count_q <= '0;
                    resident_event_count_q <= '0;
                    count_raw_plans <= count_raw_plans + 1'b1;
                    state_q <= ST_PLAN;
                end
            end
            if (cache_lookup_fire)
                state_q <= ST_CACHE_META;

            if (cache_meta_fire) begin
                if (cache_meta_hit) begin
                    if (!resident_contract_ok) begin
                        protocol_error <= 1'b1;
                        state_q <= ST_REJECT;
                    end else begin
                        route_q <= ROUTE_RESIDENT;
                        cache_owned_q <= 1'b1;
                        slot_required_q <= event_count_comb != '0;
                        replay_start_word_q <=
                            WORD_INDEX_W'(token_start_word_comb);
                        resident_term_count_q <= cache_meta_term_count;
                        resident_event_count_q <=
                            EVENT_COUNT_W'(event_count_comb);
                        count_resident_plans <=
                            count_resident_plans + 1'b1;
                        state_q <= ST_PLAN;
                    end
                end else begin
                    route_q <= ROUTE_IPD;
                    cache_owned_q <= 1'b0;
                    slot_required_q <= 1'b1;
                    replay_start_word_q <= '0;
                    resident_term_count_q <= '0;
                    resident_event_count_q <= '0;
                    count_ipd_plans <= count_ipd_plans + 1'b1;
                    state_q <= ST_PLAN;
                end
            end

            if (plan_fire)
                state_q <= ST_IDLE;
            if (reject_fire) begin
                count_rejects <= count_rejects + 1'b1;
                state_q <= ST_IDLE;
            end
        end
    end
endmodule

`default_nettype wire
