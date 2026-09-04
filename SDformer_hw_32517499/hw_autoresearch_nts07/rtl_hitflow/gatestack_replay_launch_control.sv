`timescale 1ns/1ps
`default_nettype none

// Queries slot metadata before cache lookup, validates the selected format and
// launches exactly one resident, sequential IPD32W or RAW41 replay session.
module gatestack_replay_launch_control #(
    parameter int CONTEXTS       = 2,
    parameter int HEADS          = 24,
    parameter int HEAD_BITS      = 6642,
    parameter int TAG_W          = 32,
    parameter int SIZE_W         = 16,
    parameter int EVENT_COUNT_W  = 13,
    parameter int WORD_INDEX_W   = 7,
    parameter int COUNTER_W      = 32,
    parameter int CONTEXT_ID_W   = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W      = (HEADS <= 1) ? 1 : $clog2(HEADS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         launch_valid,
    output logic                         launch_ready,
    input  logic [CONTEXT_ID_W-1:0]      launch_context_id,
    input  logic [HEAD_ID_W-1:0]         launch_head_id,

    output logic                         launch_done_valid,
    input  logic                         launch_done_ready,
    output logic [1:0]                   launch_done_route,
    output logic [TAG_W-1:0]             launch_done_tag,
    output logic                         launch_done_error,

    output logic                         slot_inspect_valid,
    input  logic                         slot_inspect_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_inspect_context_id,
    output logic [HEAD_ID_W-1:0]         slot_inspect_head_id,
    input  logic                         slot_meta_valid,
    output logic                         slot_meta_ready,
    input  logic                         slot_meta_exists,
    input  logic [TAG_W-1:0]             slot_meta_tag,
    input  logic                         slot_meta_mode_is_csr,
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

    output logic                         slot_replay_begin_valid,
    input  logic                         slot_replay_begin_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_replay_context_id,
    output logic [HEAD_ID_W-1:0]         slot_replay_head_id,
    output logic [WORD_INDEX_W-1:0]      slot_replay_start_word,

    output logic                         resident_start_valid,
    input  logic                         resident_start_ready,
    output logic [TAG_W-1:0]             resident_start_tag,
    output logic [7:0]                   resident_start_term_count,
    output logic [EVENT_COUNT_W-1:0]     resident_start_event_count,

    output logic                         ipd_start_valid,
    input  logic                         ipd_start_ready,
    output logic                         raw_start_valid,
    input  logic                         raw_start_ready,
    output logic [TAG_W-1:0]             raw_start_tag,

    output logic                         route_start_valid,
    input  logic                         route_start_ready,
    output logic [1:0]                   route_start_select,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_launches,
    output logic [COUNTER_W-1:0]         count_resident_launches,
    output logic [COUNTER_W-1:0]         count_ipd_launches,
    output logic [COUNTER_W-1:0]         count_raw_launches,
    output logic [COUNTER_W-1:0]         count_launch_errors
);
    localparam int WORDS_PER_HEAD = (HEAD_BITS + 63) / 64;
    localparam logic [1:0] ROUTE_RESIDENT = 2'd0;
    localparam logic [1:0] ROUTE_IPD = 2'd1;
    localparam logic [1:0] ROUTE_RAW = 2'd2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_SLOT_REQUEST,
        ST_SLOT_META,
        ST_CACHE_REQUEST,
        ST_CACHE_META,
        ST_START,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [CONTEXT_ID_W-1:0] context_q;
    logic [HEAD_ID_W-1:0] head_q;
    logic [TAG_W-1:0] slot_tag_q;
    logic [SIZE_W-1:0] slot_payload_bits_q;
    logic [SIZE_W-1:0] slot_word_count_q;
    logic [1:0] route_q;
    logic [7:0] resident_term_count_q;
    logic [EVENT_COUNT_W-1:0] resident_event_count_q;
    logic [WORD_INDEX_W-1:0] replay_start_word_q;
    logic route_pending_q;
    logic decoder_pending_q;
    logic slot_pending_q;
    logic launch_error_q;

    logic launch_fire;
    logic slot_inspect_fire;
    logic slot_meta_fire;
    logic cache_lookup_fire;
    logic cache_meta_fire;
    logic route_start_fire;
    logic decoder_start_fire;
    logic slot_replay_fire;
    logic launch_done_fire;
    logic [31:0] token_start_word_comb;
    logic [31:0] token_offset_bytes_comb;
    logic [31:0] payload_bytes_comb;
    logic [31:0] event_count_comb;
    logic resident_contract_ok;
    logic raw_contract_ok;
    logic all_start_pending_clear;

    assign launch_ready = state_q == ST_IDLE;
    assign launch_fire = launch_valid && launch_ready;
    assign slot_inspect_valid = state_q == ST_SLOT_REQUEST;
    assign slot_inspect_context_id = context_q;
    assign slot_inspect_head_id = head_q;
    assign slot_inspect_fire = slot_inspect_valid && slot_inspect_ready;
    assign slot_meta_ready = state_q == ST_SLOT_META;
    assign slot_meta_fire = slot_meta_valid && slot_meta_ready;
    assign cache_lookup_valid = state_q == ST_CACHE_REQUEST;
    assign cache_lookup_context_id = context_q;
    assign cache_lookup_head_id = head_q;
    assign cache_lookup_expected_tag = slot_tag_q;
    assign cache_lookup_fire = cache_lookup_valid && cache_lookup_ready;
    assign cache_meta_ready = state_q == ST_CACHE_META;
    assign cache_meta_fire = cache_meta_valid && cache_meta_ready;

    assign slot_replay_begin_valid = state_q == ST_START && slot_pending_q;
    assign slot_replay_context_id = context_q;
    assign slot_replay_head_id = head_q;
    assign slot_replay_start_word = replay_start_word_q;
    assign slot_replay_fire = slot_replay_begin_valid &&
                              slot_replay_begin_ready;
    assign resident_start_valid = state_q == ST_START && decoder_pending_q &&
                                  route_q == ROUTE_RESIDENT;
    assign ipd_start_valid = state_q == ST_START && decoder_pending_q &&
                             route_q == ROUTE_IPD;
    assign raw_start_valid = state_q == ST_START && decoder_pending_q &&
                             route_q == ROUTE_RAW;
    assign decoder_start_fire =
        (resident_start_valid && resident_start_ready) ||
        (ipd_start_valid && ipd_start_ready) ||
        (raw_start_valid && raw_start_ready);
    assign resident_start_tag = slot_tag_q;
    assign resident_start_term_count = resident_term_count_q;
    assign resident_start_event_count = resident_event_count_q;
    assign raw_start_tag = slot_tag_q;
    assign route_start_valid = state_q == ST_START && route_pending_q;
    assign route_start_select = route_q;
    assign route_start_fire = route_start_valid && route_start_ready;

    assign launch_done_valid = state_q == ST_DONE;
    assign launch_done_route = route_q;
    assign launch_done_tag = slot_tag_q;
    assign launch_done_error = launch_error_q;
    assign launch_done_fire = launch_done_valid && launch_done_ready;

    assign token_start_word_comb = 32'd2 +
        ((32'(cache_meta_term_count) + 1) >> 1);
    assign token_offset_bytes_comb = token_start_word_comb << 3;
    assign payload_bytes_comb = 32'(slot_payload_bits_q) >> 3;
    assign event_count_comb = payload_bytes_comb - token_offset_bytes_comb;
    assign resident_contract_ok = cache_meta_hit &&
        cache_meta_tag == slot_tag_q &&
        32'(cache_meta_term_count) <= 80 &&
        slot_payload_bits_q[2:0] == 0 &&
        32'(slot_payload_bits_q) <= HEAD_BITS &&
        payload_bytes_comb >= token_offset_bytes_comb &&
        event_count_comb < (1 << EVENT_COUNT_W) &&
        ((cache_meta_term_count == 0) == (event_count_comb == 0)) &&
        token_start_word_comb <= 32'(slot_word_count_q) &&
        (event_count_comb == 0 ||
         token_start_word_comb < 32'(slot_word_count_q));
    assign raw_contract_ok = 32'(slot_meta_payload_bits) == HEAD_BITS &&
                             32'(slot_meta_word_count) == WORDS_PER_HEAD;
    assign all_start_pending_clear =
        (!route_pending_q || route_start_fire) &&
        (!decoder_pending_q || decoder_start_fire) &&
        (!slot_pending_q || slot_replay_fire);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            head_q <= '0;
            slot_tag_q <= '0;
            slot_payload_bits_q <= '0;
            slot_word_count_q <= '0;
            route_q <= ROUTE_RESIDENT;
            resident_term_count_q <= '0;
            resident_event_count_q <= '0;
            replay_start_word_q <= '0;
            route_pending_q <= 1'b0;
            decoder_pending_q <= 1'b0;
            slot_pending_q <= 1'b0;
            launch_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_launches <= '0;
            count_resident_launches <= '0;
            count_ipd_launches <= '0;
            count_raw_launches <= '0;
            count_launch_errors <= '0;
        end else begin
            if (launch_fire) begin
                context_q <= launch_context_id;
                head_q <= launch_head_id;
                slot_tag_q <= '0;
                slot_payload_bits_q <= '0;
                slot_word_count_q <= '0;
                route_q <= ROUTE_RESIDENT;
                launch_error_q <= 1'b0;
                count_launches <= count_launches + 1'b1;
                state_q <= ST_SLOT_REQUEST;
            end
            if (slot_inspect_fire) begin
                state_q <= ST_SLOT_META;
            end
            if (slot_meta_fire) begin
                slot_tag_q <= slot_meta_tag;
                slot_payload_bits_q <= slot_meta_payload_bits;
                slot_word_count_q <= slot_meta_word_count;
                if (!slot_meta_exists) begin
                    launch_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    count_launch_errors <= count_launch_errors + 1'b1;
                    state_q <= ST_DONE;
                end else if (slot_meta_mode_is_csr) begin
                    state_q <= ST_CACHE_REQUEST;
                end else if (!raw_contract_ok) begin
                    launch_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                    count_launch_errors <= count_launch_errors + 1'b1;
                    state_q <= ST_DONE;
                end else begin
                    route_q <= ROUTE_RAW;
                    replay_start_word_q <= '0;
                    route_pending_q <= 1'b1;
                    decoder_pending_q <= 1'b1;
                    slot_pending_q <= 1'b1;
                    state_q <= ST_START;
                end
            end
            if (cache_lookup_fire) begin
                state_q <= ST_CACHE_META;
            end
            if (cache_meta_fire) begin
                route_pending_q <= 1'b1;
                decoder_pending_q <= 1'b1;
                if (cache_meta_hit) begin
                    if (!resident_contract_ok) begin
                        launch_error_q <= 1'b1;
                        protocol_error <= 1'b1;
                        count_launch_errors <= count_launch_errors + 1'b1;
                        route_pending_q <= 1'b0;
                        decoder_pending_q <= 1'b0;
                        slot_pending_q <= 1'b0;
                        state_q <= ST_DONE;
                    end else begin
                        route_q <= ROUTE_RESIDENT;
                        resident_term_count_q <= cache_meta_term_count;
                        resident_event_count_q <=
                            EVENT_COUNT_W'(event_count_comb);
                        replay_start_word_q <=
                            WORD_INDEX_W'(token_start_word_comb);
                        slot_pending_q <= event_count_comb != 0;
                        state_q <= ST_START;
                    end
                end else begin
                    route_q <= ROUTE_IPD;
                    replay_start_word_q <= '0;
                    slot_pending_q <= 1'b1;
                    state_q <= ST_START;
                end
            end
            if (state_q == ST_START) begin
                if (route_start_fire) route_pending_q <= 1'b0;
                if (decoder_start_fire) decoder_pending_q <= 1'b0;
                if (slot_replay_fire) slot_pending_q <= 1'b0;
                if (all_start_pending_clear) begin
                    unique case (route_q)
                        ROUTE_RESIDENT: count_resident_launches <=
                            count_resident_launches + 1'b1;
                        ROUTE_IPD: count_ipd_launches <=
                            count_ipd_launches + 1'b1;
                        ROUTE_RAW: count_raw_launches <=
                            count_raw_launches + 1'b1;
                        default: protocol_error <= 1'b1;
                    endcase
                    state_q <= ST_DONE;
                end
            end
            if (launch_done_fire) begin
                state_q <= ST_IDLE;
            end
        end
    end
endmodule

`default_nettype wire
