`timescale 1ns/1ps
`default_nettype none

// Separates persistent payload identity from per-replay execution identity.
// Decoder completion checks payload_tag; backend completion checks execution_tag.
module gatestack_dualtag_replay_lifecycle_manager #(
    parameter int CONTEXTS       = 2,
    parameter int HEADS          = 24,
    parameter int TAG_W          = 32,
    parameter int COUNTER_W      = 32,
    parameter int CONTEXT_ID_W   = (CONTEXTS <= 1) ?
                                    1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W      = (HEADS <= 1) ? 1 : $clog2(HEADS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         session_valid,
    output logic                         session_ready,
    input  logic [CONTEXT_ID_W-1:0]      session_context_id,
    input  logic [HEAD_ID_W-1:0]         session_head_id,
    input  logic [TAG_W-1:0]             session_payload_tag,
    input  logic [TAG_W-1:0]             session_execution_tag,
    input  logic                         session_cache_owned,
    input  logic                         session_last_output_tile,

    input  logic                         decoder_done_valid,
    output logic                         decoder_done_ready,
    input  logic [TAG_W-1:0]             decoder_done_payload_tag,
    input  logic                         decoder_done_error,
    input  logic                         backend_done_valid,
    output logic                         backend_done_ready,
    input  logic [TAG_W-1:0]             backend_done_execution_tag,
    input  logic                         backend_done_error,

    output logic                         slot_release_valid,
    input  logic                         slot_release_ready,
    output logic [CONTEXT_ID_W-1:0]      slot_release_context_id,
    output logic [HEAD_ID_W-1:0]         slot_release_head_id,
    output logic                         cache_release_valid,
    input  logic                         cache_release_ready,
    output logic [CONTEXT_ID_W-1:0]      cache_release_context_id,
    output logic [HEAD_ID_W-1:0]         cache_release_head_id,
    output logic [TAG_W-1:0]             cache_release_payload_tag,

    output logic                         session_done_valid,
    input  logic                         session_done_ready,
    output logic [TAG_W-1:0]             session_done_payload_tag,
    output logic [TAG_W-1:0]             session_done_execution_tag,
    output logic                         session_done_error,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_sessions,
    output logic [COUNTER_W-1:0]         count_final_tile_releases,
    output logic [COUNTER_W-1:0]         count_cache_releases,
    output logic [COUNTER_W-1:0]         count_session_errors
);
    typedef enum logic [1:0] {ST_IDLE, ST_WAIT, ST_RELEASE, ST_DONE} state_t;
    state_t state_q;
    logic [CONTEXT_ID_W-1:0] context_q;
    logic [HEAD_ID_W-1:0] head_q;
    logic [TAG_W-1:0] payload_tag_q, execution_tag_q;
    logic cache_owned_q, last_tile_q;
    logic decoder_seen_q, backend_seen_q;
    logic slot_release_pending_q, cache_release_pending_q;
    logic session_error_q;
    logic session_fire, decoder_fire, backend_fire;
    logic slot_release_fire, cache_release_fire, done_fire;
    logic decoder_mismatch, backend_mismatch, both_done_comb;
    logic completion_error_comb;

    assign session_ready = state_q == ST_IDLE;
    assign session_fire = session_valid && session_ready;
    assign decoder_done_ready = state_q == ST_WAIT && !decoder_seen_q;
    assign backend_done_ready = state_q == ST_WAIT && !backend_seen_q;
    assign decoder_fire = decoder_done_valid && decoder_done_ready;
    assign backend_fire = backend_done_valid && backend_done_ready;
    assign decoder_mismatch = decoder_done_payload_tag != payload_tag_q ||
                              decoder_done_error;
    assign backend_mismatch =
        backend_done_execution_tag != execution_tag_q || backend_done_error;
    assign completion_error_comb = session_error_q ||
        (decoder_fire && decoder_mismatch) ||
        (backend_fire && backend_mismatch);
    assign both_done_comb = (decoder_seen_q || decoder_fire) &&
                            (backend_seen_q || backend_fire);

    assign slot_release_valid = state_q == ST_RELEASE &&
                                slot_release_pending_q;
    assign cache_release_valid = state_q == ST_RELEASE &&
                                 cache_release_pending_q;
    assign slot_release_context_id = context_q;
    assign slot_release_head_id = head_q;
    assign cache_release_context_id = context_q;
    assign cache_release_head_id = head_q;
    assign cache_release_payload_tag = payload_tag_q;
    assign slot_release_fire = slot_release_valid && slot_release_ready;
    assign cache_release_fire = cache_release_valid && cache_release_ready;

    assign session_done_valid = state_q == ST_DONE;
    assign session_done_payload_tag = payload_tag_q;
    assign session_done_execution_tag = execution_tag_q;
    assign session_done_error = session_error_q;
    assign done_fire = session_done_valid && session_done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            context_q <= '0;
            head_q <= '0;
            payload_tag_q <= '0;
            execution_tag_q <= '0;
            cache_owned_q <= 1'b0;
            last_tile_q <= 1'b0;
            decoder_seen_q <= 1'b0;
            backend_seen_q <= 1'b0;
            slot_release_pending_q <= 1'b0;
            cache_release_pending_q <= 1'b0;
            session_error_q <= 1'b0;
            protocol_error <= 1'b0;
            count_sessions <= '0;
            count_final_tile_releases <= '0;
            count_cache_releases <= '0;
            count_session_errors <= '0;
        end else begin
            if (session_fire) begin
                context_q <= session_context_id;
                head_q <= session_head_id;
                payload_tag_q <= session_payload_tag;
                execution_tag_q <= session_execution_tag;
                cache_owned_q <= session_cache_owned;
                last_tile_q <= session_last_output_tile;
                decoder_seen_q <= 1'b0;
                backend_seen_q <= 1'b0;
                slot_release_pending_q <= 1'b0;
                cache_release_pending_q <= 1'b0;
                session_error_q <= 1'b0;
                count_sessions <= count_sessions + 1'b1;
                state_q <= ST_WAIT;
            end
            if (decoder_fire) begin
                decoder_seen_q <= 1'b1;
                if (decoder_mismatch) begin
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                end
            end
            if (backend_fire) begin
                backend_seen_q <= 1'b1;
                if (backend_mismatch) begin
                    session_error_q <= 1'b1;
                    protocol_error <= 1'b1;
                end
            end
            if (state_q == ST_WAIT && both_done_comb) begin
                session_error_q <= completion_error_comb;
                if (last_tile_q) begin
                    slot_release_pending_q <= 1'b1;
                    cache_release_pending_q <= cache_owned_q;
                    count_final_tile_releases <=
                        count_final_tile_releases + 1'b1;
                    state_q <= ST_RELEASE;
                end else begin
                    if (completion_error_comb)
                        count_session_errors <= count_session_errors + 1'b1;
                    state_q <= ST_DONE;
                end
            end
            if (slot_release_fire)
                slot_release_pending_q <= 1'b0;
            if (cache_release_fire) begin
                cache_release_pending_q <= 1'b0;
                count_cache_releases <= count_cache_releases + 1'b1;
            end
            if (state_q == ST_RELEASE &&
                (!slot_release_pending_q || slot_release_fire) &&
                (!cache_release_pending_q || cache_release_fire)) begin
                if (session_error_q)
                    count_session_errors <= count_session_errors + 1'b1;
                state_q <= ST_DONE;
            end
            if (done_fire)
                state_q <= ST_IDLE;
        end
    end
endmodule

`default_nettype wire
