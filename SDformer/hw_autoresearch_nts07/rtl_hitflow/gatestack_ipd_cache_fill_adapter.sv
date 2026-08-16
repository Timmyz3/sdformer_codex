`timescale 1ns/1ps
`default_nettype none

// Converts the IPD decoder's header/term tap into one descriptor-cache fill.
// Oversized heads consume the tap in bypass mode without stalling projection.
module gatestack_ipd_cache_fill_adapter #(
    parameter int TAG_W = 32,
    parameter int HEAD_ID_W = 5,
    parameter int COUNTER_W = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         begin_valid,
    output logic                         begin_ready,
    input  logic [HEAD_ID_W-1:0]         begin_head_id,
    input  logic [TAG_W-1:0]             begin_tag,
    input  logic [7:0]                   begin_term_count,
    input  logic                         begin_cache_allowed,
    input  logic                         entry_valid,
    output logic                         entry_ready,
    input  logic [8:0]                   entry_gate_code,
    input  logic [4:0]                   entry_lane_id,
    input  logic [7:0]                   entry_destination_count,
    input  logic                         entry_last,

    output logic                         cache_begin_valid,
    input  logic                         cache_begin_ready,
    output logic [HEAD_ID_W-1:0]         cache_begin_head_id,
    output logic [TAG_W-1:0]             cache_begin_tag,
    output logic [7:0]                   cache_begin_term_count,
    input  logic                         cache_begin_cacheable,
    output logic                         cache_entry_valid,
    input  logic                         cache_entry_ready,
    output logic [8:0]                   cache_entry_gate_code,
    output logic [4:0]                   cache_entry_lane_id,
    output logic [7:0]                   cache_entry_destination_count,
    output logic                         cache_entry_last,
    output logic                         session_active,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_cacheable_fills,
    output logic [COUNTER_W-1:0]         count_bypass_fills
);
    typedef enum logic [1:0] {ST_IDLE, ST_CACHE, ST_BYPASS} state_t;
    state_t state_q;
    logic [7:0] expected_terms_q, entry_index_q;
    logic begin_fire, entry_fire, expected_last;

    assign cache_begin_valid = state_q == ST_IDLE && begin_valid &&
                               begin_cache_allowed;
    assign begin_ready = state_q == ST_IDLE &&
                         (!begin_cache_allowed || cache_begin_ready);
    assign begin_fire = begin_valid && begin_ready;
    assign cache_begin_head_id = begin_head_id;
    assign cache_begin_tag = begin_tag;
    assign cache_begin_term_count = begin_term_count;
    assign cache_entry_valid = state_q == ST_CACHE && entry_valid;
    assign entry_ready = state_q == ST_CACHE ? cache_entry_ready :
                         state_q == ST_BYPASS;
    assign cache_entry_gate_code = entry_gate_code;
    assign cache_entry_lane_id = entry_lane_id;
    assign cache_entry_destination_count = entry_destination_count;
    assign cache_entry_last = entry_last;
    assign entry_fire = entry_valid && entry_ready;
    assign expected_last = 32'(entry_index_q) + 1 ==
                           32'(expected_terms_q);
    assign session_active = state_q != ST_IDLE;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            expected_terms_q <= '0;
            entry_index_q <= '0;
            protocol_error <= 1'b0;
            count_cacheable_fills <= '0;
            count_bypass_fills <= '0;
        end else begin
            if (begin_fire) begin
                expected_terms_q <= begin_term_count;
                entry_index_q <= '0;
                if (begin_cache_allowed && cache_begin_cacheable) begin
                    count_cacheable_fills <= count_cacheable_fills + 1'b1;
                    if (begin_term_count != 0)
                        state_q <= ST_CACHE;
                end else begin
                    count_bypass_fills <= count_bypass_fills + 1'b1;
                    if (begin_term_count != 0)
                        state_q <= ST_BYPASS;
                end
            end
            if (entry_valid && state_q == ST_IDLE)
                protocol_error <= 1'b1;
            if (entry_fire) begin
                if (entry_last != expected_last) begin
                    protocol_error <= 1'b1;
                    state_q <= ST_IDLE;
                end else if (expected_last) begin
                    state_q <= ST_IDLE;
                end else begin
                    entry_index_q <= entry_index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
