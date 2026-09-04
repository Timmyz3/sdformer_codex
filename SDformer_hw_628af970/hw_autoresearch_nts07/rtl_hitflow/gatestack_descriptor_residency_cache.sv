`timescale 1ns/1ps
`default_nettype none

// Bounded, exact descriptor residency. Heads beyond CACHE_TERMS are reported
// as bypasses and remain replayable through the sequential IPD32W path.
module gatestack_descriptor_residency_cache #(
    parameter int CONTEXTS       = 2,
    parameter int HEADS          = 24,
    parameter int CACHE_TERMS    = 80,
    parameter int TAG_W          = 32,
    parameter int COUNTER_W      = 32,
    parameter int CONTEXT_ID_W   = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int HEAD_ID_W      = (HEADS <= 1) ? 1 : $clog2(HEADS),
    parameter int TERM_INDEX_W   = (CACHE_TERMS <= 1) ? 1 : $clog2(CACHE_TERMS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         fill_begin_valid,
    output logic                         fill_begin_ready,
    input  logic [CONTEXT_ID_W-1:0]      fill_context_id,
    input  logic [HEAD_ID_W-1:0]         fill_head_id,
    input  logic [TAG_W-1:0]             fill_tag,
    input  logic [7:0]                   fill_term_count,
    output logic                         fill_begin_cacheable,

    input  logic                         fill_entry_valid,
    output logic                         fill_entry_ready,
    input  logic [8:0]                   fill_gate_code,
    input  logic [4:0]                   fill_lane_id,
    input  logic [7:0]                   fill_destination_count,
    input  logic                         fill_entry_last,

    input  logic                         lookup_valid,
    output logic                         lookup_ready,
    input  logic [CONTEXT_ID_W-1:0]      lookup_context_id,
    input  logic [HEAD_ID_W-1:0]         lookup_head_id,
    input  logic [TAG_W-1:0]             lookup_expected_tag,

    output logic                         lookup_meta_valid,
    input  logic                         lookup_meta_ready,
    output logic                         lookup_hit,
    output logic [TAG_W-1:0]             lookup_tag,
    output logic [7:0]                   lookup_term_count,

    output logic                         lookup_entry_valid,
    input  logic                         lookup_entry_ready,
    output logic [8:0]                   lookup_gate_code,
    output logic [4:0]                   lookup_lane_id,
    output logic [7:0]                   lookup_destination_count,
    output logic [TERM_INDEX_W-1:0]      lookup_term_index,
    output logic                         lookup_entry_last,

    input  logic                         release_valid,
    output logic                         release_ready,
    input  logic [CONTEXT_ID_W-1:0]      release_context_id,
    input  logic [HEAD_ID_W-1:0]         release_head_id,
    input  logic [TAG_W-1:0]             release_expected_tag,

    output logic [(CONTEXTS*HEADS)-1:0]  cache_valid_flat,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_cached_heads,
    output logic [COUNTER_W-1:0]         count_bypass_heads,
    output logic [COUNTER_W-1:0]         count_lookup_hits,
    output logic [COUNTER_W-1:0]         count_lookup_misses,
    output logic [COUNTER_W-1:0]         count_releases,
    output logic [COUNTER_W-1:0]         count_release_noops,
    output logic [COUNTER_W-1:0]         count_release_tag_mismatches
);

    localparam int TOTAL_SLOTS = CONTEXTS * HEADS;
    localparam int TOTAL_ENTRIES = TOTAL_SLOTS * CACHE_TERMS;
    localparam int SLOT_INDEX_W = (TOTAL_SLOTS <= 1) ? 1 : $clog2(TOTAL_SLOTS);
    localparam int ENTRY_ADDR_W = (TOTAL_ENTRIES <= 1) ?
                                  1 : $clog2(TOTAL_ENTRIES);

    logic [23:0] entry_mem [0:TOTAL_ENTRIES-1];
    logic [TOTAL_SLOTS-1:0] cache_valid_q;
    logic [TAG_W-1:0] cache_tag_q [0:TOTAL_SLOTS-1];
    logic [7:0] cache_term_count_q [0:TOTAL_SLOTS-1];

    logic fill_active_q;
    logic [SLOT_INDEX_W-1:0] fill_slot_q;
    logic [ENTRY_ADDR_W-1:0] fill_base_q;
    logic [TERM_INDEX_W-1:0] fill_index_q;
    logic [7:0] fill_expected_count_q;
    logic [TAG_W-1:0] fill_tag_q;

    logic lookup_meta_valid_q;
    logic lookup_hit_q;
    logic [TAG_W-1:0] lookup_tag_q;
    logic [7:0] lookup_count_q;
    logic lookup_stream_active_q;
    logic [SLOT_INDEX_W-1:0] lookup_slot_q;
    logic [ENTRY_ADDR_W-1:0] lookup_base_q;
    logic [TERM_INDEX_W-1:0] lookup_index_q;

    logic fill_context_in_range;
    logic fill_head_in_range;
    logic lookup_context_in_range;
    logic lookup_head_in_range;
    logic release_context_in_range;
    logic release_head_in_range;
    logic [SLOT_INDEX_W-1:0] fill_slot_comb;
    logic [SLOT_INDEX_W-1:0] lookup_slot_comb;
    logic [SLOT_INDEX_W-1:0] release_slot_comb;
    logic fill_begin_fire;
    logic fill_entry_fire;
    logic lookup_fire;
    logic lookup_meta_fire;
    logic lookup_entry_fire;
    logic release_fire;
    logic release_slot_valid;
    logic release_tag_matches;
    logic fill_expected_last;

    assign fill_context_in_range = 32'(fill_context_id) < CONTEXTS;
    assign fill_head_in_range = 32'(fill_head_id) < HEADS;
    assign lookup_context_in_range = 32'(lookup_context_id) < CONTEXTS;
    assign lookup_head_in_range = 32'(lookup_head_id) < HEADS;
    assign release_context_in_range = 32'(release_context_id) < CONTEXTS;
    assign release_head_in_range = 32'(release_head_id) < HEADS;
    assign fill_begin_cacheable = 32'(fill_term_count) <= CACHE_TERMS;

    always_comb begin
        fill_slot_comb = '0;
        lookup_slot_comb = '0;
        release_slot_comb = '0;
        if (fill_context_in_range && fill_head_in_range) begin
            fill_slot_comb = SLOT_INDEX_W'(
                (32'(fill_context_id) * 32'(HEADS)) + 32'(fill_head_id));
        end
        if (lookup_context_in_range && lookup_head_in_range) begin
            lookup_slot_comb = SLOT_INDEX_W'(
                (32'(lookup_context_id) * 32'(HEADS)) +
                32'(lookup_head_id));
        end
        if (release_context_in_range && release_head_in_range) begin
            release_slot_comb = SLOT_INDEX_W'(
                (32'(release_context_id) * 32'(HEADS)) +
                32'(release_head_id));
        end
    end

    assign fill_begin_ready = !fill_active_q &&
                              fill_context_in_range && fill_head_in_range &&
                              !cache_valid_q[fill_slot_comb] &&
                              !(lookup_stream_active_q &&
                                lookup_slot_q == fill_slot_comb);
    assign fill_entry_ready = fill_active_q;
    assign fill_begin_fire = fill_begin_valid && fill_begin_ready;
    assign fill_entry_fire = fill_entry_valid && fill_entry_ready;
    assign fill_expected_last =
        32'(fill_index_q) + 1 == 32'(fill_expected_count_q);

    assign lookup_ready = !lookup_meta_valid_q && !lookup_stream_active_q &&
                          lookup_context_in_range && lookup_head_in_range &&
                          !(fill_active_q && fill_slot_q == lookup_slot_comb);
    assign lookup_fire = lookup_valid && lookup_ready;
    assign lookup_meta_valid = lookup_meta_valid_q;
    assign lookup_meta_fire = lookup_meta_valid && lookup_meta_ready;
    assign lookup_hit = lookup_hit_q;
    assign lookup_tag = lookup_tag_q;
    assign lookup_term_count = lookup_count_q;

    assign lookup_entry_valid = lookup_stream_active_q;
    assign lookup_entry_fire = lookup_entry_valid && lookup_entry_ready;
    assign lookup_term_index = lookup_index_q;
    assign lookup_entry_last = lookup_entry_valid &&
        (32'(lookup_index_q) + 1 == 32'(lookup_count_q));
    assign lookup_gate_code = entry_mem[
        lookup_base_q + ENTRY_ADDR_W'(lookup_index_q)][8:0];
    assign lookup_lane_id = entry_mem[
        lookup_base_q + ENTRY_ADDR_W'(lookup_index_q)][13:9];
    assign lookup_destination_count = entry_mem[
        lookup_base_q + ENTRY_ADDR_W'(lookup_index_q)][21:14];

    assign release_ready = release_context_in_range && release_head_in_range &&
                           !(fill_active_q && fill_slot_q == release_slot_comb) &&
                           !(lookup_meta_valid_q &&
                             lookup_slot_q == release_slot_comb) &&
                           !(lookup_stream_active_q &&
                             lookup_slot_q == release_slot_comb);
    assign release_fire = release_valid && release_ready;
    assign release_slot_valid = cache_valid_q[release_slot_comb];
    assign release_tag_matches = release_slot_valid &&
        cache_tag_q[release_slot_comb] == release_expected_tag;
    assign cache_valid_flat = cache_valid_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cache_valid_q <= '0;
            fill_active_q <= 1'b0;
            fill_slot_q <= '0;
            fill_base_q <= '0;
            fill_index_q <= '0;
            fill_expected_count_q <= '0;
            fill_tag_q <= '0;
            lookup_meta_valid_q <= 1'b0;
            lookup_hit_q <= 1'b0;
            lookup_tag_q <= '0;
            lookup_count_q <= '0;
            lookup_stream_active_q <= 1'b0;
            lookup_slot_q <= '0;
            lookup_base_q <= '0;
            lookup_index_q <= '0;
            protocol_error <= 1'b0;
            count_cached_heads <= '0;
            count_bypass_heads <= '0;
            count_lookup_hits <= '0;
            count_lookup_misses <= '0;
            count_releases <= '0;
            count_release_noops <= '0;
            count_release_tag_mismatches <= '0;
        end else begin
            if (fill_begin_fire) begin
                if (fill_begin_cacheable) begin
                    if (fill_term_count == 0) begin
                        cache_valid_q[fill_slot_comb] <= 1'b1;
                        cache_tag_q[fill_slot_comb] <= fill_tag;
                        cache_term_count_q[fill_slot_comb] <= '0;
                        count_cached_heads <= count_cached_heads + 1'b1;
                    end else begin
                        fill_active_q <= 1'b1;
                        fill_slot_q <= fill_slot_comb;
                        fill_base_q <= ENTRY_ADDR_W'(
                            32'(fill_slot_comb) * 32'(CACHE_TERMS));
                        fill_index_q <= '0;
                        fill_expected_count_q <= fill_term_count;
                        fill_tag_q <= fill_tag;
                    end
                end else begin
                    count_bypass_heads <= count_bypass_heads + 1'b1;
                end
            end

            if (fill_entry_fire) begin
                entry_mem[fill_base_q + ENTRY_ADDR_W'(fill_index_q)] <= {
                    2'b00, fill_destination_count, fill_lane_id,
                    fill_gate_code
                };
                if (fill_entry_last != fill_expected_last ||
                    fill_destination_count == 0 ||
                    32'(fill_lane_id) >= 32) begin
                    fill_active_q <= 1'b0;
                    protocol_error <= 1'b1;
                end else if (fill_expected_last) begin
                    fill_active_q <= 1'b0;
                    cache_valid_q[fill_slot_q] <= 1'b1;
                    cache_tag_q[fill_slot_q] <= fill_tag_q;
                    cache_term_count_q[fill_slot_q] <= fill_expected_count_q;
                    count_cached_heads <= count_cached_heads + 1'b1;
                end else begin
                    fill_index_q <= fill_index_q + 1'b1;
                end
            end

            if (lookup_fire) begin
                lookup_meta_valid_q <= 1'b1;
                lookup_slot_q <= lookup_slot_comb;
                lookup_base_q <= ENTRY_ADDR_W'(
                    32'(lookup_slot_comb) * 32'(CACHE_TERMS));
                lookup_index_q <= '0;
                lookup_hit_q <= cache_valid_q[lookup_slot_comb] &&
                    cache_tag_q[lookup_slot_comb] == lookup_expected_tag;
                lookup_tag_q <= cache_valid_q[lookup_slot_comb] ?
                    cache_tag_q[lookup_slot_comb] : '0;
                lookup_count_q <= cache_valid_q[lookup_slot_comb] ?
                    cache_term_count_q[lookup_slot_comb] : '0;
                if (cache_valid_q[lookup_slot_comb] &&
                    cache_tag_q[lookup_slot_comb] == lookup_expected_tag) begin
                    count_lookup_hits <= count_lookup_hits + 1'b1;
                end else begin
                    count_lookup_misses <= count_lookup_misses + 1'b1;
                end
            end

            if (lookup_meta_fire) begin
                lookup_meta_valid_q <= 1'b0;
                if (lookup_hit_q && lookup_count_q != 0) begin
                    lookup_stream_active_q <= 1'b1;
                end
            end

            if (lookup_entry_fire) begin
                if (lookup_entry_last) begin
                    lookup_stream_active_q <= 1'b0;
                end else begin
                    lookup_index_q <= lookup_index_q + 1'b1;
                end
            end

            if (release_fire) begin
                if (release_tag_matches) begin
                    cache_valid_q[release_slot_comb] <= 1'b0;
                    count_releases <= count_releases + 1'b1;
                end else if (!release_slot_valid) begin
                    count_release_noops <= count_release_noops + 1'b1;
                end else begin
                    // A stale lifecycle must never evict a newer payload tag.
                    protocol_error <= 1'b1;
                    count_release_tag_mismatches <=
                        count_release_tag_mismatches + 1'b1;
                end
            end

            if ((fill_begin_valid &&
                 (!fill_context_in_range || !fill_head_in_range)) ||
                (lookup_valid &&
                 (!lookup_context_in_range || !lookup_head_in_range)) ||
                (release_valid &&
                 (!release_context_in_range || !release_head_in_range))) begin
                protocol_error <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
