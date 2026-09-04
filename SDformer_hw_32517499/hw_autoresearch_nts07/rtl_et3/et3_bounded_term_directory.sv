`timescale 1ns/1ps
`default_nettype none

// Bounded online directory for exact SET/MULTISET term formation.
// Entries with the same key may occupy multiple fixed-size destination
// segments. When the key directory is full, source items are preserved in a
// bounded fallback FIFO and emitted as single-destination terms.
module et3_bounded_term_directory #(
    parameter int KEY_CAP = 4,
    parameter int SEG_DEPTH = 4,
    parameter int FALLBACK_DEPTH = 8,
    parameter int TAG_W = 16,
    parameter int GATE_W = 9,
    parameter int LANE_W = 5,
    parameter int MULT_W = 3,
    parameter int DEST_W = 8,
    parameter int COUNTER_W = 32,
    parameter int KEY_ADDR_W = (KEY_CAP <= 1) ? 1 : $clog2(KEY_CAP),
    parameter int SEG_ADDR_W = (SEG_DEPTH <= 1) ? 1 : $clog2(SEG_DEPTH),
    parameter int FB_ADDR_W = (FALLBACK_DEPTH <= 1) ? 1 :
                              $clog2(FALLBACK_DEPTH),
    parameter int KEY_COUNT_W = (KEY_CAP <= 1) ? 1 : $clog2(KEY_CAP + 1),
    parameter int SEG_COUNT_W = (SEG_DEPTH <= 1) ? 1 : $clog2(SEG_DEPTH + 1),
    parameter int FB_COUNT_W = (FALLBACK_DEPTH <= 1) ? 1 :
                               $clog2(FALLBACK_DEPTH + 1)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         flush,

    input  logic                         source_valid,
    output logic                         source_ready,
    input  logic [TAG_W-1:0]             source_group_tag,
    input  logic                         source_mode_multiset,
    input  logic [GATE_W-1:0]            source_gate_code,
    input  logic [LANE_W-1:0]            source_lane_id,
    input  logic [MULT_W-1:0]            source_multiplicity,
    input  logic [DEST_W-1:0]            source_destination,
    input  logic                         source_head_last,

    input  logic                         group_close_valid,
    output logic                         group_close_ready,
    input  logic [TAG_W-1:0]             group_close_tag,

    output logic                         cmd_valid,
    input  logic                         cmd_ready,
    output logic [TAG_W-1:0]             cmd_group_tag,
    output logic                         cmd_mode_multiset,
    output logic [GATE_W-1:0]            cmd_gate_code,
    output logic [LANE_W-1:0]            cmd_lane_id,
    output logic [MULT_W-1:0]            cmd_multiplicity,
    output logic [DEST_W-1:0]            cmd_destination,
    output logic                         cmd_term_first,
    output logic                         cmd_term_last,
    output logic                         cmd_head_last,
    output logic                         cmd_fallback,

    output logic                         group_emit_done,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_source_items,
    output logic [COUNTER_W-1:0]         count_directory_entries,
    output logic [COUNTER_W-1:0]         count_fallback_items,
    output logic [COUNTER_W-1:0]         count_typed_terms,
    output logic [COUNTER_W-1:0]         count_destination_beats,
    output logic [COUNTER_W-1:0]         count_partial_drains
);
    typedef enum logic [1:0] {
        ST_COLLECT,
        ST_EMIT_DIRECTORY,
        ST_EMIT_FALLBACK
    } state_t;

    state_t state_q;

    logic entry_mode_q [0:KEY_CAP-1];
    logic [TAG_W-1:0] entry_tag_q [0:KEY_CAP-1];
    logic [GATE_W-1:0] entry_gate_q [0:KEY_CAP-1];
    logic [LANE_W-1:0] entry_lane_q [0:KEY_CAP-1];
    logic [MULT_W-1:0] entry_mult_q [0:KEY_CAP-1];
    logic [SEG_COUNT_W-1:0] entry_dest_count_q [0:KEY_CAP-1];
    logic [DEST_W-1:0] entry_dest_q [0:KEY_CAP-1][0:SEG_DEPTH-1];
    logic [KEY_COUNT_W-1:0] entry_count_q;

    logic fb_mode_q [0:FALLBACK_DEPTH-1];
    logic [TAG_W-1:0] fb_tag_q [0:FALLBACK_DEPTH-1];
    logic [GATE_W-1:0] fb_gate_q [0:FALLBACK_DEPTH-1];
    logic [LANE_W-1:0] fb_lane_q [0:FALLBACK_DEPTH-1];
    logic [MULT_W-1:0] fb_mult_q [0:FALLBACK_DEPTH-1];
    logic [DEST_W-1:0] fb_dest_q [0:FALLBACK_DEPTH-1];
    logic [FB_COUNT_W-1:0] fb_count_q;

    logic [KEY_ADDR_W-1:0] emit_entry_q;
    logic [SEG_ADDR_W-1:0] emit_dest_q;
    logic [FB_ADDR_W-1:0] emit_fb_q;
    logic source_head_last_q;
    logic group_active_q;
    logic [TAG_W-1:0] group_tag_q;

    logic match_found;
    logic [KEY_ADDR_W-1:0] match_index;
    logic duplicate_destination;
    logic key_matches;
    logic source_contract_ok;
    logic directory_capacity;
    logic fallback_capacity;
    logic use_fallback;
    logic source_fire;
    logic group_close_fire;
    logic cmd_fire;
    logic need_partial_drain;

    always_comb begin
        match_found = 1'b0;
        match_index = '0;
        duplicate_destination = 1'b0;
        for (int entry = 0; entry < KEY_CAP; entry++) begin
            key_matches =
                (entry < 32'(entry_count_q)) &&
                (entry_mode_q[entry] == source_mode_multiset) &&
                (entry_tag_q[entry] == source_group_tag) &&
                (entry_gate_q[entry] == source_gate_code) &&
                (entry_lane_q[entry] == source_lane_id) &&
                (entry_mult_q[entry] == source_multiplicity);
            if (key_matches) begin
                for (int dest = 0; dest < SEG_DEPTH; dest++) begin
                    if ((dest < 32'(entry_dest_count_q[entry])) &&
                        (entry_dest_q[entry][dest] ==
                         source_destination)) begin
                        duplicate_destination = 1'b1;
                    end
                end
                if (!match_found &&
                    (32'(entry_dest_count_q[entry]) < SEG_DEPTH)) begin
                    match_found = 1'b1;
                    match_index = KEY_ADDR_W'(entry);
                end
            end
        end
        for (int fallback = 0; fallback < FALLBACK_DEPTH; fallback++) begin
            if ((fallback < 32'(fb_count_q)) &&
                (fb_mode_q[fallback] == source_mode_multiset) &&
                (fb_tag_q[fallback] == source_group_tag) &&
                (fb_gate_q[fallback] == source_gate_code) &&
                (fb_lane_q[fallback] == source_lane_id) &&
                (fb_mult_q[fallback] == source_multiplicity) &&
                (fb_dest_q[fallback] == source_destination)) begin
                duplicate_destination = 1'b1;
            end
        end
    end

    assign source_contract_ok =
        (source_gate_code != '0) &&
        (source_multiplicity != '0) &&
        (32'(source_multiplicity) <= 5) &&
        (source_mode_multiset || (source_multiplicity == MULT_W'(1))) &&
        (!group_active_q || (source_group_tag == group_tag_q)) &&
        !duplicate_destination;
    assign directory_capacity = match_found ||
                                (32'(entry_count_q) < KEY_CAP);
    assign fallback_capacity = 32'(fb_count_q) < FALLBACK_DEPTH;
    assign use_fallback = !directory_capacity;
    assign source_ready = (state_q == ST_COLLECT) &&
                          source_contract_ok &&
                          (directory_capacity || fallback_capacity);
    assign source_fire = source_valid && source_ready;
    assign group_close_ready = (state_q == ST_COLLECT) &&
                               (
                                   (
                                       !source_valid &&
                                       (!group_active_q ||
                                        (group_close_tag == group_tag_q))
                                   ) ||
                                   (
                                       source_valid &&
                                       source_ready &&
                                       source_head_last &&
                                       (group_close_tag ==
                                        source_group_tag)
                                   )
                               );
    assign group_close_fire = group_close_valid && group_close_ready;
    assign need_partial_drain = (state_q == ST_COLLECT) &&
                                source_valid &&
                                source_contract_ok &&
                                !directory_capacity &&
                                !fallback_capacity;

    always_comb begin
        cmd_valid = 1'b0;
        cmd_group_tag = '0;
        cmd_mode_multiset = 1'b0;
        cmd_gate_code = '0;
        cmd_lane_id = '0;
        cmd_multiplicity = '0;
        cmd_destination = '0;
        cmd_term_first = 1'b0;
        cmd_term_last = 1'b0;
        cmd_head_last = 1'b0;
        cmd_fallback = 1'b0;

        if ((state_q == ST_EMIT_DIRECTORY) &&
            (32'(emit_entry_q) < 32'(entry_count_q))) begin
            cmd_valid = 1'b1;
            cmd_group_tag = entry_tag_q[emit_entry_q];
            cmd_mode_multiset = entry_mode_q[emit_entry_q];
            cmd_gate_code = entry_gate_q[emit_entry_q];
            cmd_lane_id = entry_lane_q[emit_entry_q];
            cmd_multiplicity = entry_mult_q[emit_entry_q];
            cmd_destination = entry_dest_q[emit_entry_q][emit_dest_q];
            cmd_term_first = emit_dest_q == '0;
            cmd_term_last =
                (32'(emit_dest_q) + 1) ==
                32'(entry_dest_count_q[emit_entry_q]);
            cmd_head_last = source_head_last_q && cmd_term_last &&
                ((32'(emit_entry_q) + 1) == 32'(entry_count_q)) &&
                (fb_count_q == '0);
        end else if ((state_q == ST_EMIT_FALLBACK) &&
                     (32'(emit_fb_q) < 32'(fb_count_q))) begin
            cmd_valid = 1'b1;
            cmd_group_tag = fb_tag_q[emit_fb_q];
            cmd_mode_multiset = fb_mode_q[emit_fb_q];
            cmd_gate_code = fb_gate_q[emit_fb_q];
            cmd_lane_id = fb_lane_q[emit_fb_q];
            cmd_multiplicity = fb_mult_q[emit_fb_q];
            cmd_destination = fb_dest_q[emit_fb_q];
            cmd_term_first = 1'b1;
            cmd_term_last = 1'b1;
            cmd_head_last = source_head_last_q &&
                            ((32'(emit_fb_q) + 1) == 32'(fb_count_q));
            cmd_fallback = 1'b1;
        end
    end

    assign cmd_fire = cmd_valid && cmd_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core || flush) begin
            state_q <= ST_COLLECT;
            entry_count_q <= '0;
            fb_count_q <= '0;
            emit_entry_q <= '0;
            emit_dest_q <= '0;
            emit_fb_q <= '0;
            source_head_last_q <= 1'b0;
            group_active_q <= 1'b0;
            group_tag_q <= '0;
            group_emit_done <= 1'b0;
            protocol_error <= 1'b0;
            count_source_items <= '0;
            count_directory_entries <= '0;
            count_fallback_items <= '0;
            count_typed_terms <= '0;
            count_destination_beats <= '0;
            count_partial_drains <= '0;
            for (int entry = 0; entry < KEY_CAP; entry++) begin
                entry_mode_q[entry] <= 1'b0;
                entry_tag_q[entry] <= '0;
                entry_gate_q[entry] <= '0;
                entry_lane_q[entry] <= '0;
                entry_mult_q[entry] <= '0;
                entry_dest_count_q[entry] <= '0;
                for (int dest = 0; dest < SEG_DEPTH; dest++) begin
                    entry_dest_q[entry][dest] <= '0;
                end
            end
            for (int fallback = 0; fallback < FALLBACK_DEPTH; fallback++) begin
                fb_mode_q[fallback] <= 1'b0;
                fb_tag_q[fallback] <= '0;
                fb_gate_q[fallback] <= '0;
                fb_lane_q[fallback] <= '0;
                fb_mult_q[fallback] <= '0;
                fb_dest_q[fallback] <= '0;
            end
        end else begin
            group_emit_done <= 1'b0;

            if ((state_q == ST_COLLECT) && source_valid &&
                !source_contract_ok) begin
                protocol_error <= 1'b1;
            end
            if ((state_q == ST_COLLECT) && group_close_valid &&
                !group_close_ready && !source_valid) begin
                protocol_error <= 1'b1;
            end

            if (need_partial_drain) begin
                source_head_last_q <= 1'b0;
                emit_entry_q <= '0;
                emit_dest_q <= '0;
                emit_fb_q <= '0;
                state_q <= ST_EMIT_DIRECTORY;
                count_partial_drains <= count_partial_drains + 1'b1;
            end

            if (group_close_fire) begin
                if (!group_active_q) begin
                    group_active_q <= 1'b1;
                    group_tag_q <= group_close_tag;
                end
                source_head_last_q <= 1'b1;
                emit_entry_q <= '0;
                emit_dest_q <= '0;
                emit_fb_q <= '0;
                state_q <= ST_EMIT_DIRECTORY;
            end

            if (source_fire) begin
                count_source_items <= count_source_items + 1'b1;
                if (!group_active_q) begin
                    group_active_q <= 1'b1;
                    group_tag_q <= source_group_tag;
                end
                if (use_fallback) begin
                    fb_mode_q[FB_ADDR_W'(fb_count_q)] <=
                        source_mode_multiset;
                    fb_tag_q[FB_ADDR_W'(fb_count_q)] <= source_group_tag;
                    fb_gate_q[FB_ADDR_W'(fb_count_q)] <= source_gate_code;
                    fb_lane_q[FB_ADDR_W'(fb_count_q)] <= source_lane_id;
                    fb_mult_q[FB_ADDR_W'(fb_count_q)] <=
                        source_multiplicity;
                    fb_dest_q[FB_ADDR_W'(fb_count_q)] <=
                        source_destination;
                    fb_count_q <= fb_count_q + 1'b1;
                    count_fallback_items <= count_fallback_items + 1'b1;
                end else if (match_found) begin
                    entry_dest_q[match_index][
                        SEG_ADDR_W'(entry_dest_count_q[match_index])
                    ] <= source_destination;
                    entry_dest_count_q[match_index] <=
                        entry_dest_count_q[match_index] + 1'b1;
                end else begin
                    entry_mode_q[KEY_ADDR_W'(entry_count_q)] <=
                        source_mode_multiset;
                    entry_tag_q[KEY_ADDR_W'(entry_count_q)] <=
                        source_group_tag;
                    entry_gate_q[KEY_ADDR_W'(entry_count_q)] <=
                        source_gate_code;
                    entry_lane_q[KEY_ADDR_W'(entry_count_q)] <=
                        source_lane_id;
                    entry_mult_q[KEY_ADDR_W'(entry_count_q)] <=
                        source_multiplicity;
                    entry_dest_q[KEY_ADDR_W'(entry_count_q)][0] <=
                        source_destination;
                    entry_dest_count_q[KEY_ADDR_W'(entry_count_q)] <=
                        SEG_COUNT_W'(1);
                    entry_count_q <= entry_count_q + 1'b1;
                    count_directory_entries <=
                        count_directory_entries + 1'b1;
                end

                if (source_head_last) begin
                    source_head_last_q <= 1'b1;
                    emit_entry_q <= '0;
                    emit_dest_q <= '0;
                    emit_fb_q <= '0;
                    state_q <= ST_EMIT_DIRECTORY;
                end
            end

            if (state_q == ST_EMIT_DIRECTORY) begin
                if (entry_count_q == '0) begin
                    state_q <= ST_EMIT_FALLBACK;
                end else if (cmd_fire) begin
                    count_destination_beats <=
                        count_destination_beats + 1'b1;
                    if (cmd_term_first) begin
                        count_typed_terms <= count_typed_terms + 1'b1;
                    end
                    if (cmd_term_last) begin
                        emit_dest_q <= '0;
                        if ((32'(emit_entry_q) + 1) <
                            32'(entry_count_q)) begin
                            emit_entry_q <= emit_entry_q + 1'b1;
                        end else if (fb_count_q != '0) begin
                            emit_fb_q <= '0;
                            state_q <= ST_EMIT_FALLBACK;
                        end else begin
                            state_q <= ST_COLLECT;
                            entry_count_q <= '0;
                            fb_count_q <= '0;
                            if (source_head_last_q) begin
                                group_active_q <= 1'b0;
                                source_head_last_q <= 1'b0;
                                group_emit_done <= 1'b1;
                            end
                        end
                    end else begin
                        emit_dest_q <= emit_dest_q + 1'b1;
                    end
                end
            end

            if (state_q == ST_EMIT_FALLBACK) begin
                if (fb_count_q == '0) begin
                    state_q <= ST_COLLECT;
                    entry_count_q <= '0;
                    fb_count_q <= '0;
                    if (source_head_last_q) begin
                        group_active_q <= 1'b0;
                        source_head_last_q <= 1'b0;
                        group_emit_done <= 1'b1;
                    end
                end else if (cmd_fire) begin
                    count_destination_beats <=
                        count_destination_beats + 1'b1;
                    count_typed_terms <= count_typed_terms + 1'b1;
                    if ((32'(emit_fb_q) + 1) < 32'(fb_count_q)) begin
                        emit_fb_q <= emit_fb_q + 1'b1;
                    end else begin
                        state_q <= ST_COLLECT;
                        entry_count_q <= '0;
                        fb_count_q <= '0;
                        if (source_head_last_q) begin
                            group_active_q <= 1'b0;
                            source_head_last_q <= 1'b0;
                            group_emit_done <= 1'b1;
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
