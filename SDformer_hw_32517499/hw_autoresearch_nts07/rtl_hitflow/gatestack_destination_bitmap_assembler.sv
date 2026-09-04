`timescale 1ns/1ps
`default_nettype none

// Builds one exact destination bitmap per ordered term while buffering one
// prefetched term. Product generation may proceed from the term stream in
// parallel; the completed bitmap rendezvous with that product downstream.
module gatestack_destination_bitmap_assembler #(
    parameter int TOKENS          = 162,
    parameter int EVENT_WAYS      = 4,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = 5,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int TAG_W           = 32,
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int COUNTER_W       = 32
) (
    input  logic                               clk_core,
    input  logic                               rst_core,

    input  logic                               term_valid,
    output logic                               term_ready,
    input  logic [TAG_W-1:0]                   term_tag,
    input  logic [8:0]                         term_gate_code,
    input  logic [LANE_ID_W-1:0]               term_lane_id,
    input  logic [7:0]                         term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]             term_issue_seq,
    input  logic                               term_head_last,

    input  logic                               event_valid,
    output logic                               event_ready,
    input  logic [8:0]                         event_gate_code,
    input  logic [LANE_ID_W-1:0]               event_lane_id,
    input  logic [EVENT_WAYS-1:0]              event_token_valid,
    input  logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    input  logic [WAY_COUNT_W-1:0]             event_count,
    input  logic [ISSUE_SEQ_W-1:0]             event_issue_seq,
    input  logic                               event_term_first,
    input  logic                               event_term_last,
    input  logic                               event_head_last,

    output logic                               bitmap_valid,
    input  logic                               bitmap_ready,
    output logic [TAG_W-1:0]                   bitmap_tag,
    output logic [8:0]                         bitmap_gate_code,
    output logic [LANE_ID_W-1:0]               bitmap_lane_id,
    output logic [ISSUE_SEQ_W-1:0]             bitmap_issue_seq,
    output logic                               bitmap_head_last,
    output logic [TOKENS-1:0]                  bitmap_destinations,

    output logic                               protocol_error,
    output logic [COUNTER_W-1:0]               count_terms,
    output logic [COUNTER_W-1:0]               count_events,
    output logic [COUNTER_W-1:0]               count_bitmaps,
    output logic [COUNTER_W-1:0]               count_term_stall_cycles,
    output logic [COUNTER_W-1:0]               count_event_stall_cycles,
    output logic [COUNTER_W-1:0]               count_bitmap_stall_cycles
);
    localparam int BITMAP_INDEX_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS);

    logic current_valid_q;
    logic [TAG_W-1:0] current_tag_q;
    logic [8:0] current_gate_q;
    logic [LANE_ID_W-1:0] current_lane_q;
    logic [7:0] current_destination_count_q;
    logic [ISSUE_SEQ_W-1:0] current_issue_seq_q;
    logic current_head_last_q;
    logic [7:0] current_received_q;
    logic [TOKENS-1:0] current_bitmap_q;
    logic bitmap_valid_q;

    logic next_valid_q;
    logic [TAG_W-1:0] next_tag_q;
    logic [8:0] next_gate_q;
    logic [LANE_ID_W-1:0] next_lane_q;
    logic [7:0] next_destination_count_q;
    logic [ISSUE_SEQ_W-1:0] next_issue_seq_q;
    logic next_head_last_q;

    logic term_fire;
    logic event_fire;
    logic bitmap_fire;
    logic event_metadata_matches;
    logic event_first_matches;
    logic event_last_matches;
    logic event_head_last_matches;
    logic event_tokens_in_range;
    logic event_valid_mask_matches_count;
    logic event_has_duplicate;
    logic event_contract_ok;
    logic [TOKENS-1:0] event_bitmap_comb;
    logic [7:0] received_after_event;
    logic [TOKENS-1:0] bitmap_after_event;
    logic [WAY_COUNT_W-1:0] valid_count_comb;

    assign term_ready = !bitmap_valid_q && !next_valid_q;
    assign term_fire = term_valid && term_ready;
    assign event_ready = current_valid_q && !bitmap_valid_q;
    assign event_fire = event_valid && event_ready;
    assign bitmap_valid = bitmap_valid_q;
    assign bitmap_tag = current_tag_q;
    assign bitmap_gate_code = current_gate_q;
    assign bitmap_lane_id = current_lane_q;
    assign bitmap_issue_seq = current_issue_seq_q;
    assign bitmap_head_last = current_head_last_q;
    assign bitmap_destinations = current_bitmap_q;
    assign bitmap_fire = bitmap_valid && bitmap_ready;

    assign event_metadata_matches = event_gate_code == current_gate_q &&
        event_lane_id == current_lane_q &&
        event_issue_seq == current_issue_seq_q;
    assign event_first_matches = event_term_first ==
                                 (current_received_q == 0);
    assign received_after_event = current_received_q + 8'(event_count);
    assign event_last_matches = event_term_last ==
        (received_after_event == current_destination_count_q);
    assign event_head_last_matches = event_head_last ==
        (event_term_last && current_head_last_q);
    assign bitmap_after_event = current_bitmap_q | event_bitmap_comb;
    assign event_contract_ok = event_metadata_matches &&
        event_first_matches && event_last_matches && event_head_last_matches &&
        event_tokens_in_range && event_valid_mask_matches_count &&
        !event_has_duplicate && event_count != 0 &&
        received_after_event <= current_destination_count_q;

    always_comb begin
        event_bitmap_comb = '0;
        valid_count_comb = '0;
        event_tokens_in_range = 1'b1;
        event_has_duplicate = 1'b0;
        for (int way = 0; way < 4; way = way + 1) begin
            if (event_token_valid[way]) begin
                valid_count_comb = valid_count_comb + 1'b1;
                if (32'(event_token_ids[(way*TOKEN_ID_W) +: TOKEN_ID_W]) >=
                    TOKENS) begin
                    event_tokens_in_range = 1'b0;
                end else begin
                    if (current_bitmap_q[
                        BITMAP_INDEX_W'(event_token_ids[
                            (way*TOKEN_ID_W) +: TOKEN_ID_W])] ||
                        event_bitmap_comb[
                        BITMAP_INDEX_W'(event_token_ids[
                            (way*TOKEN_ID_W) +: TOKEN_ID_W])]) begin
                        event_has_duplicate = 1'b1;
                    end
                    event_bitmap_comb[
                        BITMAP_INDEX_W'(event_token_ids[
                            (way*TOKEN_ID_W) +: TOKEN_ID_W])] = 1'b1;
                end
            end
        end
    end
    assign event_valid_mask_matches_count = valid_count_comb == event_count;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            current_valid_q <= 1'b0;
            current_tag_q <= '0;
            current_gate_q <= '0;
            current_lane_q <= '0;
            current_destination_count_q <= '0;
            current_issue_seq_q <= '0;
            current_head_last_q <= 1'b0;
            current_received_q <= '0;
            current_bitmap_q <= '0;
            bitmap_valid_q <= 1'b0;
            next_valid_q <= 1'b0;
            next_tag_q <= '0;
            next_gate_q <= '0;
            next_lane_q <= '0;
            next_destination_count_q <= '0;
            next_issue_seq_q <= '0;
            next_head_last_q <= 1'b0;
            protocol_error <= 1'b0;
            count_terms <= '0;
            count_events <= '0;
            count_bitmaps <= '0;
            count_term_stall_cycles <= '0;
            count_event_stall_cycles <= '0;
            count_bitmap_stall_cycles <= '0;
        end else begin
            if (term_fire) begin
                count_terms <= count_terms + 1'b1;
                if (!current_valid_q) begin
                    current_valid_q <= 1'b1;
                    current_tag_q <= term_tag;
                    current_gate_q <= term_gate_code;
                    current_lane_q <= term_lane_id;
                    current_destination_count_q <= term_destination_count;
                    current_issue_seq_q <= term_issue_seq;
                    current_head_last_q <= term_head_last;
                    current_received_q <= '0;
                    current_bitmap_q <= '0;
                end else begin
                    next_valid_q <= 1'b1;
                    next_tag_q <= term_tag;
                    next_gate_q <= term_gate_code;
                    next_lane_q <= term_lane_id;
                    next_destination_count_q <= term_destination_count;
                    next_issue_seq_q <= term_issue_seq;
                    next_head_last_q <= term_head_last;
                end
                if (term_destination_count == 0 ||
                    (current_valid_q &&
                     term_issue_seq != current_issue_seq_q + 1'b1)) begin
                    protocol_error <= 1'b1;
                end
            end

            if (event_fire) begin
                count_events <= count_events + COUNTER_W'(event_count);
                if (!event_contract_ok) begin
                    protocol_error <= 1'b1;
                end else begin
                    current_received_q <= received_after_event;
                    current_bitmap_q <= bitmap_after_event;
                    if (event_term_last) begin
                        bitmap_valid_q <= 1'b1;
                    end
                end
            end

            if (bitmap_fire) begin
                count_bitmaps <= count_bitmaps + 1'b1;
                bitmap_valid_q <= 1'b0;
                if (next_valid_q) begin
                    current_valid_q <= 1'b1;
                    current_tag_q <= next_tag_q;
                    current_gate_q <= next_gate_q;
                    current_lane_q <= next_lane_q;
                    current_destination_count_q <= next_destination_count_q;
                    current_issue_seq_q <= next_issue_seq_q;
                    current_head_last_q <= next_head_last_q;
                    current_received_q <= '0;
                    current_bitmap_q <= '0;
                    next_valid_q <= 1'b0;
                end else begin
                    current_valid_q <= 1'b0;
                    current_received_q <= '0;
                    current_bitmap_q <= '0;
                end
            end

            if (term_valid && !term_ready) begin
                count_term_stall_cycles <= count_term_stall_cycles + 1'b1;
            end
            if (event_valid && !event_ready) begin
                count_event_stall_cycles <= count_event_stall_cycles + 1'b1;
            end
            if (bitmap_valid && !bitmap_ready) begin
                count_bitmap_stall_cycles <= count_bitmap_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
