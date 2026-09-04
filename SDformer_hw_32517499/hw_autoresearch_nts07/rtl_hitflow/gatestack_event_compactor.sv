`timescale 1ns/1ps
`default_nettype none

// Converts one active token's K bitmap into up to WAYS lane events per cycle.
module gatestack_event_compactor #(
    parameter int LANES       = 32,
    parameter int WAYS        = 4,
    parameter int TAG_W       = 32,
    parameter int TOKEN_ID_W  = 8,
    parameter int SLOT_ID_W   = 2,
    parameter int COUNTER_W   = 32,
    parameter int LANE_ID_W   = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int COUNT_W     = (WAYS + 1 <= 2) ? 1 : $clog2(WAYS + 1)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [TAG_W-1:0]             token_tag,
    input  logic [TOKEN_ID_W-1:0]        token_id,
    input  logic [SLOT_ID_W-1:0]         token_slot_id,
    input  logic [LANES-1:0]             token_k_bits,

    output logic                         event_valid,
    input  logic                         event_ready,
    output logic [TAG_W-1:0]             event_tag,
    output logic [TOKEN_ID_W-1:0]        event_token_id,
    output logic [SLOT_ID_W-1:0]         event_slot_id,
    output logic [WAYS-1:0]              event_lane_valid,
    output logic [(WAYS*LANE_ID_W)-1:0]  event_lane_ids,
    output logic [COUNT_W-1:0]           event_count,
    output logic                         event_last_for_token,

    output logic [COUNTER_W-1:0]         count_tokens,
    output logic [COUNTER_W-1:0]         count_events,
    output logic [COUNTER_W-1:0]         count_event_stall_cycles
);

    logic active_q;
    logic [TAG_W-1:0] tag_q;
    logic [TOKEN_ID_W-1:0] token_q;
    logic [SLOT_ID_W-1:0] slot_q;
    logic [LANES-1:0] pending_q;

    logic [LANES-1:0] selected_mask;
    logic [LANES-1:0] remaining_mask;
    logic [WAYS-1:0] way_found;
    logic token_fire;
    logic event_fire;

    assign token_ready = !active_q;
    assign token_fire = token_valid && token_ready;
    assign event_valid = active_q && (pending_q != '0);
    assign event_fire = event_valid && event_ready;
    assign event_tag = tag_q;
    assign event_token_id = token_q;
    assign event_slot_id = slot_q;
    assign event_last_for_token = event_valid && (pending_q == selected_mask);

    always_comb begin
        selected_mask = '0;
        remaining_mask = pending_q;
        way_found = '0;
        event_lane_valid = '0;
        event_lane_ids = '0;
        event_count = '0;

        for (int way = 0; way < WAYS; way = way + 1) begin
            for (int lane = 0; lane < LANES; lane = lane + 1) begin
                if (!way_found[way] && remaining_mask[lane]) begin
                    way_found[way] = 1'b1;
                    event_lane_valid[way] = 1'b1;
                    event_lane_ids[(way*LANE_ID_W) +: LANE_ID_W] =
                        LANE_ID_W'(lane);
                    selected_mask[lane] = 1'b1;
                    remaining_mask[lane] = 1'b0;
                end
            end
        end

        for (int way = 0; way < WAYS; way = way + 1) begin
            event_count = event_count + COUNT_W'(event_lane_valid[way]);
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            tag_q <= '0;
            token_q <= '0;
            slot_q <= '0;
            pending_q <= '0;
            count_tokens <= '0;
            count_events <= '0;
            count_event_stall_cycles <= '0;
        end else begin
            if (token_fire) begin
                active_q <= (token_k_bits != '0);
                tag_q <= token_tag;
                token_q <= token_id;
                slot_q <= token_slot_id;
                pending_q <= token_k_bits;
                count_tokens <= count_tokens + 1'b1;
            end else if (event_fire) begin
                pending_q <= pending_q & ~selected_mask;
                count_events <= count_events + COUNTER_W'(event_count);
                if (event_last_for_token) begin
                    active_q <= 1'b0;
                end
            end

            if (event_valid && !event_ready) begin
                count_event_stall_cycles <= count_event_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
