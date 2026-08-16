`timescale 1ns/1ps
`default_nettype none

module hitflow_nmf_g1_builder #(
    parameter int TOKENS    = 162,
    parameter int LANES     = 32,
    parameter int GATE_W    = 9,
    parameter int SLOTS     = 4,
    parameter int TAG_W     = 32,
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W  = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         group_valid,
    output logic                         group_ready,
    input  logic [TAG_W-1:0]             group_tag,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [TOKEN_ID_W-1:0]        token_id,
    input  logic [GATE_W-1:0]            token_gate_code,
    input  logic [LANES-1:0]             token_k_bits,
    input  logic                         token_last,

    output logic                         term_valid,
    input  logic                         term_ready,
    output logic [TAG_W-1:0]             term_tag,
    output logic [GATE_W-1:0]            term_gate_code,
    output logic [LANE_ID_W-1:0]         term_lane,
    output logic [TOKENS-1:0]            term_destination_bitmap,

    output logic                         fallback_valid,
    input  logic                         fallback_ready,
    output logic [TAG_W-1:0]             fallback_tag,
    output logic [TOKEN_ID_W-1:0]        fallback_token_id,
    output logic [GATE_W-1:0]            fallback_gate_code,
    output logic [LANES-1:0]             fallback_k_bits,

    output logic                         group_done_valid,
    input  logic                         group_done_ready,
    output logic [TAG_W-1:0]             group_done_tag,
    output logic                         overflow_seen,
    output logic                         protocol_error,

    output logic [COUNTER_W-1:0]         count_tokens,
    output logic [COUNTER_W-1:0]         count_active_lanes,
    output logic [COUNTER_W-1:0]         count_terms,
    output logic [COUNTER_W-1:0]         count_fallback_tokens
);

    localparam int SLOT_ID_W  = (SLOTS <= 1) ? 1 : $clog2(SLOTS);
    localparam int DESTINATION_BITS = SLOTS * LANES * TOKENS;
    localparam int SCAN_BASE_W =
        (DESTINATION_BITS <= 1) ? 1 : $clog2(DESTINATION_BITS);
    localparam logic [TOKEN_ID_W:0] TOKEN_COUNT_VALUE =
        (TOKEN_ID_W + 1)'(TOKENS);
    localparam logic [TOKEN_ID_W:0] LAST_TOKEN_VALUE =
        (TOKEN_ID_W + 1)'(TOKENS - 1);
    localparam logic [SLOT_ID_W-1:0] LAST_SLOT_VALUE =
        SLOT_ID_W'(SLOTS - 1);
    localparam logic [LANE_ID_W-1:0] LAST_LANE_VALUE =
        LANE_ID_W'(LANES - 1);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_BUILD,
        ST_DRAIN_DIRECTORY,
        ST_DRAIN_FALLBACK,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [TOKEN_ID_W:0] expected_token_q;
    logic [SLOTS-1:0] slot_valid_q;
    logic [GATE_W-1:0] slot_gate_q [0:SLOTS-1];
    logic [DESTINATION_BITS-1:0] destination_flat;
    logic fallback_pending_q;
    logic [TOKEN_ID_W-1:0] fallback_token_q;
    logic [GATE_W-1:0] fallback_gate_q;
    logic [LANES-1:0] fallback_k_q;
    logic [SLOT_ID_W-1:0] scan_slot_q;
    logic [LANE_ID_W-1:0] scan_lane_q;

    logic match_found;
    logic free_found;
    logic [SLOT_ID_W-1:0] match_slot;
    logic [SLOT_ID_W-1:0] free_slot;
    logic token_has_product;
    logic token_order_ok;
    logic token_last_ok;
    logic token_protocol_ok;
    logic token_fire;
    logic allocate_new_slot;
    logic [TOKENS-1:0] token_onehot;
    logic current_term_nonempty;
    logic directory_advance;
    logic directory_last;
    logic [COUNTER_W-1:0] token_active_lane_count;
    logic [SCAN_BASE_W-1:0] scan_destination_base_q;

    always_comb begin
        match_found = 1'b0;
        free_found = 1'b0;
        match_slot = '0;
        free_slot = '0;
        for (int slot = 0; slot < SLOTS; slot = slot + 1) begin
            if (!match_found && slot_valid_q[slot] &&
                (slot_gate_q[slot] == token_gate_code)) begin
                match_found = 1'b1;
                match_slot = slot[SLOT_ID_W-1:0];
            end
            if (!free_found && !slot_valid_q[slot]) begin
                free_found = 1'b1;
                free_slot = slot[SLOT_ID_W-1:0];
            end
        end
    end

    always_comb begin
        token_active_lane_count = '0;
        for (int lane = 0; lane < LANES; lane = lane + 1) begin
            if (token_k_bits[lane]) begin
                token_active_lane_count = token_active_lane_count + 1'b1;
            end
        end
    end

    always_comb begin
        token_onehot = '0;
        if ({1'b0, token_id} < TOKEN_COUNT_VALUE) begin
            token_onehot[token_id] = 1'b1;
        end
    end

    assign group_ready = (state_q == ST_IDLE);
    assign token_has_product = (token_gate_code != '0) && (|token_k_bits);
    assign token_order_ok = ({1'b0, token_id} == expected_token_q);
    assign token_last_ok = token_last == (expected_token_q == LAST_TOKEN_VALUE);
    assign token_protocol_ok = token_order_ok && token_last_ok;
    assign token_ready = (state_q == ST_BUILD) && token_protocol_ok &&
                         !fallback_pending_q;
    assign token_fire = token_valid && token_ready;
    assign allocate_new_slot = token_has_product && !match_found && free_found;
    assign protocol_error = token_valid &&
                            ((state_q != ST_BUILD) || !token_protocol_ok);

    assign current_term_nonempty = slot_valid_q[scan_slot_q] &&
        (|destination_flat[scan_destination_base_q +: TOKENS]);
    assign term_valid = (state_q == ST_DRAIN_DIRECTORY) && current_term_nonempty;
    assign term_tag = tag_q;
    assign term_gate_code = slot_gate_q[scan_slot_q];
    assign term_lane = scan_lane_q;
    assign term_destination_bitmap =
        destination_flat[scan_destination_base_q +: TOKENS];
    assign directory_advance = (state_q == ST_DRAIN_DIRECTORY) &&
                               (!current_term_nonempty || term_ready);
    assign directory_last = (scan_slot_q == LAST_SLOT_VALUE) &&
                            (scan_lane_q == LAST_LANE_VALUE);

    assign fallback_valid = fallback_pending_q;
    assign fallback_tag = tag_q;
    assign fallback_token_id = fallback_token_q;
    assign fallback_gate_code = fallback_gate_q;
    assign fallback_k_bits = fallback_k_q;

    assign group_done_valid = (state_q == ST_DONE);
    assign group_done_tag = tag_q;

    for (genvar slot = 0; slot < SLOTS; slot = slot + 1) begin : g_destination_slot
        for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : g_destination_lane
            localparam int DESTINATION_BASE = (slot * LANES + lane) * TOKENS;
            logic [TOKENS-1:0] bitmap_q;

            assign destination_flat[DESTINATION_BASE +: TOKENS] = bitmap_q;

            always_ff @(posedge clk_core) begin
                if (!rst_core && token_fire) begin
                    if (allocate_new_slot && (free_slot == slot[SLOT_ID_W-1:0])) begin
                        bitmap_q <=
                            token_k_bits[lane] ? token_onehot : '0;
                    end else if (token_has_product && match_found &&
                                 (match_slot == slot[SLOT_ID_W-1:0]) &&
                                 token_k_bits[lane]) begin
                        bitmap_q[token_id] <= 1'b1;
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q              <= ST_IDLE;
            tag_q                <= '0;
            expected_token_q     <= '0;
            slot_valid_q         <= '0;
            fallback_pending_q   <= 1'b0;
            fallback_token_q     <= '0;
            fallback_gate_q      <= '0;
            fallback_k_q         <= '0;
            scan_slot_q          <= '0;
            scan_lane_q          <= '0;
            scan_destination_base_q <= '0;
            overflow_seen        <= 1'b0;
            count_tokens         <= '0;
            count_active_lanes   <= '0;
            count_terms          <= '0;
            count_fallback_tokens <= '0;
        end else begin
            if (fallback_valid && fallback_ready) begin
                fallback_pending_q <= 1'b0;
                count_fallback_tokens <= count_fallback_tokens + 1'b1;
            end
            unique case (state_q)
                ST_IDLE: begin
                    if (group_valid && group_ready) begin
                        state_q              <= ST_BUILD;
                        tag_q                <= group_tag;
                        expected_token_q     <= '0;
                        slot_valid_q         <= '0;
                        fallback_pending_q   <= 1'b0;
                        scan_slot_q          <= '0;
                        scan_lane_q          <= '0;
                        scan_destination_base_q <= '0;
                        overflow_seen        <= 1'b0;
                        count_tokens         <= '0;
                        count_active_lanes   <= '0;
                        count_terms          <= '0;
                        count_fallback_tokens <= '0;
                    end
                end

                ST_BUILD: begin
                    if (token_fire) begin
                        count_tokens <= count_tokens + 1'b1;
                        count_active_lanes <= count_active_lanes +
                                              token_active_lane_count;
                        if (token_has_product) begin
                            if (!match_found) begin
                                if (free_found) begin
                                    slot_valid_q[free_slot] <= 1'b1;
                                    slot_gate_q[free_slot] <= token_gate_code;
                                end else begin
                                    fallback_token_q <= token_id;
                                    fallback_gate_q <= token_gate_code;
                                    fallback_k_q <= token_k_bits;
                                    fallback_pending_q <= 1'b1;
                                    overflow_seen <= 1'b1;
                                end
                            end
                        end
                        if (token_last) begin
                            state_q <= ST_DRAIN_DIRECTORY;
                            scan_slot_q <= '0;
                            scan_lane_q <= '0;
                            scan_destination_base_q <= '0;
                        end else begin
                            expected_token_q <= expected_token_q + 1'b1;
                        end
                    end
                end

                ST_DRAIN_DIRECTORY: begin
                    if (directory_advance) begin
                        if (current_term_nonempty && term_ready) begin
                            count_terms <= count_terms + 1'b1;
                        end
                        if (directory_last) begin
                            state_q <= fallback_pending_q ?
                                       ST_DRAIN_FALLBACK : ST_DONE;
                        end else if (scan_lane_q == LAST_LANE_VALUE) begin
                            scan_lane_q <= '0;
                            scan_slot_q <= scan_slot_q + 1'b1;
                            scan_destination_base_q <= scan_destination_base_q +
                                                       SCAN_BASE_W'(TOKENS);
                        end else begin
                            scan_lane_q <= scan_lane_q + 1'b1;
                            scan_destination_base_q <= scan_destination_base_q +
                                                       SCAN_BASE_W'(TOKENS);
                        end
                    end
                end

                ST_DRAIN_FALLBACK: begin
                    if (!fallback_pending_q ||
                        (fallback_valid && fallback_ready)) begin
                        state_q <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    if (group_done_valid && group_done_ready) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
