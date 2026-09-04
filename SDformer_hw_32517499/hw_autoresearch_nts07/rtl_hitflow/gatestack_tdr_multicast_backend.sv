`timescale 1ns/1ps
`default_nettype none

// TDR backend for one replay session. Product generation and destination
// decoding proceed independently, rendezvous by tag/sequence, then multicast.
module gatestack_tdr_multicast_backend #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int EVENT_WAYS      = 4,
    parameter int OUT_TILE        = 8,
    parameter int BANKS           = 2,
    parameter int SEGMENT_TOKENS  = 18,
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int TAG_W           = 32,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int WAY_COUNT_W     = $clog2(EVENT_WAYS + 1),
    parameter int OUTSTANDING_W   = ISSUE_SEQ_W + 1
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      session_start_valid,
    output logic                                      session_start_ready,
    input  logic [TAG_W-1:0]                          session_tag,
    input  logic [INPUT_CH_W-1:0]                     session_input_channel_base,
    input  logic [OUTPUT_TILE_W-1:0]                  session_output_tile,

    input  logic                                      term_valid,
    output logic                                      term_ready,
    input  logic [GATE_W-1:0]                         term_gate_code,
    input  logic [LANE_ID_W-1:0]                      term_lane_id,
    input  logic [7:0]                                term_destination_count,
    input  logic [ISSUE_SEQ_W-1:0]                    term_issue_seq,
    input  logic                                      term_head_last,

    input  logic                                      event_valid,
    output logic                                      event_ready,
    input  logic [GATE_W-1:0]                         event_gate_code,
    input  logic [LANE_ID_W-1:0]                      event_lane_id,
    input  logic [EVENT_WAYS-1:0]                     event_token_valid,
    input  logic [(EVENT_WAYS*TOKEN_ID_W)-1:0]        event_token_ids,
    input  logic [WAY_COUNT_W-1:0]                    event_count,
    input  logic [ISSUE_SEQ_W-1:0]                    event_issue_seq,
    input  logic                                      event_term_first,
    input  logic                                      event_term_last,
    input  logic                                      event_head_last,

    input  logic                                      source_done_valid,
    output logic                                      source_done_ready,
    input  logic [TAG_W-1:0]                          source_done_tag,
    input  logic                                      source_done_error,
    output logic                                      decoder_done_valid,
    input  logic                                      decoder_done_ready,
    output logic [TAG_W-1:0]                          decoder_done_tag,
    output logic                                      decoder_done_error,

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

    output logic [BANKS-1:0]                          update_valid,
    input  logic [BANKS-1:0]                          update_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]             update_token_ids,
    output logic [TAG_W-1:0]                          update_tag,
    output logic [(OUT_TILE*PRODUCT_W)-1:0]           update_values,

    output logic                                      backend_done_valid,
    input  logic                                      backend_done_ready,
    output logic [TAG_W-1:0]                          backend_done_tag,
    output logic                                      backend_done_error,
    output logic                                      protocol_error,
    output logic [OUTSTANDING_W-1:0]                  outstanding_terms,
    output logic [COUNTER_W-1:0]                      count_sessions,
    output logic [COUNTER_W-1:0]                      count_terms,
    output logic [COUNTER_W-1:0]                      count_completed_terms,
    output logic [COUNTER_W-1:0]                      count_empty_sessions
);
    logic active_q;
    logic decoder_seen_q;
    logic decoder_error_q;
    logic term_seen_q;
    logic last_bitmap_seen_q;
    logic backend_done_q;
    logic [TAG_W-1:0] tag_q;
    logic [INPUT_CH_W-1:0] input_channel_base_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic [OUTSTANDING_W-1:0] outstanding_q;
    logic [ISSUE_SEQ_W-1:0] joined_seq_expected_q;

    logic session_start_fire;
    logic source_done_fire;
    logic term_fire;
    logic multicast_done_valid;
    logic multicast_done_ready;
    logic [TAG_W-1:0] multicast_done_tag;
    logic multicast_done_fire;
    logic [OUTSTANDING_W-1:0] outstanding_after;
    logic completion_after;
    logic input_channel_in_range;
    logic term_lane_in_range;
    logic [INPUT_CH_W:0] input_channel_sum;

    logic fork_term_ready;
    logic fork_product_valid;
    logic fork_product_ready;
    logic [TAG_W-1:0] fork_product_tag;
    logic [GATE_W-1:0] fork_product_gate;
    logic [INPUT_CH_W-1:0] fork_product_input_channel;
    logic [OUTPUT_TILE_W-1:0] fork_product_output_tile;
    logic [ISSUE_SEQ_W-1:0] fork_product_issue_seq;
    logic fork_bitmap_valid;
    logic fork_bitmap_ready;
    logic [TAG_W-1:0] fork_bitmap_tag;
    logic [GATE_W-1:0] fork_bitmap_gate;
    logic [LANE_ID_W-1:0] fork_bitmap_lane;
    logic [7:0] fork_bitmap_destination_count;
    logic [ISSUE_SEQ_W-1:0] fork_bitmap_issue_seq;
    logic fork_bitmap_head_last;

    logic product_valid;
    logic product_ready;
    logic [TAG_W-1:0] product_tag;
    logic [INPUT_CH_W-1:0] product_input_channel;
    logic [OUTPUT_TILE_W-1:0] product_output_tile;
    logic [ISSUE_SEQ_W-1:0] product_issue_seq;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values;
    logic product_protocol_error;

    logic bitmap_valid;
    logic bitmap_ready;
    logic [TAG_W-1:0] bitmap_tag;
    logic [GATE_W-1:0] bitmap_gate;
    logic [LANE_ID_W-1:0] bitmap_lane;
    logic [ISSUE_SEQ_W-1:0] bitmap_issue_seq;
    logic bitmap_head_last;
    logic [TOKENS-1:0] bitmap_destinations;
    logic bitmap_protocol_error;

    logic joined_valid;
    logic joined_ready;
    logic [TAG_W-1:0] joined_tag;
    logic [INPUT_CH_W-1:0] joined_input_channel;
    logic [OUTPUT_TILE_W-1:0] joined_output_tile;
    logic [ISSUE_SEQ_W-1:0] joined_issue_seq;
    logic [TOKENS-1:0] joined_destinations;
    logic [(OUT_TILE*PRODUCT_W)-1:0] joined_values;
    logic join_protocol_error;
    logic multicast_protocol_error;
    logic bitmap_event_ready;
    logic bitmap_fire;
    logic joined_fire;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] fork_count_terms;
    logic [COUNTER_W-1:0] fork_count_product_wait;
    logic [COUNTER_W-1:0] fork_count_bitmap_wait;
    logic [COUNTER_W-1:0] product_count_terms;
    logic [COUNTER_W-1:0] product_count_weight_requests;
    logic [COUNTER_W-1:0] product_count_products;
    logic [COUNTER_W-1:0] product_count_weight_wait;
    logic [COUNTER_W-1:0] product_count_output_stall;
    logic [COUNTER_W-1:0] bitmap_count_terms;
    logic [COUNTER_W-1:0] bitmap_count_events;
    logic [COUNTER_W-1:0] bitmap_count_bitmaps;
    logic [COUNTER_W-1:0] bitmap_count_term_stall;
    logic [COUNTER_W-1:0] bitmap_count_event_stall;
    logic [COUNTER_W-1:0] bitmap_count_output_stall;
    logic [COUNTER_W-1:0] join_count_terms;
    logic [COUNTER_W-1:0] join_count_product_wait;
    logic [COUNTER_W-1:0] join_count_bitmap_wait;
    logic [COUNTER_W-1:0] join_count_output_stall;
    logic [COUNTER_W-1:0] multicast_count_products;
    logic [COUNTER_W-1:0] multicast_count_destinations;
    logic [COUNTER_W-1:0] multicast_count_issue_cycles;
    logic [COUNTER_W-1:0] multicast_count_segment_advances;
    logic [COUNTER_W-1:0] multicast_count_bank_stall;
    /* verilator lint_on UNUSEDSIGNAL */

    assign session_start_ready = !active_q;
    assign session_start_fire = session_start_valid && session_start_ready;
    assign input_channel_sum = {1'b0, input_channel_base_q} +
                               (INPUT_CH_W+1)'(term_lane_id);
    assign input_channel_in_range = !input_channel_sum[INPUT_CH_W];
    assign term_lane_in_range = 32'(term_lane_id) < LANES;
    assign term_ready = active_q && !decoder_seen_q && !backend_done_q &&
                        term_lane_in_range && input_channel_in_range &&
                        fork_term_ready;
    assign term_fire = term_valid && term_ready;

    assign source_done_ready = active_q && !decoder_seen_q &&
                               decoder_done_ready;
    assign source_done_fire = source_done_valid && source_done_ready;
    assign decoder_done_valid = source_done_valid && active_q &&
                                !decoder_seen_q;
    assign decoder_done_tag = source_done_tag;
    assign decoder_done_error = source_done_error ||
                                (source_done_tag != tag_q);

    assign multicast_done_ready = 1'b1;
    assign multicast_done_fire = multicast_done_valid &&
                                 multicast_done_ready;
    assign bitmap_fire = bitmap_valid && bitmap_ready;
    assign joined_fire = joined_valid && joined_ready;
    always_comb begin
        outstanding_after = outstanding_q;
        unique case ({term_fire, multicast_done_fire})
            2'b10: outstanding_after = outstanding_q + 1'b1;
            2'b01: outstanding_after = outstanding_q - 1'b1;
            default: outstanding_after = outstanding_q;
        endcase
    end
    assign completion_after = (decoder_seen_q || source_done_fire) &&
                              outstanding_after == {OUTSTANDING_W{1'b0}};
    assign outstanding_terms = outstanding_q;
    assign backend_done_valid = backend_done_q;
    assign backend_done_tag = tag_q;
    assign backend_done_error = decoder_error_q ||
        (term_seen_q && !last_bitmap_seen_q) || product_protocol_error ||
        bitmap_protocol_error || join_protocol_error ||
        multicast_protocol_error;

    gatestack_term_fork #(
        .TAG_W(TAG_W),
        .LANE_ID_W(LANE_ID_W),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .COUNTER_W(COUNTER_W)
    ) u_term_fork (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid && active_q && !decoder_seen_q &&
                    term_lane_in_range && input_channel_in_range),
        .term_ready(fork_term_ready),
        .term_tag(tag_q),
        .term_gate_code(term_gate_code),
        .term_lane_id(term_lane_id),
        .term_input_channel(input_channel_base_q + INPUT_CH_W'(term_lane_id)),
        .term_output_tile(output_tile_q),
        .term_destination_count(term_destination_count),
        .term_issue_seq(term_issue_seq),
        .term_head_last(term_head_last),
        .product_term_valid(fork_product_valid),
        .product_term_ready(fork_product_ready),
        .product_term_tag(fork_product_tag),
        .product_term_gate_code(fork_product_gate),
        .product_term_input_channel(fork_product_input_channel),
        .product_term_output_tile(fork_product_output_tile),
        .product_term_issue_seq(fork_product_issue_seq),
        .bitmap_term_valid(fork_bitmap_valid),
        .bitmap_term_ready(fork_bitmap_ready),
        .bitmap_term_tag(fork_bitmap_tag),
        .bitmap_term_gate_code(fork_bitmap_gate),
        .bitmap_term_lane_id(fork_bitmap_lane),
        .bitmap_term_destination_count(fork_bitmap_destination_count),
        .bitmap_term_issue_seq(fork_bitmap_issue_seq),
        .bitmap_term_head_last(fork_bitmap_head_last),
        .count_terms(fork_count_terms),
        .count_product_wait_cycles(fork_count_product_wait),
        .count_bitmap_wait_cycles(fork_count_bitmap_wait)
    );

    gatestack_decoupled_product_engine #(
        .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W),
        .OUT_TILE(OUT_TILE), .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .TAG_W(TAG_W), .COUNTER_W(COUNTER_W)
    ) u_product_engine (
        .clk_core(clk_core), .rst_core(rst_core),
        .clear_error(1'b0),
        .term_valid(fork_product_valid), .term_ready(fork_product_ready),
        .term_tag(fork_product_tag), .term_gate_code(fork_product_gate),
        .term_input_channel(fork_product_input_channel),
        .term_output_tile(fork_product_output_tile),
        .term_issue_seq(fork_product_issue_seq),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready), .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .product_valid(product_valid), .product_ready(product_ready),
        .product_tag(product_tag),
        .product_input_channel(product_input_channel),
        .product_output_tile(product_output_tile),
        .product_issue_seq(product_issue_seq),
        .product_values(product_values),
        .protocol_error(product_protocol_error),
        .count_terms(product_count_terms),
        .count_weight_requests(product_count_weight_requests),
        .count_products(product_count_products),
        .count_weight_wait_cycles(product_count_weight_wait),
        .count_output_stall_cycles(product_count_output_stall)
    );

    gatestack_destination_bitmap_assembler #(
        .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(LANE_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .TAG_W(TAG_W),
        .WAY_COUNT_W(WAY_COUNT_W), .COUNTER_W(COUNTER_W)
    ) u_bitmap_assembler (
        .clk_core(clk_core), .rst_core(rst_core),
        .term_valid(fork_bitmap_valid), .term_ready(fork_bitmap_ready),
        .term_tag(fork_bitmap_tag), .term_gate_code(fork_bitmap_gate),
        .term_lane_id(fork_bitmap_lane),
        .term_destination_count(fork_bitmap_destination_count),
        .term_issue_seq(fork_bitmap_issue_seq),
        .term_head_last(fork_bitmap_head_last),
        .event_valid(event_valid && active_q && !decoder_seen_q),
        .event_ready(bitmap_event_ready), .event_gate_code(event_gate_code),
        .event_lane_id(event_lane_id),
        .event_token_valid(event_token_valid),
        .event_token_ids(event_token_ids), .event_count(event_count),
        .event_issue_seq(event_issue_seq),
        .event_term_first(event_term_first),
        .event_term_last(event_term_last), .event_head_last(event_head_last),
        .bitmap_valid(bitmap_valid), .bitmap_ready(bitmap_ready),
        .bitmap_tag(bitmap_tag), .bitmap_gate_code(bitmap_gate),
        .bitmap_lane_id(bitmap_lane), .bitmap_issue_seq(bitmap_issue_seq),
        .bitmap_head_last(bitmap_head_last),
        .bitmap_destinations(bitmap_destinations),
        .protocol_error(bitmap_protocol_error),
        .count_terms(bitmap_count_terms),
        .count_events(bitmap_count_events),
        .count_bitmaps(bitmap_count_bitmaps),
        .count_term_stall_cycles(bitmap_count_term_stall),
        .count_event_stall_cycles(bitmap_count_event_stall),
        .count_bitmap_stall_cycles(bitmap_count_output_stall)
    );
    assign event_ready = active_q && !decoder_seen_q && bitmap_event_ready;

    gatestack_product_bitmap_join #(
        .TOKENS(TOKENS), .OUT_TILE(OUT_TILE), .PRODUCT_W(PRODUCT_W),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W)
    ) u_product_bitmap_join (
        .clk_core(clk_core), .rst_core(rst_core),
        .product_valid(product_valid), .product_ready(product_ready),
        .product_tag(product_tag),
        .product_input_channel(product_input_channel),
        .product_output_tile(product_output_tile),
        .product_issue_seq(product_issue_seq),
        .product_values(product_values), .bitmap_valid(bitmap_valid),
        .bitmap_ready(bitmap_ready), .bitmap_tag(bitmap_tag),
        .bitmap_issue_seq(bitmap_issue_seq),
        .bitmap_destinations(bitmap_destinations),
        .joined_valid(joined_valid), .joined_ready(joined_ready),
        .joined_tag(joined_tag),
        .joined_input_channel(joined_input_channel),
        .joined_output_tile(joined_output_tile),
        .joined_issue_seq(joined_issue_seq),
        .joined_destinations(joined_destinations),
        .joined_values(joined_values), .protocol_error(join_protocol_error),
        .count_joined_terms(join_count_terms),
        .count_product_wait_cycles(join_count_product_wait),
        .count_bitmap_wait_cycles(join_count_bitmap_wait),
        .count_output_stall_cycles(join_count_output_stall)
    );

    hitflow_segmented_multicast #(
        .TOKENS(TOKENS), .SEGMENT_TOKENS(SEGMENT_TOKENS),
        .BANKS(BANKS), .PRODUCT_W(PRODUCT_W), .OUT_TILE(OUT_TILE),
        .TAG_W(TAG_W), .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W)
    ) u_multicast (
        .clk_core(clk_core), .rst_core(rst_core),
        .product_valid(joined_valid), .product_ready(joined_ready),
        .product_tag(joined_tag),
        .product_destination_bitmap(joined_destinations),
        .product_values(joined_values), .update_valid(update_valid),
        .update_ready(update_ready), .update_token_ids(update_token_ids),
        .update_tag(update_tag), .update_values(update_values),
        .product_done_valid(multicast_done_valid),
        .product_done_ready(multicast_done_ready),
        .product_done_tag(multicast_done_tag),
        .protocol_error(multicast_protocol_error),
        .count_products(multicast_count_products),
        .count_destinations(multicast_count_destinations),
        .count_issue_cycles(multicast_count_issue_cycles),
        .count_segment_advances(multicast_count_segment_advances),
        .count_bank_stall_cycles(multicast_count_bank_stall)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            decoder_seen_q <= 1'b0;
            decoder_error_q <= 1'b0;
            term_seen_q <= 1'b0;
            last_bitmap_seen_q <= 1'b0;
            backend_done_q <= 1'b0;
            tag_q <= '0;
            input_channel_base_q <= '0;
            output_tile_q <= '0;
            outstanding_q <= '0;
            joined_seq_expected_q <= '0;
            protocol_error <= 1'b0;
            count_sessions <= '0;
            count_terms <= '0;
            count_completed_terms <= '0;
            count_empty_sessions <= '0;
        end else begin
            if (session_start_fire) begin
                active_q <= 1'b1;
                decoder_seen_q <= 1'b0;
                decoder_error_q <= 1'b0;
                term_seen_q <= 1'b0;
                last_bitmap_seen_q <= 1'b0;
                backend_done_q <= 1'b0;
                tag_q <= session_tag;
                input_channel_base_q <= session_input_channel_base;
                output_tile_q <= session_output_tile;
                outstanding_q <= '0;
                joined_seq_expected_q <= '0;
                count_sessions <= count_sessions + 1'b1;
            end else if (active_q) begin
                outstanding_q <= outstanding_after;
                if (term_fire) begin
                    term_seen_q <= 1'b1;
                    count_terms <= count_terms + 1'b1;
                end
                if (multicast_done_fire)
                    count_completed_terms <= count_completed_terms + 1'b1;
                if (bitmap_fire && bitmap_head_last)
                    last_bitmap_seen_q <= 1'b1;
                if (joined_fire)
                    joined_seq_expected_q <= joined_seq_expected_q + 1'b1;
                if (source_done_fire) begin
                    decoder_seen_q <= 1'b1;
                    decoder_error_q <= decoder_done_error;
                    if (!term_seen_q && !term_fire)
                        count_empty_sessions <= count_empty_sessions + 1'b1;
                end
                if (!backend_done_q && completion_after)
                    backend_done_q <= 1'b1;
                if (backend_done_valid && backend_done_ready) begin
                    active_q <= 1'b0;
                    backend_done_q <= 1'b0;
                end
            end

            if ((multicast_done_fire && outstanding_q == 0 && !term_fire) ||
                (multicast_done_valid && multicast_done_tag != tag_q) ||
                (term_valid && active_q &&
                 (!term_lane_in_range || !input_channel_in_range)) ||
                (bitmap_fire &&
                 (bitmap_gate == 0 || 32'(bitmap_lane) >= LANES)) ||
                (joined_fire &&
                 (joined_tag != tag_q ||
                  joined_output_tile != output_tile_q ||
                  joined_input_channel < input_channel_base_q ||
                  joined_issue_seq != joined_seq_expected_q)) ||
                product_protocol_error || bitmap_protocol_error ||
                join_protocol_error || multicast_protocol_error) begin
                protocol_error <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
