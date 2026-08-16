`timescale 1ns/1ps
`default_nettype none

// G1 projection integration:
//   tokens -> NMF directory -> gate*weight product -> segmented multicast
//          -> banked accumulator -> bias-commit finals
// Fallback/overflow is reported but not expanded in this slice; tests must keep
// unique final-gate codes within SLOTS so overflow_seen stays 0.
module hitflow_g1_projection_top #(
    parameter int TOKENS         = 162,
    parameter int LANES          = 32,
    parameter int SLOTS          = 4,
    parameter int GATE_W         = 9,
    parameter int WEIGHT_W       = 8,
    parameter int PRODUCT_W      = GATE_W + WEIGHT_W,
    parameter int ACC_W          = 32,
    parameter int OUT_TILE       = 8,
    parameter int BANKS          = 2,
    parameter int SEGMENT_TOKENS = 18,
    parameter int TAG_W          = 32,
    parameter int COUNTER_W      = 32,
    parameter int TOKEN_ID_W     = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W      = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int INPUT_CH_W     = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter int OUTPUT_TILE_W  = 4
) (
    input  logic                            clk_core,
    input  logic                            rst_core,

    input  logic                            group_valid,
    output logic                            group_ready,
    input  logic [TAG_W-1:0]                group_tag,

    input  logic                            token_valid,
    output logic                            token_ready,
    input  logic [TOKEN_ID_W-1:0]           token_id,
    input  logic [GATE_W-1:0]               token_gate_code,
    input  logic [LANES-1:0]                token_k_bits,
    input  logic                            token_last,

    output logic                            weight_req_valid,
    input  logic                            weight_req_ready,
    output logic [TAG_W-1:0]                weight_req_tag,
    output logic [INPUT_CH_W-1:0]           weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]        weight_req_output_tile,

    input  logic                            weight_rsp_valid,
    output logic                            weight_rsp_ready,
    input  logic [TAG_W-1:0]                weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]           weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]        weight_rsp_output_tile,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]  weight_rsp_weights,

    // After products drain, host provides one bias vector per token in order 0..TOKENS-1.
    output logic                            bias_req_valid,
    input  logic                            bias_req_ready,
    output logic [TOKEN_ID_W-1:0]           bias_req_token_id,
    input  logic [(OUT_TILE*PRODUCT_W)-1:0] bias_req_values,

    output logic [BANKS-1:0]                final_valid,
    input  logic [BANKS-1:0]                final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]   final_token_ids,
    output logic [TAG_W-1:0]                final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,

    output logic                            group_done_valid,
    input  logic                            group_done_ready,
    output logic [TAG_W-1:0]                group_done_tag,
    output logic                            overflow_seen,
    output logic                            protocol_error,
    output logic                            accumulator_overflow,

    output logic [COUNTER_W-1:0]            count_tokens,
    output logic [COUNTER_W-1:0]            count_terms,
    output logic [COUNTER_W-1:0]            count_products,
    output logic [COUNTER_W-1:0]            count_bias_commits
);

    localparam OUT_TILE_LOOP = OUT_TILE;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_RUN,
        ST_WAIT_DRAIN,
        ST_BIAS,
        ST_FINISH,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;

    logic nmf_group_valid;
    logic nmf_group_ready;
    logic nmf_token_ready;
    logic nmf_term_valid;
    logic nmf_term_ready;
    logic [TAG_W-1:0] nmf_term_tag;
    logic [GATE_W-1:0] nmf_term_gate;
    logic [LANE_ID_W-1:0] nmf_term_lane;
    logic [TOKENS-1:0] nmf_term_bitmap;
    logic nmf_fallback_valid;
    logic nmf_group_done_valid;
    logic nmf_group_done_ready;
    logic [TAG_W-1:0] nmf_group_done_tag;
    logic nmf_overflow;
    logic nmf_protocol_error;
    logic [COUNTER_W-1:0] nmf_count_tokens;
    logic [COUNTER_W-1:0] nmf_count_terms;
    logic [COUNTER_W-1:0] nmf_count_fallback;
    logic [COUNTER_W-1:0] nmf_count_active_lanes;

    logic product_term_valid;
    logic product_term_ready;
    logic product_valid;
    logic product_ready;
    logic [TAG_W-1:0] product_tag;
    logic [INPUT_CH_W-1:0] product_input_channel;
    logic [OUTPUT_TILE_W-1:0] product_output_tile;
    logic [TOKENS-1:0] product_bitmap;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values;
    logic product_protocol_error;
    logic [COUNTER_W-1:0] product_count_terms;
    logic [COUNTER_W-1:0] product_count_weight_requests;
    logic [COUNTER_W-1:0] product_count_products;
    logic [COUNTER_W-1:0] product_count_weight_wait;
    logic [COUNTER_W-1:0] product_count_output_stall;

    logic [BANKS-1:0] mcast_update_valid;
    logic [BANKS-1:0] mcast_update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] mcast_update_token_ids;
    logic [TAG_W-1:0] mcast_update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] mcast_update_values;
    logic mcast_done_valid;
    logic mcast_done_ready;
    logic [TAG_W-1:0] mcast_done_tag;
    logic mcast_protocol_error;
    logic [COUNTER_W-1:0] mcast_count_products;
    logic [COUNTER_W-1:0] mcast_count_destinations;
    logic [COUNTER_W-1:0] mcast_count_issue;
    logic [COUNTER_W-1:0] mcast_count_seg;
    logic [COUNTER_W-1:0] mcast_count_bank_stall;

    logic acc_group_start_valid;
    logic acc_group_start_ready;
    logic [BANKS-1:0] acc_update_valid;
    logic [BANKS-1:0] acc_update_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [TAG_W-1:0] acc_update_tag;
    logic acc_update_is_bias;
    logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic [(OUT_TILE*ACC_W)-1:0] acc_update_bias_values;
    logic acc_group_finish_valid;
    logic acc_group_finish_ready;
    logic [TAG_W-1:0] acc_group_finish_tag;
    logic acc_protocol_error;
    logic [COUNTER_W-1:0] acc_count_updates;
    logic [COUNTER_W-1:0] acc_count_writes;
    logic [COUNTER_W-1:0] acc_count_bias;
    logic [COUNTER_W-1:0] acc_count_bank_stall;
    logic [COUNTER_W-1:0] acc_count_final_stall;

    logic [TOKEN_ID_W:0] bias_token_q;
    logic bias_issue_fire;
    logic pipe_idle;
    logic sticky_overflow_q;
    logic sticky_protocol_q;

    logic mcast_product_ready;
    logic allow_stream;

    assign allow_stream = (state_q == ST_RUN) || (state_q == ST_WAIT_DRAIN);
    assign group_ready = (state_q == ST_IDLE) && acc_group_start_ready &&
                         nmf_group_ready;
    assign token_ready = (state_q == ST_RUN) && nmf_token_ready;
    assign nmf_group_valid = (state_q == ST_IDLE) && group_valid &&
                             acc_group_start_ready && nmf_group_ready;
    assign acc_group_start_valid = nmf_group_valid;
    // Keep accepting NMF group_done until NMF leaves ST_DONE; otherwise NMF
    // cannot return to IDLE for the next window.
    assign nmf_group_done_ready = nmf_group_done_valid;
    assign mcast_done_ready = 1'b1;

    // Directory path only in this slice.
    assign product_term_valid = allow_stream && nmf_term_valid;
    assign nmf_term_ready = allow_stream && product_term_ready;

    // Idle when no in-flight directory term/product/multicast work remains.
    assign pipe_idle = !nmf_term_valid && !product_valid &&
                       mcast_product_ready && !mcast_done_valid &&
                       (mcast_update_valid == '0) && allow_stream;

    assign bias_req_valid = (state_q == ST_BIAS);
    assign bias_req_token_id = bias_token_q[TOKEN_ID_W-1:0];
    assign bias_issue_fire = bias_req_valid && bias_req_ready &&
                             (acc_update_ready[bias_token_q % BANKS]);

    // Arbiter: product path vs bias path.
    always_comb begin
        acc_update_valid = '0;
        acc_update_token_ids = '0;
        acc_update_tag = tag_q;
        acc_update_is_bias = 1'b0;
        acc_update_values = '0;
        acc_update_bias_values = '0;
        mcast_update_ready = '0;

        if (state_q == ST_BIAS) begin
            acc_update_is_bias = 1'b1;
            acc_update_values = bias_req_values;
            for (int lane = 32'd0;
                 lane < OUT_TILE_LOOP;
                 lane = lane + 32'd1) begin
                acc_update_bias_values[(lane*ACC_W) +: ACC_W] = {
                    {(ACC_W-PRODUCT_W){
                        bias_req_values[(lane*PRODUCT_W)+PRODUCT_W-1]
                    }},
                    bias_req_values[(lane*PRODUCT_W) +: PRODUCT_W]
                };
            end
            acc_update_valid[bias_token_q % BANKS] = bias_req_valid &&
                                                     bias_req_ready;
            acc_update_token_ids[(bias_token_q % BANKS)*TOKEN_ID_W +: TOKEN_ID_W] =
                bias_token_q[TOKEN_ID_W-1:0];
        end else begin
            acc_update_valid = mcast_update_valid;
            acc_update_token_ids = mcast_update_token_ids;
            acc_update_tag = mcast_update_tag;
            acc_update_is_bias = 1'b0;
            acc_update_values = mcast_update_values;
            mcast_update_ready = acc_update_ready;
        end
    end

    assign acc_group_finish_valid = (state_q == ST_FINISH);
    assign group_done_valid = (state_q == ST_DONE);
    assign group_done_tag = tag_q;
    assign overflow_seen = sticky_overflow_q;
    assign protocol_error = sticky_protocol_q || nmf_protocol_error ||
                            product_protocol_error || mcast_protocol_error ||
                            acc_protocol_error;
    assign count_tokens = nmf_count_tokens;
    assign count_terms = nmf_count_terms;
    assign count_products = product_count_products;
    assign count_bias_commits = acc_count_bias;

    hitflow_nmf_g1_builder #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .GATE_W(GATE_W),
        .SLOTS(SLOTS),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W)
    ) u_nmf (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .group_valid(nmf_group_valid),
        .group_ready(nmf_group_ready),
        .group_tag(group_tag),
        .token_valid(token_valid && (state_q == ST_RUN)),
        .token_ready(nmf_token_ready),
        .token_id(token_id),
        .token_gate_code(token_gate_code),
        .token_k_bits(token_k_bits),
        .token_last(token_last),
        .term_valid(nmf_term_valid),
        .term_ready(nmf_term_ready),
        .term_tag(nmf_term_tag),
        .term_gate_code(nmf_term_gate),
        .term_lane(nmf_term_lane),
        .term_destination_bitmap(nmf_term_bitmap),
        .fallback_valid(nmf_fallback_valid),
        .fallback_ready(1'b1),
        .fallback_tag(),
        .fallback_token_id(),
        .fallback_gate_code(),
        .fallback_k_bits(),
        .group_done_valid(nmf_group_done_valid),
        .group_done_ready(nmf_group_done_ready),
        .group_done_tag(nmf_group_done_tag),
        .overflow_seen(nmf_overflow),
        .protocol_error(nmf_protocol_error),
        .count_tokens(nmf_count_tokens),
        .count_active_lanes(nmf_count_active_lanes),
        .count_terms(nmf_count_terms),
        .count_fallback_tokens(nmf_count_fallback)
    );

    hitflow_gate_product_engine #(
        .TOKENS(TOKENS),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .OUT_TILE(OUT_TILE),
        .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W)
    ) u_product (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(product_term_valid),
        .term_ready(product_term_ready),
        .term_tag(nmf_term_tag),
        .term_gate_code(nmf_term_gate),
        .term_input_channel(INPUT_CH_W'(nmf_term_lane)),
        .term_output_tile('0),
        .term_destination_bitmap(nmf_term_bitmap),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready),
        .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready),
        .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .product_valid(product_valid),
        .product_ready(mcast_product_ready && allow_stream),
        .product_tag(product_tag),
        .product_input_channel(product_input_channel),
        .product_output_tile(product_output_tile),
        .product_destination_bitmap(product_bitmap),
        .product_values(product_values),
        .protocol_error(product_protocol_error),
        .count_terms(product_count_terms),
        .count_weight_requests(product_count_weight_requests),
        .count_products(product_count_products),
        .count_weight_wait_cycles(product_count_weight_wait),
        .count_output_stall_cycles(product_count_output_stall)
    );

    hitflow_segmented_multicast #(
        .TOKENS(TOKENS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS),
        .BANKS(BANKS),
        .PRODUCT_W(PRODUCT_W),
        .OUT_TILE(OUT_TILE),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W)
    ) u_mcast (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .product_valid(product_valid && allow_stream),
        .product_ready(mcast_product_ready),
        .product_tag(product_tag),
        .product_destination_bitmap(product_bitmap),
        .product_values(product_values),
        .update_valid(mcast_update_valid),
        .update_ready(mcast_update_ready),
        .update_token_ids(mcast_update_token_ids),
        .update_tag(mcast_update_tag),
        .update_values(mcast_update_values),
        .product_done_valid(mcast_done_valid),
        .product_done_ready(mcast_done_ready),
        .product_done_tag(mcast_done_tag),
        .protocol_error(mcast_protocol_error),
        .count_products(mcast_count_products),
        .count_destinations(mcast_count_destinations),
        .count_issue_cycles(mcast_count_issue),
        .count_segment_advances(mcast_count_seg),
        .count_bank_stall_cycles(mcast_count_bank_stall)
    );

    hitflow_banked_accumulator #(
        .TOKENS(TOKENS),
        .BANKS(BANKS),
        .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W),
        .OUT_TILE(OUT_TILE),
        .TAG_W(TAG_W),
        .COUNTER_W(COUNTER_W),
        .TOKEN_ID_W(TOKEN_ID_W)
    ) u_acc (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(1'b0),
        .group_start_valid(acc_group_start_valid),
        .group_start_ready(acc_group_start_ready),
        .group_start_tag(group_tag),
        .update_valid(acc_update_valid),
        .update_ready(acc_update_ready),
        .update_token_ids(acc_update_token_ids),
        .update_tag(acc_update_tag),
        .update_is_bias(acc_update_is_bias),
        .update_values(acc_update_values),
        .update_bias_values(acc_update_bias_values),
        .final_valid(final_valid),
        .final_ready(final_ready),
        .final_token_ids(final_token_ids),
        .final_tag(final_tag),
        .final_values(final_values),
        .group_finish_valid(acc_group_finish_valid),
        .group_finish_ready(acc_group_finish_ready),
        .group_finish_tag(acc_group_finish_tag),
        .protocol_error(acc_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_updates(acc_count_updates),
        .count_writes(acc_count_writes),
        .count_bias_commits(acc_count_bias),
        .count_bank_stall_cycles(acc_count_bank_stall),
        .count_final_stall_cycles(acc_count_final_stall)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            bias_token_q <= '0;
            sticky_overflow_q <= 1'b0;
            sticky_protocol_q <= 1'b0;
        end else begin
            if (nmf_overflow || nmf_fallback_valid) begin
                sticky_overflow_q <= 1'b1;
            end
            if (nmf_protocol_error || product_protocol_error ||
                mcast_protocol_error || acc_protocol_error) begin
                sticky_protocol_q <= 1'b1;
            end

            // Use explicit if-else (not unique case) for iverilog-friendly control.
            if (state_q == ST_IDLE) begin
                // Align controller start with NMF/accumulator group accept.
                if (nmf_group_valid) begin
                    tag_q <= group_tag;
                    bias_token_q <= '0;
                    state_q <= ST_RUN;
                end
            end else if (state_q == ST_RUN) begin
                if (nmf_group_done_valid && nmf_group_done_ready) begin
                    state_q <= ST_WAIT_DRAIN;
                end
            end else if (state_q == ST_WAIT_DRAIN) begin
                if (pipe_idle) begin
                    bias_token_q <= '0;
                    state_q <= ST_BIAS;
                end
            end else if (state_q == ST_BIAS) begin
                if (bias_issue_fire) begin
                    if (bias_token_q == (TOKEN_ID_W+1)'(TOKENS - 1)) begin
                        state_q <= ST_FINISH;
                    end else begin
                        bias_token_q <= bias_token_q + 1'b1;
                    end
                end
            end else if (state_q == ST_FINISH) begin
                if (acc_group_finish_ready) begin
                    state_q <= ST_DONE;
                end
            end else if (state_q == ST_DONE) begin
                if (group_done_ready) begin
                    state_q <= ST_IDLE;
                end
            end else begin
                state_q <= ST_IDLE;
            end
        end
    end

endmodule

`default_nettype wire
