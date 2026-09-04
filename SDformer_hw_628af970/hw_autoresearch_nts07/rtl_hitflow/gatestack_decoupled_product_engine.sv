`timescale 1ns/1ps
`default_nettype none

// Generates gate*weight products independently of destination decoding.
// The issue sequence later rendezvous with the exact destination bitmap.
module gatestack_decoupled_product_engine #(
    parameter int GATE_W          = 9,
    parameter int WEIGHT_W        = 8,
    parameter int PRODUCT_W       = GATE_W + WEIGHT_W,
    parameter int OUT_TILE        = 8,
    parameter int INPUT_CH_W      = 10,
    parameter int OUTPUT_TILE_W   = 8,
    parameter int ISSUE_SEQ_W     = 13,
    parameter int TAG_W           = 32,
    parameter int COUNTER_W       = 32
) (
    input  logic                              clk_core,
    input  logic                              rst_core,
    input  logic                              clear_error,

    input  logic                              term_valid,
    output logic                              term_ready,
    input  logic [TAG_W-1:0]                  term_tag,
    input  logic [GATE_W-1:0]                 term_gate_code,
    input  logic [INPUT_CH_W-1:0]             term_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]          term_output_tile,
    input  logic [ISSUE_SEQ_W-1:0]            term_issue_seq,

    output logic                              weight_req_valid,
    input  logic                              weight_req_ready,
    output logic [TAG_W-1:0]                  weight_req_tag,
    output logic [INPUT_CH_W-1:0]             weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]          weight_req_output_tile,

    input  logic                              weight_rsp_valid,
    output logic                              weight_rsp_ready,
    input  logic [TAG_W-1:0]                  weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]             weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]          weight_rsp_output_tile,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]    weight_rsp_weights,

    output logic                              product_valid,
    input  logic                              product_ready,
    output logic [TAG_W-1:0]                  product_tag,
    output logic [INPUT_CH_W-1:0]             product_input_channel,
    output logic [OUTPUT_TILE_W-1:0]          product_output_tile,
    output logic [ISSUE_SEQ_W-1:0]            product_issue_seq,
    output logic [(OUT_TILE*PRODUCT_W)-1:0]   product_values,

    output logic                              protocol_error,
    output logic [COUNTER_W-1:0]              count_terms,
    output logic [COUNTER_W-1:0]              count_weight_requests,
    output logic [COUNTER_W-1:0]              count_products,
    output logic [COUNTER_W-1:0]              count_weight_wait_cycles,
    output logic [COUNTER_W-1:0]              count_output_stall_cycles
);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_WEIGHT_REQUEST,
        ST_WEIGHT_RESPONSE,
        ST_PRODUCT_OUTPUT
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [GATE_W-1:0] gate_q;
    logic [INPUT_CH_W-1:0] input_channel_q;
    logic [OUTPUT_TILE_W-1:0] output_tile_q;
    logic [ISSUE_SEQ_W-1:0] issue_seq_q;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values_q;
    logic term_protocol_ok;
    logic response_matches;
    logic term_fire;
    logic weight_req_fire;
    logic weight_rsp_fire;
    logic product_fire;

    assign term_protocol_ok = term_gate_code != 0;
    assign term_ready = state_q == ST_IDLE &&
                        (!term_valid || term_protocol_ok);
    assign term_fire = term_valid && term_ready;

    assign weight_req_valid = state_q == ST_WEIGHT_REQUEST;
    assign weight_req_tag = tag_q;
    assign weight_req_input_channel = input_channel_q;
    assign weight_req_output_tile = output_tile_q;
    assign weight_req_fire = weight_req_valid && weight_req_ready;

    assign response_matches = weight_rsp_tag == tag_q &&
        weight_rsp_input_channel == input_channel_q &&
        weight_rsp_output_tile == output_tile_q;
    assign weight_rsp_ready = state_q == ST_WEIGHT_RESPONSE &&
                              response_matches;
    assign weight_rsp_fire = weight_rsp_valid && weight_rsp_ready;

    assign product_valid = state_q == ST_PRODUCT_OUTPUT;
    assign product_tag = tag_q;
    assign product_input_channel = input_channel_q;
    assign product_output_tile = output_tile_q;
    assign product_issue_seq = issue_seq_q;
    assign product_values = product_values_q;
    assign product_fire = product_valid && product_ready;

    for (genvar output_lane = 0; output_lane < OUT_TILE;
         output_lane = output_lane + 1) begin : g_product_lane
        logic signed [WEIGHT_W-1:0] weight_value;
        logic signed [GATE_W:0] gate_positive;
        logic signed [PRODUCT_W-1:0] exact_product;
        assign weight_value =
            weight_rsp_weights[(output_lane*WEIGHT_W) +: WEIGHT_W];
        assign gate_positive = $signed({1'b0, gate_q});
        assign exact_product = PRODUCT_W'(gate_positive * weight_value);
        always_ff @(posedge clk_core) begin
            if (rst_core) begin
                product_values_q[(output_lane*PRODUCT_W) +: PRODUCT_W] <= '0;
            end else if (weight_rsp_fire) begin
                product_values_q[(output_lane*PRODUCT_W) +: PRODUCT_W] <=
                    exact_product;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            gate_q <= '0;
            input_channel_q <= '0;
            output_tile_q <= '0;
            issue_seq_q <= '0;
            protocol_error <= 1'b0;
            count_terms <= '0;
            count_weight_requests <= '0;
            count_products <= '0;
            count_weight_wait_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (clear_error)
                protocol_error <= 1'b0;
            unique case (state_q)
                ST_IDLE: begin
                    if (term_fire) begin
                        tag_q <= term_tag;
                        gate_q <= term_gate_code;
                        input_channel_q <= term_input_channel;
                        output_tile_q <= term_output_tile;
                        issue_seq_q <= term_issue_seq;
                        count_terms <= count_terms + 1'b1;
                        state_q <= ST_WEIGHT_REQUEST;
                    end
                end
                ST_WEIGHT_REQUEST: begin
                    if (weight_req_fire) begin
                        count_weight_requests <=
                            count_weight_requests + 1'b1;
                        state_q <= ST_WEIGHT_RESPONSE;
                    end
                end
                ST_WEIGHT_RESPONSE: begin
                    if (weight_rsp_fire) begin
                        state_q <= ST_PRODUCT_OUTPUT;
                    end
                end
                ST_PRODUCT_OUTPUT: begin
                    if (product_fire) begin
                        count_products <= count_products + 1'b1;
                        state_q <= ST_IDLE;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase

            if (!clear_error &&
                ((term_valid && state_q == ST_IDLE && !term_protocol_ok) ||
                 (weight_rsp_valid && state_q != ST_WEIGHT_RESPONSE) ||
                 (weight_rsp_valid && state_q == ST_WEIGHT_RESPONSE &&
                  !response_matches))) begin
                protocol_error <= 1'b1;
            end
            if (state_q == ST_WEIGHT_RESPONSE && !weight_rsp_fire) begin
                count_weight_wait_cycles <= count_weight_wait_cycles + 1'b1;
            end
            if (product_valid && !product_ready) begin
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
