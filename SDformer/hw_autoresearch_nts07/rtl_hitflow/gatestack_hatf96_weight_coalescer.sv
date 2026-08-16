`timescale 1ns/1ps
`default_nettype none

// Converts one logical HATF supertile weight request into three independent
// 32-lane SRAM-bank transactions and atomically joins their responses.
module gatestack_hatf96_weight_coalescer #(
    parameter int BANK_COUNT = 3,
    parameter int LANES_PER_BANK = 32,
    parameter int WEIGHT_W = 8,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int COUNTER_W = 32
) (
    input  logic                                          clk_core,
    input  logic                                          rst_core,

    input  logic                                          req_valid,
    output logic                                          req_ready,
    input  logic [TAG_W-1:0]                              req_tag,
    input  logic [INPUT_CH_W-1:0]                         req_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]                      req_supertile,

    output logic [BANK_COUNT-1:0]                         bank_req_valid,
    input  logic [BANK_COUNT-1:0]                         bank_req_ready,
    output logic [(BANK_COUNT*TAG_W)-1:0]                 bank_req_tags,
    output logic [(BANK_COUNT*INPUT_CH_W)-1:0]            bank_req_input_channels,
    output logic [(BANK_COUNT*OUTPUT_TILE_W)-1:0]         bank_req_output_tiles,

    input  logic [BANK_COUNT-1:0]                         bank_rsp_valid,
    output logic [BANK_COUNT-1:0]                         bank_rsp_ready,
    input  logic [(BANK_COUNT*TAG_W)-1:0]                 bank_rsp_tags,
    input  logic [(BANK_COUNT*INPUT_CH_W)-1:0]            bank_rsp_input_channels,
    input  logic [(BANK_COUNT*OUTPUT_TILE_W)-1:0]         bank_rsp_output_tiles,
    input  logic [(BANK_COUNT*LANES_PER_BANK*WEIGHT_W)-1:0]
                                                           bank_rsp_weights,

    output logic                                          rsp_valid,
    input  logic                                          rsp_ready,
    output logic [TAG_W-1:0]                              rsp_tag,
    output logic [INPUT_CH_W-1:0]                         rsp_input_channel,
    output logic [OUTPUT_TILE_W-1:0]                      rsp_supertile,
    output logic [(BANK_COUNT*LANES_PER_BANK*WEIGHT_W)-1:0]
                                                           rsp_weights,
    output logic                                          rsp_error,
    output logic                                          protocol_error,
    output logic [COUNTER_W-1:0]                          count_requests,
    output logic [COUNTER_W-1:0]                          count_bank_requests,
    output logic [COUNTER_W-1:0]                          count_bank_responses,
    output logic [COUNTER_W-1:0]                          count_response_stalls
);
    localparam BANK_LIMIT = BANK_COUNT;
    logic active_q, rsp_valid_q, rsp_error_q;
    logic [BANK_COUNT-1:0] issued_q, received_q;
    logic [TAG_W-1:0] req_tag_q;
    logic [INPUT_CH_W-1:0] req_input_channel_q;
    logic [OUTPUT_TILE_W-1:0] req_supertile_q;
    logic [(BANK_COUNT*LANES_PER_BANK*WEIGHT_W)-1:0]
        bank_weights_q;
    logic [BANK_COUNT-1:0] bank_req_fire, bank_rsp_fire;
    logic [BANK_COUNT-1:0] bank_rsp_identity_ok;
    logic req_fire, rsp_fire;
    logic [COUNTER_W-1:0] bank_req_fire_count;
    logic [COUNTER_W-1:0] bank_rsp_fire_count;

    assign req_ready = !active_q && !rsp_valid_q;
    assign req_fire = req_valid && req_ready;
    assign bank_req_valid = {BANK_COUNT{active_q}} & ~issued_q;
    assign bank_req_fire = bank_req_valid & bank_req_ready;
    assign bank_rsp_ready = {BANK_COUNT{active_q && !rsp_valid_q}} &
                            issued_q & ~received_q;
    assign bank_rsp_fire = bank_rsp_valid & bank_rsp_ready;
    assign rsp_valid = rsp_valid_q;
    assign rsp_fire = rsp_valid && rsp_ready;
    assign rsp_tag = req_tag_q;
    assign rsp_input_channel = req_input_channel_q;
    assign rsp_supertile = req_supertile_q;
    assign rsp_weights = bank_weights_q;
    assign rsp_error = rsp_error_q;

    always_comb begin
        bank_rsp_identity_ok = '0;
        bank_req_fire_count = '0;
        bank_rsp_fire_count = '0;
        for (int bank = 32'd0; bank < BANK_LIMIT; bank = bank + 32'd1) begin
            bank_req_fire_count = bank_req_fire_count +
                                  COUNTER_W'(bank_req_fire[bank]);
            bank_rsp_fire_count = bank_rsp_fire_count +
                                  COUNTER_W'(bank_rsp_fire[bank]);
            bank_req_tags[(bank*TAG_W) +: TAG_W] = req_tag_q;
            bank_req_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] =
                req_input_channel_q;
            bank_req_output_tiles[(bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                OUTPUT_TILE_W'(32'(req_supertile_q) * BANK_COUNT + bank);
            bank_rsp_identity_ok[bank] =
                bank_rsp_tags[(bank*TAG_W) +: TAG_W] == req_tag_q &&
                bank_rsp_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] ==
                    req_input_channel_q &&
                bank_rsp_output_tiles[
                    (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] ==
                    OUTPUT_TILE_W'(
                        32'(req_supertile_q) * BANK_COUNT + bank);
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            rsp_valid_q <= 1'b0;
            rsp_error_q <= 1'b0;
            issued_q <= '0;
            received_q <= '0;
            req_tag_q <= '0;
            req_input_channel_q <= '0;
            req_supertile_q <= '0;
            bank_weights_q <= '0;
            protocol_error <= 1'b0;
            count_requests <= '0;
            count_bank_requests <= '0;
            count_bank_responses <= '0;
            count_response_stalls <= '0;
        end else begin
            if (req_fire) begin
                active_q <= 1'b1;
                issued_q <= '0;
                received_q <= '0;
                req_tag_q <= req_tag;
                req_input_channel_q <= req_input_channel;
                req_supertile_q <= req_supertile;
                rsp_error_q <= 1'b0;
                count_requests <= count_requests + 1'b1;
            end else begin
                issued_q <= issued_q | bank_req_fire;
            end
            count_bank_requests <= count_bank_requests + bank_req_fire_count;
            count_bank_responses <= count_bank_responses + bank_rsp_fire_count;
            for (int bank = 32'd0; bank < BANK_LIMIT; bank = bank + 32'd1) begin
                if (bank_rsp_fire[bank]) begin
                    received_q[bank] <= 1'b1;
                    bank_weights_q[
                        (bank*LANES_PER_BANK*WEIGHT_W) +:
                        (LANES_PER_BANK*WEIGHT_W)] <= bank_rsp_weights[
                        (bank*LANES_PER_BANK*WEIGHT_W) +:
                        (LANES_PER_BANK*WEIGHT_W)];
                    if (!bank_rsp_identity_ok[bank]) begin
                        rsp_error_q <= 1'b1;
                        protocol_error <= 1'b1;
                    end
                end
            end

            if (active_q &&
                ((received_q | bank_rsp_fire) == {BANK_COUNT{1'b1}})) begin
                active_q <= 1'b0;
                rsp_valid_q <= 1'b1;
            end
            if (rsp_fire)
                rsp_valid_q <= 1'b0;
            if (rsp_valid && !rsp_ready)
                count_response_stalls <= count_response_stalls + 1'b1;
        end
    end
endmodule

`default_nettype wire
