`timescale 1ns/1ps
`default_nettype none

// Numeric island: M119 three-beat tail-bypass mapper feeds the M123
// same-address-forwarding W384 signed19 accumulator with no software-visible
// delta port between them.  Window sequencing is fail-closed, while prior
// accepted mapper updates are allowed to drain if a later wrapper request is
// malformed.  M117 descriptor scheduling and foundry macros remain port cuts.
module m124_pwp_tail_mapper_signed19_forwarding_accumulator_island (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,

    input  logic                         service_valid,
    output logic                         service_ready,
    input  logic                         service_is_event,
    input  logic [3:0]                   service_source,
    input  logic [2:0]                   service_block,
    input  logic [1:0]                   service_load_beat,
    input  logic [8:0]                   service_row_offset,
    input  logic                         service_negate,
    input  logic                         service_last_for_key,
    output logic                         service_accept,

    output logic                         weight_rd_en,
    output logic [6:0]                   weight_rd_key,
    output logic [1:0]                   weight_rd_beat,
    input  logic [255:0]                 weight_rd_data,

    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_block,
    output logic [8:0]                   commit_row,
    output logic [1823:0]                commit_data,
    output logic                         commit_last,
    output logic                         window_done,

    output logic                         lane_mem_rd_en,
    output logic [11:0]                  lane_mem_rd_addr,
    input  logic [18:0]                  lane_mem_rd_data [0:95],
    output logic                         lane_mem_wr_en,
    output logic [11:0]                  lane_mem_wr_addr,
    output logic [18:0]                  lane_mem_wr_data [0:95],

    output logic                         mapped_update_accept,
    output logic                         tail_bypass_available,
    output logic                         mapper_busy,
    output logic                         accumulator_window_active,
    output logic                         protocol_error,
    output logic                         busy
);
    logic wrapper_fault_q;
    logic wrapper_illegal_request;
    logic request_collision;

    logic mapper_service_valid;
    logic mapper_service_ready;
    logic mapper_service_accept;
    logic mapper_update_valid;
    logic mapper_update_ready;
    logic [2:0] mapper_update_block;
    logic [8:0] mapper_update_row;
    logic [1823:0] mapper_update_delta;
    logic mapper_update_accept;
    logic accumulator_update_accept;
    logic mapper_payload_active;
    logic mapper_protocol_error;

    logic accumulator_start_valid;
    logic accumulator_start_ready;
    logic accumulator_start_accept;
    logic accumulator_end_valid;
    logic accumulator_end_ready;
    logic accumulator_end_accept;
    logic accumulator_protocol_error;
    logic accumulator_busy;

    assign request_collision = (window_start_valid && service_valid)
                             || (window_start_valid && window_end_valid)
                             || (service_valid && window_end_valid);
    assign wrapper_illegal_request = request_collision
        || (service_valid && !accumulator_window_active)
        || (window_start_valid
            && (mapper_busy || accumulator_window_active))
        || (window_end_valid
            && (mapper_busy || !accumulator_window_active));
    assign protocol_error = wrapper_fault_q || wrapper_illegal_request
                          || mapper_protocol_error
                          || accumulator_protocol_error;

    assign accumulator_start_valid = window_start_valid
                                   && !service_valid && !window_end_valid
                                   && !wrapper_fault_q
                                   && !mapper_protocol_error;
    assign window_start_ready = accumulator_start_ready
                              && !service_valid && !window_end_valid
                              && !mapper_busy && !protocol_error;
    assign window_start_accept = accumulator_start_accept;

    assign mapper_service_valid = service_valid
                                && accumulator_window_active
                                && !window_start_valid && !window_end_valid
                                && !wrapper_fault_q
                                && !accumulator_protocol_error;
    assign service_ready = mapper_service_ready
                         && accumulator_window_active
                         && !window_start_valid && !window_end_valid
                         && !wrapper_fault_q
                         && !accumulator_protocol_error;
    assign service_accept = mapper_service_accept;

    assign accumulator_end_valid = window_end_valid
                                 && !window_start_valid && !service_valid
                                 && !mapper_busy
                                 && !wrapper_fault_q
                                 && !mapper_protocol_error;
    assign window_end_ready = accumulator_end_ready
                            && !window_start_valid && !service_valid
                            && !mapper_busy && !protocol_error;
    assign window_end_accept = accumulator_end_accept;
    assign mapped_update_accept = accumulator_update_accept;
    assign busy = mapper_busy || accumulator_busy;

    always_ff @(posedge clk_core) begin
        if (rst_core)
            wrapper_fault_q <= 1'b0;
        else if (wrapper_illegal_request)
            wrapper_fault_q <= 1'b1;
    end

    m119_pwp_weight_tail_bypass_mapper mapper (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .service_valid(mapper_service_valid),
        .service_ready(mapper_service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_accept(mapper_service_accept),
        .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .update_valid(mapper_update_valid),
        .update_ready(mapper_update_ready),
        .update_block(mapper_update_block),
        .update_row(mapper_update_row),
        .update_delta(mapper_update_delta),
        .update_accept(mapper_update_accept),
        .payload_active(mapper_payload_active),
        .tail_bypass_available(tail_bypass_available),
        .protocol_error(mapper_protocol_error),
        .busy(mapper_busy)
    );

    m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter accumulator (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(accumulator_start_valid),
        .window_start_ready(accumulator_start_ready),
        .window_start_accept(accumulator_start_accept),
        .update_valid(mapper_update_valid),
        .update_ready(mapper_update_ready),
        .update_block(mapper_update_block),
        .update_row(mapper_update_row),
        .update_delta(mapper_update_delta),
        .update_accept(accumulator_update_accept),
        .window_end_valid(accumulator_end_valid),
        .window_end_ready(accumulator_end_ready),
        .window_end_accept(accumulator_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .protocol_error(accumulator_protocol_error),
        .window_active(accumulator_window_active),
        .busy(accumulator_busy)
    );
endmodule

`default_nettype wire

