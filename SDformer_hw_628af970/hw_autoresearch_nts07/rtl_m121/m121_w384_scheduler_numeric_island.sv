`timescale 1ns/1ps
`default_nettype none

// End-to-end standalone W384 service island.  The M117 transpose scheduler's
// counted load/event stream directly drives the M120 synchronous-weight
// tail-bypass mapper and signed19 accumulator.  This wrapper adds no service
// queue or cycle scheduler; backpressure is visible at the exact composition
// cut.  Foundry memories and full-network producer scheduling remain external.
module m121_w384_scheduler_numeric_island (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         accumulator_window_start_valid,
    output logic                         accumulator_window_start_ready,
    output logic                         accumulator_window_start_accept,

    input  logic                         event_valid,
    output logic                         event_ready,
    input  logic [3:0]                   event_source,
    input  logic [2:0]                   event_block,
    input  logic [8:0]                   event_row_offset,
    input  logic                         event_negate,
    input  logic [11:0]                  window_base_row,
    input  logic [15:0]                  window_context,
    output logic                         event_accept,
    input  logic                         descriptor_close_valid,
    output logic                         descriptor_close_ready,
    output logic                         descriptor_close_accept,

    output logic                         weight_prefetch_valid,
    input  logic                         weight_prefetch_ready,
    output logic [3:0]                   weight_prefetch_source,
    output logic [2:0]                   weight_prefetch_block,
    output logic [15:0]                  weight_prefetch_context,
    output logic                         weight_prefetch_accept,

    output logic                         weight_rd_en,
    output logic [6:0]                   weight_rd_key,
    output logic [1:0]                   weight_rd_beat,
    input  logic [255:0]                 weight_rd_data,

    input  logic                         accumulator_window_end_valid,
    output logic                         accumulator_window_end_ready,
    output logic                         accumulator_window_end_accept,
    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_block,
    output logic [8:0]                   commit_row,
    output logic [1823:0]                commit_data,
    output logic                         commit_last,
    output logic                         accumulator_window_done,

    output logic                         lane_mem_rd_en,
    output logic [11:0]                  lane_mem_rd_addr,
    input  logic [18:0]                  lane_mem_rd_data [0:95],
    output logic                         lane_mem_wr_en,
    output logic [11:0]                  lane_mem_wr_addr,
    output logic [18:0]                  lane_mem_wr_data [0:95],

    output logic                         descriptor_done,
    output logic                         descriptor_done_empty,
    output logic [11:0]                  descriptor_done_base_row,
    output logic [15:0]                  descriptor_done_context,

    output logic                         observed_service_valid,
    output logic                         observed_service_ready,
    output logic                         observed_service_accept,
    output logic                         observed_numeric_service_accept,
    output logic                         observed_service_is_event,
    output logic [3:0]                   observed_service_source,
    output logic [2:0]                   observed_service_block,
    output logic [1:0]                   observed_service_load_beat,
    output logic [8:0]                   observed_service_row_offset,
    output logic                         observed_service_negate,
    output logic                         observed_service_last_for_key,
    output logic                         mapped_update_accept,
    output logic                         tail_bypass_available,
    output logic                         scheduler_protocol_error,
    output logic                         numeric_protocol_error,
    output logic                         protocol_error,
    output logic                         busy
);
    logic scheduler_event_valid;
    logic scheduler_event_ready;
    logic scheduler_close_valid;
    logic scheduler_close_ready;
    logic service_valid;
    logic service_ready;
    logic scheduler_service_accept;
    logic numeric_service_accept;
    logic service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [8:0] service_row_offset;
    logic [11:0] service_destination_row;
    logic service_negate;
    logic service_last_for_key;
    logic [15:0] service_context;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic scheduler_busy;
    logic numeric_mapper_busy;
    logic accumulator_window_active;
    logic numeric_busy;

    assign scheduler_event_valid = event_valid && !numeric_protocol_error;
    assign event_ready = scheduler_event_ready && !numeric_protocol_error;
    assign scheduler_close_valid = descriptor_close_valid
                                 && !numeric_protocol_error;
    assign descriptor_close_ready = scheduler_close_ready
                                  && !numeric_protocol_error;
    assign protocol_error = scheduler_protocol_error
                          || numeric_protocol_error;
    assign busy = scheduler_busy || numeric_busy;

    assign observed_service_valid = service_valid;
    assign observed_service_ready = service_ready;
    assign observed_service_accept = scheduler_service_accept;
    assign observed_numeric_service_accept = numeric_service_accept;
    assign observed_service_is_event = service_is_event;
    assign observed_service_source = service_source;
    assign observed_service_block = service_block;
    assign observed_service_load_beat = service_load_beat;
    assign observed_service_row_offset = service_row_offset;
    assign observed_service_negate = service_negate;
    assign observed_service_last_for_key = service_last_for_key;

    m117_w384_prefetch_transpose_scheduler scheduler (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .event_valid(scheduler_event_valid),
        .event_ready(scheduler_event_ready),
        .event_source(event_source),
        .event_block(event_block),
        .event_row_offset(event_row_offset),
        .event_negate(event_negate),
        .window_base_row(window_base_row),
        .window_context(window_context),
        .event_accept(event_accept),
        .window_close_valid(scheduler_close_valid),
        .window_close_ready(scheduler_close_ready),
        .window_close_accept(descriptor_close_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_destination_row(service_destination_row),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_context(service_context),
        .service_accept(scheduler_service_accept),
        .weight_prefetch_valid(weight_prefetch_valid),
        .weight_prefetch_ready(weight_prefetch_ready),
        .weight_prefetch_source(weight_prefetch_source),
        .weight_prefetch_block(weight_prefetch_block),
        .weight_prefetch_context(weight_prefetch_context),
        .weight_prefetch_accept(weight_prefetch_accept),
        .descriptor_done(descriptor_done),
        .descriptor_done_empty(descriptor_done_empty),
        .descriptor_done_base_row(descriptor_done_base_row),
        .descriptor_done_context(descriptor_done_context),
        .fill_bank(fill_bank),
        .drain_bank(drain_bank),
        .bank_ready(bank_ready),
        .protocol_error(scheduler_protocol_error),
        .busy(scheduler_busy)
    );

    m120_pwp_tail_mapper_signed19_accumulator_island numeric_island (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(accumulator_window_start_valid),
        .window_start_ready(accumulator_window_start_ready),
        .window_start_accept(accumulator_window_start_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_accept(numeric_service_accept),
        .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .window_end_valid(accumulator_window_end_valid),
        .window_end_ready(accumulator_window_end_ready),
        .window_end_accept(accumulator_window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(accumulator_window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .mapper_busy(numeric_mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(numeric_protocol_error),
        .busy(numeric_busy)
    );
endmodule

`default_nettype wire
