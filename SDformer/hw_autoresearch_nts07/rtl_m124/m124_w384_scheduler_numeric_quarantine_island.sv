`timescale 1ns/1ps
`default_nettype none

// Quarantined W384 service island. The M117 scheduler drives the M119 mapper
// and M123 same-address-forwarding signed19 accumulator. A sticky same-cycle
// composite quarantine isolates lifecycle and output transactions after any
// scheduler or numeric fault.  This wrapper adds no service
// queue or cycle scheduler; backpressure is visible at the exact composition
// cut.  Foundry memories and full-network producer scheduling remain external.
module m124_w384_scheduler_numeric_quarantine_island (
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
    logic scheduler_event_accept;
    logic scheduler_close_valid;
    logic scheduler_close_ready;
    logic scheduler_close_accept;
    logic service_valid;
    logic service_ready;
    logic numeric_service_ready;
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
    logic quarantine_q;
    logic composite_fault_now;
    logic raw_window_start_ready, raw_window_start_accept;
    logic raw_window_end_ready, raw_window_end_accept;
    logic raw_weight_rd_en;
    logic [6:0] raw_weight_rd_key;
    logic [1:0] raw_weight_rd_beat;
    logic raw_commit_valid;
    logic [2:0] raw_commit_block;
    logic [8:0] raw_commit_row;
    logic [1823:0] raw_commit_data;
    logic raw_commit_last, raw_window_done;
    logic raw_mapped_update_accept, raw_tail_bypass_available;
    logic raw_weight_prefetch_valid, raw_weight_prefetch_accept;
    logic [3:0] raw_weight_prefetch_source;
    logic [2:0] raw_weight_prefetch_block;
    logic [15:0] raw_weight_prefetch_context;
    logic raw_descriptor_done, raw_descriptor_done_empty;
    logic [11:0] raw_descriptor_done_base_row;
    logic [15:0] raw_descriptor_done_context;

    assign composite_fault_now = scheduler_protocol_error
                               || numeric_protocol_error;
    assign protocol_error = quarantine_q || composite_fault_now;
    assign scheduler_event_valid = event_valid && !quarantine_q
                                 && !numeric_protocol_error;
    assign event_ready = scheduler_event_ready && !protocol_error;
    assign event_accept = scheduler_event_accept && !protocol_error;
    assign scheduler_close_valid = descriptor_close_valid
                                 && !quarantine_q
                                 && !numeric_protocol_error;
    assign descriptor_close_ready = scheduler_close_ready
                                  && !protocol_error;
    assign descriptor_close_accept = scheduler_close_accept
                                   && !protocol_error;
    assign busy = scheduler_busy || numeric_busy;

    assign service_ready = numeric_service_ready && !protocol_error;
    assign accumulator_window_start_ready = raw_window_start_ready
                                          && !protocol_error;
    assign accumulator_window_start_accept = raw_window_start_accept
                                           && !protocol_error;
    assign accumulator_window_end_ready = raw_window_end_ready
                                        && !protocol_error;
    assign accumulator_window_end_accept = raw_window_end_accept
                                         && !protocol_error;
    assign weight_rd_en = raw_weight_rd_en && !protocol_error;
    assign weight_rd_key = raw_weight_rd_key;
    assign weight_rd_beat = raw_weight_rd_beat;
    assign commit_valid = raw_commit_valid && !protocol_error;
    assign commit_block = raw_commit_block;
    assign commit_row = raw_commit_row;
    assign commit_data = raw_commit_data;
    assign commit_last = raw_commit_last;
    assign accumulator_window_done = raw_window_done && !protocol_error;
    assign mapped_update_accept = raw_mapped_update_accept
                                && !protocol_error;
    assign tail_bypass_available = raw_tail_bypass_available
                                 && !protocol_error;
    assign weight_prefetch_valid = raw_weight_prefetch_valid
                                 && !protocol_error;
    assign weight_prefetch_source = raw_weight_prefetch_source;
    assign weight_prefetch_block = raw_weight_prefetch_block;
    assign weight_prefetch_context = raw_weight_prefetch_context;
    assign weight_prefetch_accept = raw_weight_prefetch_accept
                                  && !protocol_error;
    assign descriptor_done = raw_descriptor_done && !protocol_error;
    assign descriptor_done_empty = raw_descriptor_done_empty;
    assign descriptor_done_base_row = raw_descriptor_done_base_row;
    assign descriptor_done_context = raw_descriptor_done_context;

    always_ff @(posedge clk_core) begin
        if (rst_core)
            quarantine_q <= 1'b0;
        else if (composite_fault_now)
            quarantine_q <= 1'b1;
    end

    assign observed_service_valid = service_valid && !protocol_error;
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
        .event_accept(scheduler_event_accept),
        .window_close_valid(scheduler_close_valid),
        .window_close_ready(scheduler_close_ready),
        .window_close_accept(scheduler_close_accept),
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
        .weight_prefetch_valid(raw_weight_prefetch_valid),
        .weight_prefetch_ready(weight_prefetch_ready && !protocol_error),
        .weight_prefetch_source(raw_weight_prefetch_source),
        .weight_prefetch_block(raw_weight_prefetch_block),
        .weight_prefetch_context(raw_weight_prefetch_context),
        .weight_prefetch_accept(raw_weight_prefetch_accept),
        .descriptor_done(raw_descriptor_done),
        .descriptor_done_empty(raw_descriptor_done_empty),
        .descriptor_done_base_row(raw_descriptor_done_base_row),
        .descriptor_done_context(raw_descriptor_done_context),
        .fill_bank(fill_bank),
        .drain_bank(drain_bank),
        .bank_ready(bank_ready),
        .protocol_error(scheduler_protocol_error),
        .busy(scheduler_busy)
    );

    m124_pwp_tail_mapper_signed19_forwarding_accumulator_island numeric_island (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(accumulator_window_start_valid
                            && !quarantine_q
                            && !scheduler_protocol_error),
        .window_start_ready(raw_window_start_ready),
        .window_start_accept(raw_window_start_accept),
        .service_valid(service_valid && !quarantine_q
                       && !scheduler_protocol_error),
        .service_ready(numeric_service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_accept(numeric_service_accept),
        .weight_rd_en(raw_weight_rd_en),
        .weight_rd_key(raw_weight_rd_key),
        .weight_rd_beat(raw_weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .window_end_valid(accumulator_window_end_valid
                          && !quarantine_q
                          && !scheduler_protocol_error),
        .window_end_ready(raw_window_end_ready),
        .window_end_accept(raw_window_end_accept),
        .commit_valid(raw_commit_valid),
        .commit_ready(commit_ready && !protocol_error),
        .commit_block(raw_commit_block),
        .commit_row(raw_commit_row),
        .commit_data(raw_commit_data),
        .commit_last(raw_commit_last),
        .window_done(raw_window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .mapped_update_accept(raw_mapped_update_accept),
        .tail_bypass_available(raw_tail_bypass_available),
        .mapper_busy(numeric_mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(numeric_protocol_error),
        .busy(numeric_busy)
    );
endmodule

`default_nettype wire
