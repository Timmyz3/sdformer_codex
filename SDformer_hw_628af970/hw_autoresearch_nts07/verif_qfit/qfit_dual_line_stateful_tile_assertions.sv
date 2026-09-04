`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_stateful_tile_assertions #(
    parameter int TAG_W = 24,
    parameter int COUNT_W = 9,
    parameter int OUTPUT_W = 512,
    parameter int PERF_W = 64
) (
    input logic clk_core,
    input logic rst_core,
    input logic weight_epoch_clear,
    input logic weight_valid,
    input logic weight_ready,
    input logic weights_loaded,
    input logic request_valid,
    input logic request_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [TAG_W-1:0] output_state_key,
    input logic output_use_motion,
    input logic output_force_local,
    input logic [COUNT_W-1:0] output_source_count,
    input logic [OUTPUT_W-1:0] output_acc,
    input logic [PERF_W-1:0] perf_requests,
    input logic [PERF_W-1:0] perf_state_hits,
    input logic [PERF_W-1:0] perf_state_misses,
    input logic [PERF_W-1:0] perf_local_tiles,
    input logic [PERF_W-1:0] perf_motion_tiles,
    input logic [PERF_W-1:0] perf_invalid_valid_bits
);
    property p_output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        output_valid && !output_ready |=> output_valid
            && $stable({output_state_key, output_use_motion,
                        output_force_local, output_source_count, output_acc});
    endproperty
    assert property (p_output_stable_under_stall);

    property p_mode_is_one_hot;
        @(posedge clk_core) disable iff (rst_core)
        output_valid |-> output_use_motion != output_force_local;
    endproperty
    assert property (p_mode_is_one_hot);

    property p_state_lookup_accounting;
        @(posedge clk_core) disable iff (rst_core)
        perf_requests == perf_state_hits + perf_state_misses;
    endproperty
    assert property (p_state_lookup_accounting);

    property p_retirement_not_ahead;
        @(posedge clk_core) disable iff (rst_core)
        perf_local_tiles + perf_motion_tiles <= perf_requests;
    endproperty
    assert property (p_retirement_not_ahead);

    property p_invalid_not_ahead;
        @(posedge clk_core) disable iff (rst_core)
        perf_invalid_valid_bits <= perf_requests;
    endproperty
    assert property (p_invalid_not_ahead);

    property p_weight_epoch_excludes_request;
        @(posedge clk_core) disable iff (rst_core)
        weight_epoch_clear || weight_valid |-> !request_ready;
    endproperty
    assert property (p_weight_epoch_excludes_request);

    property p_request_fire_has_stable_weight_epoch;
        @(posedge clk_core) disable iff (rst_core)
        request_valid && request_ready |-> !weight_epoch_clear && !weight_valid;
    endproperty
    assert property (p_request_fire_has_stable_weight_epoch);

    property p_loaded_epoch_is_write_closed;
        @(posedge clk_core) disable iff (rst_core)
        weights_loaded |-> !weight_ready;
    endproperty
    assert property (p_loaded_epoch_is_write_closed);

    cover property (@(posedge clk_core) disable iff (rst_core)
        output_valid && output_ready && output_use_motion);
    cover property (@(posedge clk_core) disable iff (rst_core)
        output_valid && output_ready && output_force_local
        && output_source_count == '0);
endmodule

`default_nettype wire
