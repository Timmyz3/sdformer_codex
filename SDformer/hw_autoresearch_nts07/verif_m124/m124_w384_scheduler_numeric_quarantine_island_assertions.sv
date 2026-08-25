`timescale 1ns/1ps
`default_nettype none

module m124_w384_scheduler_numeric_quarantine_island_assertions (
    input logic             clk_core,
    input logic             rst_core,
    input logic             accumulator_window_start_ready,
    input logic             accumulator_window_start_accept,
    input logic             accumulator_window_end_ready,
    input logic             accumulator_window_end_accept,
    input logic             event_valid,
    input logic             event_ready,
    input logic             event_accept,
    input logic             descriptor_close_valid,
    input logic             descriptor_close_ready,
    input logic             descriptor_close_accept,
    input logic             observed_service_valid,
    input logic             observed_service_ready,
    input logic             observed_service_accept,
    input logic             observed_numeric_service_accept,
    input logic             observed_service_is_event,
    input logic [3:0]       observed_service_source,
    input logic [2:0]       observed_service_block,
    input logic [1:0]       observed_service_load_beat,
    input logic             observed_service_last_for_key,
    input logic             weight_rd_en,
    input logic             weight_prefetch_valid,
    input logic             descriptor_done,
    input logic             mapped_update_accept,
    input logic             tail_bypass_available,
    input logic             commit_valid,
    input logic             commit_ready,
    input logic [2:0]       commit_block,
    input logic [8:0]       commit_row,
    input logic [1823:0]    commit_data,
    input logic             commit_last,
    input logic             accumulator_window_done,
    input logic             lane_mem_rd_en,
    input logic             lane_mem_wr_en,
    input logic             scheduler_protocol_error,
    input logic             numeric_protocol_error,
    input logic             protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    ap_event_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        event_accept == (event_valid && event_ready));
    ap_close_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_close_accept
        == (descriptor_close_valid && descriptor_close_ready));
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        observed_service_accept
        == (observed_service_valid && observed_service_ready));
    ap_composed_accept_agreement: assert property (
        @(posedge clk_core) disable iff (rst_core)
        observed_numeric_service_accept == observed_service_accept);
    ap_weight_read_exact_load: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_rd_en == (observed_service_accept
                         && !observed_service_is_event));
    ap_event_maps_next_cycle: assert property (@(posedge clk_core) disable iff (rst_core)
        observed_service_accept && observed_service_is_event
        |=> mapped_update_accept);
    ap_update_has_prior_event: assert property (@(posedge clk_core) disable iff (rst_core)
        mapped_update_accept
        |-> $past(observed_service_accept && observed_service_is_event));
    ap_nonfinal_key_zero_bubble: assert property (
        @(posedge clk_core) disable iff (rst_core)
        observed_service_accept && observed_service_is_event
            && observed_service_last_for_key
            && {observed_service_source, observed_service_block} != 7'd127
        |=> observed_service_accept && !observed_service_is_event
            && observed_service_load_beat == 0);
    ap_commit_stable_on_stall: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_block, commit_row, commit_data, commit_last}));
    ap_commit_last_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_last |-> commit_block == 7 && commit_row == 383);
    ap_window_done_after_last: assert property (@(posedge clk_core) disable iff (rst_core)
        accumulator_window_done
        |-> $past(commit_valid && commit_ready && commit_last));
    ap_combined_error: assert property (@(posedge clk_core) disable iff (rst_core)
        scheduler_protocol_error || numeric_protocol_error
        |-> protocol_error);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_composite_quarantine: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !accumulator_window_start_ready
            && !accumulator_window_start_accept
            && !accumulator_window_end_ready
            && !accumulator_window_end_accept
            && !event_ready && !descriptor_close_ready
            && !observed_service_ready && !observed_service_accept
            && !observed_numeric_service_accept
            && !weight_rd_en && !weight_prefetch_valid
            && !mapped_update_accept && !commit_valid
            && !accumulator_window_done && !descriptor_done);

    cp_three_loads_tail_event: cover property (@(posedge clk_core) disable iff (rst_core)
        observed_service_accept && !observed_service_is_event
            && observed_service_load_beat == 0
        ##1 observed_service_accept && !observed_service_is_event
            && observed_service_load_beat == 1
        ##1 observed_service_accept && !observed_service_is_event
            && observed_service_load_beat == 2
        ##1 observed_service_accept && observed_service_is_event
            && tail_bypass_available);
    cp_event_update_chain: cover property (@(posedge clk_core) disable iff (rst_core)
        observed_service_accept && observed_service_is_event
        ##1 mapped_update_accept);
    cp_zero_bubble_key_transition: cover property (
        @(posedge clk_core) disable iff (rst_core)
        observed_service_accept && observed_service_is_event
            && observed_service_last_for_key
            && {observed_service_source, observed_service_block} != 7'd127
        ##1 observed_service_accept && !observed_service_is_event
            && observed_service_load_beat == 0);
    cp_update_ii1: cover property (@(posedge clk_core) disable iff (rst_core)
        mapped_update_accept ##1 mapped_update_accept);
    cp_lane_rw_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_rd_en && lane_mem_wr_en);
    cp_descriptor_done: cover property (@(posedge clk_core) disable iff (rst_core)
        descriptor_done);
    cp_full_commit: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_ready && commit_last
        ##1 accumulator_window_done);
    cp_numeric_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !numeric_protocol_error ##1 numeric_protocol_error);
    cp_scheduler_fault_quarantine: cover property (
        @(posedge clk_core) disable iff (rst_core)
        !scheduler_protocol_error ##1 scheduler_protocol_error
        ##1 protocol_error && !accumulator_window_end_ready
            && !commit_valid && !weight_rd_en);
`endif
endmodule

`default_nettype wire

