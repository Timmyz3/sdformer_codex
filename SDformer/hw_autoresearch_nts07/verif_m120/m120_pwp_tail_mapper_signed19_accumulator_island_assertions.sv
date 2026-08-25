`timescale 1ns/1ps
`default_nettype none

module m120_pwp_tail_mapper_signed19_accumulator_island_assertions (
    input logic             clk_core,
    input logic             rst_core,
    input logic             window_start_valid,
    input logic             window_start_ready,
    input logic             window_start_accept,
    input logic             service_valid,
    input logic             service_ready,
    input logic             service_is_event,
    input logic [1:0]       service_load_beat,
    input logic             service_accept,
    input logic             weight_rd_en,
    input logic             window_end_valid,
    input logic             window_end_ready,
    input logic             window_end_accept,
    input logic             mapped_update_accept,
    input logic             tail_bypass_available,
    input logic             commit_valid,
    input logic             commit_ready,
    input logic [2:0]       commit_block,
    input logic [8:0]       commit_row,
    input logic [1823:0]    commit_data,
    input logic             commit_last,
    input logic             window_done,
    input logic             lane_mem_rd_en,
    input logic [11:0]      lane_mem_rd_addr,
    input logic             lane_mem_wr_en,
    input logic [11:0]      lane_mem_wr_addr,
    input logic             mapper_busy,
    input logic             accumulator_window_active,
    input logic             protocol_error,
    input logic             busy
);
`ifdef SVA_RUNTIME_ENABLED
    ap_start_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_start_accept == (window_start_valid && window_start_ready));
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_end_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_end_accept == (window_end_valid && window_end_ready));
    ap_weight_read_for_load_only: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_rd_en == (service_accept && !service_is_event));
    ap_event_maps_exactly_next_cycle: assert property (
        @(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event |=> mapped_update_accept);
    ap_update_has_unique_prior_event: assert property (
        @(posedge clk_core) disable iff (rst_core)
        mapped_update_accept |-> $past(service_accept && service_is_event));
    ap_update_during_active_window: assert property (
        @(posedge clk_core) disable iff (rst_core)
        mapped_update_accept |-> accumulator_window_active);
    ap_commit_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_block, commit_row, commit_data, commit_last}));
    ap_commit_last_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_last |-> commit_block == 7 && commit_row == 383);
    ap_window_done_after_last: assert property (@(posedge clk_core) disable iff (rst_core)
        window_done |-> $past(commit_valid && commit_ready && commit_last));
    ap_lane_read_range: assert property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_rd_en |-> lane_mem_rd_addr < 3072);
    ap_lane_write_range: assert property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_wr_en |-> lane_mem_wr_addr < 3072);
    ap_end_waits_for_mapper: assert property (@(posedge clk_core) disable iff (rst_core)
        window_end_accept |-> !mapper_busy);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_fault_quarantine: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !window_start_ready && !service_ready
            && !window_end_ready);

    cp_three_loads_tail_event: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && !service_is_event && service_load_beat == 0
        ##1 service_accept && !service_is_event && service_load_beat == 1
        ##1 service_accept && !service_is_event && service_load_beat == 2
        ##1 service_accept && service_is_event && tail_bypass_available);
    cp_event_update_chain: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event ##1 mapped_update_accept);
    cp_update_ii1: cover property (@(posedge clk_core) disable iff (rst_core)
        mapped_update_accept ##1 mapped_update_accept);
    cp_lane_read_write_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        lane_mem_rd_en && lane_mem_wr_en);
    cp_commit_stall_release: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && !commit_ready ##1 commit_valid && commit_ready);
    cp_full_window: cover property (@(posedge clk_core) disable iff (rst_core)
        commit_valid && commit_ready && commit_last ##1 window_done);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
    cp_busy: cover property (@(posedge clk_core) disable iff (rst_core)
        busy);
`endif
endmodule

`default_nettype wire
