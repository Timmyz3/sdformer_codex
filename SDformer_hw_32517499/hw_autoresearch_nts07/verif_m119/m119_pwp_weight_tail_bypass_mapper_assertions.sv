`timescale 1ns/1ps
`default_nettype none

module m119_pwp_weight_tail_bypass_mapper_assertions #(
    parameter int DELTA_BITS = 1824
) (
    input logic                     clk_core,
    input logic                     rst_core,
    input logic                     service_valid,
    input logic                     service_ready,
    input logic                     service_is_event,
    input logic [1:0]               service_load_beat,
    input logic                     service_accept,
    input logic                     weight_rd_en,
    input logic [6:0]               weight_rd_key,
    input logic [1:0]               weight_rd_beat,
    input logic                     update_valid,
    input logic                     update_ready,
    input logic [2:0]               update_block,
    input logic [8:0]               update_row,
    input logic [DELTA_BITS-1:0]    update_delta,
    input logic                     update_accept,
    input logic                     payload_active,
    input logic                     tail_bypass_available,
    input logic                     protocol_error,
    input logic                     busy
);
`ifdef SVA_RUNTIME_ENABLED
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_update_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        update_accept == (update_valid && update_ready));
    ap_weight_read_exactly_for_load: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_rd_en == (service_accept && !service_is_event));
    ap_weight_beat_legal: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_rd_en |-> weight_rd_beat < 3);
    ap_update_stable_on_stall: assert property (@(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready |=> update_valid
            && $stable({update_block, update_row, update_delta}));
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_fault_quarantines_new_service: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !service_ready && !service_accept && !weight_rd_en);
    ap_event_requires_payload: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event |-> payload_active);

    cp_three_loads_then_tail_event: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_accept && !service_is_event && service_load_beat == 0
        ##1 service_accept && !service_is_event && service_load_beat == 1
        ##1 service_accept && !service_is_event && service_load_beat == 2
        ##1 service_accept && service_is_event && tail_bypass_available);
    cp_event_ii1: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event
        ##1 service_accept && service_is_event);
    cp_update_stall: cover property (@(posedge clk_core) disable iff (rst_core)
        update_valid && !update_ready ##1 update_valid && update_ready);
    cp_signed_map_accept: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event ##1 update_accept);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
    cp_busy: cover property (@(posedge clk_core) disable iff (rst_core)
        busy);
`endif
endmodule

`default_nettype wire
