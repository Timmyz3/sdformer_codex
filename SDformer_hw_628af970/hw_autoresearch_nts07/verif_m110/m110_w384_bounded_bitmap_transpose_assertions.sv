`timescale 1ns/1ps
`default_nettype none

module m110_w384_bounded_bitmap_transpose_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic window_close_valid,
    input logic window_close_ready,
    input logic window_close_accept,
    input logic service_valid,
    input logic service_ready,
    input logic service_accept,
    input logic service_is_event,
    input logic [3:0] service_source,
    input logic [2:0] service_block,
    input logic [1:0] service_load_beat,
    input logic [8:0] service_row_offset,
    input logic [11:0] service_destination_row,
    input logic service_negate,
    input logic service_last_for_key,
    input logic [15:0] service_context,
    input logic fill_bank,
    input logic drain_bank,
    input logic [1:0] bank_ready,
    input logic protocol_error,
    input logic busy
);
`ifdef SVA_RUNTIME_ENABLED
    ap_event_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        event_accept == (event_valid && event_ready));
    ap_close_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_close_accept == (window_close_valid && window_close_ready));
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_no_request_collision_accept: assert property (@(posedge clk_core) disable iff (rst_core)
        event_valid && window_close_valid |-> !event_accept && !window_close_accept);
    ap_fault_quarantine: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |-> !event_ready && !window_close_ready && !service_valid);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_service_stable_on_stall: assert property (@(posedge clk_core) disable iff (rst_core)
        service_valid && !service_ready |=> service_valid
            && $stable({service_is_event, service_source, service_block,
                        service_load_beat, service_row_offset,
                        service_destination_row, service_negate,
                        service_last_for_key, service_context}));
    ap_load_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        service_valid && !service_is_event |-> service_row_offset == 0
            && service_destination_row == 0 && !service_negate
            && !service_last_for_key && service_load_beat <= 2);
    ap_event_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        service_valid && service_is_event |-> service_row_offset < 384);

    cp_pingpong_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        event_valid && service_valid && fill_bank != drain_bank);
    cp_last_row: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event && service_row_offset == 383
            && service_last_for_key);
    cp_stall: cover property (@(posedge clk_core) disable iff (rst_core)
        service_valid && !service_ready ##1 service_valid && service_ready);
    cp_full_key_identity: cover property (@(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event && service_source == 15
            && service_block == 7 && service_row_offset == 383);
    cp_fault: cover property (@(posedge clk_core) disable iff (rst_core)
        !protocol_error ##1 protocol_error);
`endif
endmodule

`default_nettype wire
