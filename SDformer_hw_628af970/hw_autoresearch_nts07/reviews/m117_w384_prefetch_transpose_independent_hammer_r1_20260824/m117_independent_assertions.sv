`timescale 1ns/1ps
`default_nettype none

module m117_independent_assertions (
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
    input logic weight_prefetch_valid,
    input logic weight_prefetch_ready,
    input logic weight_prefetch_accept,
    input logic [3:0] weight_prefetch_source,
    input logic [2:0] weight_prefetch_block,
    input logic [15:0] weight_prefetch_context,
    input logic descriptor_done,
    input logic descriptor_done_empty,
    input logic [11:0] descriptor_done_base_row,
    input logic [15:0] descriptor_done_context,
    input logic fill_bank,
    input logic drain_bank,
    input logic protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    ap_event_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        event_accept == (event_valid && event_ready));
    ap_close_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        window_close_accept == (window_close_valid && window_close_ready));
    ap_service_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        service_accept == (service_valid && service_ready));
    ap_prefetch_handshake: assert property (@(posedge clk_core) disable iff (rst_core)
        weight_prefetch_accept
            == (weight_prefetch_valid && weight_prefetch_ready));

    ap_service_identity_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        service_valid && !service_ready |=> service_valid
            && $stable({service_is_event, service_source, service_block,
                        service_load_beat, service_row_offset,
                        service_destination_row, service_negate,
                        service_last_for_key, service_context}));
    ap_prefetch_identity_stable_on_stall: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        weight_prefetch_valid && !weight_prefetch_ready
        |=> weight_prefetch_valid
            && $stable({weight_prefetch_source, weight_prefetch_block,
                        weight_prefetch_context}));

    ap_dispatch_prefetch_to_exact_load0: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        weight_prefetch_accept && !service_valid
        |=> service_valid && !service_is_event && service_load_beat == 0
            && service_source == $past(weight_prefetch_source)
            && service_block == $past(weight_prefetch_block)
            && service_context == $past(weight_prefetch_context));
    ap_simultaneous_final_prefetch_to_exact_load0: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        service_accept && service_is_event && service_last_for_key
            && weight_prefetch_accept
        |=> service_valid && !service_is_event && service_load_beat == 0
            && service_source == $past(weight_prefetch_source)
            && service_block == $past(weight_prefetch_block)
            && service_context == $past(weight_prefetch_context));
    ap_early_prefetch_is_not_reissued_while_final_stalls: assert property (
        @(posedge clk_core) disable iff (rst_core || protocol_error)
        service_valid && service_is_event && service_last_for_key
            && !service_ready && weight_prefetch_accept
        |=> service_valid && service_is_event && service_last_for_key
            && !weight_prefetch_valid);

    ap_empty_done_shape: assert property (@(posedge clk_core) disable iff (rst_core)
        descriptor_done_empty |-> descriptor_done);
    cp_dispatch_prefetch: cover property (@(posedge clk_core) disable iff (rst_core)
        weight_prefetch_accept && !service_valid ##1
        service_valid && !service_is_event && service_load_beat == 0);
    cp_simultaneous_zero_bubble_subset: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_accept && service_is_event && service_last_for_key
            && weight_prefetch_accept ##1
        service_valid && !service_is_event && service_load_beat == 0);
    cp_early_prefetch_final_stall: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_valid && service_is_event && service_last_for_key
            && !service_ready && weight_prefetch_accept ##1
        service_valid && service_is_event && service_last_for_key
            && !weight_prefetch_valid);
    cp_repeated_service_stall_release: cover property (
        @(posedge clk_core) disable iff (rst_core)
        service_valid && !service_ready [*3] ##1 service_valid && service_ready);
    cp_prefetch_repeated_stall_release: cover property (
        @(posedge clk_core) disable iff (rst_core)
        weight_prefetch_valid && !weight_prefetch_ready [*3] ##1
        weight_prefetch_valid && weight_prefetch_ready);
    cp_empty_done: cover property (@(posedge clk_core) disable iff (rst_core)
        descriptor_done && descriptor_done_empty
            && descriptor_done_base_row != 0
            && descriptor_done_context != 0);
    cp_nonempty_done: cover property (@(posedge clk_core) disable iff (rst_core)
        descriptor_done && !descriptor_done_empty
            && descriptor_done_base_row != 0
            && descriptor_done_context != 0);
    cp_back_to_back_done_identities: cover property (
        @(posedge clk_core) disable iff (rst_core)
        descriptor_done ##1 descriptor_done
            && $changed({descriptor_done_empty, descriptor_done_base_row,
                         descriptor_done_context}));
    cp_pingpong_overlap: cover property (@(posedge clk_core) disable iff (rst_core)
        event_valid && service_valid && fill_bank != drain_bank);
    cp_last_event_exact_grace_then_close: cover property (
        @(posedge clk_core) disable iff (rst_core)
        event_accept ##1 event_valid && !event_ready && !event_accept
            && !protocol_error ##1
        !event_valid && window_close_valid && window_close_accept);
`endif
endmodule

`default_nettype wire
