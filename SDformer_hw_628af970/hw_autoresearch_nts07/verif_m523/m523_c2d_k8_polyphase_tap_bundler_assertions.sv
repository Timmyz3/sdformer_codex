`timescale 1ns/1ps
`default_nettype none

module m523_c2d_k8_polyphase_tap_bundler_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int COORD_BITS = 10,
    parameter int TIME_BITS = 4,
    parameter int BUNDLE_LANES = 8,
    parameter int FIFO_DEPTH = 18
) (
    input logic clk_core,
    input logic rst_core,
    input logic event_valid,
    input logic event_ready,
    input logic event_accept,
    input logic [TAG_BITS-1:0] event_tag,
    input logic [TIME_BITS-1:0] event_time,
    input logic [CHANNEL_BITS-1:0] event_source_channel,
    input logic [COORD_BITS-1:0] event_source_y,
    input logic [COORD_BITS-1:0] event_source_x,
    input logic [COORD_BITS-1:0] event_input_height,
    input logic [COORD_BITS-1:0] event_input_width,
    input logic event_last,
    input logic bundle_valid,
    input logic bundle_ready,
    input logic bundle_accept,
    input logic [TAG_BITS-1:0] bundle_tag,
    input logic [TIME_BITS-1:0] bundle_time,
    input logic [3:0] bundle_count,
    input logic [BUNDLE_LANES-1:0] tap_lane_valid,
    input logic [BUNDLE_LANES-1:0] tap_event_last,
    input logic [CHANNEL_BITS-1:0] tap_source_channel [0:BUNDLE_LANES-1],
    input logic [COORD_BITS-1:0] tap_source_y [0:BUNDLE_LANES-1],
    input logic [COORD_BITS-1:0] tap_source_x [0:BUNDLE_LANES-1],
    input logic [1:0] tap_kernel_y [0:BUNDLE_LANES-1],
    input logic [1:0] tap_kernel_x [0:BUNDLE_LANES-1],
    input logic [3:0] tap_kernel_index [0:BUNDLE_LANES-1],
    input logic [COORD_BITS-1:0] tap_destination_y [0:BUNDLE_LANES-1],
    input logic [COORD_BITS-1:0] tap_destination_x [0:BUNDLE_LANES-1],
    input logic [1:0] tap_phase_bank [0:BUNDLE_LANES-1],
    input logic bundle_last_for_event,
    input logic bundle_stream_last,
    input logic protocol_error,
    input logic busy,
    input logic [31:0] debug_event_count,
    input logic [31:0] debug_bundle_count,
    input logic [31:0] debug_tap_count,
    input logic [5:0] debug_fifo_count
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_event_accept_definition:
        assert property (event_accept == (event_valid && event_ready));
    ap_bundle_accept_definition:
        assert property (bundle_accept == (bundle_valid && bundle_ready));
    ap_event_payload_stable_while_waiting:
        assert property (event_valid && !event_accept
            |=> protocol_error
                || (event_valid && $stable({event_tag, event_time,
                    event_source_channel, event_source_y, event_source_x,
                    event_input_height, event_input_width, event_last})));
    ap_valid_implies_busy:
        assert property (bundle_valid |-> busy);
    ap_count_range:
        assert property (bundle_valid |-> bundle_count inside {[1:8]});
    ap_lane_count:
        assert property (bundle_valid
            |-> $countones(tap_lane_valid) == bundle_count);
    ap_lane_prefix:
        assert property (bundle_valid
            |-> tap_lane_valid == (8'hff >> (8-bundle_count)));
    ap_event_boundary_subset:
        assert property ((tap_event_last & ~tap_lane_valid) == 0);
    ap_last_scalar_matches_last_lane:
        assert property (bundle_valid |-> bundle_last_for_event
            == tap_event_last[bundle_count-1]);
    ap_stream_last_subset:
        assert property (bundle_stream_last |-> bundle_last_for_event);
    ap_stream_last_terminates_bundle:
        assert property (bundle_stream_last
            |-> tap_event_last[bundle_count-1]);
    ap_fifo_count_range:
        assert property (debug_fifo_count <= FIFO_DEPTH);
    ap_fault_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_no_event_after_fault:
        assert property (protocol_error |-> !event_accept);
    ap_bundle_scalar_stable_on_stall:
        assert property (bundle_valid && !bundle_ready
            |=> bundle_valid && $stable({bundle_tag, bundle_time,
                bundle_count, tap_lane_valid, tap_event_last,
                bundle_last_for_event, bundle_stream_last}));
    ap_event_counter:
        assert property (event_accept
            |=> debug_event_count == $past(debug_event_count) + 1);
    ap_bundle_counter:
        assert property (bundle_accept
            |=> debug_bundle_count == $past(debug_bundle_count) + 1);
    ap_tap_counter:
        assert property (bundle_accept
            |=> debug_tap_count == $past(debug_tap_count)
                + $past(bundle_count));

    generate
        for (genvar lane = 0; lane < BUNDLE_LANES; lane++) begin : g_lane_sva
            ap_lane_payload_stable_on_stall:
                assert property (bundle_valid && !bundle_ready
                    |=> $stable({tap_source_channel[lane],
                        tap_source_y[lane], tap_source_x[lane],
                        tap_kernel_y[lane], tap_kernel_x[lane],
                        tap_kernel_index[lane], tap_destination_y[lane],
                        tap_destination_x[lane], tap_phase_bank[lane],
                        tap_event_last[lane]}));
            ap_kernel_index:
                assert property (bundle_valid && tap_lane_valid[lane]
                    |-> tap_kernel_index[lane]
                        == tap_kernel_y[lane] * 3 + tap_kernel_x[lane]);
            ap_phase_bank:
                assert property (bundle_valid && tap_lane_valid[lane]
                    |-> tap_phase_bank[lane]
                        == {tap_destination_y[lane][0],
                            tap_destination_x[lane][0]});
        end
    endgenerate

    cp_full_eight_tap_bundle:
        cover property (bundle_accept && bundle_count == 8);
    cp_one_tap_boundary_flush:
        cover property (bundle_accept && bundle_count == 1
            && bundle_last_for_event && !bundle_stream_last);
    cp_cross_event_tail_fill:
        cover property (bundle_accept
            && |(tap_event_last[6:0] & tap_lane_valid[7:1]));
    cp_stream_last_flush:
        cover property (bundle_accept && bundle_stream_last);
    cp_partial_bundle_flush:
        cover property (bundle_accept && bundle_count inside {[1:7]});
    cp_stall:
        cover property (bundle_valid && !bundle_ready
            ##1 bundle_valid && bundle_ready);
    cp_same_edge_input_output:
        cover property (event_accept && bundle_accept);
    cp_fifo_full:
        cover property (debug_fifo_count == FIFO_DEPTH);
    cp_protocol_fault_during_busy:
        cover property (busy && event_valid && !event_ready
            ##1 protocol_error && busy);
    cp_fault_drain_complete:
        cover property (protocol_error && $fell(busy));
endmodule

`default_nettype wire
