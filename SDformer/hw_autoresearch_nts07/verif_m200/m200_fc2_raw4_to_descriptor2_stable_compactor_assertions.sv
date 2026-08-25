`timescale 1ns/1ps
`default_nettype none

module m200_fc2_raw4_to_descriptor2_stable_compactor_assertions #(
    parameter int TAG_BITS = 24,
    parameter int QUEUE_DEPTH = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic header_valid,
    input logic header_ready,
    input logic header_accept,
    input logic raw_valid,
    input logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [4:0] raw_beat_index [0:3],
    input logic [95:0] raw_bitmap [0:3],
    input logic raw_last,
    input logic raw_accept,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic [1:0] descriptor_count,
    input logic [4:0] descriptor_beat_index [0:1],
    input logic [95:0] descriptor_bitmap [0:1],
    input logic [1:0] descriptor_window_last,
    input logic descriptor_accept,
    input logic token_done_valid,
    input logic token_done_ready,
    input logic token_done_accept,
    input logic protocol_error,
    input logic busy
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_handshake:
        assert property (header_accept == (header_valid && header_ready));
    ap_raw_handshake:
        assert property (raw_accept == (raw_valid && raw_ready));
    ap_descriptor_handshake:
        assert property (descriptor_accept
            == (descriptor_valid && descriptor_ready));
    ap_done_handshake:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_descriptor_count:
        assert property (descriptor_valid
            |-> descriptor_count inside {2'd1, 2'd2});
    ap_descriptor_nonzero0:
        assert property (descriptor_valid |-> descriptor_bitmap[0] != 0);
    ap_descriptor_nonzero1:
        assert property (descriptor_valid && descriptor_count == 2
            |-> descriptor_bitmap[1] != 0);
    ap_descriptor_order:
        assert property (descriptor_valid && descriptor_count == 2
            |-> descriptor_beat_index[1] > descriptor_beat_index[0]);
    ap_no_cross_window_packet:
        assert property (descriptor_valid && descriptor_count == 2
            |-> !descriptor_window_last[0]);
    ap_invalid_lane_zero:
        assert property (descriptor_valid && descriptor_count == 1
            |-> descriptor_bitmap[1] == 0
                && descriptor_window_last[1] == 0);
    ap_done_after_drain:
        assert property (token_done_valid |-> !descriptor_valid);
    ap_busy_when_done:
        assert property (token_done_valid |-> busy);
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_closes_inputs:
        assert property (protocol_error |=> !header_ready && !raw_ready
            && !descriptor_valid && !token_done_valid);
    ap_hold_raw_on_stall:
        assert property (raw_valid && !raw_ready && !protocol_error |=>
            $stable({raw_valid, raw_lane_valid, raw_last,
                     raw_beat_index[0], raw_beat_index[1],
                     raw_beat_index[2], raw_beat_index[3],
                     raw_bitmap[0], raw_bitmap[1],
                     raw_bitmap[2], raw_bitmap[3]}));
    ap_hold_descriptor_on_stall:
        assert property (descriptor_valid && !descriptor_ready |=>
            $stable({descriptor_valid, descriptor_count,
                     descriptor_beat_index[0], descriptor_beat_index[1],
                     descriptor_bitmap[0], descriptor_bitmap[1],
                     descriptor_window_last}));

    cp_raw4_all_nonzero:
        cover property (raw_accept && raw_lane_valid == 4'hf
            && raw_bitmap[0] != 0 && raw_bitmap[1] != 0
            && raw_bitmap[2] != 0 && raw_bitmap[3] != 0);
    cp_raw4_all_zero:
        cover property (raw_accept && raw_lane_valid == 4'hf
            && raw_bitmap[0] == 0 && raw_bitmap[1] == 0
            && raw_bitmap[2] == 0 && raw_bitmap[3] == 0);
    cp_descriptor2:
        cover property (descriptor_accept && descriptor_count == 2);
    cp_window_boundary:
        cover property (descriptor_accept && |descriptor_window_last);
    cp_descriptor_stall:
        cover property (descriptor_valid && !descriptor_ready
            ##1 descriptor_accept);
    cp_raw_backpressure:
        cover property (raw_valid && !raw_ready && !protocol_error);
    cp_simultaneous_push_pop:
        cover property (raw_accept && descriptor_accept);
    cp_zero_token_done:
        cover property (raw_accept && raw_last
            && raw_bitmap[0] == 0 ##[1:12] token_done_accept);
    cp_bad_header_attack:
        cover property (header_valid && !header_ready && protocol_error);
    cp_bad_raw_attack:
        cover property (raw_valid && !raw_ready && protocol_error);
`endif
endmodule

`default_nettype wire
