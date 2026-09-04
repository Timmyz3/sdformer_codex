`timescale 1ns/1ps
`default_nettype none

module m188_fc2_dual_window_k7_weakest_bank_defer_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int MAX_WINDOW_DESCRIPTORS = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic header_valid,
    input logic header_ready,
    input logic header_accept,
    input logic [3:0] header_output_blocks,
    input logic [5:0] header_descriptor_count,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic descriptor_accept,
    input logic [4:0] descriptor_beat_index,
    input logic [95:0] descriptor_bitmap,
    input logic group_valid,
    input logic group_ready,
    input logic group_accept,
    input logic [2:0] group_output_block,
    input logic [3:0] group_source_count,
    input logic [7:0] group_bank_valid,
    input logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    input logic token_done_valid,
    input logic token_done_ready,
    input logic token_done_accept,
    input logic [TAG_BITS-1:0] token_done_tag,
    input logic token_done_had_event,
    input logic protocol_error,
    input logic busy,
    input logic token_active_q,
    input logic token_has_index_q,
    input logic [4:0] last_beat_index_q,
    input logic [3:0] token_output_blocks_q,
    input logic [5:0] token_descriptor_count_q,
    input logic [5:0] descriptors_accepted_q,
    input logic [3:0] entry_count_q [0:1],
    input logic window_closed_q [0:1],
    input logic fill_window_releasing,
    input logic current_window_release,
    input logic candidate_load,
    input logic exclude_one_bank,
    input logic [2:0] excluded_bank,
    input logic [7:0] selected_valid,
    input logic [2:0] selected_entry [0:7]
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_handshake:
        assert property (header_accept == (header_valid && header_ready));
    ap_descriptor_handshake:
        assert property (descriptor_accept
            == (descriptor_valid && descriptor_ready));
    ap_group_handshake:
        assert property (group_accept == (group_valid && group_ready));
    ap_done_handshake:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_stops_inputs:
        assert property (protocol_error |=> !header_ready && !descriptor_ready);
    ap_header_extent_legal:
        assert property (header_accept |->
            (header_output_blocks == 1 && header_descriptor_count <= 4)
            || (header_output_blocks == 2 && header_descriptor_count <= 8)
            || (header_output_blocks == 4 && header_descriptor_count <= 16)
            || (header_output_blocks == 8 && header_descriptor_count <= 32));
    ap_descriptor_nonzero:
        assert property (descriptor_accept |-> descriptor_bitmap != 0);
    ap_index_increases:
        assert property (descriptor_accept && token_has_index_q
            |-> descriptor_beat_index > last_beat_index_q);
    ap_descriptor_count_bounded:
        assert property (descriptor_accept
            |-> descriptors_accepted_q < token_descriptor_count_q);
    ap_entry_bounds:
        assert property (entry_count_q[0] <= MAX_WINDOW_DESCRIPTORS
            && entry_count_q[1] <= MAX_WINDOW_DESCRIPTORS);
    ap_group_nonempty_bounded:
        assert property (group_valid |-> group_source_count inside {[1:7]});
    ap_group_count_matches_mask:
        assert property (group_valid |-> group_source_count
            == $countones(group_bank_valid));
    ap_group_block_bounded:
        assert property (group_valid
            |-> {1'b0, group_output_block} < token_output_blocks_q);
    ap_busy_covers_outputs:
        assert property ((group_valid || token_done_valid) |-> busy);
    ap_hold_group_on_stall:
        assert property (group_valid && !group_ready |=>
            $stable({group_valid, group_output_block, group_source_count,
                     group_bank_valid,
                     group_source_channel[0], group_source_channel[1],
                     group_source_channel[2], group_source_channel[3],
                     group_source_channel[4], group_source_channel[5],
                     group_source_channel[6], group_source_channel[7]}));
    ap_hold_done_on_stall:
        assert property (token_done_valid && !token_done_ready |=>
            $stable({token_done_valid, token_done_tag,
                     token_done_had_event}));
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_identity
            ap_channel_bank_identity:
                assert property (group_valid && group_bank_valid[bank]
                    |-> group_source_channel[bank][2:0] == bank[2:0]);
            ap_unused_channel_zero:
                assert property (group_valid && !group_bank_valid[bank]
                    |-> group_source_channel[bank] == 0);
        end
    endgenerate

    cp_one_source_group:
        cover property (group_accept && group_source_count == 1);
    cp_two_source_group:
        cover property (group_accept && group_source_count == 2);
    cp_three_source_group:
        cover property (group_accept && group_source_count == 3);
    cp_four_source_group:
        cover property (group_accept && group_source_count == 4);
    cp_five_source_group:
        cover property (group_accept && group_source_count == 5);
    cp_six_source_group:
        cover property (group_accept && group_source_count == 6);
    cp_seven_source_group:
        cover property (group_accept && group_source_count == 7);
    ap_defer_exactly_one:
        assert property (candidate_load && exclude_one_bank
            |-> $countones(selected_valid) == 7);
    cp_weakest_bank_defer:
        cover property (candidate_load && exclude_one_bank
            && $countones(selected_valid) == 7);
    cp_group_stall_then_accept:
        cover property (group_valid && !group_ready ##1 group_accept);
    cp_descriptor_backpressure:
        cover property (descriptor_valid && !descriptor_ready);
    cp_cross_descriptor_group:
        cover property (candidate_load && selected_valid[0]
            && ((selected_valid[1] && selected_entry[0] != selected_entry[1])
                || (selected_valid[2]
                    && selected_entry[0] != selected_entry[2])
                || (selected_valid[3]
                    && selected_entry[0] != selected_entry[3])
                || (selected_valid[4]
                    && selected_entry[0] != selected_entry[4])
                || (selected_valid[5]
                    && selected_entry[0] != selected_entry[5])
                || (selected_valid[6]
                    && selected_entry[0] != selected_entry[6])
                || (selected_valid[7]
                    && selected_entry[0] != selected_entry[7])));
    cp_window_to_window_replace:
        cover property (current_window_release && candidate_load);
    cp_release_and_refill:
        cover property (current_window_release && descriptor_accept
            && fill_window_releasing);
    cp_both_windows_closed:
        cover property (window_closed_q[0] && window_closed_q[1]);
    cp_stage0:
        cover property (group_accept && token_output_blocks_q == 1);
    cp_stage1:
        cover property (group_accept && token_output_blocks_q == 2);
    cp_stage2:
        cover property (group_accept && token_output_blocks_q == 4);
    cp_stage3:
        cover property (group_accept && token_output_blocks_q == 8);
    cp_zero_token:
        cover property (token_done_accept && !token_done_had_event);
    cp_nonzero_token:
        cover property (token_done_accept && token_done_had_event);
    cp_same_cycle_header_rearm:
        cover property (token_done_accept && header_accept);
`endif
endmodule

`default_nettype wire
