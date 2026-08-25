`timescale 1ns/1ps
`default_nettype none

module m175_fc2_bitmap96_hierarchical_k4_replay_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int BASE_ROW_BITS = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic scan_valid,
    input logic scan_ready,
    input logic scan_accept,
    input logic [BASE_ROW_BITS-1:0] scan_base_row,
    input logic group_valid,
    input logic group_ready,
    input logic group_accept,
    input logic [TAG_BITS-1:0] group_tag,
    input logic [2:0] group_output_block,
    input logic [2:0] group_source_count,
    input logic [2:0] group_bank_id [0:3],
    input logic [CHANNEL_BITS-1:0] group_source_channel [0:3],
    input logic token_done_valid,
    input logic token_done_ready,
    input logic token_done_accept,
    input logic [TAG_BITS-1:0] token_done_tag,
    input logic token_done_had_event,
    input logic protocol_error,
    input logic busy,
    input logic token_last_seen_q,
    input logic [3:0] token_output_blocks_q,
    input logic residual_valid_q,
    input logic group_final_accept
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_scan_handshake:
        assert property (scan_accept == (scan_valid && scan_ready));
    ap_group_handshake:
        assert property (group_accept == (group_valid && group_ready));
    ap_done_handshake:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_stops_scan:
        assert property (protocol_error |=> !scan_ready);
    ap_scan_alignment:
        assert property (scan_accept |-> scan_base_row[1:0] == 0);
    ap_group_nonempty_bounded:
        assert property (group_valid |-> group_source_count inside {[1:4]});
    ap_group_block_bounded:
        assert property (group_valid
            |-> {1'b0, group_output_block} < token_output_blocks_q);
    ap_busy_covers_outputs:
        assert property ((group_valid || token_done_valid) |-> busy);
    ap_hold_group_on_stall:
        assert property (group_valid && !group_ready |=>
            $stable({group_valid, group_tag, group_output_block,
                     group_source_count,
                     group_bank_id[0], group_bank_id[1],
                     group_bank_id[2], group_bank_id[3],
                     group_source_channel[0], group_source_channel[1],
                     group_source_channel[2], group_source_channel[3]}));
    ap_hold_done_on_stall:
        assert property (token_done_valid && !token_done_ready |=>
            $stable({token_done_valid, token_done_tag,
                     token_done_had_event}));
    generate
        for (genvar slot = 0; slot < 4; slot++) begin : g_bank_identity
            ap_channel_bank_identity:
                assert property (group_valid && group_source_count > slot
                    |-> group_source_channel[slot][2:0]
                        == group_bank_id[slot]);
        end
    endgenerate
    ap_unique_01:
        assert property (group_valid && group_source_count > 1
            |-> group_bank_id[0] != group_bank_id[1]);
    ap_unique_02:
        assert property (group_valid && group_source_count > 2
            |-> group_bank_id[0] != group_bank_id[2]
                && group_bank_id[1] != group_bank_id[2]);
    ap_unique_03:
        assert property (group_valid && group_source_count > 3
            |-> group_bank_id[0] != group_bank_id[3]
                && group_bank_id[1] != group_bank_id[3]
                && group_bank_id[2] != group_bank_id[3]);

    cp_four_source_group:
        cover property (group_accept && group_source_count == 4);
    cp_single_source_group:
        cover property (group_accept && group_source_count == 1);
    cp_same_cycle_group_replace:
        cover property (group_final_accept ##1 group_valid
            && group_output_block == 0);
    cp_raw_beat_prefetch_during_replay:
        cover property (scan_accept && group_valid
            && !group_final_accept);
    cp_group_stall_then_accept:
        cover property (group_valid && !group_ready ##1 group_accept);
    cp_stage0_final:
        cover property (group_final_accept && token_output_blocks_q == 1);
    cp_stage1_final:
        cover property (group_final_accept && token_output_blocks_q == 2);
    cp_stage2_final:
        cover property (group_final_accept && token_output_blocks_q == 4);
    cp_stage3_final:
        cover property (group_final_accept && token_output_blocks_q == 8);
    cp_zero_token_done:
        cover property (token_done_accept && !token_done_had_event);
    cp_nonzero_token_done:
        cover property (token_done_accept && token_done_had_event);
    cp_same_cycle_token_rearm:
        cover property (token_done_accept && scan_accept);
    cp_last_seen_with_pending_work:
        cover property (token_last_seen_q
            && (residual_valid_q || group_valid));
`endif
endmodule

`default_nettype wire
