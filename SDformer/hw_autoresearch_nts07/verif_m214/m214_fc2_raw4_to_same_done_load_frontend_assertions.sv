`timescale 1ns/1ps
`default_nettype none

module m214_fc2_raw4_to_same_done_load_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic header_shape_legal,
    input logic m202_header_accept, input logic m204_header_accept,
    input logic raw_valid, input logic raw_ready, input logic raw_accept,
    input logic raw_last,
    input logic [3:0] raw_lane_valid,
    input logic descriptor_valid, input logic descriptor_ready,
    input logic descriptor_accept, input logic [2:0] descriptor_count,
    input logic [3:0] descriptor_window_last,
    input logic descriptor_token_last,
    input logic m202_descriptor_accept, input logic m204_descriptor_accept,
    input logic compact_done_accept, input logic m202_compact_done_accept,
    input logic upstream_done_accept,
    input logic group_valid, input logic group_ready, input logic group_accept,
    input logic [TAG_BITS-1:0] group_tag,
    input logic [2:0] group_output_block,
    input logic [3:0] group_source_count,
    input logic [7:0] group_bank_valid,
    input logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept, input logic protocol_error,
    input logic pair_has_two, input logic terminal_pair_release,
    input logic stage0_handoff_load,
    input logic old_pair_available, input logic pair_available,
    input logic same_cycle_done_fence, input logic same_cycle_done_load,
    input logic candidate_load, input logic upstream_done_seen,
    input logic [3:0] output_blocks,
    input logic pair_first_closed,
    input logic terminal_partial_close,
    input logic compactor_raw_done,
    input logic compactor_queue_empty_next,
    input logic descriptor_bank_capacity_legal,
    input logic [5:0] descriptor_bank_sum [0:7]
);
`ifdef SVA_RUNTIME_ENABLED
    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_accept_definition:
        assert property (header_accept == (header_valid && header_ready));
    ap_joint_header_atomicity:
        assert property (header_accept
            == (m202_header_accept && m204_header_accept));
    ap_child_header_lockstep:
        assert property (m202_header_accept == m204_header_accept);
    ap_raw_accept_definition:
        assert property (raw_accept == (raw_valid && raw_ready));
    ap_descriptor_accept_lockstep:
        assert property (m202_descriptor_accept
            == m204_descriptor_accept);
    ap_terminal_hint_only_with_descriptor:
        assert property (descriptor_token_last |-> descriptor_valid);
    ap_terminal_hint_stable_under_stall:
        assert property (descriptor_valid && !descriptor_ready
            |=> descriptor_valid && $stable(descriptor_token_last));
    ap_accepted_terminal_hint_exact_tail:
        assert property (descriptor_accept && descriptor_token_last
            |-> compactor_queue_empty_next
                && (compactor_raw_done || (raw_accept && raw_last)));
    ap_terminal_tail_must_raise_hint:
        assert property (descriptor_accept && compactor_queue_empty_next
                && (compactor_raw_done || (raw_accept && raw_last))
            |-> descriptor_token_last);
    ap_terminal_partial_close_definition:
        assert property (terminal_partial_close
            == (descriptor_accept && descriptor_token_last
                && descriptor_window_last == 0));
    ap_compact_done_lockstep:
        assert property (m202_compact_done_accept == upstream_done_accept);
    ap_same_cycle_fence_is_authoritative_accept:
        assert property (same_cycle_done_fence == upstream_done_accept);
    ap_pair_availability_exact_extension:
        assert property (pair_available
            == (old_pair_available
                || (upstream_done_accept && pair_first_closed)));
    ap_causal_same_cycle_done_load:
        assert property (same_cycle_done_load
            == (candidate_load && upstream_done_accept
                && !old_pair_available));
    ap_same_cycle_done_load_is_lone_paired_stage:
        assert property (same_cycle_done_load
            |-> !upstream_done_seen && output_blocks != 1
                && pair_first_closed && !pair_has_two);
    ap_group_accept_definition:
        assert property (group_accept == (group_valid && group_ready));
    ap_group_mask_count:
        assert property (group_valid
            |-> group_source_count == popcount8(group_bank_valid));
    ap_group_stable_under_stall:
        assert property (group_valid && !group_ready
            |=> group_valid && $stable(group_tag)
                && $stable(group_output_block)
                && $stable(group_source_count)
                && $stable(group_bank_valid)
                && $stable(group_source_channel));
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);
    ap_terminal_release_exposes_done:
        assert property (terminal_pair_release |-> token_done_valid);
    ap_terminal_release_accepts_when_ready:
        assert property (terminal_pair_release && token_done_ready
            |-> token_done_accept);
    ap_handoff_is_nonterminal_release:
        assert property (stage0_handoff_load
            |-> group_accept && !token_done_accept);
    ap_handoff_keeps_group_resident:
        assert property (stage0_handoff_load |=> group_valid);
    ap_accepted_descriptor_bank_capacity:
        assert property (descriptor_accept
            |-> descriptor_bank_capacity_legal);
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_sum_bound
            ap_descriptor_bank_sum_bound:
                assert property (descriptor_bank_sum[bank] <= 6'd48);
        end
    endgenerate

    cp_joint_header: cover property (header_accept);
    cp_raw4_dense: cover property (raw_accept && raw_lane_valid == 4'b1111);
    cp_raw_backpressure: cover property (raw_valid && !raw_ready);
    cp_descriptor4: cover property (descriptor_accept
        && descriptor_count == 4);
    cp_paired_window: cover property (pair_has_two);
    cp_group_stall: cover property (group_valid && !group_ready);
    cp_group_accept: cover property (group_accept);
    cp_compact_done: cover property (compact_done_accept);
    cp_token_done: cover property (token_done_accept);
    cp_terminal_collapse: cover property (terminal_pair_release
        && token_done_accept);
    cp_terminal_header_chain: cover property (terminal_pair_release
        && token_done_accept && header_accept);
    cp_stage0_handoff: cover property (stage0_handoff_load);
    cp_terminal_partial_close: cover property (terminal_partial_close);
    cp_terminal_partial_close_while_group_releases: cover property (
        terminal_partial_close && group_accept);
    cp_same_cycle_done_load: cover property (same_cycle_done_load);
    cp_descriptor_bank_sum_48: cover property (descriptor_accept
        && descriptor_bank_sum[0] == 6'd48);
    cp_protocol_attack: cover property (protocol_error);
`endif
endmodule

bind m214_fc2_raw4_to_same_done_load_frontend
    m214_fc2_raw4_to_same_done_load_frontend_assertions sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_accept(header_accept),
        .header_shape_legal(header_shape_legal),
        .m202_header_accept(m202_header_accept),
        .m204_header_accept(m204_header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_accept(raw_accept), .raw_last(raw_last),
        .raw_lane_valid(raw_lane_valid),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_accept(descriptor_accept),
        .descriptor_count(descriptor_count),
        .descriptor_window_last(descriptor_window_last),
        .descriptor_token_last(descriptor_token_last),
        .m202_descriptor_accept(m202_descriptor_accept),
        .m204_descriptor_accept(m204_descriptor_accept),
        .compact_done_accept(compact_done_accept),
        .m202_compact_done_accept(m202_compact_done_accept),
        .upstream_done_accept(m204_upstream_done_accept),
        .group_valid(group_valid), .group_ready(group_ready),
        .group_accept(group_accept), .group_tag(group_tag),
        .group_output_block(group_output_block),
        .group_source_count(group_source_count),
        .group_bank_valid(group_bank_valid),
        .group_source_channel(group_source_channel),
        .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),
        .token_done_accept(token_done_accept),
        .protocol_error(protocol_error),
        .pair_has_two(paired_sink.pair_has_two),
        .terminal_pair_release(paired_sink.terminal_pair_release),
        .stage0_handoff_load(paired_sink.stage0_handoff_load),
        .old_pair_available(paired_sink.old_pair_available),
        .pair_available(paired_sink.pair_available),
        .same_cycle_done_fence(paired_sink.same_cycle_done_fence),
        .same_cycle_done_load(paired_sink.same_cycle_done_load),
        .candidate_load(paired_sink.candidate_load),
        .upstream_done_seen(paired_sink.upstream_done_seen_q),
        .output_blocks(paired_sink.output_blocks_q),
        .pair_first_closed(paired_sink.pair_first_closed),
        .terminal_partial_close(paired_sink.terminal_partial_close),
        .compactor_raw_done(compactor.raw_done_q),
        .compactor_queue_empty_next(compactor.queue_count_next == 0),
        .descriptor_bank_capacity_legal(
            paired_sink.descriptor_bank_capacity_legal),
        .descriptor_bank_sum(paired_sink.descriptor_bank_sum));

`default_nettype wire
