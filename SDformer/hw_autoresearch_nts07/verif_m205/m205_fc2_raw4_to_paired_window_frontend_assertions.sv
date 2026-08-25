`timescale 1ns/1ps
`default_nettype none

module m205_fc2_raw4_to_paired_window_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic header_shape_legal,
    input logic m202_header_accept, input logic m204_header_accept,
    input logic raw_valid, input logic raw_ready, input logic raw_accept,
    input logic [3:0] raw_lane_valid,
    input logic descriptor_accept, input logic [2:0] descriptor_count,
    input logic m202_descriptor_accept, input logic m204_descriptor_accept,
    input logic compact_done_accept, input logic m202_compact_done_accept,
    input logic upstream_done_accept,
    input logic group_valid, input logic group_ready, input logic group_accept,
    input logic [TAG_BITS-1:0] group_tag,
    input logic [2:0] group_output_block,
    input logic [3:0] group_source_count,
    input logic [7:0] group_bank_valid,
    input logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    input logic token_done_accept, input logic protocol_error,
    input logic pair_has_two
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
    ap_compact_done_lockstep:
        assert property (m202_compact_done_accept == upstream_done_accept);
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
    cp_protocol_attack: cover property (protocol_error);
`endif
endmodule

bind m205_fc2_raw4_to_paired_window_frontend
    m205_fc2_raw4_to_paired_window_frontend_assertions sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_accept(header_accept),
        .header_shape_legal(header_shape_legal),
        .m202_header_accept(m202_header_accept),
        .m204_header_accept(m204_header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_accept(raw_accept), .raw_lane_valid(raw_lane_valid),
        .descriptor_accept(descriptor_accept),
        .descriptor_count(descriptor_count),
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
        .token_done_accept(token_done_accept),
        .protocol_error(protocol_error),
        .pair_has_two(paired_sink.pair_has_two));

`default_nettype wire
