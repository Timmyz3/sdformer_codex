`timescale 1ns/1ps
`default_nettype none

// M216: the exact M214 compactor feeding a SOURCE_CAP={1,8} shadow sink.
// SOURCE_CAP=8 must be cycle-identical to M214.  SOURCE_CAP=1 retains the same
// raw scan, queue8, two D8-capable windows, tags, stalls, terminal hint and
// authoritative same-done load, changing only fixed-bank group service width.
module m216_fc2_raw4_to_source_cap_frontend #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int SOURCE_CAP = 8
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, output logic header_ready,
    input logic [TAG_BITS-1:0] header_tag,
    input logic [5:0] header_raw_beat_count,
    input logic [3:0] header_window_depth,
    input logic [3:0] header_output_blocks,
    output logic header_accept,
    input logic raw_valid, output logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [4:0] raw_beat_index [0:3],
    input logic [95:0] raw_bitmap [0:3],
    input logic raw_last, output logic raw_accept,
    output logic group_valid, input logic group_ready,
    output logic [TAG_BITS-1:0] group_tag,
    output logic [2:0] group_output_block,
    output logic [3:0] group_source_count,
    output logic [7:0] group_bank_valid,
    output logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    output logic group_accept,
    output logic token_done_valid, input logic token_done_ready,
    output logic [TAG_BITS-1:0] token_done_tag,
    output logic [5:0] token_done_descriptor_count,
    output logic token_done_had_event,
    output logic token_done_accept,
    output logic protocol_error, output logic busy
);
    logic local_fault_q, header_shape_legal;
    logic m202_header_valid, m202_header_ready, m202_header_accept;
    logic m204_header_valid, m204_header_ready, m204_header_accept;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic m202_descriptor_accept, m204_descriptor_accept;
    logic [2:0] descriptor_count;
    logic [TAG_BITS-1:0] descriptor_token_tag;
    logic [4:0] descriptor_beat_index [0:3];
    logic [95:0] descriptor_bitmap [0:3];
    logic [3:0] descriptor_window_last;
    logic descriptor_token_last;
    logic compact_done_valid, compact_done_ready, compact_done_accept;
    logic m202_compact_done_accept, m204_upstream_done_accept;
    logic [TAG_BITS-1:0] compact_done_tag;
    logic [5:0] compact_done_descriptor_count;
    logic m202_protocol_error, m204_protocol_error;
    logic m202_busy, m204_busy;

    always_comb begin
        header_shape_legal = 0;
        case (header_output_blocks)
            1: header_shape_legal = header_raw_beat_count == 4
                && header_window_depth == 2;
            2: header_shape_legal = header_raw_beat_count == 8
                && header_window_depth == 4;
            4: header_shape_legal = header_raw_beat_count == 16
                && header_window_depth == 8;
            8: header_shape_legal = header_raw_beat_count == 32
                && header_window_depth == 8;
            default: header_shape_legal = 0;
        endcase
        header_shape_legal = header_shape_legal
            && (SOURCE_CAP == 1 || SOURCE_CAP == 8);
    end
    assign m202_header_valid = header_valid && header_shape_legal
        && m204_header_ready && !local_fault_q;
    assign m204_header_valid = header_valid && header_shape_legal
        && m202_header_ready && !local_fault_q;
    assign header_ready = header_shape_legal && m202_header_ready
        && m204_header_ready && !local_fault_q;
    assign header_accept = header_valid && header_ready;
    // Both producer and consumer expose an observational accept output.  Keep
    // those outputs on separate nets and audit their lockstep explicitly.
    assign descriptor_accept = m202_descriptor_accept
        && m204_descriptor_accept;
    assign compact_done_accept = m202_compact_done_accept
        && m204_upstream_done_accept;
    // Quarantine an illegal composite header in the request cycle as well as
    // latching it.  Neither child sees the request, so this local term is the
    // only same-cycle fail-closed indication at the composition boundary.
    assign protocol_error = local_fault_q
        || (header_valid && !header_shape_legal)
        || m202_protocol_error || m204_protocol_error;
    assign busy = m202_busy || m204_busy;

    always_ff @(posedge clk_core) begin
        if (rst_core) local_fault_q <= 0;
        else if (header_valid && !header_shape_legal)
            local_fault_q <= 1;
    end

    m214_fc2_raw4_to_descriptor4_terminal_hint_compactor #(
        .TAG_BITS(TAG_BITS), .QUEUE_DEPTH(8)
    ) compactor (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(m202_header_valid), .header_ready(m202_header_ready),
        .header_token_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_accept(m202_header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index),
        .raw_bitmap(raw_bitmap), .raw_last(raw_last), .raw_accept(raw_accept),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_count(descriptor_count),
        .descriptor_token_tag(descriptor_token_tag),
        .descriptor_beat_index(descriptor_beat_index),
        .descriptor_bitmap(descriptor_bitmap),
        .descriptor_window_last(descriptor_window_last),
        .descriptor_token_last(descriptor_token_last),
        .descriptor_accept(m202_descriptor_accept),
        .token_done_valid(compact_done_valid),
        .token_done_ready(compact_done_ready),
        .token_done_tag(compact_done_tag),
        .token_done_descriptor_count(compact_done_descriptor_count),
        .token_done_accept(m202_compact_done_accept),
        .protocol_error(m202_protocol_error), .busy(m202_busy)
    );

    m216_fc2_descriptor4_source_cap_frontend #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .MAX_WINDOW_DESCRIPTORS(8), .SOURCE_CAP(SOURCE_CAP)
    ) paired_sink (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(m204_header_valid), .header_ready(m204_header_ready),
        .header_tag(header_tag), .header_output_blocks(header_output_blocks),
        .header_accept(m204_header_accept),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_count(descriptor_count),
        .descriptor_token_tag(descriptor_token_tag),
        .descriptor_beat_index(descriptor_beat_index),
        .descriptor_bitmap(descriptor_bitmap),
        .descriptor_window_last(descriptor_window_last),
        .descriptor_token_last(descriptor_token_last),
        .descriptor_accept(m204_descriptor_accept),
        .upstream_done_valid(compact_done_valid),
        .upstream_done_ready(compact_done_ready),
        .upstream_done_tag(compact_done_tag),
        .upstream_done_descriptor_count(compact_done_descriptor_count),
        .upstream_done_accept(m204_upstream_done_accept),
        .group_valid(group_valid), .group_ready(group_ready),
        .group_tag(group_tag), .group_output_block(group_output_block),
        .group_source_count(group_source_count),
        .group_bank_valid(group_bank_valid),
        .group_source_channel(group_source_channel),
        .group_accept(group_accept),
        .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready), .token_done_tag(token_done_tag),
        .token_done_descriptor_count(token_done_descriptor_count),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),
        .protocol_error(m204_protocol_error), .busy(m204_busy)
    );
endmodule

`default_nettype wire
