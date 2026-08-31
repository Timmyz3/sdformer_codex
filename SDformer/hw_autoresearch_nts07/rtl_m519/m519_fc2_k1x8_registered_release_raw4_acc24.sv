`timescale 1ns/1ps
`default_nettype none

// M519 registered-release equal-bandwidth scalar baseline.
//
// The frozen M216 SOURCE_CAP=8 frontend is used only as a lossless eight-bank
// dispatcher.  Every active bank is sent to one native-cropped M219 scalar
// service.  The eight services have independent request/response ports, so up
// to eight 128-bit words can be issued and returned per cycle.  Their final
// Acc24 slices are atomically joined and summed.  This is deliberately a strong
// K1x8 baseline: aggregate service capacity is O64/FIFO32 versus M218 O8/FIFO4.
module m519_fc2_k1x8_registered_release_raw4_acc24 #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         header_valid,
    output logic                         header_ready,
    input  logic [TAG_BITS-1:0]          header_tag,
    input  logic [5:0]                   header_raw_beat_count,
    input  logic [3:0]                   header_window_depth,
    input  logic [3:0]                   header_output_blocks,
    output logic                         header_accept,

    input  logic                         raw_valid,
    output logic                         raw_ready,
    input  logic [3:0]                   raw_lane_valid,
    input  logic [4:0]                   raw_beat_index [0:3],
    input  logic [95:0]                  raw_bitmap [0:3],
    input  logic                         raw_last,
    output logic                         raw_accept,

    output logic [7:0]                   mem_req_valid,
    input  logic [7:0]                   mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch [0:7],
    output logic [2:0]                   mem_req_slot [0:7],
    output logic [GENERATION_BITS-1:0]   mem_req_generation [0:7],
    output logic [TAG_BITS-1:0]          mem_req_tag [0:7],
    output logic [2:0]                   mem_req_output_block [0:7],
    output logic [2:0]                   mem_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel [0:7],
    output logic [7:0]                   mem_req_accept,

    input  logic [7:0]                   mem_rsp_valid,
    output logic [7:0]                   mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch [0:7],
    input  logic [2:0]                   mem_rsp_slot [0:7],
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation [0:7],
    input  logic [TAG_BITS-1:0]          mem_rsp_tag [0:7],
    input  logic signed [7:0]            mem_rsp_weight [0:7][0:SLICE_LANES-1],
    output logic [7:0]                   mem_rsp_accept,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic [2:0]                   result_output_block,
    output logic [2:0]                   result_slice,
    output logic signed [23:0]           result_accumulator [0:SLICE_LANES-1],
    output logic                         result_last,
    output logic                         result_accept,

    output logic                         token_done_valid,
    input  logic                         token_done_ready,
    output logic [TAG_BITS-1:0]          token_done_tag,
    output logic                         token_done_had_event,
    output logic                         token_done_accept,

    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         stale_response_seen,
    output logic                         busy,
    output logic [5:0]                   debug_fifo_count,
    output logic [6:0]                   debug_outstanding_count,
    output logic [31:0]                  debug_group_accept_count,
    output logic [31:0]                  debug_request_accept_count,
    output logic [31:0]                  debug_response_accept_count,
    output logic [31:0]                  debug_context_write_count,
    output logic [31:0]                  debug_result_accept_count,
    output logic [31:0]                  debug_active_bank_read_count
);
    logic integration_header_legal;
    logic adapter_fault_q, adapter_fault_now;
    logic join_overflow_q, join_overflow_now;

    logic fe_header_valid, fe_header_ready, fe_header_accept;
    logic fe_raw_valid, fe_raw_ready, fe_raw_accept;
    logic fe_group_valid, fe_group_ready, fe_group_accept;
    logic [TAG_BITS-1:0] fe_group_tag;
    logic [2:0] fe_group_output_block;
    logic [3:0] fe_group_source_count;
    logic [7:0] fe_group_bank_valid;
    logic [CHANNEL_BITS-1:0] fe_group_source_channel [0:7];
    logic fe_done_valid, fe_done_ready, fe_done_accept;
    logic [TAG_BITS-1:0] fe_done_tag;
    logic [5:0] fe_done_descriptor_count;
    logic fe_done_had_event;
    logic fe_protocol_error, fe_busy;

    logic [7:0] lane_header_valid, lane_header_ready, lane_header_accept;
    logic [7:0] lane_group_valid, lane_group_ready, lane_group_accept;
    logic [7:0] lane_frontend_done_valid, lane_frontend_done_ready;
    logic [7:0] lane_frontend_done_accept;
    logic [7:0] lane_result_valid, lane_result_ready, lane_result_accept;
    logic [TAG_BITS-1:0] lane_result_tag [0:7];
    logic [2:0] lane_result_block [0:7], lane_result_slice [0:7];
    logic signed [23:0] lane_result_accumulator [0:7][0:SLICE_LANES-1];
    logic [7:0] lane_result_last;
    logic [7:0] lane_done_valid, lane_done_ready, lane_done_accept;
    logic [TAG_BITS-1:0] lane_done_tag [0:7];
    logic [7:0] lane_done_had_event;
    logic [7:0] lane_protocol_error, lane_numeric_overflow;
    logic [7:0] lane_stale_response_seen, lane_busy;
    logic [2:0] lane_fifo_count [0:7];
    logic [3:0] lane_outstanding_count [0:7];
    logic [31:0] lane_group_count [0:7], lane_request_count [0:7];
    logic [31:0] lane_response_count [0:7], lane_context_count [0:7];
    logic [31:0] lane_result_count [0:7], lane_active_read_count [0:7];
    logic [7:0] lane_group_seen_q;

    logic all_lane_header_ready, all_active_group_ready;
    logic all_lane_frontend_done_ready, all_lane_result_valid;
    logic all_lane_done_valid, result_identity_legal, done_identity_legal;

    always_comb begin
        integration_header_legal = 0;
        case (header_output_blocks)
            1: integration_header_legal = header_raw_beat_count == 4
                && header_window_depth == 2;
            2: integration_header_legal = header_raw_beat_count == 8
                && header_window_depth == 4;
            4: integration_header_legal = header_raw_beat_count == 16
                && header_window_depth == 8;
            8: integration_header_legal = header_raw_beat_count == 32
                && header_window_depth == 8;
            default: integration_header_legal = 0;
        endcase

        all_lane_header_ready = &lane_header_ready;
        fe_header_valid = header_valid && integration_header_legal
            && all_lane_header_ready && !adapter_fault_q;
        for (int bank = 0; bank < 8; bank++)
            lane_header_valid[bank] = header_valid
                && integration_header_legal && fe_header_ready
                && all_lane_header_ready && !adapter_fault_q;
        header_ready = integration_header_legal && fe_header_ready
            && all_lane_header_ready && !adapter_fault_q;
        header_accept = header_valid && header_ready;

        all_active_group_ready = fe_group_source_count >= 1
            && fe_group_source_count <= 8
            && fe_group_source_count == $countones(fe_group_bank_valid);
        for (int bank = 0; bank < 8; bank++) begin
            if (fe_group_bank_valid[bank]
                    && (!lane_group_ready[bank]
                        || fe_group_source_channel[bank][2:0] != bank[2:0]))
                all_active_group_ready = 0;
        end
        fe_group_ready = all_active_group_ready && !adapter_fault_q;
        for (int bank = 0; bank < 8; bank++)
            lane_group_valid[bank] = fe_group_valid
                && fe_group_bank_valid[bank] && all_active_group_ready
                && !adapter_fault_q;

        all_lane_frontend_done_ready = &lane_frontend_done_ready;
        fe_done_ready = all_lane_frontend_done_ready && !adapter_fault_q;
        for (int bank = 0; bank < 8; bank++)
            lane_frontend_done_valid[bank] = fe_done_valid
                && all_lane_frontend_done_ready && !adapter_fault_q;

        all_lane_result_valid = &lane_result_valid;
        result_identity_legal = 1;
        for (int bank = 1; bank < 8; bank++) begin
            if (lane_result_tag[bank] != lane_result_tag[0]
                    || lane_result_block[bank] != lane_result_block[0]
                    || lane_result_slice[bank] != lane_result_slice[0]
                    || lane_result_last[bank] != lane_result_last[0])
                result_identity_legal = 0;
        end
        result_valid = all_lane_result_valid && result_identity_legal
            && !adapter_fault_q && !join_overflow_q;
        result_accept = result_valid && result_ready;
        result_tag = lane_result_tag[0];
        result_output_block = lane_result_block[0];
        result_slice = lane_result_slice[0];
        result_last = lane_result_last[0];
        for (int bank = 0; bank < 8; bank++)
            lane_result_ready[bank] = result_ready && all_lane_result_valid
                && result_identity_legal && !adapter_fault_q
                && !join_overflow_q;

        join_overflow_now = 0;
        for (int lane = 0; lane < SLICE_LANES; lane++) begin
            logic signed [26:0] sum;
            sum = 0;
            for (int bank = 0; bank < 8; bank++)
                sum = sum + $signed({{3{lane_result_accumulator[bank][lane][23]}},
                    lane_result_accumulator[bank][lane]});
            result_accumulator[lane] = sum[23:0];
            if (all_lane_result_valid
                    && sum[26:23] != {4{sum[23]}})
                join_overflow_now = 1;
        end

        all_lane_done_valid = &lane_done_valid;
        done_identity_legal = 1;
        for (int bank = 1; bank < 8; bank++) begin
            if (lane_done_tag[bank] != lane_done_tag[0])
                done_identity_legal = 0;
        end
        for (int bank = 0; bank < 8; bank++) begin
            if (all_lane_done_valid
                    && lane_done_had_event[bank] != lane_group_seen_q[bank])
                done_identity_legal = 0;
        end
        token_done_valid = all_lane_done_valid && done_identity_legal
            && !adapter_fault_q && !join_overflow_q;
        token_done_accept = token_done_valid && token_done_ready;
        token_done_tag = lane_done_tag[0];
        token_done_had_event = |lane_done_had_event;
        for (int bank = 0; bank < 8; bank++)
            lane_done_ready[bank] = token_done_ready && all_lane_done_valid
                && done_identity_legal && !adapter_fault_q
                && !join_overflow_q;

        adapter_fault_now = 0;
        if (fe_header_accept != header_accept) adapter_fault_now = 1;
        if (header_accept && lane_header_accept != 8'hff)
            adapter_fault_now = 1;
        if (fe_group_accept
                && lane_group_accept != fe_group_bank_valid)
            adapter_fault_now = 1;
        if (!fe_group_accept && lane_group_accept != 0)
            adapter_fault_now = 1;
        if (fe_done_accept
                && lane_frontend_done_accept != 8'hff)
            adapter_fault_now = 1;
        if (!fe_done_accept && lane_frontend_done_accept != 0)
            adapter_fault_now = 1;
        if (all_lane_result_valid && !result_identity_legal)
            adapter_fault_now = 1;
        if (all_lane_done_valid && !done_identity_legal)
            adapter_fault_now = 1;
    end

    assign fe_raw_valid = raw_valid && !adapter_fault_q
        && !(|lane_protocol_error) && !(|lane_numeric_overflow);
    assign raw_ready = fe_raw_ready && !adapter_fault_q
        && !(|lane_protocol_error) && !(|lane_numeric_overflow);
    assign raw_accept = fe_raw_accept;
    assign protocol_error = adapter_fault_q
        || (header_valid && !integration_header_legal)
        || fe_protocol_error || (|lane_protocol_error);
    assign numeric_overflow = join_overflow_q || (|lane_numeric_overflow);
    assign stale_response_seen = |lane_stale_response_seen;
    assign busy = fe_busy || (|lane_busy);

    always_comb begin
        debug_fifo_count = 0;
        debug_outstanding_count = 0;
        debug_group_accept_count = 0;
        debug_request_accept_count = 0;
        debug_response_accept_count = 0;
        debug_context_write_count = 0;
        debug_result_accept_count = 0;
        debug_active_bank_read_count = 0;
        for (int bank = 0; bank < 8; bank++) begin
            debug_fifo_count = debug_fifo_count + lane_fifo_count[bank];
            debug_outstanding_count = debug_outstanding_count
                + lane_outstanding_count[bank];
            debug_group_accept_count = debug_group_accept_count
                + lane_group_count[bank];
            debug_request_accept_count = debug_request_accept_count
                + lane_request_count[bank];
            debug_response_accept_count = debug_response_accept_count
                + lane_response_count[bank];
            debug_context_write_count = debug_context_write_count
                + lane_context_count[bank];
            debug_result_accept_count = debug_result_accept_count
                + lane_result_count[bank];
            debug_active_bank_read_count = debug_active_bank_read_count
                + lane_active_read_count[bank];
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            adapter_fault_q <= 0;
            join_overflow_q <= 0;
            lane_group_seen_q <= 0;
        end else begin
            if (adapter_fault_now || (header_valid && !integration_header_legal))
                adapter_fault_q <= 1;
            if (join_overflow_now) join_overflow_q <= 1;
            if (header_accept) begin
                lane_group_seen_q <= 0;
                join_overflow_q <= 0;
            end
            for (int bank = 0; bank < 8; bank++) begin
                if (lane_group_accept[bank]) lane_group_seen_q[bank] <= 1;
            end
        end
    end

    m216_fc2_raw4_to_source_cap_frontend #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS), .SOURCE_CAP(8)
    ) frontend (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(fe_header_valid), .header_ready(fe_header_ready),
        .header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(fe_header_accept),
        .raw_valid(fe_raw_valid), .raw_ready(fe_raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index),
        .raw_bitmap(raw_bitmap), .raw_last(raw_last),
        .raw_accept(fe_raw_accept),
        .group_valid(fe_group_valid), .group_ready(fe_group_ready),
        .group_tag(fe_group_tag),
        .group_output_block(fe_group_output_block),
        .group_source_count(fe_group_source_count),
        .group_bank_valid(fe_group_bank_valid),
        .group_source_channel(fe_group_source_channel),
        .group_accept(fe_group_accept),
        .token_done_valid(fe_done_valid), .token_done_ready(fe_done_ready),
        .token_done_tag(fe_done_tag),
        .token_done_descriptor_count(fe_done_descriptor_count),
        .token_done_had_event(fe_done_had_event),
        .token_done_accept(fe_done_accept),
        .protocol_error(fe_protocol_error), .busy(fe_busy));

    for (genvar bank = 0; bank < 8; bank++) begin : g_lane
        m519_fc2_k1_registered_release_service_island #(
            .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
            .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
            .OUTSTANDING(8), .GROUP_FIFO_DEPTH(4),
            .SLICE_LANES(SLICE_LANES), .FLUSH_ACK_TIMEOUT_CYCLES(1024)
        ) service (
            .clk_core(clk_core), .rst_core(rst_core),
            .soft_flush(1'b0), .mem_flush_valid(),
            .mem_flush_ready(1'b1), .mem_flush_epoch(),
            .mem_flush_ack_valid(1'b0), .mem_flush_ack_ready(),
            .mem_flush_ack_epoch('0),
            .header_valid(lane_header_valid[bank]),
            .header_ready(lane_header_ready[bank]),
            .header_tag(header_tag),
            .header_output_blocks(header_output_blocks),
            .header_accept(lane_header_accept[bank]),
            .group_valid(lane_group_valid[bank]),
            .group_ready(lane_group_ready[bank]),
            .group_tag(fe_group_tag),
            .group_output_block(fe_group_output_block),
            .group_bank_id(bank[2:0]),
            .group_source_channel(fe_group_source_channel[bank]),
            .group_accept(lane_group_accept[bank]),
            .frontend_done_valid(lane_frontend_done_valid[bank]),
            .frontend_done_ready(lane_frontend_done_ready[bank]),
            .frontend_done_tag(fe_done_tag),
            .frontend_done_had_event(lane_group_seen_q[bank]
                || lane_group_accept[bank]),
            .frontend_done_accept(lane_frontend_done_accept[bank]),
            .mem_req_valid(mem_req_valid[bank]),
            .mem_req_ready(mem_req_ready[bank]),
            .mem_req_epoch(mem_req_epoch[bank]),
            .mem_req_slot(mem_req_slot[bank]),
            .mem_req_generation(mem_req_generation[bank]),
            .mem_req_tag(mem_req_tag[bank]),
            .mem_req_output_block(mem_req_output_block[bank]),
            .mem_req_slice(mem_req_slice[bank]),
            .mem_req_bank_id(),
            .mem_req_source_channel(mem_req_source_channel[bank]),
            .mem_req_accept(mem_req_accept[bank]),
            .mem_rsp_valid(mem_rsp_valid[bank]),
            .mem_rsp_ready(mem_rsp_ready[bank]),
            .mem_rsp_epoch(mem_rsp_epoch[bank]),
            .mem_rsp_slot(mem_rsp_slot[bank]),
            .mem_rsp_generation(mem_rsp_generation[bank]),
            .mem_rsp_tag(mem_rsp_tag[bank]),
            .mem_rsp_bank_id(bank[2:0]),
            .mem_rsp_weight(mem_rsp_weight[bank]),
            .mem_rsp_accept(mem_rsp_accept[bank]),
            .result_valid(lane_result_valid[bank]),
            .result_ready(lane_result_ready[bank]),
            .result_tag(lane_result_tag[bank]),
            .result_output_block(lane_result_block[bank]),
            .result_slice(lane_result_slice[bank]),
            .result_accumulator(lane_result_accumulator[bank]),
            .result_last(lane_result_last[bank]),
            .result_accept(lane_result_accept[bank]),
            .token_done_valid(lane_done_valid[bank]),
            .token_done_ready(lane_done_ready[bank]),
            .token_done_tag(lane_done_tag[bank]),
            .token_done_had_event(lane_done_had_event[bank]),
            .token_done_accept(lane_done_accept[bank]),
            .protocol_error(lane_protocol_error[bank]),
            .numeric_overflow(lane_numeric_overflow[bank]),
            .stale_response_seen(lane_stale_response_seen[bank]),
            .busy(lane_busy[bank]),
            .debug_fifo_count(lane_fifo_count[bank]),
            .debug_outstanding_count(lane_outstanding_count[bank]),
            .debug_group_accept_count(lane_group_count[bank]),
            .debug_request_accept_count(lane_request_count[bank]),
            .debug_response_accept_count(lane_response_count[bank]),
            .debug_context_write_count(lane_context_count[bank]),
            .debug_result_accept_count(lane_result_count[bank]),
            .debug_active_bank_read_count(lane_active_read_count[bank]));
    end
endmodule

`default_nettype wire
