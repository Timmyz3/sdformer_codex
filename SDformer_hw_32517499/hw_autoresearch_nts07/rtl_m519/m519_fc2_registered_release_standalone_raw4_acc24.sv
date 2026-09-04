`timescale 1ns/1ps
`default_nettype none

// M519 registered-release clone of M342: standalone raw4 -> Acc24 FC2 integration boundary.
//
// SOURCE_CAP=8 joins the frozen M216 frontend to the frozen M218 service.
// SOURCE_CAP=1 is the scope-matched baseline: the same M216 raw frontend,
// a lossless onehot-to-scalar adapter, and the frozen M219 cropped service.
// Both variants expose the same logical eight-bank memory interface and the
// same committed Acc24 result interface.  BN2/SN2/scaling are deliberately
// outside this module.
module m519_fc2_registered_release_standalone_raw4_acc24 #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SOURCE_CAP = 8,
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

    output logic                         mem_req_valid,
    input  logic                         mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch,
    output logic [2:0]                   mem_req_slot,
    output logic [GENERATION_BITS-1:0]   mem_req_generation,
    output logic [TAG_BITS-1:0]          mem_req_tag,
    output logic [2:0]                   mem_req_output_block,
    output logic [2:0]                   mem_req_slice,
    output logic [3:0]                   mem_req_source_count,
    output logic [7:0]                   mem_req_bank_valid,
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel [0:7],
    output logic                         mem_req_accept,

    input  logic                         mem_rsp_valid,
    output logic                         mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch,
    input  logic [2:0]                   mem_rsp_slot,
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation,
    input  logic [TAG_BITS-1:0]          mem_rsp_tag,
    input  logic [7:0]                   mem_rsp_bank_valid,
    input  logic signed [7:0]            mem_rsp_weight [0:7][0:SLICE_LANES-1],
    output logic                         mem_rsp_accept,

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
    output logic [2:0]                   debug_fifo_count,
    output logic [3:0]                   debug_outstanding_count,
    output logic [31:0]                  debug_group_accept_count,
    output logic [31:0]                  debug_request_accept_count,
    output logic [31:0]                  debug_response_accept_count,
    output logic [31:0]                  debug_context_write_count,
    output logic [31:0]                  debug_result_accept_count,
    output logic [31:0]                  debug_active_bank_read_count
);
    localparam bit PARAMETERS_LEGAL = (SOURCE_CAP == 1 || SOURCE_CAP == 8)
        && SLICE_LANES == 16;

    logic integration_header_legal, adapter_fault_q, adapter_fault_now;
    logic group_seen_q;

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

    logic svc_header_valid, svc_header_ready, svc_header_accept;
    logic svc_group_valid, svc_group_ready, svc_group_accept;
    logic svc_frontend_done_valid, svc_frontend_done_ready;
    logic svc_frontend_done_accept;
    logic svc_protocol_error, svc_numeric_overflow;
    logic svc_stale_response_seen, svc_busy;
    logic svc_soft_flush;

    logic k1_group_mask_legal, k1_rsp_mask_legal;
    logic [2:0] k1_group_bank, k1_rsp_bank;

    // Design Compiler V-2023.12 does not synthesize the SystemVerilog
    // $onehot system function.  Keep the legality check explicit so VCS and
    // synthesis share the same one-of-eight hardware predicate.  Unknown or
    // high-impedance masks fall through to illegal instead of entering the
    // scalar K1 path.
    function automatic logic onehot8(input logic [7:0] value);
        case (value)
            8'b0000_0001,
            8'b0000_0010,
            8'b0000_0100,
            8'b0000_1000,
            8'b0001_0000,
            8'b0010_0000,
            8'b0100_0000,
            8'b1000_0000: onehot8 = 1'b1;
            default:      onehot8 = 1'b0;
        endcase
    endfunction

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M342 supports only SOURCE_CAP={1,8}, SLICE16");
        end
    endgenerate

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
        integration_header_legal = integration_header_legal
            && PARAMETERS_LEGAL;

        k1_group_mask_legal = fe_group_source_count == 1
            && onehot8(fe_group_bank_valid);
        k1_group_bank = 0;
        for (int bank = 0; bank < 8; bank++) begin
            if (fe_group_bank_valid[bank]) k1_group_bank = bank[2:0];
        end
        k1_rsp_mask_legal = onehot8(mem_rsp_bank_valid);
        k1_rsp_bank = 0;
        for (int bank = 0; bank < 8; bank++) begin
            if (mem_rsp_bank_valid[bank]) k1_rsp_bank = bank[2:0];
        end
    end

    // Header payload is presented to both consumers.  Mutual ready gating
    // makes a header acceptance indivisible: neither child can advance alone.
    assign fe_header_valid = header_valid && integration_header_legal
        && svc_header_ready && !adapter_fault_q;
    assign svc_header_valid = header_valid && integration_header_legal
        && fe_header_ready && !adapter_fault_q;
    assign header_ready = integration_header_legal && fe_header_ready
        && svc_header_ready && !adapter_fault_q;
    assign header_accept = header_valid && header_ready;

    assign fe_raw_valid = raw_valid && !svc_protocol_error
        && !svc_numeric_overflow && !adapter_fault_q;
    assign raw_ready = fe_raw_ready && !svc_protocol_error
        && !svc_numeric_overflow && !adapter_fault_q;
    assign raw_accept = fe_raw_accept;

    // M216 token completion is not externally visible.  It is the sole
    // frontend_done event consumed by M218/M219.  The top-level token_done
    // ports below come only from the selected service island.
    assign svc_frontend_done_valid = fe_done_valid;
    assign fe_done_ready = svc_frontend_done_ready;

    assign svc_soft_flush = 1'b0;
    assign protocol_error = adapter_fault_q
        || (header_valid && !integration_header_legal)
        || fe_protocol_error || svc_protocol_error;
    assign numeric_overflow = svc_numeric_overflow;
    assign stale_response_seen = svc_stale_response_seen;
    assign busy = fe_busy || svc_busy;

    always_comb begin
        adapter_fault_now = 0;
        if (fe_header_accept != svc_header_accept) adapter_fault_now = 1;
        if (fe_group_accept != svc_group_accept) adapter_fault_now = 1;
        if (fe_done_accept != svc_frontend_done_accept)
            adapter_fault_now = 1;
        if (svc_frontend_done_accept
                && fe_done_had_event != (group_seen_q || fe_group_accept))
            adapter_fault_now = 1;
        if (SOURCE_CAP == 1 && fe_group_valid && !k1_group_mask_legal)
            adapter_fault_now = 1;
        if (SOURCE_CAP == 1 && mem_rsp_valid && !k1_rsp_mask_legal)
            adapter_fault_now = 1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            adapter_fault_q <= 0;
            group_seen_q <= 0;
        end else begin
            if (adapter_fault_now || (header_valid && !integration_header_legal))
                adapter_fault_q <= 1;
            if (header_accept) group_seen_q <= 0;
            if (fe_group_accept) group_seen_q <= 1;
        end
    end

    m216_fc2_raw4_to_source_cap_frontend #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .SOURCE_CAP(SOURCE_CAP)
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

    generate
        if (SOURCE_CAP == 8) begin : g_k8
            assign svc_group_valid = fe_group_valid;
            assign fe_group_ready = svc_group_ready;

            m218_fc2_tagged_slice_service_island #(
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS),
                .GENERATION_BITS(GENERATION_BITS),
                .OUTSTANDING(8), .GROUP_FIFO_DEPTH(4),
                .SLICE_LANES(SLICE_LANES),
                .FLUSH_ACK_TIMEOUT_CYCLES(1024)
            ) service (
                .clk_core(clk_core), .rst_core(rst_core),
                .soft_flush(svc_soft_flush), .mem_flush_valid(),
                .mem_flush_ready(1'b1), .mem_flush_epoch(),
                .mem_flush_ack_valid(1'b0), .mem_flush_ack_ready(),
                .mem_flush_ack_epoch('0),
                .header_valid(svc_header_valid),
                .header_ready(svc_header_ready), .header_tag(header_tag),
                .header_output_blocks(header_output_blocks),
                .header_accept(svc_header_accept),
                .group_valid(svc_group_valid),
                .group_ready(svc_group_ready), .group_tag(fe_group_tag),
                .group_output_block(fe_group_output_block),
                .group_source_count(fe_group_source_count),
                .group_bank_valid(fe_group_bank_valid),
                .group_source_channel(fe_group_source_channel),
                .group_accept(svc_group_accept),
                .frontend_done_valid(svc_frontend_done_valid),
                .frontend_done_ready(svc_frontend_done_ready),
                .frontend_done_tag(fe_done_tag),
                .frontend_done_had_event(fe_done_had_event),
                .frontend_done_accept(svc_frontend_done_accept),
                .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
                .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot),
                .mem_req_generation(mem_req_generation),
                .mem_req_tag(mem_req_tag),
                .mem_req_output_block(mem_req_output_block),
                .mem_req_slice(mem_req_slice),
                .mem_req_source_count(mem_req_source_count),
                .mem_req_bank_valid(mem_req_bank_valid),
                .mem_req_source_channel(mem_req_source_channel),
                .mem_req_accept(mem_req_accept),
                .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
                .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
                .mem_rsp_generation(mem_rsp_generation),
                .mem_rsp_tag(mem_rsp_tag),
                .mem_rsp_bank_valid(mem_rsp_bank_valid),
                .mem_rsp_weight(mem_rsp_weight),
                .mem_rsp_accept(mem_rsp_accept),
                .result_valid(result_valid), .result_ready(result_ready),
                .result_tag(result_tag),
                .result_output_block(result_output_block),
                .result_slice(result_slice),
                .result_accumulator(result_accumulator),
                .result_last(result_last), .result_accept(result_accept),
                .token_done_valid(token_done_valid),
                .token_done_ready(token_done_ready),
                .token_done_tag(token_done_tag),
                .token_done_had_event(token_done_had_event),
                .token_done_accept(token_done_accept),
                .protocol_error(svc_protocol_error),
                .numeric_overflow(svc_numeric_overflow),
                .stale_response_seen(svc_stale_response_seen),
                .busy(svc_busy), .debug_fifo_count(debug_fifo_count),
                .debug_outstanding_count(debug_outstanding_count),
                .debug_group_accept_count(debug_group_accept_count),
                .debug_request_accept_count(debug_request_accept_count),
                .debug_response_accept_count(debug_response_accept_count),
                .debug_context_write_count(debug_context_write_count),
                .debug_result_accept_count(debug_result_accept_count),
                .debug_active_bank_read_count(debug_active_bank_read_count));
        end else begin : g_k1
            logic [2:0] k1_mem_req_bank_id;
            logic [CHANNEL_BITS-1:0] k1_mem_req_source_channel;
            logic signed [7:0] k1_mem_rsp_weight [0:SLICE_LANES-1];
            logic k1_mem_rsp_ready;

            assign svc_group_valid = fe_group_valid && k1_group_mask_legal;
            assign fe_group_ready = svc_group_ready && k1_group_mask_legal;

            always_comb begin
                mem_req_source_count = mem_req_valid ? 1 : 0;
                mem_req_bank_valid = mem_req_valid
                    ? (8'b1 << k1_mem_req_bank_id) : 0;
                for (int bank = 0; bank < 8; bank++) begin
                    mem_req_source_channel[bank] = 0;
                    if (mem_req_valid && bank[2:0] == k1_mem_req_bank_id)
                        mem_req_source_channel[bank]
                            = k1_mem_req_source_channel;
                end
                for (int lane = 0; lane < SLICE_LANES; lane++)
                    k1_mem_rsp_weight[lane]
                        = mem_rsp_weight[k1_rsp_bank][lane];
            end
            assign mem_rsp_ready = k1_rsp_mask_legal && k1_mem_rsp_ready;

            m519_fc2_k1_registered_release_service_island #(
                .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
                .EPOCH_BITS(EPOCH_BITS),
                .GENERATION_BITS(GENERATION_BITS),
                .OUTSTANDING(8), .GROUP_FIFO_DEPTH(4),
                .SLICE_LANES(SLICE_LANES),
                .FLUSH_ACK_TIMEOUT_CYCLES(1024)
            ) service (
                .clk_core(clk_core), .rst_core(rst_core),
                .soft_flush(svc_soft_flush), .mem_flush_valid(),
                .mem_flush_ready(1'b1), .mem_flush_epoch(),
                .mem_flush_ack_valid(1'b0), .mem_flush_ack_ready(),
                .mem_flush_ack_epoch('0),
                .header_valid(svc_header_valid),
                .header_ready(svc_header_ready), .header_tag(header_tag),
                .header_output_blocks(header_output_blocks),
                .header_accept(svc_header_accept),
                .group_valid(svc_group_valid),
                .group_ready(svc_group_ready), .group_tag(fe_group_tag),
                .group_output_block(fe_group_output_block),
                .group_bank_id(k1_group_bank),
                .group_source_channel(
                    fe_group_source_channel[k1_group_bank]),
                .group_accept(svc_group_accept),
                .frontend_done_valid(svc_frontend_done_valid),
                .frontend_done_ready(svc_frontend_done_ready),
                .frontend_done_tag(fe_done_tag),
                .frontend_done_had_event(fe_done_had_event),
                .frontend_done_accept(svc_frontend_done_accept),
                .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
                .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot),
                .mem_req_generation(mem_req_generation),
                .mem_req_tag(mem_req_tag),
                .mem_req_output_block(mem_req_output_block),
                .mem_req_slice(mem_req_slice),
                .mem_req_bank_id(k1_mem_req_bank_id),
                .mem_req_source_channel(k1_mem_req_source_channel),
                .mem_req_accept(mem_req_accept),
                .mem_rsp_valid(mem_rsp_valid && k1_rsp_mask_legal),
                .mem_rsp_ready(k1_mem_rsp_ready),
                .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
                .mem_rsp_generation(mem_rsp_generation),
                .mem_rsp_tag(mem_rsp_tag),
                .mem_rsp_bank_id(k1_rsp_bank),
                .mem_rsp_weight(k1_mem_rsp_weight),
                .mem_rsp_accept(mem_rsp_accept),
                .result_valid(result_valid), .result_ready(result_ready),
                .result_tag(result_tag),
                .result_output_block(result_output_block),
                .result_slice(result_slice),
                .result_accumulator(result_accumulator),
                .result_last(result_last), .result_accept(result_accept),
                .token_done_valid(token_done_valid),
                .token_done_ready(token_done_ready),
                .token_done_tag(token_done_tag),
                .token_done_had_event(token_done_had_event),
                .token_done_accept(token_done_accept),
                .protocol_error(svc_protocol_error),
                .numeric_overflow(svc_numeric_overflow),
                .stale_response_seen(svc_stale_response_seen),
                .busy(svc_busy), .debug_fifo_count(debug_fifo_count),
                .debug_outstanding_count(debug_outstanding_count),
                .debug_group_accept_count(debug_group_accept_count),
                .debug_request_accept_count(debug_request_accept_count),
                .debug_response_accept_count(debug_response_accept_count),
                .debug_context_write_count(debug_context_write_count),
                .debug_result_accept_count(debug_result_accept_count),
                .debug_active_bank_read_count(debug_active_bank_read_count));
        end
    endgenerate
endmodule

`default_nettype wire
