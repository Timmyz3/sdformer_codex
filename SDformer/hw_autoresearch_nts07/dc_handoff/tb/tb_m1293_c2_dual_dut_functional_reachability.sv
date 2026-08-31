`timescale 1ns/1ps
`default_nettype none

module m1293_c2_k1_diagnostic_system #(
    parameter bit VALID_QUALIFIED_ENDPOINT = 1'b0,
    parameter int LANES = 16
) (
    input  logic clk_core,
    input  logic rst_core,
    input  logic header_valid,
    output logic header_ready,
    input  logic [23:0] header_tag,
    input  logic [5:0] header_raw_beat_count,
    input  logic [3:0] header_window_depth,
    input  logic [3:0] header_output_blocks,
    output logic header_accept,
    input  logic raw_valid,
    output logic raw_ready,
    input  logic [3:0] raw_lane_valid,
    input  logic [4:0] raw_beat_index [0:3],
    input  logic [95:0] raw_bitmap [0:3],
    input  logic raw_last,
    output logic raw_accept,
    output logic protocol_error,
    output logic numeric_overflow,
    output logic stale_response_seen,
    output logic busy,
    output logic result_valid,
    output logic result_accept,
    output logic [23:0] result_tag,
    output logic [2:0] result_output_block,
    output logic [2:0] result_slice,
    output logic signed [23:0] result_accumulator [0:LANES-1],
    output logic result_last,
    output logic token_done_valid,
    output logic token_done_accept,
    output logic [23:0] token_done_tag,
    output logic token_done_had_event,
    output logic [12:0] semantic_taps,
    output logic [7:0] endpoint_protocol_fault_now,
    output logic [7:0] observed_mem_req_valid,
    output logic [7:0] observed_mem_req_ready,
    output logic [7:0] observed_mem_req_accept,
    output logic [23:0] observed_mem_req_slot
);
    logic [7:0] mem_req_valid, mem_req_ready, mem_req_accept;
    logic [15:0] mem_req_epoch [0:7];
    logic [2:0] mem_req_slot [0:7];
    logic [31:0] mem_req_generation [0:7];
    logic [23:0] mem_req_tag [0:7];
    logic [2:0] mem_req_output_block [0:7];
    logic [2:0] mem_req_slice [0:7];
    logic [11:0] mem_req_source_channel [0:7];
    logic [7:0] mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [15:0] mem_rsp_epoch [0:7];
    logic [2:0] mem_rsp_slot [0:7];
    logic [31:0] mem_rsp_generation [0:7];
    logic [23:0] mem_rsp_tag [0:7];
    logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1];
    logic [5:0] unused_fifo_count;
    logic [6:0] unused_outstanding_count;
    logic [31:0] unused_group_count, unused_request_count;
    logic [31:0] unused_response_count, unused_context_count;
    logic [31:0] unused_result_count, unused_active_read_count;
    logic [3:0] unused_adapter_live_slots;
    logic [31:0] unused_adapter_bundle_request_count;
    logic [31:0] unused_adapter_bank_request_count;
    logic [31:0] unused_adapter_bank_response_count;
    logic [31:0] unused_adapter_bundle_response_count;
    logic [31:0] bank_request_count [0:7];
    logic [31:0] bank_response_count [0:7];
    logic [3:0] bank_pending_count [0:7];
    logic bank_reuse_error [0:7];

    m1279_c2_k1_semantic_tap_wrapper dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_tag(header_tag), .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index),
        .raw_bitmap(raw_bitmap), .raw_last(raw_last), .raw_accept(raw_accept),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation), .mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(mem_req_accept),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight), .mem_rsp_accept(mem_rsp_accept),
        .result_valid(result_valid), .result_ready(!rst_core),
        .result_tag(result_tag), .result_output_block(result_output_block),
        .result_slice(result_slice), .result_accumulator(result_accumulator),
        .result_last(result_last), .result_accept(result_accept),
        .token_done_valid(token_done_valid), .token_done_ready(!rst_core),
        .token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),
        .protocol_error(protocol_error), .numeric_overflow(numeric_overflow),
        .stale_response_seen(stale_response_seen), .busy(busy),
        .debug_fifo_count(unused_fifo_count),
        .debug_outstanding_count(unused_outstanding_count),
        .debug_group_accept_count(unused_group_count),
        .debug_request_accept_count(unused_request_count),
        .debug_response_accept_count(unused_response_count),
        .debug_context_write_count(unused_context_count),
        .debug_result_accept_count(unused_result_count),
        .debug_active_bank_read_count(unused_active_read_count),
        .debug_adapter_live_slots(unused_adapter_live_slots),
        .debug_adapter_bundle_request_count(unused_adapter_bundle_request_count),
        .debug_adapter_bank_request_count(unused_adapter_bank_request_count),
        .debug_adapter_bank_response_count(unused_adapter_bank_response_count),
        .debug_adapter_bundle_response_count(unused_adapter_bundle_response_count),
        .tap_frontend_compactor_fault_q(semantic_taps[0]),
        .tap_frontend_paired_sink_fault_q(semantic_taps[1]),
        .tap_core_adapter_fault_q(semantic_taps[2]),
        .tap_service_fault_q(semantic_taps[3]),
        .tap_memory_adapter_fault_q(semantic_taps[4]),
        .tap_core_mem_req_accept(semantic_taps[5]),
        .tap_adapter_core_mem_req_accept(semantic_taps[6]),
        .tap_core_mem_rsp_accept(semantic_taps[7]),
        .tap_adapter_core_mem_rsp_accept(semantic_taps[8]),
        .tap_consistency_fault_now(semantic_taps[9]),
        .tap_consistency_fault_q(semantic_taps[10]),
        .tap_core_protocol_error(semantic_taps[11]),
        .tap_adapter_protocol_error(semantic_taps[12]));

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_memory
            if (VALID_QUALIFIED_ENDPOINT) begin : g_qualified
                m1293_valid_qualified_scalar_bank_endpoint #(
                    .BANK_ID(bank), .LATENCY(4)
                ) memory (
                    .clk_core(clk_core), .rst_core(rst_core), .enable(1'b1),
                    .request_allow(!rst_core), .newest_first(1'b1),
                    .spurious_valid(1'b0),
                    .mem_req_valid(mem_req_valid[bank]),
                    .mem_req_ready(mem_req_ready[bank]),
                    .mem_req_epoch(mem_req_epoch[bank]),
                    .mem_req_slot(mem_req_slot[bank]),
                    .mem_req_generation(mem_req_generation[bank]),
                    .mem_req_tag(mem_req_tag[bank]),
                    .mem_req_output_block(mem_req_output_block[bank]),
                    .mem_req_slice(mem_req_slice[bank]),
                    .mem_req_source_channel(mem_req_source_channel[bank]),
                    .mem_req_accept(mem_req_accept[bank]),
                    .endpoint_protocol_fault_now(endpoint_protocol_fault_now[bank]),
                    .mem_rsp_valid(mem_rsp_valid[bank]),
                    .mem_rsp_ready(mem_rsp_ready[bank]),
                    .mem_rsp_epoch(mem_rsp_epoch[bank]),
                    .mem_rsp_slot(mem_rsp_slot[bank]),
                    .mem_rsp_generation(mem_rsp_generation[bank]),
                    .mem_rsp_tag(mem_rsp_tag[bank]),
                    .mem_rsp_weight(mem_rsp_weight[bank]),
                    .mem_rsp_accept(mem_rsp_accept[bank]),
                    .request_count(bank_request_count[bank]),
                    .response_count(bank_response_count[bank]),
                    .pending_count(bank_pending_count[bank]),
                    .live_slot_reuse_error(bank_reuse_error[bank]));
            end else begin : g_original
                assign endpoint_protocol_fault_now[bank] = 1'b0;
                m349_fc2_scalar_bank_memory_model #(
                    .BANK_ID(bank), .LATENCY(4)
                ) memory (
                    .clk_core(clk_core), .rst_core(rst_core), .enable(1'b1),
                    .request_allow(!rst_core), .newest_first(1'b1),
                    .spurious_valid(1'b0),
                    .mem_req_valid(mem_req_valid[bank]),
                    .mem_req_ready(mem_req_ready[bank]),
                    .mem_req_epoch(mem_req_epoch[bank]),
                    .mem_req_slot(mem_req_slot[bank]),
                    .mem_req_generation(mem_req_generation[bank]),
                    .mem_req_tag(mem_req_tag[bank]),
                    .mem_req_output_block(mem_req_output_block[bank]),
                    .mem_req_slice(mem_req_slice[bank]),
                    .mem_req_source_channel(mem_req_source_channel[bank]),
                    .mem_req_accept(mem_req_accept[bank]),
                    .mem_rsp_valid(mem_rsp_valid[bank]),
                    .mem_rsp_ready(mem_rsp_ready[bank]),
                    .mem_rsp_epoch(mem_rsp_epoch[bank]),
                    .mem_rsp_slot(mem_rsp_slot[bank]),
                    .mem_rsp_generation(mem_rsp_generation[bank]),
                    .mem_rsp_tag(mem_rsp_tag[bank]),
                    .mem_rsp_weight(mem_rsp_weight[bank]),
                    .mem_rsp_accept(mem_rsp_accept[bank]),
                    .request_count(bank_request_count[bank]),
                    .response_count(bank_response_count[bank]),
                    .pending_count(bank_pending_count[bank]),
                    .live_slot_reuse_error(bank_reuse_error[bank]));
            end
            assign observed_mem_req_slot[bank*3 +: 3] = mem_req_slot[bank];
        end
    endgenerate

    assign observed_mem_req_valid = mem_req_valid;
    assign observed_mem_req_ready = mem_req_ready;
    assign observed_mem_req_accept = mem_req_accept;
endmodule

module tb_m1293_c2_dual_dut_functional_reachability;
    localparam int LANES = 16;
    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic header_valid_original, header_valid_qualified;
    logic header_ready_original, header_ready_qualified;
    logic header_accept_original, header_accept_qualified;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid_original, raw_valid_qualified;
    logic raw_ready_original, raw_ready_qualified;
    logic raw_accept_original, raw_accept_qualified;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic raw_last;
    logic protocol_error_original, protocol_error_qualified;
    logic numeric_overflow_original, numeric_overflow_qualified;
    logic stale_original, stale_qualified, busy_original, busy_qualified;
    logic result_valid_original, result_valid_qualified;
    logic result_accept_original, result_accept_qualified;
    logic [23:0] result_tag_original, result_tag_qualified;
    logic [2:0] result_block_original, result_block_qualified;
    logic [2:0] result_slice_original, result_slice_qualified;
    logic signed [23:0] result_acc_original [0:LANES-1];
    logic signed [23:0] result_acc_qualified [0:LANES-1];
    logic result_last_original, result_last_qualified;
    logic done_valid_original, done_valid_qualified;
    logic done_accept_original, done_accept_qualified;
    logic [23:0] done_tag_original, done_tag_qualified;
    logic done_event_original, done_event_qualified;
    logic [12:0] semantic_original, semantic_qualified;
    logic [7:0] endpoint_fault_original, endpoint_fault_qualified;
    logic [7:0] req_valid_original, req_valid_qualified;
    logic [7:0] req_ready_original, req_ready_qualified;
    logic [7:0] req_accept_original, req_accept_qualified;
    logic [23:0] req_slot_original, req_slot_qualified;

`define M1293_SYSTEM_INPUTS(HV,RV) \
        .clk_core(clk_core), .rst_core(rst_core), .header_valid(HV), \
        .header_tag(header_tag), .header_raw_beat_count(header_raw_beat_count), \
        .header_window_depth(header_window_depth), \
        .header_output_blocks(header_output_blocks), .raw_valid(RV), \
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index), \
        .raw_bitmap(raw_bitmap), .raw_last(raw_last)

    m1293_c2_k1_diagnostic_system #(.VALID_QUALIFIED_ENDPOINT(1'b0)) original (
        `M1293_SYSTEM_INPUTS(header_valid_original, raw_valid_original),
        .header_ready(header_ready_original), .header_accept(header_accept_original),
        .raw_ready(raw_ready_original), .raw_accept(raw_accept_original),
        .protocol_error(protocol_error_original),
        .numeric_overflow(numeric_overflow_original),
        .stale_response_seen(stale_original), .busy(busy_original),
        .result_valid(result_valid_original), .result_accept(result_accept_original),
        .result_tag(result_tag_original), .result_output_block(result_block_original),
        .result_slice(result_slice_original), .result_accumulator(result_acc_original),
        .result_last(result_last_original), .token_done_valid(done_valid_original),
        .token_done_accept(done_accept_original), .token_done_tag(done_tag_original),
        .token_done_had_event(done_event_original), .semantic_taps(semantic_original),
        .endpoint_protocol_fault_now(endpoint_fault_original),
        .observed_mem_req_valid(req_valid_original),
        .observed_mem_req_ready(req_ready_original),
        .observed_mem_req_accept(req_accept_original),
        .observed_mem_req_slot(req_slot_original));

    m1293_c2_k1_diagnostic_system #(.VALID_QUALIFIED_ENDPOINT(1'b1)) qualified (
        `M1293_SYSTEM_INPUTS(header_valid_qualified, raw_valid_qualified),
        .header_ready(header_ready_qualified), .header_accept(header_accept_qualified),
        .raw_ready(raw_ready_qualified), .raw_accept(raw_accept_qualified),
        .protocol_error(protocol_error_qualified),
        .numeric_overflow(numeric_overflow_qualified),
        .stale_response_seen(stale_qualified), .busy(busy_qualified),
        .result_valid(result_valid_qualified), .result_accept(result_accept_qualified),
        .result_tag(result_tag_qualified), .result_output_block(result_block_qualified),
        .result_slice(result_slice_qualified), .result_accumulator(result_acc_qualified),
        .result_last(result_last_qualified), .token_done_valid(done_valid_qualified),
        .token_done_accept(done_accept_qualified), .token_done_tag(done_tag_qualified),
        .token_done_had_event(done_event_qualified), .semantic_taps(semantic_qualified),
        .endpoint_protocol_fault_now(endpoint_fault_qualified),
        .observed_mem_req_valid(req_valid_qualified),
        .observed_mem_req_ready(req_ready_qualified),
        .observed_mem_req_accept(req_accept_qualified),
        .observed_mem_req_slot(req_slot_qualified));

`undef M1293_SYSTEM_INPUTS

    logic [63:0] sample_unknown_bitmap, unknown_union_bitmap, first_x_bitmap;
    integer first_x_cycle, window_cycle;
    integer request_count_original, request_count_qualified;
    integer result_count_original, result_count_qualified;
    integer done_count_original, done_count_qualified;
    integer request_class_mismatch_count, result_class_mismatch_count;
    integer done_class_mismatch_count;
    integer first_request_cycle, first_result_cycle, first_done_cycle;
    logic first_x_seen, headers_seen, raw_seen_original, raw_seen_qualified;

    always_comb begin : sample_all_classes
        sample_unknown_bitmap = '0;
        for (integer tap = 0; tap < 13; tap++) begin
            sample_unknown_bitmap[tap] = $isunknown(semantic_original[tap]);
            sample_unknown_bitmap[13+tap] = $isunknown(semantic_qualified[tap]);
        end
        sample_unknown_bitmap[26] = $isunknown({req_valid_original,
            req_ready_original, req_accept_original, req_slot_original});
        sample_unknown_bitmap[27] = $isunknown({req_valid_qualified,
            req_ready_qualified, req_accept_qualified, req_slot_qualified});
        sample_unknown_bitmap[28] = $isunknown({protocol_error_original,
            numeric_overflow_original, stale_original, busy_original});
        sample_unknown_bitmap[29] = $isunknown({protocol_error_qualified,
            numeric_overflow_qualified, stale_qualified, busy_qualified});
        sample_unknown_bitmap[30] = $isunknown(endpoint_fault_original);
        sample_unknown_bitmap[31] = $isunknown(endpoint_fault_qualified);
        sample_unknown_bitmap[32] = $isunknown({result_valid_original,
            result_accept_original, result_tag_original, result_block_original,
            result_slice_original, result_last_original});
        sample_unknown_bitmap[33] = $isunknown({result_valid_qualified,
            result_accept_qualified, result_tag_qualified, result_block_qualified,
            result_slice_qualified, result_last_qualified});
        for (integer lane = 0; lane < LANES; lane++) begin
            sample_unknown_bitmap[32] = sample_unknown_bitmap[32] |
                $isunknown(result_acc_original[lane]);
            sample_unknown_bitmap[33] = sample_unknown_bitmap[33] |
                $isunknown(result_acc_qualified[lane]);
        end
        sample_unknown_bitmap[34] = $isunknown({done_valid_original,
            done_accept_original, done_tag_original, done_event_original});
        sample_unknown_bitmap[35] = $isunknown({done_valid_qualified,
            done_accept_qualified, done_tag_qualified, done_event_qualified});
    end

    always @(posedge clk_core) begin : transaction_class_compare
        if (rst_core) begin
            request_count_original = 0;
            request_count_qualified = 0;
            result_count_original = 0;
            result_count_qualified = 0;
            done_count_original = 0;
            done_count_qualified = 0;
            request_class_mismatch_count = 0;
            result_class_mismatch_count = 0;
            done_class_mismatch_count = 0;
            first_request_cycle = -1;
            first_result_cycle = -1;
            first_done_cycle = -1;
        end else if (headers_seen) begin
            if ((|req_accept_original) || (|req_accept_qualified)) begin
                if (req_accept_original !== req_accept_qualified) begin
                    request_class_mismatch_count = request_class_mismatch_count + 1;
                    $fatal(1, "M1293 request-class accept vector mismatch");
                end
                for (integer bank = 0; bank < 8; bank++) begin
                    if (req_accept_original[bank] === 1'b1) begin
                        request_count_original = request_count_original + 1;
                        if (first_request_cycle < 0) first_request_cycle = window_cycle;
                        if (req_slot_original[bank*3 +: 3] !==
                                req_slot_qualified[bank*3 +: 3]) begin
                            request_class_mismatch_count =
                                request_class_mismatch_count + 1;
                            $fatal(1, "M1293 request-class bank/slot mismatch");
                        end
                    end
                    if (req_accept_qualified[bank] === 1'b1)
                        request_count_qualified = request_count_qualified + 1;
                end
            end

            if (result_accept_original || result_accept_qualified) begin
                if (result_accept_original !== result_accept_qualified) begin
                    result_class_mismatch_count = result_class_mismatch_count + 1;
                    $fatal(1, "M1293 result-class accept mismatch");
                end
                if (result_accept_original === 1'b1) begin
                    result_count_original = result_count_original + 1;
                    result_count_qualified = result_count_qualified + 1;
                    if (first_result_cycle < 0) first_result_cycle = window_cycle;
                    if ({result_tag_original, result_block_original,
                            result_slice_original, result_last_original} !==
                        {result_tag_qualified, result_block_qualified,
                            result_slice_qualified, result_last_qualified}) begin
                        result_class_mismatch_count = result_class_mismatch_count + 1;
                        $fatal(1, "M1293 result-class header mismatch");
                    end
                    for (integer lane = 0; lane < LANES; lane++) begin
                        if (result_acc_original[lane] !== result_acc_qualified[lane]) begin
                            result_class_mismatch_count =
                                result_class_mismatch_count + 1;
                            $fatal(1, "M1293 result-class accumulator mismatch");
                        end
                    end
                end
            end

            if (done_accept_original || done_accept_qualified) begin
                if (done_accept_original !== done_accept_qualified) begin
                    done_class_mismatch_count = done_class_mismatch_count + 1;
                    $fatal(1, "M1293 token-done-class accept mismatch");
                end
                if (done_accept_original === 1'b1) begin
                    done_count_original = done_count_original + 1;
                    done_count_qualified = done_count_qualified + 1;
                    if (first_done_cycle < 0) first_done_cycle = window_cycle;
                    if ({done_tag_original, done_event_original} !==
                            {done_tag_qualified, done_event_qualified}) begin
                        done_class_mismatch_count = done_class_mismatch_count + 1;
                        $fatal(1, "M1293 token-done-class payload mismatch");
                    end
                end
            end
        end
    end

    always @(posedge clk_core) begin : atomic_window
        logic [63:0] next_union;
        logic original_unknown, qualified_unknown;
        if (rst_core) begin
            unknown_union_bitmap = '0;
            first_x_bitmap = '0;
            first_x_cycle = -1;
            first_x_seen = 1'b0;
            window_cycle = 0;
            headers_seen = 1'b0;
            raw_seen_original = 1'b0;
            raw_seen_qualified = 1'b0;
        end else begin
            if (header_accept_original && header_accept_qualified)
                headers_seen = 1'b1;
            if (headers_seen) begin
                next_union = unknown_union_bitmap | sample_unknown_bitmap;
                unknown_union_bitmap = next_union;
                if ((sample_unknown_bitmap != '0) && !first_x_seen) begin
                    first_x_seen = 1'b1;
                    first_x_bitmap = sample_unknown_bitmap;
                    first_x_cycle = window_cycle;
                    $display("M1293_FIRST_X cycle=%0d bitmap=%016h",
                        window_cycle, sample_unknown_bitmap);
                end
                if (raw_accept_original) raw_seen_original = 1'b1;
                if (raw_accept_qualified) raw_seen_qualified = 1'b1;
                window_cycle = window_cycle + 1;
                if (window_cycle == 256) begin
                    original_unknown = |{next_union[34], next_union[32],
                        next_union[30], next_union[28], next_union[26],
                        next_union[12:0]};
                    qualified_unknown = |{next_union[35], next_union[33],
                        next_union[31], next_union[29], next_union[27],
                        next_union[25:13]};
                    if (!raw_seen_original || !raw_seen_qualified)
                        $fatal(1, "M1293 both DUTs must accept raw input");
                    if (request_count_original <= 0 || request_count_qualified <= 0)
                        $fatal(1, "M1293 endpoint did not participate: no bank request");
                    if (result_count_original <= 0 || result_count_qualified <= 0)
                        $fatal(1, "M1293 request did not reach result class");
                    if (done_count_original <= 0 || done_count_qualified <= 0)
                        $fatal(1, "M1293 request did not reach token-done class");
                    if (request_count_original != request_count_qualified ||
                            result_count_original != result_count_qualified ||
                            done_count_original != done_count_qualified)
                        $fatal(1, "M1293 dual-DUT transaction counts differ");
                    if (request_class_mismatch_count != 0 ||
                            result_class_mismatch_count != 0 ||
                            done_class_mismatch_count != 0)
                        $fatal(1, "M1293 class-aware functional compare failed");
                    if (first_request_cycle < 0 ||
                            first_result_cycle <= first_request_cycle ||
                            first_done_cycle < first_request_cycle)
                        $fatal(1, "M1293 request-to-completion ordering not reached");
                    if (endpoint_fault_qualified != 8'b0)
                        $fatal(1, "M1293 qualified endpoint fault");
                    if (qualified_unknown)
                        $fatal(1, "M1293 qualified path retains X");
                    if (protocol_error_qualified || numeric_overflow_qualified ||
                            stale_qualified)
                        $fatal(1, "M1293 qualified functional fault");
                    if (original_unknown)
                        $display("PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY classification=ORIGINAL_X_QUALIFIED_CLEAN req=%0d result=%0d done=%0d first=%0d/%0d/%0d",
                            request_count_qualified, result_count_qualified,
                            done_count_qualified, first_request_cycle,
                            first_result_cycle, first_done_cycle);
                    else
                        $display("PASS_M1293_DUAL_DUT_FUNCTIONAL_REACHABILITY classification=BOTH_CLEAN_FUNCTIONALLY_EQUAL req=%0d result=%0d done=%0d first=%0d/%0d/%0d",
                            request_count_qualified, result_count_qualified,
                            done_count_qualified, first_request_cycle,
                            first_result_cycle, first_done_cycle);
                    $finish;
                end
            end
        end
    end

    initial begin : stimulus
        integer wait_cycles;
        rst_core = 1'b1;
        header_valid_original = 1'b0;
        header_valid_qualified = 1'b0;
        raw_valid_original = 1'b0;
        raw_valid_qualified = 1'b0;
        header_tag = 24'h129300;
        header_raw_beat_count = 6'd4;
        header_window_depth = 4'd2;
        header_output_blocks = 4'd1;
        raw_lane_valid = 4'b0;
        raw_last = 1'b0;
        for (integer lane = 0; lane < 4; lane++) begin
            raw_beat_index[lane] = '0;
            raw_bitmap[lane] = '0;
        end
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        header_valid_original = 1'b1;
        header_valid_qualified = 1'b1;
        wait_cycles = 0;
        while (!(header_accept_original && header_accept_qualified) &&
                wait_cycles < 16) begin
            @(posedge clk_core);
            wait_cycles++;
        end
        if (!(header_accept_original && header_accept_qualified))
            $fatal(1, "M1293 both headers not accepted within 16 cycles");
        @(negedge clk_core);
        header_valid_original = 1'b0;
        header_valid_qualified = 1'b0;
        raw_lane_valid = 4'b1111;
        for (integer lane = 0; lane < 4; lane++) begin
            raw_beat_index[lane] = lane[4:0];
            raw_bitmap[lane] = 96'h000000000000000000000101 << (lane*8);
        end
        raw_last = 1'b1;
        raw_valid_original = 1'b1;
        raw_valid_qualified = 1'b1;
        wait_cycles = 0;
        while (!(raw_accept_original && raw_accept_qualified) &&
                wait_cycles < 32) begin
            @(posedge clk_core);
            wait_cycles++;
        end
        if (!(raw_accept_original && raw_accept_qualified))
            $fatal(1, "M1293 both raw packets not accepted within 32 cycles");
        @(negedge clk_core);
        raw_valid_original = 1'b0;
        raw_valid_qualified = 1'b0;
        raw_lane_valid = 4'b0;
        raw_last = 1'b0;
    end

    initial begin
        #2000 $fatal(1, "M1293 absolute watchdog");
    end
endmodule

`default_nettype wire
