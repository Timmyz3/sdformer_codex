`timescale 1ns/1ps
`default_nettype none

interface m497_fc2_candidate_if #(
    parameter int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16,
    parameter int GENERATION_BITS=32, SLICE_LANES=16
);
    logic header_valid, header_ready, header_accept;
    logic [TAG_BITS-1:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_ready, raw_accept, raw_last;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic mem_req_valid, mem_req_ready, mem_req_accept;
    logic [EPOCH_BITS-1:0] mem_req_epoch;
    logic [2:0] mem_req_slot;
    logic [GENERATION_BITS-1:0] mem_req_generation;
    logic [TAG_BITS-1:0] mem_req_tag;
    logic [2:0] mem_req_output_block, mem_req_slice;
    logic [3:0] mem_req_source_count;
    logic [7:0] mem_req_bank_valid;
    logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7];
    logic mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [EPOCH_BITS-1:0] mem_rsp_epoch;
    logic [2:0] mem_rsp_slot;
    logic [GENERATION_BITS-1:0] mem_rsp_generation;
    logic [TAG_BITS-1:0] mem_rsp_tag;
    logic [7:0] mem_rsp_bank_valid;
    logic signed [7:0] mem_rsp_weight [0:7][0:SLICE_LANES-1];
    logic result_valid, result_ready, result_accept, result_last;
    logic [TAG_BITS-1:0] result_tag;
    logic [2:0] result_output_block, result_slice;
    logic signed [23:0] result_accumulator [0:SLICE_LANES-1];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [TAG_BITS-1:0] token_done_tag;
    logic token_done_had_event;
    logic protocol_error, numeric_overflow, stale_response_seen, busy;
    logic [5:0] debug_fifo_count;
    logic [6:0] debug_outstanding_count;
    logic [31:0] debug_group_accept_count;
    logic [31:0] debug_request_accept_count;
    logic [31:0] debug_response_accept_count;
    logic [31:0] debug_context_write_count;
    logic [31:0] debug_result_accept_count;
    logic [31:0] debug_active_bank_read_count;
endinterface

`define M497_CANDIDATE_PORTS(P) \
    .clk_core(clk_core), .rst_core(rst_core), \
    .header_valid(P.header_valid), .header_ready(P.header_ready), \
    .header_tag(P.header_tag), \
    .header_raw_beat_count(P.header_raw_beat_count), \
    .header_window_depth(P.header_window_depth), \
    .header_output_blocks(P.header_output_blocks), \
    .header_accept(P.header_accept), \
    .raw_valid(P.raw_valid), .raw_ready(P.raw_ready), \
    .raw_lane_valid(P.raw_lane_valid), \
    .raw_beat_index(P.raw_beat_index), .raw_bitmap(P.raw_bitmap), \
    .raw_last(P.raw_last), .raw_accept(P.raw_accept), \
    .mem_req_valid(candidate_mem_req_valid), \
    .mem_req_ready(candidate_mem_req_ready), \
    .mem_req_epoch(candidate_mem_req_epoch), \
    .mem_req_slot(candidate_mem_req_slot), \
    .mem_req_generation(candidate_mem_req_generation), \
    .mem_req_tag(candidate_mem_req_tag), \
    .mem_req_output_block(candidate_mem_req_block), \
    .mem_req_slice(candidate_mem_req_slice), \
    .mem_req_source_channel(candidate_mem_req_channel), \
    .mem_req_accept(candidate_mem_req_accept), \
    .mem_rsp_valid(candidate_mem_rsp_valid), \
    .mem_rsp_ready(candidate_mem_rsp_ready), \
    .mem_rsp_epoch(candidate_mem_rsp_epoch), \
    .mem_rsp_slot(candidate_mem_rsp_slot), \
    .mem_rsp_generation(candidate_mem_rsp_generation), \
    .mem_rsp_tag(candidate_mem_rsp_tag), \
    .mem_rsp_weight(candidate_mem_rsp_weight), \
    .mem_rsp_accept(candidate_mem_rsp_accept), \
    .result_valid(P.result_valid), .result_ready(P.result_ready), \
    .result_tag(P.result_tag), \
    .result_output_block(P.result_output_block), \
    .result_slice(P.result_slice), \
    .result_accumulator(P.result_accumulator), \
    .result_last(P.result_last), .result_accept(P.result_accept), \
    .token_done_valid(P.token_done_valid), \
    .token_done_ready(P.token_done_ready), \
    .token_done_tag(P.token_done_tag), \
    .token_done_had_event(P.token_done_had_event), \
    .token_done_accept(P.token_done_accept), \
    .protocol_error(P.protocol_error), \
    .numeric_overflow(P.numeric_overflow), \
    .stale_response_seen(P.stale_response_seen), .busy(P.busy), \
    .debug_fifo_count(P.debug_fifo_count), \
    .debug_outstanding_count(P.debug_outstanding_count), \
    .debug_group_accept_count(P.debug_group_accept_count), \
    .debug_request_accept_count(P.debug_request_accept_count), \
    .debug_response_accept_count(P.debug_response_accept_count), \
    .debug_context_write_count(P.debug_context_write_count), \
    .debug_result_accept_count(P.debug_result_accept_count), \
    .debug_active_bank_read_count(P.debug_active_bank_read_count), \
    .debug_adapter_live_slots(candidate_debug_adapter_live_slots), \
    .debug_adapter_bundle_request_count( \
        candidate_debug_adapter_bundle_request_count), \
    .debug_adapter_bank_request_count( \
        candidate_debug_adapter_bank_request_count), \
    .debug_adapter_bank_response_count( \
        candidate_debug_adapter_bank_response_count), \
    .debug_adapter_bundle_response_count( \
        candidate_debug_adapter_bundle_response_count)

module tb_m497_fc2_canonical_k1_vs_k1x8_raw4_acc24;
    localparam int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16;
    localparam int GENERATION_BITS=32, LANES=16, MAX_CHANNELS=3072;
    logic clk_core=0, rst_core;
    always #1.5 clk_core=~clk_core;

    m497_fc2_candidate_if candidate_if();

    logic baseline_header_valid, baseline_header_ready, baseline_header_accept;
    logic [TAG_BITS-1:0] baseline_header_tag;
    logic [5:0] baseline_header_raw_beat_count;
    logic [3:0] baseline_header_window_depth, baseline_header_output_blocks;
    logic baseline_raw_valid, baseline_raw_ready, baseline_raw_accept;
    logic [3:0] baseline_raw_lane_valid;
    logic [4:0] baseline_raw_beat_index [0:3];
    logic [95:0] baseline_raw_bitmap [0:3];
    logic baseline_raw_last;
    logic [7:0] baseline_mem_req_valid, baseline_mem_req_ready;
    logic [EPOCH_BITS-1:0] baseline_mem_req_epoch [0:7];
    logic [2:0] baseline_mem_req_slot [0:7];
    logic [GENERATION_BITS-1:0] baseline_mem_req_generation [0:7];
    logic [TAG_BITS-1:0] baseline_mem_req_tag [0:7];
    logic [2:0] baseline_mem_req_block [0:7], baseline_mem_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] baseline_mem_req_channel [0:7];
    logic [7:0] baseline_mem_req_accept;
    logic [7:0] baseline_mem_rsp_valid, baseline_mem_rsp_ready;
    logic [EPOCH_BITS-1:0] baseline_mem_rsp_epoch [0:7];
    logic [2:0] baseline_mem_rsp_slot [0:7];
    logic [GENERATION_BITS-1:0] baseline_mem_rsp_generation [0:7];
    logic [TAG_BITS-1:0] baseline_mem_rsp_tag [0:7];
    logic signed [7:0] baseline_mem_rsp_weight [0:7][0:LANES-1];
    logic [7:0] baseline_mem_rsp_accept;
    logic baseline_result_valid, baseline_result_ready;
    logic baseline_result_accept, baseline_result_last;
    logic [TAG_BITS-1:0] baseline_result_tag;
    logic [2:0] baseline_result_block, baseline_result_slice;
    logic signed [23:0] baseline_result_accumulator [0:LANES-1];
    logic baseline_done_valid, baseline_done_ready, baseline_done_accept;
    logic [TAG_BITS-1:0] baseline_done_tag;
    logic baseline_done_had_event;
    logic baseline_protocol_error, baseline_numeric_overflow;
    logic baseline_stale_response_seen, baseline_busy;
    logic [5:0] baseline_debug_fifo_count;
    logic [6:0] baseline_debug_outstanding_count;
    logic [31:0] baseline_debug_group_count, baseline_debug_request_count;
    logic [31:0] baseline_debug_response_count, baseline_debug_context_count;
    logic [31:0] baseline_debug_result_count, baseline_debug_active_read_count;

    logic candidate_enable, baseline_enable, request_allow, response_allow;
    logic [7:0] candidate_mem_req_valid, candidate_mem_req_ready;
    logic [EPOCH_BITS-1:0] candidate_mem_req_epoch [0:7];
    logic [2:0] candidate_mem_req_slot [0:7];
    logic [GENERATION_BITS-1:0] candidate_mem_req_generation [0:7];
    logic [TAG_BITS-1:0] candidate_mem_req_tag [0:7];
    logic [2:0] candidate_mem_req_block [0:7], candidate_mem_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] candidate_mem_req_channel [0:7];
    logic [7:0] candidate_mem_req_accept;
    logic [7:0] candidate_mem_rsp_valid, candidate_mem_rsp_ready;
    logic [EPOCH_BITS-1:0] candidate_mem_rsp_epoch [0:7];
    logic [2:0] candidate_mem_rsp_slot [0:7];
    logic [GENERATION_BITS-1:0] candidate_mem_rsp_generation [0:7];
    logic [TAG_BITS-1:0] candidate_mem_rsp_tag [0:7];
    logic signed [7:0] candidate_mem_rsp_weight [0:7][0:LANES-1];
    logic [7:0] candidate_mem_rsp_accept;
    logic [7:0] candidate_memory_rsp_valid_internal;
    logic candidate_memory_spurious_valid;
    logic [31:0] candidate_memory_request_count;
    logic [31:0] candidate_memory_response_count;
    logic [31:0] candidate_memory_active_read_count;
    logic [3:0] candidate_memory_pending_count;
    logic candidate_memory_live_slot_reuse_error;
    logic [31:0] candidate_memory_request_count_bank [0:7];
    logic [31:0] candidate_memory_response_count_bank [0:7];
    logic [3:0] candidate_memory_pending_count_bank [0:7];
    logic [7:0] candidate_memory_live_slot_reuse_error_bank;
    logic [3:0] candidate_debug_adapter_live_slots;
    logic [31:0] candidate_debug_adapter_bundle_request_count;
    logic [31:0] candidate_debug_adapter_bank_request_count;
    logic [31:0] candidate_debug_adapter_bank_response_count;
    logic [31:0] candidate_debug_adapter_bundle_response_count;
    logic [7:0] baseline_memory_spurious_valid;
    logic [7:0] baseline_memory_rsp_valid_internal;
    logic [31:0] baseline_memory_request_count [0:7];
    logic [31:0] baseline_memory_response_count [0:7];
    logic [3:0] baseline_memory_pending_count [0:7];
    logic [7:0] baseline_memory_live_slot_reuse_error;

    m499_fc2_k1_no_reuse_8bank_raw4_acc24 candidate (
        `M497_CANDIDATE_PORTS(candidate_if));

    // The candidate and baseline both terminate in the same eight scalar-bank
    // model.  Only the controller/state organization differs.
    for (genvar bank = 0; bank < 8; bank++) begin : g_candidate_scalar_memory
        assign candidate_mem_rsp_valid[bank]
            = candidate_memory_rsp_valid_internal[bank]
                && (response_allow || candidate_memory_spurious_valid);
        m349_fc2_scalar_bank_memory_model #(.BANK_ID(bank)) memory (
            .clk_core(clk_core), .rst_core(rst_core),
            .enable(candidate_enable), .request_allow(request_allow),
            .newest_first(1'b1),
            .spurious_valid(candidate_memory_spurious_valid),
            .mem_req_valid(candidate_mem_req_valid[bank]),
            .mem_req_ready(candidate_mem_req_ready[bank]),
            .mem_req_epoch(candidate_mem_req_epoch[bank]),
            .mem_req_slot(candidate_mem_req_slot[bank]),
            .mem_req_generation(candidate_mem_req_generation[bank]),
            .mem_req_tag(candidate_mem_req_tag[bank]),
            .mem_req_output_block(candidate_mem_req_block[bank]),
            .mem_req_slice(candidate_mem_req_slice[bank]),
            .mem_req_source_channel(candidate_mem_req_channel[bank]),
            .mem_req_accept(candidate_mem_req_accept[bank]),
            .mem_rsp_valid(candidate_memory_rsp_valid_internal[bank]),
            .mem_rsp_ready(candidate_mem_rsp_ready[bank]),
            .mem_rsp_epoch(candidate_mem_rsp_epoch[bank]),
            .mem_rsp_slot(candidate_mem_rsp_slot[bank]),
            .mem_rsp_generation(candidate_mem_rsp_generation[bank]),
            .mem_rsp_tag(candidate_mem_rsp_tag[bank]),
            .mem_rsp_weight(candidate_mem_rsp_weight[bank]),
            .mem_rsp_accept(candidate_mem_rsp_accept[bank]),
            .request_count(candidate_memory_request_count_bank[bank]),
            .response_count(candidate_memory_response_count_bank[bank]),
            .pending_count(candidate_memory_pending_count_bank[bank]),
            .live_slot_reuse_error(
                candidate_memory_live_slot_reuse_error_bank[bank]));
    end

    // Retain the atomic internal-bundle monitor used by M349.  These are
    // observation-only aliases; physical traffic uses the scalar ports above.
    always_comb begin
        candidate_if.mem_req_valid = candidate.core_mem_req_valid;
        candidate_if.mem_req_ready = candidate.core_mem_req_ready;
        candidate_if.mem_req_accept = candidate.core_mem_req_accept;
        candidate_if.mem_req_epoch = candidate.core_mem_req_epoch;
        candidate_if.mem_req_slot = candidate.core_mem_req_slot;
        candidate_if.mem_req_generation = candidate.core_mem_req_generation;
        candidate_if.mem_req_tag = candidate.core_mem_req_tag;
        candidate_if.mem_req_output_block
            = candidate.core_mem_req_output_block;
        candidate_if.mem_req_slice = candidate.core_mem_req_slice;
        candidate_if.mem_req_source_count
            = candidate.core_mem_req_source_count;
        candidate_if.mem_req_bank_valid
            = candidate.core_mem_req_bank_valid;
        candidate_if.mem_rsp_valid = candidate.core_mem_rsp_valid;
        candidate_if.mem_rsp_ready = candidate.core_mem_rsp_ready;
        candidate_if.mem_rsp_accept = candidate.core_mem_rsp_accept;
        candidate_if.mem_rsp_epoch = candidate.core_mem_rsp_epoch;
        candidate_if.mem_rsp_slot = candidate.core_mem_rsp_slot;
        candidate_if.mem_rsp_generation = candidate.core_mem_rsp_generation;
        candidate_if.mem_rsp_tag = candidate.core_mem_rsp_tag;
        candidate_if.mem_rsp_bank_valid
            = candidate.core_mem_rsp_bank_valid;
        for (int bank = 0; bank < 8; bank++) begin
            candidate_if.mem_req_source_channel[bank]
                = candidate.core_mem_req_source_channel[bank];
            for (int lane = 0; lane < LANES; lane++)
                candidate_if.mem_rsp_weight[bank][lane]
                    = candidate.core_mem_rsp_weight[bank][lane];
        end

        candidate_memory_request_count = 0;
        candidate_memory_response_count = 0;
        candidate_memory_active_read_count = 0;
        candidate_memory_pending_count = 0;
        candidate_memory_live_slot_reuse_error = 0;
        for (int bank = 0; bank < 8; bank++) begin
            candidate_memory_request_count +=
                candidate_memory_request_count_bank[bank];
            candidate_memory_response_count +=
                candidate_memory_response_count_bank[bank];
            candidate_memory_active_read_count +=
                candidate_memory_request_count_bank[bank];
            candidate_memory_pending_count +=
                candidate_memory_pending_count_bank[bank];
            candidate_memory_live_slot_reuse_error |=
                candidate_memory_live_slot_reuse_error_bank[bank];
        end
    end

    m349_fc2_k1x8_raw4_acc24 baseline (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(baseline_header_valid),
        .header_ready(baseline_header_ready),
        .header_tag(baseline_header_tag),
        .header_raw_beat_count(baseline_header_raw_beat_count),
        .header_window_depth(baseline_header_window_depth),
        .header_output_blocks(baseline_header_output_blocks),
        .header_accept(baseline_header_accept),
        .raw_valid(baseline_raw_valid), .raw_ready(baseline_raw_ready),
        .raw_lane_valid(baseline_raw_lane_valid),
        .raw_beat_index(baseline_raw_beat_index),
        .raw_bitmap(baseline_raw_bitmap), .raw_last(baseline_raw_last),
        .raw_accept(baseline_raw_accept),
        .mem_req_valid(baseline_mem_req_valid),
        .mem_req_ready(baseline_mem_req_ready),
        .mem_req_epoch(baseline_mem_req_epoch),
        .mem_req_slot(baseline_mem_req_slot),
        .mem_req_generation(baseline_mem_req_generation),
        .mem_req_tag(baseline_mem_req_tag),
        .mem_req_output_block(baseline_mem_req_block),
        .mem_req_slice(baseline_mem_req_slice),
        .mem_req_source_channel(baseline_mem_req_channel),
        .mem_req_accept(baseline_mem_req_accept),
        .mem_rsp_valid(baseline_mem_rsp_valid),
        .mem_rsp_ready(baseline_mem_rsp_ready),
        .mem_rsp_epoch(baseline_mem_rsp_epoch),
        .mem_rsp_slot(baseline_mem_rsp_slot),
        .mem_rsp_generation(baseline_mem_rsp_generation),
        .mem_rsp_tag(baseline_mem_rsp_tag),
        .mem_rsp_weight(baseline_mem_rsp_weight),
        .mem_rsp_accept(baseline_mem_rsp_accept),
        .result_valid(baseline_result_valid),
        .result_ready(baseline_result_ready),
        .result_tag(baseline_result_tag),
        .result_output_block(baseline_result_block),
        .result_slice(baseline_result_slice),
        .result_accumulator(baseline_result_accumulator),
        .result_last(baseline_result_last),
        .result_accept(baseline_result_accept),
        .token_done_valid(baseline_done_valid),
        .token_done_ready(baseline_done_ready),
        .token_done_tag(baseline_done_tag),
        .token_done_had_event(baseline_done_had_event),
        .token_done_accept(baseline_done_accept),
        .protocol_error(baseline_protocol_error),
        .numeric_overflow(baseline_numeric_overflow),
        .stale_response_seen(baseline_stale_response_seen),
        .busy(baseline_busy),
        .debug_fifo_count(baseline_debug_fifo_count),
        .debug_outstanding_count(baseline_debug_outstanding_count),
        .debug_group_accept_count(baseline_debug_group_count),
        .debug_request_accept_count(baseline_debug_request_count),
        .debug_response_accept_count(baseline_debug_response_count),
        .debug_context_write_count(baseline_debug_context_count),
        .debug_result_accept_count(baseline_debug_result_count),
        .debug_active_bank_read_count(baseline_debug_active_read_count));

    for (genvar bank = 0; bank < 8; bank++) begin : g_scalar_memory
        assign baseline_mem_rsp_valid[bank]
            = baseline_memory_rsp_valid_internal[bank]
                && (response_allow || baseline_memory_spurious_valid[bank]);
        m349_fc2_scalar_bank_memory_model #(.BANK_ID(bank)) memory (
            .clk_core(clk_core), .rst_core(rst_core),
            .enable(baseline_enable), .request_allow(request_allow),
            .newest_first(1'b1),
            .spurious_valid(baseline_memory_spurious_valid[bank]),
            .mem_req_valid(baseline_mem_req_valid[bank]),
            .mem_req_ready(baseline_mem_req_ready[bank]),
            .mem_req_epoch(baseline_mem_req_epoch[bank]),
            .mem_req_slot(baseline_mem_req_slot[bank]),
            .mem_req_generation(baseline_mem_req_generation[bank]),
            .mem_req_tag(baseline_mem_req_tag[bank]),
            .mem_req_output_block(baseline_mem_req_block[bank]),
            .mem_req_slice(baseline_mem_req_slice[bank]),
            .mem_req_source_channel(baseline_mem_req_channel[bank]),
            .mem_req_accept(baseline_mem_req_accept[bank]),
            .mem_rsp_valid(baseline_memory_rsp_valid_internal[bank]),
            .mem_rsp_ready(baseline_mem_rsp_ready[bank]),
            .mem_rsp_epoch(baseline_mem_rsp_epoch[bank]),
            .mem_rsp_slot(baseline_mem_rsp_slot[bank]),
            .mem_rsp_generation(baseline_mem_rsp_generation[bank]),
            .mem_rsp_tag(baseline_mem_rsp_tag[bank]),
            .mem_rsp_weight(baseline_mem_rsp_weight[bank]),
            .mem_rsp_accept(baseline_mem_rsp_accept[bank]),
            .request_count(baseline_memory_request_count[bank]),
            .response_count(baseline_memory_response_count[bank]),
            .pending_count(baseline_memory_pending_count[bank]),
            .live_slot_reuse_error(
                baseline_memory_live_slot_reuse_error[bank]));
    end

    integer edge_ordinal_q, start_edge_q, measured_cycles_q;
    integer expected_arch, expected_blocks, result_count, done_count;
    integer error_count, numeric_mismatch_count, tuple_mismatch_count;
    integer weight_mismatch_count, clean_case_count, reset_case_count;
    integer protocol_attack_count, request_stall_count, result_stall_count;
    integer raw_stall_count, single1_request_count, k1x8_full_issue_count;
    integer candidate_ooo_count, baseline_ooo_count;
    integer candidate_cycles [0:3], baseline_cycles [0:3];
    integer candidate_result_store [0:7][0:5][0:LANES-1];
    integer signed reference_accum [0:7][0:5][0:LANES-1];
    integer candidate_request_tuple [0:7][0:5][0:MAX_CHANNELS-1];
    integer baseline_request_tuple [0:7][0:5][0:MAX_CHANNELS-1];
    integer candidate_response_tuple [0:7][0:5][0:MAX_CHANNELS-1];
    integer baseline_response_tuple [0:7][0:5][0:MAX_CHANNELS-1];
    logic [95:0] payload [0:31];
    logic [TAG_BITS-1:0] expected_tag;
    logic check_results, scoreboard_enable, done_seen_q, start_seen_q;

    logic candidate_slot_valid [0:7];
    logic [GENERATION_BITS-1:0] candidate_slot_generation [0:7];
    logic [2:0] candidate_slot_block [0:7], candidate_slot_slice [0:7];
    logic [7:0] candidate_slot_mask [0:7];
    logic [CHANNEL_BITS-1:0] candidate_slot_channel [0:7][0:7];
    logic baseline_slot_valid [0:7][0:7];
    logic [GENERATION_BITS-1:0] baseline_slot_generation [0:7][0:7];
    logic [2:0] baseline_slot_block [0:7][0:7];
    logic [2:0] baseline_slot_slice [0:7][0:7];
    logic [CHANNEL_BITS-1:0] baseline_slot_channel [0:7][0:7];

    function automatic integer block_index(input integer blocks);
        case (blocks)
            1: return 0;
            2: return 1;
            4: return 2;
            default: return 3;
        endcase
    endfunction

    function automatic integer raw_count(input integer blocks);
        case (blocks)
            1: return 4;
            2: return 8;
            4: return 16;
            default: return 32;
        endcase
    endfunction

    function automatic integer window_depth(input integer blocks);
        case (blocks)
            1: return 2;
            2: return 4;
            default: return 8;
        endcase
    endfunction

    function automatic integer signed weight_value(
        input integer bank, input integer lane, input integer channel,
        input integer block, input integer slice);
        integer value;
        begin
            value = (channel*3 + bank*5 + block*7
                + slice*11 + lane*13) % 31;
            return value - 15;
        end
    endfunction

    task automatic clear_case_evidence;
        begin
            for (integer block=0; block<8; block++) begin
                for (integer slice=0; slice<6; slice++) begin
                    for (integer channel=0; channel<MAX_CHANNELS; channel++) begin
                        candidate_request_tuple[block][slice][channel]=0;
                        baseline_request_tuple[block][slice][channel]=0;
                        candidate_response_tuple[block][slice][channel]=0;
                        baseline_response_tuple[block][slice][channel]=0;
                    end
                    for (integer lane=0; lane<LANES; lane++)
                        candidate_result_store[block][slice][lane]=0;
                end
            end
        end
    endtask

    task automatic build_payload_and_reference(
        input integer blocks, input integer mode, output integer events);
        integer beats, row, bank, channel;
        begin
            beats=raw_count(blocks); events=0;
            for (integer beat=0;beat<32;beat++) payload[beat]=0;
            for (integer block=0;block<8;block++)
                for (integer slice=0;slice<6;slice++)
                    for (integer lane=0;lane<LANES;lane++)
                        reference_accum[block][slice][lane]=0;
            if (mode != 9) begin
                for (bank=0;bank<8;bank++)
                    payload[0][(bank%12)*8+bank]=1;
                for (integer beat=0;beat<beats;beat++) begin
                    for (integer item=0;item<3+(mode%3);item++) begin
                        row=(beat*3+item*5+mode)%12;
                        bank=(beat+item*3+mode)%8;
                        payload[beat][row*8+bank]=1;
                    end
                    if ((beat+mode)%4==0) begin
                        row=(beat+7)%12; bank=(beat*5+2)%8;
                        payload[beat][row*8+bank]=1;
                    end
                end
            end
            for (integer beat=0;beat<beats;beat++) begin
                for (row=0;row<12;row++) begin
                    for (bank=0;bank<8;bank++) begin
                        if (payload[beat][row*8+bank]) begin
                            events++;
                            channel=(beat*12+row)*8+bank;
                            for (integer block=0;block<blocks;block++)
                                for (integer slice=0;slice<6;slice++)
                                    for (integer lane=0;lane<LANES;lane++)
                                        reference_accum[block][slice][lane]
                                            += weight_value(bank,lane,channel,
                                                block,slice);
                        end
                    end
                end
            end
        end
    endtask

    task automatic initialize_inputs;
        begin
            candidate_if.header_valid=0; candidate_if.raw_valid=0;
            candidate_if.raw_last=0; candidate_if.raw_lane_valid=0;
            baseline_header_valid=0; baseline_raw_valid=0;
            baseline_raw_last=0; baseline_raw_lane_valid=0;
            candidate_enable=0; baseline_enable=0;
            candidate_memory_spurious_valid=0;
            baseline_memory_spurious_valid=0;
            for(integer lane=0;lane<4;lane++)begin
                candidate_if.raw_beat_index[lane]=0;
                candidate_if.raw_bitmap[lane]=0;
                baseline_raw_beat_index[lane]=0;
                baseline_raw_bitmap[lane]=0;
            end
        end
    endtask

    task automatic reset_arch(input integer arch);
        begin
            @(negedge clk_core); rst_core=1; initialize_inputs();
            check_results=0; scoreboard_enable=0;
            repeat(4) @(negedge clk_core);
            if (arch==8) candidate_enable=1;
            else baseline_enable=1;
            rst_core=0;
            repeat(2) @(posedge clk_core);
        end
    endtask

    task automatic drive_header(input integer arch,
        input logic[TAG_BITS-1:0] tag, input integer blocks);
        integer guard;
        begin
            guard=0;
            @(negedge clk_core);
            if (arch==8) begin
                candidate_if.header_tag=tag;
                candidate_if.header_output_blocks=blocks;
                candidate_if.header_raw_beat_count=raw_count(blocks);
                candidate_if.header_window_depth=window_depth(blocks);
                candidate_if.header_valid=1;
                while(!candidate_if.header_accept&&guard<1000)begin
                    @(posedge clk_core);guard++;
                end
                if(!candidate_if.header_accept)
                    $fatal(1,"M497 candidate header watchdog");
                @(negedge clk_core); candidate_if.header_valid=0;
            end else begin
                baseline_header_tag=tag;
                baseline_header_output_blocks=blocks;
                baseline_header_raw_beat_count=raw_count(blocks);
                baseline_header_window_depth=window_depth(blocks);
                baseline_header_valid=1;
                while(!baseline_header_accept&&guard<1000)begin
                    @(posedge clk_core);guard++;
                end
                if(!baseline_header_accept)
                    $fatal(1,"M497 baseline header watchdog");
                @(negedge clk_core); baseline_header_valid=0;
            end
        end
    endtask

    task automatic drive_raw(input integer arch,input integer blocks,
        input integer packet_limit,input logic terminate_last);
        integer beats,packets,base,guard;
        begin
            beats=raw_count(blocks); packets=beats/4;
            if(packet_limit<packets) packets=packet_limit;
            for(integer packet=0;packet<packets;packet++)begin
                base=packet*4; @(negedge clk_core);
                if(arch==8)begin
                    candidate_if.raw_lane_valid=4'b1111;
                    for(integer lane=0;lane<4;lane++)begin
                        candidate_if.raw_beat_index[lane]=base+lane;
                        candidate_if.raw_bitmap[lane]=payload[base+lane];
                    end
                    candidate_if.raw_last=terminate_last
                        &&(packet+1==beats/4);
                    candidate_if.raw_valid=1;
                    guard=0;
                    while(!candidate_if.raw_accept&&guard<2000*blocks)begin
                        @(posedge clk_core);guard++;
                        if(guard%250==0)$display(
                            "M497 debug candidate raw wait guard=%0d B=%0d busy=%0b protocol=%0b req=%0d rsp=%0d",
                            guard,blocks,candidate_if.busy,
                            candidate_if.protocol_error,
                            candidate_if.debug_request_accept_count,
                            candidate_if.debug_response_accept_count);
                    end
                    if(!candidate_if.raw_accept)
                        $fatal(1,"M497 candidate raw watchdog packet=%0d B=%0d",
                            packet,blocks);
                end else begin
                    baseline_raw_lane_valid=4'b1111;
                    for(integer lane=0;lane<4;lane++)begin
                        baseline_raw_beat_index[lane]=base+lane;
                        baseline_raw_bitmap[lane]=payload[base+lane];
                    end
                    baseline_raw_last=terminate_last
                        &&(packet+1==beats/4);
                    baseline_raw_valid=1;
                    guard=0;
                    while(!baseline_raw_accept&&guard<2000*blocks)begin
                        @(posedge clk_core);guard++;
                        if(guard%250==0)$display(
                            "M497 debug baseline raw wait guard=%0d B=%0d busy=%0b protocol=%0b req=%0d rsp=%0d",
                            guard,blocks,baseline_busy,
                            baseline_protocol_error,
                            baseline_debug_request_count,
                            baseline_debug_response_count);
                    end
                    if(!baseline_raw_accept)
                        $fatal(1,"M497 baseline raw watchdog packet=%0d B=%0d",
                            packet,blocks);
                end
            end
            @(negedge clk_core);
            if(arch==8)begin
                candidate_if.raw_valid=0;candidate_if.raw_last=0;
                candidate_if.raw_lane_valid=0;
            end else begin
                baseline_raw_valid=0;baseline_raw_last=0;
                baseline_raw_lane_valid=0;
            end
        end
    endtask

    task automatic run_one(input integer arch,input integer blocks,
        input logic[TAG_BITS-1:0] tag,input integer events,
        output integer cycles);
        integer watchdog, expected_results, expected_reads;
        begin
            expected_arch=arch; expected_blocks=blocks; expected_tag=tag;
            reset_arch(arch); check_results=1; scoreboard_enable=1;
            fork
                begin drive_header(arch,tag,blocks);
                    drive_raw(arch,blocks,32,1); end
                begin
                    watchdog=0;
                    while(!done_seen_q && watchdog<2000*blocks) begin
                        @(negedge clk_core); watchdog++;
                        if(watchdog%250==0)$display(
                            "M497 debug done wait arch=%0d B=%0d guard=%0d candidate_busy=%0b baseline_busy=%0b",
                            arch,blocks,watchdog,candidate_if.busy,
                            baseline_busy);
                    end
                    if(!done_seen_q)begin
                        $error("M349 watchdog arch=%0d B=%0d",arch,blocks);
                        error_count++;
                    end
                end
            join
            @(negedge clk_core);
            check_results=0; scoreboard_enable=0;
            cycles=measured_cycles_q;
            expected_results=blocks*6; expected_reads=events*blocks*6;
            if(result_count!=expected_results)begin
                $error("M349 result count arch=%0d B=%0d got=%0d exp=%0d",
                    arch,blocks,result_count,expected_results);error_count++;
            end
            if(arch==8)begin
                if(candidate_if.protocol_error||candidate_if.numeric_overflow
                        ||candidate_memory_live_slot_reuse_error)begin
                    $error("M497 canonical K1 fault B=%0d",blocks);error_count++;
                end
                if(candidate_if.debug_response_accept_count
                            !=candidate_if.debug_request_accept_count
                        ||candidate_if.debug_context_write_count
                            !=candidate_if.debug_request_accept_count
                        ||candidate_if.debug_result_accept_count
                            !=expected_results
                        ||candidate_if.debug_active_bank_read_count
                            !=expected_reads
                        ||candidate_memory_request_count!=expected_reads
                        ||candidate_memory_response_count!=expected_reads
                        ||candidate_memory_active_read_count!=expected_reads
                        ||candidate_memory_pending_count!=0)begin
                    $error("M497 canonical K1 conservation B=%0d bundle_req=%0d bundle_rsp=%0d bank_req=%0d bank_rsp=%0d read=%0d exp=%0d",
                        blocks,candidate_if.debug_request_accept_count,
                        candidate_if.debug_response_accept_count,
                        candidate_memory_request_count,
                        candidate_memory_response_count,
                        candidate_if.debug_active_bank_read_count,
                        expected_reads);
                    error_count++;
                end
            end else begin
                integer mem_requests,mem_responses,mem_pending;
                mem_requests=0;mem_responses=0;mem_pending=0;
                for(integer bank=0;bank<8;bank++)begin
                    mem_requests+=baseline_memory_request_count[bank];
                    mem_responses+=baseline_memory_response_count[bank];
                    mem_pending+=baseline_memory_pending_count[bank];
                end
                if(baseline_protocol_error||baseline_numeric_overflow
                        ||(|baseline_memory_live_slot_reuse_error))begin
                    $error("M349 K1x8 fault B=%0d",blocks);error_count++;
                end
                if(baseline_debug_group_count!=events*blocks
                        ||baseline_debug_request_count
                            !=baseline_debug_group_count*6
                        ||baseline_debug_response_count
                            !=baseline_debug_request_count
                        ||baseline_debug_context_count
                            !=baseline_debug_request_count
                        ||baseline_debug_result_count!=expected_results*8
                        ||baseline_debug_active_read_count!=expected_reads
                        ||mem_requests!=baseline_debug_request_count
                        ||mem_responses!=baseline_debug_response_count
                        ||mem_pending!=0)begin
                    $error("M349 K1x8 conservation B=%0d group=%0d req=%0d rsp=%0d read=%0d exp=%0d",
                        blocks,baseline_debug_group_count,
                        baseline_debug_request_count,
                        baseline_debug_response_count,
                        baseline_debug_active_read_count,expected_reads);
                    error_count++;
                end
            end
        end
    endtask

    task automatic compare_case_multisets(input integer blocks);
        begin
            for(integer block=0;block<8;block++)begin
                for(integer slice=0;slice<6;slice++)begin
                    for(integer channel=0;channel<MAX_CHANNELS;channel++)begin
                        if(candidate_request_tuple[block][slice][channel]
                                !=baseline_request_tuple[block][slice][channel]
                                ||candidate_response_tuple[block][slice][channel]
                                !=baseline_response_tuple[block][slice][channel]
                                ||candidate_request_tuple[block][slice][channel]
                                !=candidate_response_tuple[block][slice][channel]
                                ||baseline_request_tuple[block][slice][channel]
                                !=baseline_response_tuple[block][slice][channel])begin
                            if(tuple_mismatch_count<8)
                                $error("M349 tuple mismatch B=%0d block=%0d slice=%0d channel=%0d cq=%0d bq=%0d cr=%0d br=%0d",
                                    blocks,block,slice,channel,
                                    candidate_request_tuple[block][slice][channel],
                                    baseline_request_tuple[block][slice][channel],
                                    candidate_response_tuple[block][slice][channel],
                                    baseline_response_tuple[block][slice][channel]);
                            tuple_mismatch_count++;
                        end
                    end
                    for(integer lane=0;lane<LANES;lane++)begin
                        if(block<blocks
                                &&candidate_result_store[block][slice][lane]
                                    !==reference_accum[block][slice][lane])begin
                            $error("M349 stored candidate result mismatch");
                            error_count++;
                        end
                    end
                end
            end
        end
    endtask

    task automatic run_clean_pair(input integer blocks,input integer mode);
        integer events,cand_cycles,base_cycles,index;
        logic[TAG_BITS-1:0]cand_tag,base_tag;
        begin
            build_payload_and_reference(blocks,mode,events);
            clear_case_evidence();
            cand_tag=24'h349800|(blocks<<4)|mode;
            base_tag=24'h349100|(blocks<<4)|mode;
            $display("M497 progress clean B=%0d candidate start",blocks);
            run_one(8,blocks,cand_tag,events,cand_cycles);
            $display("M497 progress clean B=%0d candidate done cycles=%0d",
                blocks,cand_cycles);
            run_one(1,blocks,base_tag,events,base_cycles);
            $display("M497 progress clean B=%0d baseline done cycles=%0d",
                blocks,base_cycles);
            compare_case_multisets(blocks);
            index=block_index(blocks);
            candidate_cycles[index]=cand_cycles;
            baseline_cycles[index]=base_cycles;
            clean_case_count+=2;
            $display("M497 canonical K1 versus K1x8 B=%0d events=%0d k1_cycles=%0d k1x8_cycles=%0d k1x8_speedup_vs_k1=%0f tuple_mismatches=%0d weight_mismatches=%0d",
                blocks,events,cand_cycles,base_cycles,
                cand_cycles*1.0/base_cycles,tuple_mismatch_count,
                weight_mismatch_count);
        end
    endtask

    task automatic run_reset_attack(input integer arch);
        integer events;
        logic[TAG_BITS-1:0]tag;
        begin
            // The first nonterminal packet of a B2 token is sufficient to
            // leave accepted work and memory traffic in flight before POR.
            // The inherited M492 B8
            // dense packet was harmless for K8 but made canonical K1 spend
            // thousands of irrelevant serialized source cycles before the
            // reset edge; that was test cost, not stronger reset coverage.
            build_payload_and_reference(2,0,events);tag=24'h349a00|arch;
            reset_arch(arch);drive_header(arch,tag,2);drive_raw(arch,2,1,0);
            @(negedge clk_core);rst_core=1;repeat(3)@(negedge clk_core);
            if(arch==8)begin
                if(candidate_if.busy||candidate_if.protocol_error
                        ||candidate_if.result_valid
                        ||candidate_if.token_done_valid
                        ||candidate_memory_pending_count!=0)begin
                    $error("M497 canonical K1 POR failed");error_count++;
                end
            end else begin
                integer pending;pending=0;
                for(integer bank=0;bank<8;bank++)
                    pending+=baseline_memory_pending_count[bank];
                if(baseline_busy||baseline_protocol_error
                        ||baseline_result_valid||baseline_done_valid
                        ||pending!=0)begin
                    $error("M349 K1x8 POR failed");error_count++;
                end
            end
            rst_core=0;repeat(2)@(posedge clk_core);reset_case_count++;
        end
    endtask

    task automatic run_header_attack(input integer arch);
        begin
            reset_arch(arch);@(negedge clk_core);
            if(arch==8)begin
                candidate_if.header_tag=24'h349bad;
                candidate_if.header_output_blocks=3;
                candidate_if.header_raw_beat_count=7;
                candidate_if.header_window_depth=3;
                candidate_if.header_valid=1;@(posedge clk_core);
                @(negedge clk_core);candidate_if.header_valid=0;
                @(posedge clk_core);
                if(!candidate_if.protocol_error)begin
                    $error("M497 canonical K1 illegal header escaped");error_count++;
                end
            end else begin
                baseline_header_tag=24'h349bad;
                baseline_header_output_blocks=3;
                baseline_header_raw_beat_count=7;
                baseline_header_window_depth=3;
                baseline_header_valid=1;@(posedge clk_core);
                @(negedge clk_core);baseline_header_valid=0;
                @(posedge clk_core);
                if(!baseline_protocol_error)begin
                    $error("M349 K1x8 illegal header escaped");error_count++;
                end
            end
            repeat(2)@(posedge clk_core);protocol_attack_count++;
        end
    endtask

    task automatic run_response_attack(input integer arch);
        logic[TAG_BITS-1:0]tag;
        begin
            tag=24'h349b00|arch;reset_arch(arch);drive_header(arch,tag,1);
            @(negedge clk_core);
            if(arch==8)candidate_memory_spurious_valid=1;
            else baseline_memory_spurious_valid[0]=1;
            @(posedge clk_core);@(negedge clk_core);
            if(arch==8)candidate_memory_spurious_valid=0;
            else baseline_memory_spurious_valid[0]=0;
            @(posedge clk_core);
            if(arch==8&&!candidate_if.protocol_error)begin
                $error("M497 canonical K1 spurious response escaped");error_count++;
            end
            if(arch==1&&!baseline_protocol_error)begin
                $error("M349 K1x8 spurious response escaped");error_count++;
            end
            repeat(2)@(posedge clk_core);protocol_attack_count++;
        end
    endtask

    always_comb begin
        request_allow=!rst_core&&(edge_ordinal_q%7!=2);
        // Twelve visible response cycles per 17-cycle epoch. The identical gate
        // is applied at both service boundaries, independently of DUT state.
        response_allow=!rst_core&&(edge_ordinal_q%17>=5);
        candidate_if.result_ready=!rst_core&&(edge_ordinal_q%5!=2);
        baseline_result_ready=!rst_core&&(edge_ordinal_q%5!=2);
        candidate_if.token_done_ready=!rst_core&&(edge_ordinal_q%4!=1);
        baseline_done_ready=!rst_core&&(edge_ordinal_q%4!=1);
    end

    // One clocked monitor owns both cycle endpoints, eliminating the M342 TB
    // active-region race between drive_header and cycle_count.
    always @(posedge clk_core) begin : monitor
        integer signed observed;
        integer slot,channel,block,slice;
        logic lower_pending;
        if(rst_core)begin
            edge_ordinal_q=0;start_edge_q=0;measured_cycles_q=0;
            done_seen_q=0;start_seen_q=0;result_count=0;done_count=0;
            for(integer s=0;s<8;s++)begin
                candidate_slot_valid[s]=0;
                for(integer bank=0;bank<8;bank++)
                    baseline_slot_valid[bank][s]=0;
            end
        end else begin
            edge_ordinal_q++;
            if((expected_arch==8&&candidate_if.header_accept)
                    ||(expected_arch==1&&baseline_header_accept))begin
                start_edge_q=edge_ordinal_q;start_seen_q=1;
            end
            if(check_results&&expected_arch==8&&candidate_if.result_accept)begin
                if(candidate_if.result_tag!==expected_tag
                        ||candidate_if.result_output_block>=expected_blocks
                        ||candidate_if.result_slice>=6)begin
                    $error("M497 canonical K1 result identity mismatch");error_count++;
                end
                for(integer lane=0;lane<LANES;lane++)begin
                    observed=$signed(candidate_if.result_accumulator[lane]);
                    if(observed!==reference_accum[candidate_if.result_output_block]
                            [candidate_if.result_slice][lane])begin
                        $error("M497 canonical K1 numeric mismatch");error_count++;
                        numeric_mismatch_count++;
                    end
                    candidate_result_store[candidate_if.result_output_block]
                        [candidate_if.result_slice][lane]=observed;
                end
                result_count++;
            end
            if(check_results&&expected_arch==1&&baseline_result_accept)begin
                if(baseline_result_tag!==expected_tag
                        ||baseline_result_block>=expected_blocks
                        ||baseline_result_slice>=6)begin
                    $error("M349 K1x8 result identity mismatch");error_count++;
                end
                for(integer lane=0;lane<LANES;lane++)begin
                    observed=$signed(baseline_result_accumulator[lane]);
                    if(observed!==reference_accum[baseline_result_block]
                            [baseline_result_slice][lane]
                            ||observed!==candidate_result_store
                                [baseline_result_block][baseline_result_slice]
                                [lane])begin
                        $error("M349 K1x8 numeric/pair mismatch");error_count++;
                        numeric_mismatch_count++;
                    end
                end
                result_count++;
            end
            if(check_results&&expected_arch==8&&candidate_if.token_done_accept)begin
                if(!start_seen_q||candidate_if.token_done_tag!==expected_tag)begin
                    $error("M497 canonical K1 done mismatch");error_count++;
                end
                measured_cycles_q=edge_ordinal_q-start_edge_q+1;
                done_seen_q=1;done_count++;
            end
            if(check_results&&expected_arch==1&&baseline_done_accept)begin
                if(!start_seen_q||baseline_done_tag!==expected_tag)begin
                    $error("M349 K1x8 done mismatch");error_count++;
                end
                measured_cycles_q=edge_ordinal_q-start_edge_q+1;
                done_seen_q=1;done_count++;
            end

            if(candidate_if.mem_req_valid&&!candidate_if.mem_req_ready)
                request_stall_count++;
            if(|(baseline_mem_req_valid&~baseline_mem_req_ready))
                request_stall_count++;
            if(candidate_if.result_valid&&!candidate_if.result_ready)
                result_stall_count++;
            if(baseline_result_valid&&!baseline_result_ready)
                result_stall_count++;
            if(candidate_if.raw_valid&&!candidate_if.raw_ready)
                raw_stall_count++;
            if(baseline_raw_valid&&!baseline_raw_ready)
                raw_stall_count++;
            if(candidate_if.mem_req_accept
                    &&candidate_if.mem_req_source_count==1)
                single1_request_count++;
            if($countones(baseline_mem_req_accept)==8)
                k1x8_full_issue_count++;

            if(scoreboard_enable)begin
                // Response is retired before a same-edge slot replacement.
                if(candidate_if.mem_rsp_accept)begin
                    slot=candidate_if.mem_rsp_slot;
                    if(!candidate_slot_valid[slot]
                            ||candidate_slot_generation[slot]
                                !=candidate_if.mem_rsp_generation
                            ||candidate_slot_mask[slot]
                                !=candidate_if.mem_rsp_bank_valid)begin
                        $error("M349 candidate response scoreboard mismatch");
                        error_count++;
                    end else begin
                        lower_pending=0;
                        for(integer other=0;other<8;other++)
                            if(other!=slot&&candidate_slot_valid[other]
                                    &&candidate_slot_generation[other]
                                        <candidate_if.mem_rsp_generation)
                                lower_pending=1;
                        if(lower_pending)candidate_ooo_count++;
                        block=candidate_slot_block[slot];
                        slice=candidate_slot_slice[slot];
                        for(integer bank=0;bank<8;bank++)begin
                            if(candidate_slot_mask[slot][bank])begin
                                channel=candidate_slot_channel[slot][bank];
                                candidate_response_tuple[block][slice][channel]++;
                                for(integer lane=0;lane<LANES;lane++)begin
                                    if($signed(candidate_if.mem_rsp_weight[bank][lane])
                                            !==weight_value(bank,lane,channel,
                                                block,slice))begin
                                        weight_mismatch_count++;error_count++;
                                    end
                                end
                            end
                        end
                    end
                    candidate_slot_valid[slot]=0;
                end
                for(integer bank=0;bank<8;bank++)begin
                    if(baseline_mem_rsp_accept[bank])begin
                        slot=baseline_mem_rsp_slot[bank];
                        if(!baseline_slot_valid[bank][slot]
                                ||baseline_slot_generation[bank][slot]
                                    !=baseline_mem_rsp_generation[bank])begin
                            $error("M349 baseline response scoreboard mismatch");
                            error_count++;
                        end else begin
                            lower_pending=0;
                            for(integer other=0;other<8;other++)
                                if(other!=slot&&baseline_slot_valid[bank][other]
                                        &&baseline_slot_generation[bank][other]
                                            <baseline_mem_rsp_generation[bank])
                                    lower_pending=1;
                            if(lower_pending)baseline_ooo_count++;
                            block=baseline_slot_block[bank][slot];
                            slice=baseline_slot_slice[bank][slot];
                            channel=baseline_slot_channel[bank][slot];
                            baseline_response_tuple[block][slice][channel]++;
                            for(integer lane=0;lane<LANES;lane++)begin
                                if($signed(baseline_mem_rsp_weight[bank][lane])
                                        !==weight_value(bank,lane,channel,
                                            block,slice))begin
                                    weight_mismatch_count++;error_count++;
                                end
                            end
                        end
                        baseline_slot_valid[bank][slot]=0;
                    end
                end
                if(candidate_if.mem_req_accept)begin
                    slot=candidate_if.mem_req_slot;
                    if(candidate_slot_valid[slot])begin
                        $error("M349 candidate live slot reuse scoreboard");
                        error_count++;
                    end
                    candidate_slot_valid[slot]=1;
                    candidate_slot_generation[slot]
                        =candidate_if.mem_req_generation;
                    candidate_slot_block[slot]
                        =candidate_if.mem_req_output_block;
                    candidate_slot_slice[slot]=candidate_if.mem_req_slice;
                    candidate_slot_mask[slot]=candidate_if.mem_req_bank_valid;
                    for(integer bank=0;bank<8;bank++)begin
                        candidate_slot_channel[slot][bank]
                            =candidate_if.mem_req_source_channel[bank];
                        if(candidate_if.mem_req_bank_valid[bank])begin
                            channel=candidate_if.mem_req_source_channel[bank];
                            candidate_request_tuple
                                [candidate_if.mem_req_output_block]
                                [candidate_if.mem_req_slice][channel]++;
                        end
                    end
                end
                for(integer bank=0;bank<8;bank++)begin
                    if(baseline_mem_req_accept[bank])begin
                        slot=baseline_mem_req_slot[bank];
                        if(baseline_slot_valid[bank][slot])begin
                            $error("M349 baseline live slot reuse scoreboard");
                            error_count++;
                        end
                        baseline_slot_valid[bank][slot]=1;
                        baseline_slot_generation[bank][slot]
                            =baseline_mem_req_generation[bank];
                        baseline_slot_block[bank][slot]
                            =baseline_mem_req_block[bank];
                        baseline_slot_slice[bank][slot]
                            =baseline_mem_req_slice[bank];
                        baseline_slot_channel[bank][slot]
                            =baseline_mem_req_channel[bank];
                        channel=baseline_mem_req_channel[bank];
                        baseline_request_tuple[baseline_mem_req_block[bank]]
                            [baseline_mem_req_slice[bank]][channel]++;
                    end
                end
            end
        end
    end

    initial begin
        rst_core=1;error_count=0;numeric_mismatch_count=0;
        tuple_mismatch_count=0;weight_mismatch_count=0;
        clean_case_count=0;reset_case_count=0;protocol_attack_count=0;
        request_stall_count=0;result_stall_count=0;raw_stall_count=0;
        single1_request_count=0;k1x8_full_issue_count=0;
        candidate_ooo_count=0;baseline_ooo_count=0;
        expected_arch=0;expected_blocks=0;expected_tag=0;
        check_results=0;scoreboard_enable=0;initialize_inputs();
        repeat(4)@(negedge clk_core);

        run_reset_attack(8);$display("M497 progress reset candidate done");
        run_reset_attack(1);$display("M497 progress reset baseline done");
        run_header_attack(8);run_header_attack(1);
        $display("M497 progress header attacks done");
        run_response_attack(8);run_response_attack(1);
        $display("M497 progress response attacks done");

        run_clean_pair(1,0);run_clean_pair(2,1);
        run_clean_pair(4,2);run_clean_pair(8,3);
        run_clean_pair(1,9);

        if(clean_case_count!=10||reset_case_count!=2
                ||protocol_attack_count!=4||numeric_mismatch_count!=0
                ||tuple_mismatch_count!=0||weight_mismatch_count!=0
                ||request_stall_count==0||result_stall_count==0
                ||raw_stall_count==0||single1_request_count==0
                ||k1x8_full_issue_count==0||candidate_ooo_count==0
                ||baseline_ooo_count==0)begin
            $error("M497 coverage clean=%0d reset=%0d attacks=%0d numeric=%0d tuple=%0d weight=%0d reqstall=%0d resultstall=%0d rawstall=%0d single1=%0d k1x8issue=%0d candOOO=%0d baseOOO=%0d",
                clean_case_count,reset_case_count,protocol_attack_count,
                numeric_mismatch_count,tuple_mismatch_count,
                weight_mismatch_count,request_stall_count,
                result_stall_count,raw_stall_count,single1_request_count,
                k1x8_full_issue_count,candidate_ooo_count,
                baseline_ooo_count);error_count++;
        end
        if(error_count==0)$display("PASS M497 canonical-K1 versus K1x8 FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 service_sva_bound=true adapter_sva_bound=true racefree_cycle_monitor=true request_stalls=%0d result_stalls=%0d raw_stalls=%0d single1_requests=%0d k1x8_full_issue=%0d candidate_younger_before_older=%0d baseline_younger_before_older=%0d",
            request_stall_count,result_stall_count,raw_stall_count,
            single1_request_count,k1x8_full_issue_count,
            candidate_ooo_count,baseline_ooo_count);
        else $fatal(1,"M497 failures=%0d numeric=%0d tuple=%0d weight=%0d",
            error_count,numeric_mismatch_count,tuple_mismatch_count,
            weight_mismatch_count);
        $finish;
    end
endmodule

`undef M497_CANDIDATE_PORTS
`default_nettype wire
