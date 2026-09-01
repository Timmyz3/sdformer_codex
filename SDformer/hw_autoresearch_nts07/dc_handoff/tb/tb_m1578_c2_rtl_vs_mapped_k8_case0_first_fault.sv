`timescale 1ns/1ps
`default_nettype none

// Additive, diagnostic-only dual-DUT source. The frozen RTL K8 shell and the
// frozen mapped ARCH_MODE1 shell receive the same hard-wired M979 case-0
// stimulus. Each DUT owns an independent instance of the same reset-safe test
// memory so neither endpoint can perturb the other DUT's request schedule.
module m1578_case0_memory_fabric (
    input  logic          clk_core,
    input  logic          rst_core,
    input  logic          request_allow,
    input  logic          response_allow,
    input  logic [7:0]    mem_req_valid,
    output logic [7:0]    mem_req_ready,
    input  logic [127:0]  mem_req_epoch,
    input  logic [23:0]   mem_req_slot,
    input  logic [255:0]  mem_req_generation,
    input  logic [191:0]  mem_req_tag,
    input  logic [23:0]   mem_req_output_block,
    input  logic [23:0]   mem_req_slice,
    input  logic [95:0]   mem_req_source_channel,
    input  logic [7:0]    mem_req_accept,
    output logic [7:0]    mem_rsp_valid,
    input  logic [7:0]    mem_rsp_ready,
    output logic [127:0]  mem_rsp_epoch,
    output logic [23:0]   mem_rsp_slot,
    output logic [255:0]  mem_rsp_generation,
    output logic [191:0]  mem_rsp_tag,
    output logic [1023:0] mem_rsp_weight,
    input  logic [7:0]    mem_rsp_accept,
    output logic [7:0]    endpoint_fault
);
    localparam int LANES = 16;
    logic [7:0] bank_rsp_valid;
    logic signed [7:0] bank_rsp_weight [0:7][0:LANES-1];
    logic [31:0] unused_request_count [0:7];
    logic [31:0] unused_response_count [0:7];
    logic [3:0] unused_pending_count [0:7];
    logic unused_reuse_error [0:7];

    for (genvar bank = 0; bank < 8; bank++) begin : g_memory
        m349_fc2_scalar_bank_memory_model #(.BANK_ID(bank)) memory (
            .clk_core(clk_core), .rst_core(rst_core), .enable(1'b1),
            .request_allow(request_allow), .newest_first(1'b1),
            .spurious_valid(1'b0), .mem_req_valid(mem_req_valid[bank]),
            .mem_req_ready(mem_req_ready[bank]),
            .mem_req_epoch(mem_req_epoch[127-bank*16-:16]),
            .mem_req_slot(mem_req_slot[23-bank*3-:3]),
            .mem_req_generation(mem_req_generation[255-bank*32-:32]),
            .mem_req_tag(mem_req_tag[191-bank*24-:24]),
            .mem_req_output_block(mem_req_output_block[23-bank*3-:3]),
            .mem_req_slice(mem_req_slice[23-bank*3-:3]),
            .mem_req_source_channel(mem_req_source_channel[95-bank*12-:12]),
            .mem_req_accept(mem_req_accept[bank]),
            .mem_rsp_valid(bank_rsp_valid[bank]),
            .mem_rsp_ready(mem_rsp_ready[bank]),
            .mem_rsp_epoch(mem_rsp_epoch[127-bank*16-:16]),
            .mem_rsp_slot(mem_rsp_slot[23-bank*3-:3]),
            .mem_rsp_generation(mem_rsp_generation[255-bank*32-:32]),
            .mem_rsp_tag(mem_rsp_tag[191-bank*24-:24]),
            .mem_rsp_weight(bank_rsp_weight[bank]),
            .mem_rsp_accept(mem_rsp_accept[bank]),
            .request_count(unused_request_count[bank]),
            .response_count(unused_response_count[bank]),
            .pending_count(unused_pending_count[bank]),
            .live_slot_reuse_error(unused_reuse_error[bank]));

        assign mem_rsp_valid[bank] = bank_rsp_valid[bank] && response_allow;
        assign endpoint_fault[bank] = memory.endpoint_protocol_fault_q;
        for (genvar lane = 0; lane < LANES; lane++) begin : g_flatten
            always_comb mem_rsp_weight[1023-(bank*LANES+lane)*8-:8]
                = bank_rsp_weight[bank][lane];
        end
    end
endmodule


module tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault;
    localparam int LANES = 16;

    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    // One shared input stimulus; the two DUTs expose independent ready/accept.
    logic header_valid;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_last;
    logic [3:0] raw_lane_valid;
    logic [19:0] raw_beat_index;
    logic [383:0] raw_bitmap;
    logic [4:0] rtl_raw_beat_index [0:3];
    logic [95:0] rtl_raw_bitmap [0:3];
    logic request_allow, response_allow, result_ready, token_done_ready;

    for (genvar lane = 0; lane < 4; lane++) begin : g_raw_unflatten
        assign rtl_raw_beat_index[lane] = raw_beat_index[19-lane*5-:5];
        assign rtl_raw_bitmap[lane] = raw_bitmap[383-lane*96-:96];
    end

    // RTL K8 pins, flattened at the TB boundary for a bit-exact comparison.
    logic rtl_header_ready, rtl_header_accept, rtl_raw_ready, rtl_raw_accept;
    logic [7:0] rtl_mem_req_valid, rtl_mem_req_ready, rtl_mem_req_accept;
    logic [127:0] rtl_mem_req_epoch;
    logic [23:0] rtl_mem_req_slot;
    logic [255:0] rtl_mem_req_generation;
    logic [191:0] rtl_mem_req_tag;
    logic [23:0] rtl_mem_req_output_block, rtl_mem_req_slice;
    logic [95:0] rtl_mem_req_source_channel;
    logic [7:0] rtl_mem_rsp_valid, rtl_mem_rsp_ready, rtl_mem_rsp_accept;
    logic [127:0] rtl_mem_rsp_epoch;
    logic [23:0] rtl_mem_rsp_slot;
    logic [255:0] rtl_mem_rsp_generation;
    logic [191:0] rtl_mem_rsp_tag;
    logic [1023:0] rtl_mem_rsp_weight;
    logic rtl_result_valid, rtl_result_last, rtl_result_accept;
    logic [23:0] rtl_result_tag;
    logic [2:0] rtl_result_output_block, rtl_result_slice;
    logic [383:0] rtl_result_accumulator;
    logic rtl_done_valid, rtl_done_had_event, rtl_done_accept;
    logic [23:0] rtl_done_tag;
    logic rtl_protocol_error, rtl_numeric_overflow;
    logic rtl_stale_response_seen, rtl_busy;
    logic [7:0] rtl_endpoint_fault;

    logic [15:0] rtl_mem_req_epoch_u [0:7];
    logic [2:0] rtl_mem_req_slot_u [0:7];
    logic [31:0] rtl_mem_req_generation_u [0:7];
    logic [23:0] rtl_mem_req_tag_u [0:7];
    logic [2:0] rtl_mem_req_output_block_u [0:7];
    logic [2:0] rtl_mem_req_slice_u [0:7];
    logic [11:0] rtl_mem_req_source_channel_u [0:7];
    logic [15:0] rtl_mem_rsp_epoch_u [0:7];
    logic [2:0] rtl_mem_rsp_slot_u [0:7];
    logic [31:0] rtl_mem_rsp_generation_u [0:7];
    logic [23:0] rtl_mem_rsp_tag_u [0:7];
    logic signed [7:0] rtl_mem_rsp_weight_u [0:7][0:LANES-1];
    logic signed [23:0] rtl_result_accumulator_u [0:LANES-1];

    for (genvar bank = 0; bank < 8; bank++) begin : g_rtl_flatten
        assign rtl_mem_req_epoch[127-bank*16-:16] = rtl_mem_req_epoch_u[bank];
        assign rtl_mem_req_slot[23-bank*3-:3] = rtl_mem_req_slot_u[bank];
        assign rtl_mem_req_generation[255-bank*32-:32]
            = rtl_mem_req_generation_u[bank];
        assign rtl_mem_req_tag[191-bank*24-:24] = rtl_mem_req_tag_u[bank];
        assign rtl_mem_req_output_block[23-bank*3-:3]
            = rtl_mem_req_output_block_u[bank];
        assign rtl_mem_req_slice[23-bank*3-:3] = rtl_mem_req_slice_u[bank];
        assign rtl_mem_req_source_channel[95-bank*12-:12]
            = rtl_mem_req_source_channel_u[bank];
        assign rtl_mem_rsp_epoch_u[bank] = rtl_mem_rsp_epoch[127-bank*16-:16];
        assign rtl_mem_rsp_slot_u[bank] = rtl_mem_rsp_slot[23-bank*3-:3];
        assign rtl_mem_rsp_generation_u[bank]
            = rtl_mem_rsp_generation[255-bank*32-:32];
        assign rtl_mem_rsp_tag_u[bank] = rtl_mem_rsp_tag[191-bank*24-:24];
        for (genvar lane = 0; lane < LANES; lane++) begin : g_weight
            assign rtl_mem_rsp_weight_u[bank][lane]
                = rtl_mem_rsp_weight[1023-(bank*LANES+lane)*8-:8];
        end
    end
    for (genvar lane = 0; lane < LANES; lane++) begin : g_rtl_result_flatten
        assign rtl_result_accumulator[383-lane*24-:24]
            = rtl_result_accumulator_u[lane];
    end

    m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24 #(
        .ARCH_MODE(1)
    ) rtl_dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(rtl_header_ready),
        .header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(rtl_header_accept),
        .raw_valid(raw_valid), .raw_ready(rtl_raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(rtl_raw_beat_index),
        .raw_bitmap(rtl_raw_bitmap), .raw_last(raw_last),
        .raw_accept(rtl_raw_accept),
        .mem_req_valid(rtl_mem_req_valid), .mem_req_ready(rtl_mem_req_ready),
        .mem_req_epoch(rtl_mem_req_epoch_u), .mem_req_slot(rtl_mem_req_slot_u),
        .mem_req_generation(rtl_mem_req_generation_u),
        .mem_req_tag(rtl_mem_req_tag_u),
        .mem_req_output_block(rtl_mem_req_output_block_u),
        .mem_req_slice(rtl_mem_req_slice_u),
        .mem_req_source_channel(rtl_mem_req_source_channel_u),
        .mem_req_accept(rtl_mem_req_accept),
        .mem_rsp_valid(rtl_mem_rsp_valid), .mem_rsp_ready(rtl_mem_rsp_ready),
        .mem_rsp_epoch(rtl_mem_rsp_epoch_u), .mem_rsp_slot(rtl_mem_rsp_slot_u),
        .mem_rsp_generation(rtl_mem_rsp_generation_u),
        .mem_rsp_tag(rtl_mem_rsp_tag_u),
        .mem_rsp_weight(rtl_mem_rsp_weight_u),
        .mem_rsp_accept(rtl_mem_rsp_accept),
        .result_valid(rtl_result_valid), .result_ready(result_ready),
        .result_tag(rtl_result_tag),
        .result_output_block(rtl_result_output_block),
        .result_slice(rtl_result_slice),
        .result_accumulator(rtl_result_accumulator_u),
        .result_last(rtl_result_last), .result_accept(rtl_result_accept),
        .token_done_valid(rtl_done_valid),
        .token_done_ready(token_done_ready), .token_done_tag(rtl_done_tag),
        .token_done_had_event(rtl_done_had_event),
        .token_done_accept(rtl_done_accept),
        .protocol_error(rtl_protocol_error),
        .numeric_overflow(rtl_numeric_overflow),
        .stale_response_seen(rtl_stale_response_seen), .busy(rtl_busy));

    m1578_case0_memory_fabric rtl_memory (
        .clk_core(clk_core), .rst_core(rst_core),
        .request_allow(request_allow), .response_allow(response_allow),
        .mem_req_valid(rtl_mem_req_valid), .mem_req_ready(rtl_mem_req_ready),
        .mem_req_epoch(rtl_mem_req_epoch), .mem_req_slot(rtl_mem_req_slot),
        .mem_req_generation(rtl_mem_req_generation),
        .mem_req_tag(rtl_mem_req_tag),
        .mem_req_output_block(rtl_mem_req_output_block),
        .mem_req_slice(rtl_mem_req_slice),
        .mem_req_source_channel(rtl_mem_req_source_channel),
        .mem_req_accept(rtl_mem_req_accept),
        .mem_rsp_valid(rtl_mem_rsp_valid), .mem_rsp_ready(rtl_mem_rsp_ready),
        .mem_rsp_epoch(rtl_mem_rsp_epoch), .mem_rsp_slot(rtl_mem_rsp_slot),
        .mem_rsp_generation(rtl_mem_rsp_generation),
        .mem_rsp_tag(rtl_mem_rsp_tag), .mem_rsp_weight(rtl_mem_rsp_weight),
        .mem_rsp_accept(rtl_mem_rsp_accept),
        .endpoint_fault(rtl_endpoint_fault));

    // Mapped K8 pins use the flattened post-DC module interface directly.
    logic mapped_header_ready, mapped_header_accept;
    logic mapped_raw_ready, mapped_raw_accept;
    logic [7:0] mapped_mem_req_valid, mapped_mem_req_ready;
    logic [7:0] mapped_mem_req_accept;
    logic [127:0] mapped_mem_req_epoch;
    logic [23:0] mapped_mem_req_slot;
    logic [255:0] mapped_mem_req_generation;
    logic [191:0] mapped_mem_req_tag;
    logic [23:0] mapped_mem_req_output_block, mapped_mem_req_slice;
    logic [95:0] mapped_mem_req_source_channel;
    logic [7:0] mapped_mem_rsp_valid, mapped_mem_rsp_ready;
    logic [7:0] mapped_mem_rsp_accept;
    logic [127:0] mapped_mem_rsp_epoch;
    logic [23:0] mapped_mem_rsp_slot;
    logic [255:0] mapped_mem_rsp_generation;
    logic [191:0] mapped_mem_rsp_tag;
    logic [1023:0] mapped_mem_rsp_weight;
    logic mapped_result_valid, mapped_result_last, mapped_result_accept;
    logic [23:0] mapped_result_tag;
    logic [2:0] mapped_result_output_block, mapped_result_slice;
    logic [383:0] mapped_result_accumulator;
    logic mapped_done_valid, mapped_done_had_event, mapped_done_accept;
    logic [23:0] mapped_done_tag;
    logic mapped_protocol_error, mapped_numeric_overflow;
    logic mapped_stale_response_seen, mapped_busy;
    logic [7:0] mapped_endpoint_fault;

    m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE1
        mapped_dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(mapped_header_ready),
        .header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(mapped_header_accept),
        .raw_valid(raw_valid), .raw_ready(mapped_raw_ready),
        .raw_lane_valid(raw_lane_valid), .raw_beat_index(raw_beat_index),
        .raw_bitmap(raw_bitmap), .raw_last(raw_last),
        .raw_accept(mapped_raw_accept),
        .mem_req_valid(mapped_mem_req_valid),
        .mem_req_ready(mapped_mem_req_ready),
        .mem_req_epoch(mapped_mem_req_epoch),
        .mem_req_slot(mapped_mem_req_slot),
        .mem_req_generation(mapped_mem_req_generation),
        .mem_req_tag(mapped_mem_req_tag),
        .mem_req_output_block(mapped_mem_req_output_block),
        .mem_req_slice(mapped_mem_req_slice),
        .mem_req_source_channel(mapped_mem_req_source_channel),
        .mem_req_accept(mapped_mem_req_accept),
        .mem_rsp_valid(mapped_mem_rsp_valid),
        .mem_rsp_ready(mapped_mem_rsp_ready),
        .mem_rsp_epoch(mapped_mem_rsp_epoch),
        .mem_rsp_slot(mapped_mem_rsp_slot),
        .mem_rsp_generation(mapped_mem_rsp_generation),
        .mem_rsp_tag(mapped_mem_rsp_tag),
        .mem_rsp_weight(mapped_mem_rsp_weight),
        .mem_rsp_accept(mapped_mem_rsp_accept),
        .result_valid(mapped_result_valid), .result_ready(result_ready),
        .result_tag(mapped_result_tag),
        .result_output_block(mapped_result_output_block),
        .result_slice(mapped_result_slice),
        .result_accumulator(mapped_result_accumulator),
        .result_last(mapped_result_last),
        .result_accept(mapped_result_accept),
        .token_done_valid(mapped_done_valid),
        .token_done_ready(token_done_ready), .token_done_tag(mapped_done_tag),
        .token_done_had_event(mapped_done_had_event),
        .token_done_accept(mapped_done_accept),
        .protocol_error(mapped_protocol_error),
        .numeric_overflow(mapped_numeric_overflow),
        .stale_response_seen(mapped_stale_response_seen),
        .busy(mapped_busy));

    m1578_case0_memory_fabric mapped_memory (
        .clk_core(clk_core), .rst_core(rst_core),
        .request_allow(request_allow), .response_allow(response_allow),
        .mem_req_valid(mapped_mem_req_valid),
        .mem_req_ready(mapped_mem_req_ready),
        .mem_req_epoch(mapped_mem_req_epoch),
        .mem_req_slot(mapped_mem_req_slot),
        .mem_req_generation(mapped_mem_req_generation),
        .mem_req_tag(mapped_mem_req_tag),
        .mem_req_output_block(mapped_mem_req_output_block),
        .mem_req_slice(mapped_mem_req_slice),
        .mem_req_source_channel(mapped_mem_req_source_channel),
        .mem_req_accept(mapped_mem_req_accept),
        .mem_rsp_valid(mapped_mem_rsp_valid),
        .mem_rsp_ready(mapped_mem_rsp_ready),
        .mem_rsp_epoch(mapped_mem_rsp_epoch),
        .mem_rsp_slot(mapped_mem_rsp_slot),
        .mem_rsp_generation(mapped_mem_rsp_generation),
        .mem_rsp_tag(mapped_mem_rsp_tag),
        .mem_rsp_weight(mapped_mem_rsp_weight),
        .mem_rsp_accept(mapped_mem_rsp_accept),
        .endpoint_fault(mapped_endpoint_fault));

    // Bit order: compactor, paired sink, core adapter, service, memory adapter,
    // memory-adapter stale. No tap is coerced from X to zero.
    logic [5:0] rtl_internal_fault_taps, mapped_internal_fault_taps;
    assign rtl_internal_fault_taps = {
        rtl_dut.g_k8.implementation.core.frontend.compactor.fault_q,
        rtl_dut.g_k8.implementation.core.frontend.paired_sink.fault_q,
        rtl_dut.g_k8.implementation.core.adapter_fault_q,
        rtl_dut.g_k8.implementation.core.g_k8.service.fault_q,
        rtl_dut.g_k8.implementation.memory_adapter.fault_q,
        rtl_dut.g_k8.implementation.memory_adapter.stale_q};
    assign mapped_internal_fault_taps = {
        mapped_dut.g_k8_implementation_core_frontend_compactor_fault_q,
        mapped_dut.g_k8_implementation_core_frontend_paired_sink_fault_q,
        mapped_dut.g_k8_implementation_core_adapter_fault_q,
        mapped_dut.g_k8_implementation_core_g_k8_service_fault_q,
        mapped_dut.g_k8_implementation_memory_adapter_fault_q,
        mapped_dut.g_k8_implementation_memory_adapter_stale_q};

    integer cycle_ordinal;
    integer first_difference_cycle, first_fault_cycle;
    logic difference_now, rtl_fault_now, mapped_fault_now;
    logic control_unknown_now;
    logic [95:0] payload [0:3];

    function automatic [7:0] tri(input logic value);
        if (value === 1'b0) tri = "0";
        else if (value === 1'b1) tri = "1";
        else tri = "X";
    endfunction

    function automatic [7:0] event8(input logic [7:0] value);
        if ($isunknown(value)) event8 = "X";
        else if (|value) event8 = "1";
        else event8 = "0";
    endfunction

    always_comb begin
        request_allow = !rst_core && (cycle_ordinal % 7 != 2);
        response_allow = !rst_core && (cycle_ordinal % 17 >= 5);
        result_ready = !rst_core && (cycle_ordinal % 5 != 2);
        token_done_ready = !rst_core && (cycle_ordinal % 4 != 1);

        difference_now = ({rtl_header_accept, rtl_raw_accept,
            rtl_mem_req_accept, rtl_mem_rsp_accept, rtl_result_accept,
            rtl_done_accept, rtl_protocol_error, rtl_numeric_overflow,
            rtl_stale_response_seen, rtl_endpoint_fault,
            rtl_internal_fault_taps}
            !== {mapped_header_accept, mapped_raw_accept,
            mapped_mem_req_accept, mapped_mem_rsp_accept,
            mapped_result_accept, mapped_done_accept,
            mapped_protocol_error, mapped_numeric_overflow,
            mapped_stale_response_seen, mapped_endpoint_fault,
            mapped_internal_fault_taps});
        rtl_fault_now = (rtl_protocol_error !== 1'b0)
            || (rtl_numeric_overflow !== 1'b0)
            || (rtl_stale_response_seen !== 1'b0)
            || (rtl_endpoint_fault !== 8'b0)
            || (rtl_internal_fault_taps !== 6'b0);
        mapped_fault_now = (mapped_protocol_error !== 1'b0)
            || (mapped_numeric_overflow !== 1'b0)
            || (mapped_stale_response_seen !== 1'b0)
            || (mapped_endpoint_fault !== 8'b0)
            || (mapped_internal_fault_taps !== 6'b0);
        control_unknown_now = $isunknown({rtl_header_accept, rtl_raw_accept,
            rtl_mem_req_accept, rtl_mem_rsp_accept, rtl_result_accept,
            rtl_done_accept, mapped_header_accept, mapped_raw_accept,
            mapped_mem_req_accept, mapped_mem_rsp_accept,
            mapped_result_accept, mapped_done_accept});
    end

    task automatic trace_edge;
        begin
            $display("M1578_TRACE cycle=%0d header=%s/%s source=%s/%s endpoint=%s/%s mem=%s/%s commit=%s/%s done=%s/%s top_pns=%s%s%s/%s%s%s endpoint_fault=%b/%b taps_csfamS=%b/%b",
                cycle_ordinal,
                tri(rtl_header_accept), tri(mapped_header_accept),
                tri(rtl_raw_accept), tri(mapped_raw_accept),
                event8(rtl_mem_req_accept), event8(mapped_mem_req_accept),
                event8(rtl_mem_rsp_accept), event8(mapped_mem_rsp_accept),
                tri(rtl_result_accept), tri(mapped_result_accept),
                tri(rtl_done_accept), tri(mapped_done_accept),
                tri(rtl_protocol_error), tri(rtl_numeric_overflow),
                tri(rtl_stale_response_seen), tri(mapped_protocol_error),
                tri(mapped_numeric_overflow), tri(mapped_stale_response_seen),
                rtl_endpoint_fault, mapped_endpoint_fault,
                rtl_internal_fault_taps, mapped_internal_fault_taps);
        end
    endtask

    task automatic print_stop(input [8*32-1:0] reason);
        begin
            $display("M1578_FIRST_STOP reason=%s cycle=%0d first_difference_cycle=%0d first_fault_cycle=%0d rtl_top_pns=%s%s%s mapped_top_pns=%s%s%s rtl_endpoint_fault=%b mapped_endpoint_fault=%b rtl_taps=%b mapped_taps=%b",
                reason, cycle_ordinal, first_difference_cycle,
                first_fault_cycle, tri(rtl_protocol_error),
                tri(rtl_numeric_overflow), tri(rtl_stale_response_seen),
                tri(mapped_protocol_error), tri(mapped_numeric_overflow),
                tri(mapped_stale_response_seen), rtl_endpoint_fault,
                mapped_endpoint_fault, rtl_internal_fault_taps,
                mapped_internal_fault_taps);
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_ordinal = 0;
            first_difference_cycle = -1;
            first_fault_cycle = -1;
        end else begin
            cycle_ordinal = cycle_ordinal + 1;
            trace_edge();
            if (difference_now && first_difference_cycle < 0)
                first_difference_cycle = cycle_ordinal;
            if ((rtl_fault_now || mapped_fault_now || control_unknown_now)
                    && first_fault_cycle < 0)
                first_fault_cycle = cycle_ordinal;
            if (rtl_fault_now || mapped_fault_now || control_unknown_now) begin
                print_stop("FAULT_OR_X");
                $finish;
            end
            if (difference_now) begin
                print_stop("FIRST_RTL_MAPPED_DIFFERENCE");
                $finish;
            end
            if (rtl_done_accept === 1'b1
                    && mapped_done_accept === 1'b1) begin
                print_stop("BOTH_CLEAN_TO_DONE");
                $finish;
            end
            if (cycle_ordinal >= 4096)
                $fatal(1, "M1578 diagnostic watchdog");
        end
    end

    task automatic initialize_inputs;
        begin
            header_valid = 0;
            header_tag = 24'h979000;
            header_raw_beat_count = 6'd4;
            header_window_depth = 4'd2;
            header_output_blocks = 4'd1;
            raw_valid = 0;
            raw_lane_valid = 0;
            raw_beat_index = 0;
            raw_bitmap = 0;
            raw_last = 0;
        end
    endtask

    task automatic build_case0;
        integer row, bank;
        begin
            for (integer beat = 0; beat < 4; beat++) payload[beat] = 0;
            for (bank = 0; bank < 8; bank++)
                payload[0][(bank % 12)*8+bank] = 1'b1;
            for (integer beat = 0; beat < 4; beat++) begin
                for (integer item = 0; item < 3; item++) begin
                    row = (beat*3 + item*5) % 12;
                    bank = (beat + item*3) % 8;
                    payload[beat][row*8+bank] = 1'b1;
                end
                if (beat % 4 == 0) begin
                    row = (beat+7) % 12;
                    bank = (beat*5+2) % 8;
                    payload[beat][row*8+bank] = 1'b1;
                end
            end
        end
    endtask

    task automatic drive_case0;
        begin
            @(negedge clk_core);
            header_valid = 1'b1;
            while (!(rtl_header_accept === 1'b1
                    && mapped_header_accept === 1'b1))
                @(posedge clk_core);
            @(negedge clk_core);
            header_valid = 1'b0;
            raw_lane_valid = 4'b1111;
            for (integer lane = 0; lane < 4; lane++) begin
                raw_beat_index[19-lane*5-:5] = lane;
                raw_bitmap[383-lane*96-:96] = payload[lane];
            end
            raw_last = 1'b1;
            raw_valid = 1'b1;
            while (!(rtl_raw_accept === 1'b1
                    && mapped_raw_accept === 1'b1))
                @(posedge clk_core);
            @(negedge clk_core);
            raw_valid = 1'b0;
            raw_lane_valid = 0;
            raw_last = 1'b0;
        end
    endtask

    initial begin
        initialize_inputs();
        build_case0();
        rst_core = 1'b1;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        drive_case0();
    end

    initial begin
        #1000000;
        $fatal(1, "M1578 absolute watchdog");
    end
endmodule

`default_nettype wire
