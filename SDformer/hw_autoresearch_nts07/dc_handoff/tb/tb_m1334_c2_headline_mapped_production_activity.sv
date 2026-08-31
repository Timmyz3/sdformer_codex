`timescale 1ns/1ps
`default_nettype none

// Additive production-activity wrapper.  The workload/reference implementation
// remains the exact frozen M979 source; only the external test-memory provider
// and fail-closed observation layer differ.
module tb_m1334_c2_headline_mapped_production_activity;
    tb_m979_c2_three_axis_mapped_gate_case_saif core();

    logic [7:0] endpoint_fault;
    for (genvar bank = 0; bank < 8; bank++) begin : g_fault_tap
        assign endpoint_fault[bank] =
            core.g_memory[bank].memory.endpoint_protocol_fault_q;
    end

    m1334_c2_production_activity_assertions checks (
        .clk_core(core.clk_core), .rst_core(core.rst_core),
        .header_valid(core.header_valid), .header_ready(core.header_ready),
        .header_accept(core.header_accept), .raw_valid(core.raw_valid),
        .raw_ready(core.raw_ready), .raw_accept(core.raw_accept),
        .raw_lane_valid(core.raw_lane_valid),
        .raw_beat_index(core.raw_beat_index), .raw_bitmap(core.raw_bitmap),
        .raw_last(core.raw_last), .mem_req_valid(core.mem_req_valid),
        .mem_req_ready(core.mem_req_ready),
        .mem_req_accept(core.mem_req_accept),
        .mem_req_epoch(core.mem_req_epoch), .mem_req_slot(core.mem_req_slot),
        .mem_req_generation(core.mem_req_generation),
        .mem_req_tag(core.mem_req_tag),
        .mem_req_output_block(core.mem_req_output_block),
        .mem_req_slice(core.mem_req_slice),
        .mem_req_source_channel(core.mem_req_source_channel),
        .mem_rsp_valid(core.mem_rsp_valid),
        .mem_rsp_ready(core.mem_rsp_ready),
        .mem_rsp_accept(core.mem_rsp_accept),
        .mem_rsp_epoch(core.mem_rsp_epoch), .mem_rsp_slot(core.mem_rsp_slot),
        .mem_rsp_generation(core.mem_rsp_generation),
        .mem_rsp_tag(core.mem_rsp_tag),
        .mem_rsp_weight(core.mem_rsp_weight),
        .result_valid(core.result_valid), .result_ready(core.result_ready),
        .result_accept(core.result_accept), .result_tag(core.result_tag),
        .result_output_block(core.result_output_block),
        .result_slice(core.result_slice),
        .result_accumulator(core.result_accumulator),
        .result_last(core.result_last),
        .token_done_valid(core.token_done_valid),
        .token_done_ready(core.token_done_ready),
        .token_done_accept(core.token_done_accept),
        .token_done_tag(core.token_done_tag),
        .token_done_had_event(core.token_done_had_event),
        .protocol_error(core.protocol_error),
        .numeric_overflow(core.numeric_overflow),
        .stale_response_seen(core.stale_response_seen),
        .endpoint_fault(endpoint_fault));
endmodule

`default_nettype wire
