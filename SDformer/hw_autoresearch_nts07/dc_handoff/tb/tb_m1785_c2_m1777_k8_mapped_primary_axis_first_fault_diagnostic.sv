`timescale 1ns/1ps
`default_nettype none

// M1785 is an additive diagnostic around the immutable M1684/M979 mapped K8
// production wrapper.  The wrapped M1684 assertion remains active and exact;
// this monitor only records which public field first becomes four-state.  It
// never initializes, forces, masks, or coerces DUT or endpoint state.
module m1785_c2_public_first_fault_monitor (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic busy,
    input logic raw_valid, input logic raw_ready, input logic raw_accept,
    input logic raw_last, input logic [3:0] raw_lane_valid,
    input logic [19:0] raw_beat_index, input logic [383:0] raw_bitmap,
    input logic [7:0] mem_req_valid, input logic [7:0] mem_req_ready,
    input logic [7:0] mem_req_accept, input logic [127:0] mem_req_epoch,
    input logic [23:0] mem_req_slot, input logic [255:0] mem_req_generation,
    input logic [191:0] mem_req_tag,
    input logic [23:0] mem_req_output_block, input logic [23:0] mem_req_slice,
    input logic [95:0] mem_req_source_channel,
    input logic [7:0] mem_rsp_valid, input logic [7:0] mem_rsp_ready,
    input logic [7:0] mem_rsp_accept,
    input logic result_valid, input logic result_ready,
    input logic result_accept, input logic [23:0] result_tag,
    input logic [2:0] result_output_block, input logic [2:0] result_slice,
    input logic [383:0] result_accumulator, input logic result_last,
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept, input logic [23:0] token_done_tag,
    input logic token_done_had_event,
    input logic protocol_error, input logic numeric_overflow,
    input logic stale_response_seen, input logic [7:0] endpoint_fault,
    input logic [5:0] registered_fault_taps
);
    integer settled_samples;
    integer first_unknown_code;
    time first_unknown_time;
    logic first_unknown_seen;
    logic [63:0] reported;

    // Stable numeric codes are intentionally part of the diagnostic contract.
    // 1..3 top faults, 4..11 endpoint faults, 12..16 source, 17..22 endpoint
    // handshakes/payload, 23..27 result, 28..31 done, 32..36 status/header.
    task automatic flag_unknown(
        input integer code, input string class_name, input string field_name);
        begin
            if (!reported[code-1]) begin
                reported[code-1] = 1'b1;
                $display("M1785_FIRST_UNKNOWN code=%0d class=%s field=%s time_ps=%0t",
                    code, class_name, field_name, $time);
            end
            if (!first_unknown_seen) begin
                first_unknown_seen = 1'b1;
                first_unknown_code = code;
                first_unknown_time = $time;
            end
        end
    endtask

    task automatic inspect_public_fields;
        begin
            if ($isunknown(protocol_error))
                flag_unknown(1, "FAULT", "protocol_error");
            if ($isunknown(numeric_overflow))
                flag_unknown(2, "FAULT", "numeric_overflow");
            if ($isunknown(stale_response_seen))
                flag_unknown(3, "FAULT", "stale_response_seen");
            if ($isunknown(registered_fault_taps))
                flag_unknown(37, "DIAGNOSTIC_TAP",
                    "registered_fault_taps");
            for (integer bank = 0; bank < 8; bank++) begin
                if ($isunknown(endpoint_fault[bank]))
                    flag_unknown(4 + bank, "ENDPOINT_FAULT",
                        $sformatf("endpoint_fault[%0d]", bank));
            end
            if ($isunknown(raw_valid))
                flag_unknown(12, "SOURCE", "raw_valid");
            if ($isunknown(raw_ready))
                flag_unknown(13, "SOURCE", "raw_ready");
            if ($isunknown(raw_accept))
                flag_unknown(14, "SOURCE", "raw_accept");
            if ($isunknown({raw_last, raw_lane_valid}))
                flag_unknown(15, "SOURCE", "raw_control");
            if (raw_valid === 1'b1
                    && $isunknown({raw_beat_index, raw_bitmap}))
                flag_unknown(16, "SOURCE", "raw_payload_when_valid");
            if ($isunknown(mem_req_valid))
                flag_unknown(17, "ENDPOINT", "mem_req_valid");
            if ($isunknown(mem_req_ready))
                flag_unknown(18, "ENDPOINT", "mem_req_ready");
            if ($isunknown(mem_req_accept))
                flag_unknown(19, "ENDPOINT", "mem_req_accept");
            if ((|mem_req_valid) === 1'b1
                    && $isunknown({mem_req_epoch, mem_req_slot,
                        mem_req_generation, mem_req_tag, mem_req_output_block,
                        mem_req_slice, mem_req_source_channel}))
                flag_unknown(20, "ENDPOINT", "mem_req_payload_when_valid");
            if ($isunknown(mem_rsp_valid))
                flag_unknown(21, "ENDPOINT", "mem_rsp_valid");
            if ($isunknown({mem_rsp_ready, mem_rsp_accept}))
                flag_unknown(22, "ENDPOINT", "mem_rsp_ready_accept");
            if ($isunknown(result_valid))
                flag_unknown(23, "RESULT", "result_valid");
            if ($isunknown(result_ready))
                flag_unknown(24, "RESULT", "result_ready");
            if ($isunknown(result_accept))
                flag_unknown(25, "RESULT", "result_accept");
            if (result_valid === 1'b1
                    && $isunknown({result_tag, result_output_block,
                        result_slice, result_accumulator, result_last}))
                flag_unknown(26, "RESULT", "result_payload_when_valid");
            if ($isunknown(result_last))
                flag_unknown(27, "RESULT", "result_last");
            if ($isunknown(token_done_valid))
                flag_unknown(28, "DONE", "token_done_valid");
            if ($isunknown(token_done_ready))
                flag_unknown(29, "DONE", "token_done_ready");
            if ($isunknown(token_done_accept))
                flag_unknown(30, "DONE", "token_done_accept");
            if (token_done_valid === 1'b1
                    && $isunknown({token_done_tag, token_done_had_event}))
                flag_unknown(31, "DONE", "token_done_payload_when_valid");
            if ($isunknown(rst_core))
                flag_unknown(32, "STATUS", "rst_core");
            if ($isunknown(busy))
                flag_unknown(33, "STATUS", "busy");
            if ($isunknown(header_valid))
                flag_unknown(34, "STATUS", "header_valid");
            if ($isunknown(header_ready))
                flag_unknown(35, "STATUS", "header_ready");
            if ($isunknown(header_accept))
                flag_unknown(36, "STATUS", "header_accept");
        end
    endtask

    initial begin
        settled_samples = 0;
        first_unknown_code = 0;
        first_unknown_time = 0;
        first_unknown_seen = 1'b0;
        reported = '0;
    end

    // Continuous fault observation captures a transition even when the frozen
    // M1684 assertion terminates the same simulation time slot.
    always @(protocol_error or numeric_overflow or stale_response_seen
            or endpoint_fault) begin
        if (rst_core === 1'b0)
            inspect_public_fields();
    end

    // A one-timeprecision settled sample avoids the active-region race already
    // proven by M1594, while retaining the opposite-phase M1684 exact checker.
    always @(posedge clk_core) begin
        if (rst_core === 1'b0) begin
            #1ps;
            settled_samples = settled_samples + 1;
            inspect_public_fields();
            $display("M1785_SETTLED_TRACE sample=%0d time_ps=%0t source_vra=%b%b%b endpoint_req_vra=%b/%b/%b endpoint_rsp_vra=%b/%b/%b result_vra=%b%b%b done_vra=%b%b%b fault_pns=%b%b%b endpoint_fault=%b registered_fault_taps=%b status_hab=%b%b%b",
                settled_samples, $time, raw_valid, raw_ready, raw_accept,
                mem_req_valid, mem_req_ready, mem_req_accept,
                mem_rsp_valid, mem_rsp_ready, mem_rsp_accept,
                result_valid, result_ready, result_accept,
                token_done_valid, token_done_ready, token_done_accept,
                protocol_error, numeric_overflow, stale_response_seen,
                endpoint_fault, registered_fault_taps,
                header_valid, header_accept, busy);
        end
    end

    final begin
        $display("M1785_FINAL first_unknown_seen=%0d first_unknown_code=%0d first_unknown_time_ps=%0t settled_samples=%0d exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0",
            first_unknown_seen, first_unknown_code, first_unknown_time,
            settled_samples);
    end
endmodule


module tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic;
    // The exact original wrapper still contains both M1334 and M1684 monitors.
    tb_m1684_c2_m1609_fresh_mapped_production_energy sealed();

    // Registered implementation taps are supplemental cone evidence. Public
    // classification remains the top/endpoint vector above.
    logic [5:0] registered_fault_taps;
    assign registered_fault_taps = {
        sealed.core.dut.g_k8_implementation_core_frontend_m202_protocol_error,
        sealed.core.dut.g_k8_implementation_core_frontend_paired_sink_fault_q,
        sealed.core.dut.g_k8_implementation_core_adapter_fault_q,
        sealed.core.dut.g_k8_implementation_core_g_k8_service_fault_q,
        sealed.core.dut.g_k8_implementation_memory_adapter_fault_q,
        sealed.core.dut.g_k8_implementation_memory_adapter_stale_q};

    m1785_c2_public_first_fault_monitor diagnostic (
        .clk_core(sealed.core.clk_core), .rst_core(sealed.core.rst_core),
        .header_valid(sealed.core.header_valid),
        .header_ready(sealed.core.header_ready),
        .header_accept(sealed.core.header_accept), .busy(sealed.core.busy),
        .raw_valid(sealed.core.raw_valid), .raw_ready(sealed.core.raw_ready),
        .raw_accept(sealed.core.raw_accept), .raw_last(sealed.core.raw_last),
        .raw_lane_valid(sealed.core.raw_lane_valid),
        .raw_beat_index(sealed.core.raw_beat_index),
        .raw_bitmap(sealed.core.raw_bitmap),
        .mem_req_valid(sealed.core.mem_req_valid),
        .mem_req_ready(sealed.core.mem_req_ready),
        .mem_req_accept(sealed.core.mem_req_accept),
        .mem_req_epoch(sealed.core.mem_req_epoch),
        .mem_req_slot(sealed.core.mem_req_slot),
        .mem_req_generation(sealed.core.mem_req_generation),
        .mem_req_tag(sealed.core.mem_req_tag),
        .mem_req_output_block(sealed.core.mem_req_output_block),
        .mem_req_slice(sealed.core.mem_req_slice),
        .mem_req_source_channel(sealed.core.mem_req_source_channel),
        .mem_rsp_valid(sealed.core.mem_rsp_valid),
        .mem_rsp_ready(sealed.core.mem_rsp_ready),
        .mem_rsp_accept(sealed.core.mem_rsp_accept),
        .result_valid(sealed.core.result_valid),
        .result_ready(sealed.core.result_ready),
        .result_accept(sealed.core.result_accept),
        .result_tag(sealed.core.result_tag),
        .result_output_block(sealed.core.result_output_block),
        .result_slice(sealed.core.result_slice),
        .result_accumulator(sealed.core.result_accumulator),
        .result_last(sealed.core.result_last),
        .token_done_valid(sealed.core.token_done_valid),
        .token_done_ready(sealed.core.token_done_ready),
        .token_done_accept(sealed.core.token_done_accept),
        .token_done_tag(sealed.core.token_done_tag),
        .token_done_had_event(sealed.core.token_done_had_event),
        .protocol_error(sealed.core.protocol_error),
        .numeric_overflow(sealed.core.numeric_overflow),
        .stale_response_seen(sealed.core.stale_response_seen),
        .endpoint_fault(sealed.endpoint_fault),
        .registered_fault_taps(registered_fault_taps));
endmodule

`default_nettype wire
