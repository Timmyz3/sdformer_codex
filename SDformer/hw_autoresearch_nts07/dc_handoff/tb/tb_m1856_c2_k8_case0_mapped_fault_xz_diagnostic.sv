`timescale 1ns/1ps
`default_nettype none

// Diagnostic-only wrapper around the exact M1811 K8 mapped netlist and the
// frozen M979 case-0 stimulus.  This wrapper does not suppress or reinterpret
// a fault.  It prints every member of the M1845 11-bit unknown boundary at
// every post-reset clock edge, identifies the first non-binary member using
// case equality, and stops immediately.  A completed run is localization
// evidence only: it is not production function, power, performance, or paper
// evidence.
module tb_m1856_c2_k8_case0_mapped_fault_xz_diagnostic;
    tb_m979_c2_three_axis_mapped_gate_case_saif core();

    logic [7:0] endpoint_fault;
    for (genvar bank = 0; bank < 8; bank++) begin : g_fault_tap
        assign endpoint_fault[bank] =
            core.g_memory[bank].memory.endpoint_protocol_fault_q;
    end

    // These exact-netlist XMRs are observation-only.  They bind existing,
    // named M1809 mapped fault-state nets.  They are never used to decide
    // localization success, so their interpretation cannot replace the public
    // interface or endpoint checks above.
    wire mapped_protocol_error_q_tap =
        core.dut.implementation.protocol_error;
    wire mapped_numeric_overflow_q_tap =
        core.dut.implementation.numeric_overflow;
    wire mapped_memory_adapter_stale_q_tap =
        core.dut.implementation.g_k8_implementation_memory_adapter_stale_q;
    wire mapped_memory_adapter_fault_q_tap =
        core.dut.implementation.g_k8_implementation_memory_adapter_fault_q;
    wire mapped_service_fault_q_tap =
        core.dut.implementation.g_k8_implementation_core_g_k8_service_fault_q;
    wire mapped_core_adapter_fault_q_tap =
        core.dut.implementation.g_k8_implementation_core_adapter_fault_q;

    integer sample_ordinal = 0;

    function automatic logic is_binary(input logic value);
        is_binary = (value === 1'b0) || (value === 1'b1);
    endfunction

    task automatic print_and_localize(input string edge_name);
        begin
            sample_ordinal = sample_ordinal + 1;
            $display("M1856_SAMPLE time_ps=%0t sample=%0d edge=%s", $time,
                sample_ordinal, edge_name);
            $display("M1856_BIT name=protocol_error value=%b binary=%0d",
                core.protocol_error, is_binary(core.protocol_error));
            $display("M1856_BIT name=numeric_overflow value=%b binary=%0d",
                core.numeric_overflow, is_binary(core.numeric_overflow));
            $display("M1856_BIT name=stale_response_seen value=%b binary=%0d",
                core.stale_response_seen, is_binary(core.stale_response_seen));
            for (integer bank = 0; bank < 8; bank++) begin
                $display("M1856_BIT name=endpoint_fault[%0d] value=%b binary=%0d",
                    bank, endpoint_fault[bank], is_binary(endpoint_fault[bank]));
            end
            $display("M1856_AUX mapped_protocol_error_q=%b mapped_numeric_overflow_q=%b mapped_memory_adapter_stale_q=%b mapped_memory_adapter_fault_q=%b mapped_service_fault_q=%b mapped_core_adapter_fault_q=%b",
                mapped_protocol_error_q_tap, mapped_numeric_overflow_q_tap,
                mapped_memory_adapter_stale_q_tap,
                mapped_memory_adapter_fault_q_tap, mapped_service_fault_q_tap,
                mapped_core_adapter_fault_q_tap);

            if (!is_binary(core.protocol_error)) begin
                $display("M1856_FIRST_NONBINARY time_ps=%0t edge=%s name=protocol_error value=%b",
                    $time, edge_name, core.protocol_error);
                $finish;
            end
            if (!is_binary(core.numeric_overflow)) begin
                $display("M1856_FIRST_NONBINARY time_ps=%0t edge=%s name=numeric_overflow value=%b",
                    $time, edge_name, core.numeric_overflow);
                $finish;
            end
            if (!is_binary(core.stale_response_seen)) begin
                $display("M1856_FIRST_NONBINARY time_ps=%0t edge=%s name=stale_response_seen value=%b",
                    $time, edge_name, core.stale_response_seen);
                $finish;
            end
            for (integer bank = 0; bank < 8; bank++) begin
                if (!is_binary(endpoint_fault[bank])) begin
                    $display("M1856_FIRST_NONBINARY time_ps=%0t edge=%s name=endpoint_fault[%0d] value=%b",
                        $time, edge_name, bank, endpoint_fault[bank]);
                    $finish;
                end
            end
        end
    endtask

    // One picosecond avoids active/NBA race ambiguity while remaining well
    // inside the 1.5 ns half-cycle of the frozen M979 clock.
    always @(posedge core.clk_core) begin
        #1ps;
        if (core.rst_core === 1'b0)
            print_and_localize("posedge");
    end

    always @(negedge core.clk_core) begin
        #1ps;
        if (core.rst_core === 1'b0)
            print_and_localize("negedge");
    end
endmodule

`default_nettype wire
