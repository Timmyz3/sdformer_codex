`timescale 1ns/1ps
`default_nettype none

// Directed unit campaign for the M1801 fault-export boundary.  It uses only
// public ports and contains no force/deposit/initreg/X suppression.
module tb_m1801_c2_registered_public_fault_export_directed;
    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic core_fault_sample_enable, core_fault_event_raw;
    logic adapter_fault_sample_enable, adapter_fault_event_raw;
    logic core_req_valid, core_req_accept, adapter_req_accept;
    logic core_rsp_valid, core_rsp_accept, adapter_rsp_accept;
    logic protocol_error;

    int legal_case0;
    int invalid_payload_cases;
    int attack_classes;
    int posedge_checks;
    int negedge_checks;
    int half_cycle_checks;
    int reset_recoveries;

    m1801_c2_registered_public_fault_export dut (.*);

    task automatic drive_zero;
        core_fault_sample_enable = 1'b0;
        core_fault_event_raw = 1'b0;
        adapter_fault_sample_enable = 1'b0;
        adapter_fault_event_raw = 1'b0;
        core_req_valid = 1'b0;
        core_req_accept = 1'b0;
        adapter_req_accept = 1'b0;
        core_rsp_valid = 1'b0;
        core_rsp_accept = 1'b0;
        adapter_rsp_accept = 1'b0;
    endtask

    task automatic require_binary_zero(input string label);
        if ($isunknown(protocol_error) || protocol_error !== 1'b0)
            $fatal(1, "M1801 %s expected binary zero", label);
    endtask

    task automatic require_binary_one(input string label);
        if ($isunknown(protocol_error) || protocol_error !== 1'b1)
            $fatal(1, "M1801 %s expected binary sticky one", label);
    endtask

    task automatic reset_boundary;
        @(negedge clk_core);
        drive_zero();
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        #1ps;
        require_binary_zero("reset asserted");
        @(negedge clk_core);
        rst_core = 1'b0;
        #1ps;
        require_binary_zero("reset recovery");
        reset_recoveries++;
    endtask

    task automatic check_fault_after_sampling(input string label);
        #1ps;
        require_binary_zero({label, " pre-edge"});
        half_cycle_checks++;
        @(posedge clk_core);
        #1ps;
        require_binary_one({label, " posedge+1ps"});
        posedge_checks++;
        @(negedge clk_core);
        #1ps;
        require_binary_one({label, " negedge+1ps"});
        negedge_checks++;
        drive_zero();
        repeat (2) @(posedge clk_core);
        #1ps;
        require_binary_one({label, " sticky"});
        attack_classes++;
    endtask

    initial begin
        rst_core = 1'b1;
        drive_zero();
        repeat (3) @(posedge clk_core);
        #1ps;
        require_binary_zero("POR");
        @(negedge clk_core);
        rst_core = 1'b0;

        // Legal case0/quiescence.  The exported bit must be binary on both
        // phases and no current-cycle term may leak onto the public pin.
        repeat (2) begin
            @(posedge clk_core); #1ps;
            require_binary_zero("legal case0 posedge");
            posedge_checks++;
            @(negedge clk_core); #1ps;
            require_binary_zero("legal case0 negedge");
            negedge_checks++;
        end
        legal_case0++;

        // Invalid payload is semantically absent while valid/owner enable is
        // zero.  This is the exact four-state shape that the frozen
        // unconditional accept comparison failed to isolate.
        @(negedge clk_core);
        core_fault_event_raw = 1'bx;
        adapter_fault_event_raw = 1'bx;
        core_req_accept = 1'bx;
        adapter_req_accept = 1'bx;
        core_rsp_accept = 1'bx;
        adapter_rsp_accept = 1'bx;
        #1ps;
        require_binary_zero("invalid payload half-cycle");
        half_cycle_checks++;
        @(posedge clk_core); #1ps;
        require_binary_zero("invalid payload posedge");
        posedge_checks++;
        @(negedge clk_core); #1ps;
        require_binary_zero("invalid payload negedge");
        negedge_checks++;
        drive_zero();
        invalid_payload_cases++;

        // Matched accepts under a real request are legal.
        @(negedge clk_core);
        core_req_valid = 1'b1;
        core_req_accept = 1'b1;
        adapter_req_accept = 1'b1;
        @(posedge clk_core); #1ps;
        require_binary_zero("matched request accepts");
        posedge_checks++;
        drive_zero();

        // Every event input gets an independent attack and reset recovery.
        @(negedge clk_core);
        core_fault_sample_enable = 1'b1;
        core_fault_event_raw = 1'b1;
        check_fault_after_sampling("core legality event");
        reset_boundary();

        @(negedge clk_core);
        adapter_fault_sample_enable = 1'b1;
        adapter_fault_event_raw = 1'b1;
        check_fault_after_sampling("adapter legality event");
        reset_boundary();

        @(negedge clk_core);
        core_req_valid = 1'b1;
        core_req_accept = 1'b1;
        adapter_req_accept = 1'b0;
        check_fault_after_sampling("request accept mismatch");
        reset_boundary();

        @(negedge clk_core);
        core_rsp_valid = 1'b1;
        core_rsp_accept = 1'b0;
        adapter_rsp_accept = 1'b1;
        check_fault_after_sampling("response accept mismatch");
        reset_boundary();

        if (legal_case0 != 1 || invalid_payload_cases != 1
                || attack_classes != 4 || reset_recoveries != 4
                || posedge_checks < 7 || negedge_checks < 7
                || half_cycle_checks < 5)
            $fatal(1, "M1801 coverage incomplete legal=%0d invalid=%0d attacks=%0d reset=%0d pos=%0d neg=%0d half=%0d",
                legal_case0, invalid_payload_cases, attack_classes,
                reset_recoveries, posedge_checks, negedge_checks,
                half_cycle_checks);
        $display("PASS M1801 registered public fault export legal_case0=1 invalid_payload_cases=1 attack_classes=4 reset_recoveries=4 posedge_checks=%0d negedge_checks=%0d half_cycle_checks=%0d public_fault_binary=true force=false initreg=false ignore_x=false",
            posedge_checks, negedge_checks, half_cycle_checks);
        $finish;
    end

    ap_public_fault_binary_posedge: assert property (
        @(posedge clk_core) disable iff (rst_core)
        !$isunknown(protocol_error));
    ap_public_fault_binary_negedge: assert property (
        @(negedge clk_core) disable iff (rst_core)
        !$isunknown(protocol_error));
    ap_sticky_until_reset: assert property (
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error);
    ap_no_invalid_req_payload_fault: assert property (
        @(posedge clk_core) disable iff (rst_core)
        !core_req_valid && !core_fault_sample_enable
            && !adapter_fault_sample_enable |->
        !$isunknown(protocol_error));
endmodule

`default_nettype wire
