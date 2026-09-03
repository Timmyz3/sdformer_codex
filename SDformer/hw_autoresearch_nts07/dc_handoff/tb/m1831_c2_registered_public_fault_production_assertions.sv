`timescale 1ns/1ps
`default_nettype none

// Additive identity-renamed successor of the M1684 public-fault monitor.
// It preserves the frozen five-case counts while binding the M1809 registered
// public-fault mapped identity.  This source draft has not been simulated.
module m1831_c2_registered_public_fault_production_assertions (
    input logic clk_core, input logic rst_core,
    input logic raw_accept, input logic [3:0] raw_lane_valid,
    input logic [383:0] raw_bitmap,
    input logic [7:0] mem_req_accept,
    input logic result_accept, input logic token_done_accept,
    input logic protocol_error, input logic numeric_overflow,
    input logic stale_response_seen, input logic [7:0] endpoint_fault
);
    integer case_id;
    integer accepted_sources, source_packets, endpoint_accepts;
    integer result_accepts, done_accepts, binary_fault_checks;
    integer source_increment;
    logic check_pending;

    function automatic integer expected_sources(input integer value);
        case (value)
            0: expected_sources = 20;
            1: expected_sources = 41;
            2: expected_sources = 90;
            3: expected_sources = 110;
            4: expected_sources = 0;
            default: expected_sources = -1;
        endcase
    endfunction

    function automatic integer expected_packets(input integer value);
        case (value)
            0: expected_packets = 1;
            1: expected_packets = 2;
            2: expected_packets = 4;
            3: expected_packets = 8;
            4: expected_packets = 1;
            default: expected_packets = -1;
        endcase
    endfunction

    task automatic check_fault_vector;
        begin
            if ($isunknown({protocol_error, numeric_overflow,
                    stale_response_seen, endpoint_fault}))
                $fatal(1, "M1831 mapped registered/public fault contains X/Z");
            if (protocol_error || numeric_overflow || stale_response_seen
                    || (|endpoint_fault))
                $fatal(1, "M1831 legal production workload raised fault");
            binary_fault_checks = binary_fault_checks + 1;
        end
    endtask

`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);
    ap_public_fault_binary:
        assert property (!$isunknown({protocol_error, numeric_overflow,
            stale_response_seen, endpoint_fault}));
    ap_registered_public_fault_zero:
        assert property (!(protocol_error || numeric_overflow
            || stale_response_seen || (|endpoint_fault)));
    cp_source_packet: cover property (raw_accept);
    cp_endpoint_accept: cover property (|mem_req_accept);
    cp_result_accept: cover property (result_accept);
    cp_done_accept: cover property (token_done_accept);
`endif

    initial begin
        if (!$value$plusargs("M979_CASE=%d", case_id))
            $fatal(1, "M1831 requires M979_CASE");
        accepted_sources = 0; source_packets = 0; endpoint_accepts = 0;
        result_accepts = 0; done_accepts = 0; binary_fault_checks = 0;
        check_pending = 1'b0;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            accepted_sources <= 0; source_packets <= 0;
            endpoint_accepts <= 0; result_accepts <= 0; done_accepts <= 0;
            binary_fault_checks <= 0; check_pending <= 1'b0;
        end else begin
            check_fault_vector();
            source_increment = 0;
            if (raw_accept) begin
                source_packets <= source_packets + 1;
                for (integer lane = 0; lane < 4; lane++)
                    if (raw_lane_valid[lane])
                        source_increment = source_increment
                            + $countones(raw_bitmap[383-lane*96-:96]);
            end
            accepted_sources <= accepted_sources + source_increment;
            endpoint_accepts <= endpoint_accepts + $countones(mem_req_accept);
            if (result_accept) result_accepts <= result_accepts + 1;
            if (token_done_accept) begin
                done_accepts <= done_accepts + 1;
                check_pending <= 1'b1;
            end
        end
    end

    always @(negedge clk_core) begin
        if (!rst_core) begin
            check_fault_vector();
            if (check_pending) begin
                if (case_id < 0 || case_id > 4)
                    $fatal(1, "M1831 case outside 0..4");
                if (accepted_sources != expected_sources(case_id))
                    $fatal(1, "M1831 accepted-source count mismatch");
                if (source_packets != expected_packets(case_id))
                    $fatal(1, "M1831 source-packet count mismatch");
                if ((case_id < 4 && endpoint_accepts == 0)
                        || (case_id == 4 && endpoint_accepts != 0))
                    $fatal(1, "M1831 endpoint activity mismatch");
                if (result_accepts == 0 || done_accepts != 1
                        || binary_fault_checks == 0)
                    $fatal(1, "M1831 production coverage incomplete");
                $display("PASS M1831 registered-fault production case=%0d accepted_sources=%0d source_packets=%0d endpoint_accepts=%0d result_accepts=%0d done_accepts=%0d fault_binary_clean=1 registered_fault_public_zero=1",
                    case_id, accepted_sources, source_packets,
                    endpoint_accepts, result_accepts, done_accepts);
                check_pending <= 1'b0;
            end
        end
    end

    initial begin #1000000; $fatal(1, "M1831 assertion watchdog"); end
endmodule

`default_nettype wire
