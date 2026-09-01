`timescale 1ns/1ps
`default_nettype none

// M1613 source-only directed stimulus for the additive M1609 successor.
// It deliberately targets the compactor boundary so the public fault latency
// is isolated from M216/service error OR chains.  No performance is measured.
module tb_m1613_c2_m1609_registered_fault_directed;
    logic clk_core = 0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic header_valid, header_ready, header_accept;
    logic [23:0] header_token_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth;
    logic raw_valid, raw_ready, raw_accept;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic raw_last;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [2:0] descriptor_count;
    logic [23:0] descriptor_token_tag;
    logic [4:0] descriptor_beat_index [0:3];
    logic [95:0] descriptor_bitmap [0:3];
    logic [3:0] descriptor_window_last;
    logic descriptor_token_last;
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic [5:0] token_done_descriptor_count;
    logic protocol_error, busy;

    int legal_terminal_no_false_pulse;
    int legal_descriptor_accepts;
    int illegal_header_latched;
    int illegal_raw_latched;
    int sticky_checks;

    m214_fc2_raw4_to_descriptor4_terminal_hint_compactor dut (.*);

    task automatic drive_idle;
        begin
            header_valid = 0;
            header_token_tag = 0;
            header_raw_beat_count = 0;
            header_window_depth = 0;
            raw_valid = 0;
            raw_lane_valid = 0;
            raw_last = 0;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = 0;
                raw_bitmap[lane] = 0;
            end
            descriptor_ready = 1;
            token_done_ready = 1;
        end
    endtask

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            drive_idle();
            rst_core = 1;
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            @(posedge clk_core);
            #1ps;
            if (protocol_error !== 0 || dut.fault_q !== 0)
                $fatal(1, "M1613 reset failed to clear registered fault");
        end
    endtask

    task automatic send_legal_header;
        begin
            @(negedge clk_core);
            header_token_tag = 24'h161301;
            header_raw_beat_count = 6'd4;
            header_window_depth = 4'd2;
            header_valid = 1;
            #1ps;
            if (header_ready !== 1 || header_accept !== 1
                    || protocol_error !== 0)
                $fatal(1, "M1613 legal header was not ready/accepted");
            @(posedge clk_core);
            #1ps;
            if (protocol_error !== 0)
                $fatal(1, "M1613 legal header raised registered fault");
            @(negedge clk_core);
            header_valid = 0;
        end
    endtask

    task automatic legal_terminal_linger_case;
        begin
            apply_reset();
            send_legal_header();
            @(negedge clk_core);
            raw_lane_valid = 4'b1111;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = lane;
                raw_bitmap[lane] = 0;
            end
            raw_bitmap[0][5] = 1;
            raw_last = 1;
            raw_valid = 1;
            #1ps;
            if (raw_ready !== 1 || raw_accept !== 1
                    || protocol_error !== 0)
                $fatal(1, "M1613 legal terminal packet was not accepted");
            if (descriptor_accept !== 1)
                $fatal(1, "M1613 legal terminal descriptor did not bypass");
            legal_descriptor_accepts++;
            @(posedge clk_core);
            #1ps;
            // raw_valid intentionally lingers until the following negedge, as
            // in the frozen directed driver.  After the terminal state update
            // this makes the combinational illegal_request high; M1609 must not
            // expose that post-accept linger as a false public error pulse.
            if (dut.illegal_request !== 1)
                $fatal(1, "M1613 did not exercise post-terminal linger seam");
            if (protocol_error !== 0 || dut.fault_q !== 0)
                $fatal(1, "M1613 legal terminal produced a false fault pulse");
            legal_terminal_no_false_pulse++;
            @(negedge clk_core);
            if (protocol_error !== 0 || token_done_accept !== 1)
                $fatal(1, "M1613 legal terminal completion mismatch");
            raw_valid = 0;
            raw_lane_valid = 0;
            raw_last = 0;
            @(posedge clk_core);
            #1ps;
            if (protocol_error !== 0)
                $fatal(1, "M1613 legal completion polluted sticky fault");
        end
    endtask

    task automatic illegal_header_case;
        begin
            apply_reset();
            @(negedge clk_core);
            header_token_tag = 24'hbad131;
            header_raw_beat_count = 0;
            header_window_depth = 3;
            header_valid = 1;
            #1ps;
            if (dut.illegal_request !== 1 || header_ready !== 0
                    || header_accept !== 0 || protocol_error !== 0)
                $fatal(1, "M1613 illegal header pre-edge contract mismatch");
            @(posedge clk_core);
            #1ps;
            if (dut.fault_q !== 1 || protocol_error !== 1)
                $fatal(1, "M1613 illegal header did not latch after edge");
            illegal_header_latched++;
            @(negedge clk_core);
            header_valid = 0;
            repeat (2) begin
                @(posedge clk_core);
                #1ps;
                if (protocol_error !== 1) $fatal(1, "M1613 fault not sticky");
                sticky_checks++;
            end
        end
    endtask

    task automatic illegal_raw_case;
        begin
            apply_reset();
            send_legal_header();
            @(negedge clk_core);
            raw_lane_valid = 4'b1011;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = lane;
                raw_bitmap[lane] = 0;
            end
            raw_bitmap[0][9] = 1;
            raw_last = 0;
            raw_valid = 1;
            #1ps;
            if (dut.illegal_request !== 1 || raw_ready !== 0
                    || raw_accept !== 0 || protocol_error !== 0)
                $fatal(1, "M1613 illegal raw pre-edge contract mismatch");
            @(posedge clk_core);
            #1ps;
            if (dut.fault_q !== 1 || protocol_error !== 1)
                $fatal(1, "M1613 illegal raw did not latch after edge");
            illegal_raw_latched++;
            @(negedge clk_core);
            raw_valid = 0;
            raw_lane_valid = 0;
            @(posedge clk_core);
            #1ps;
            if (protocol_error !== 1) $fatal(1, "M1613 raw fault not sticky");
            sticky_checks++;
        end
    endtask

    property p_header_accept_requires_ready;
        @(posedge clk_core) disable iff (rst_core)
        header_accept |-> header_ready;
    endproperty
    ap_header_accept_requires_ready: assert property (
        p_header_accept_requires_ready);

    property p_raw_accept_requires_ready;
        @(posedge clk_core) disable iff (rst_core)
        raw_accept |-> raw_ready;
    endproperty
    ap_raw_accept_requires_ready: assert property (
        p_raw_accept_requires_ready);

    property p_registered_fault_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error |=> protocol_error;
    endproperty
    ap_registered_fault_sticky: assert property (p_registered_fault_sticky);

    initial begin
        rst_core = 1;
        legal_terminal_no_false_pulse = 0;
        legal_descriptor_accepts = 0;
        illegal_header_latched = 0;
        illegal_raw_latched = 0;
        sticky_checks = 0;
        drive_idle();
        legal_terminal_linger_case();
        illegal_header_case();
        illegal_raw_case();
        if (legal_terminal_no_false_pulse != 1
                || legal_descriptor_accepts != 1
                || illegal_header_latched != 1
                || illegal_raw_latched != 1
                || sticky_checks != 3)
            $fatal(1, "M1613 required directed coverage missing");
        $display("PASS M1613 M1609 registered-fault directed legal_terminal_no_false_pulse=%0d legal_descriptor_accepts=%0d illegal_header_latched=%0d illegal_raw_latched=%0d sticky_checks=%0d source_only=false performance=false",
            legal_terminal_no_false_pulse, legal_descriptor_accepts,
            illegal_header_latched, illegal_raw_latched, sticky_checks);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "M1613 watchdog timeout");
    end
endmodule

`default_nettype wire
