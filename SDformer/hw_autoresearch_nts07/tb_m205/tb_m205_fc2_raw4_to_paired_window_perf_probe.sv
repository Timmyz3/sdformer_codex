`timescale 1ns/1ps
`default_nettype none

// Non-sealed throughput probe: drives consecutive raw4 packets without the
// task-level bubbles used by the directed protocol-stress testbench.
module tb_m205_fc2_raw4_to_paired_window_perf_probe;
    logic clk_core = 0, rst_core;
    always #1.5 clk_core = ~clk_core;
    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_ready, raw_accept;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic raw_last;
    logic group_valid, group_ready, group_accept;
    logic [23:0] group_tag;
    logic [2:0] group_output_block;
    logic [3:0] group_source_count;
    logic [7:0] group_bank_valid;
    logic [11:0] group_source_channel [0:7];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic [5:0] token_done_descriptor_count;
    logic token_done_had_event, protocol_error, busy;
    logic [95:0] payload [0:31];
    int total_groups, total_raw_backpressure;

    m205_fc2_raw4_to_paired_window_frontend dut (.*);

    function automatic logic [95:0] event_pattern(
        input int beat, input int seed
    );
        logic [95:0] value;
        int bank0, bank1, bank2, row0, row1, row2;
        begin
            value = 0;
            bank0 = (beat + seed) % 8;
            bank1 = (beat * 5 + seed + 3) % 8;
            bank2 = (beat * 3 + seed + 6) % 8;
            row0 = (beat * 3 + seed + 1) % 12;
            row1 = (beat * 7 + seed + 2) % 12;
            row2 = (beat * 11 + seed + 5) % 12;
            value[row0*8+bank0] = 1;
            value[row1*8+bank1] = 1;
            if ((beat + seed) % 3 == 0)
                value[row2*8+bank2] = 1;
            return value;
        end
    endfunction

    task automatic derive_shape(
        input int blocks, output int raw_count, output int depth
    );
        begin
            case (blocks)
                1: begin raw_count = 4; depth = 2; end
                2: begin raw_count = 8; depth = 4; end
                4: begin raw_count = 16; depth = 8; end
                8: begin raw_count = 32; depth = 8; end
                default: $fatal(1, "bad performance-probe shape");
            endcase
        end
    endtask

    task automatic drive_packet(input int base, input int raw_count);
        begin
            raw_lane_valid = 4'b1111;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = base + lane;
                raw_bitmap[lane] = payload[base + lane];
            end
            raw_last = base + 4 == raw_count;
            raw_valid = 1;
        end
    endtask

    task automatic run_token(
        input logic [23:0] tag, input int blocks,
        input int mode, input int seed, input int expected_groups
    );
        int raw_count, depth, base, group_start, backpressure_start;
        logic accepted_snapshot;
        time start_time;
        begin
            derive_shape(blocks, raw_count, depth);
            for (int beat = 0; beat < raw_count; beat++) begin
                case (mode)
                    0: payload[beat] = event_pattern(beat, seed);
                    1: payload[beat] = ((beat + seed) % 3 == 0)
                        ? 0 : event_pattern(beat, seed);
                    default: payload[beat] = 0;
                endcase
            end
            @(negedge clk_core);
            header_tag = tag; header_raw_beat_count = raw_count;
            header_window_depth = depth; header_output_blocks = blocks;
            header_valid = 1;
            do @(posedge clk_core); while (!header_accept);
            start_time = $time; group_start = total_groups;
            backpressure_start = total_raw_backpressure;
            @(negedge clk_core); header_valid = 0;
            base = 0; drive_packet(base, raw_count);
            while (base < raw_count) begin
                @(posedge clk_core); accepted_snapshot = raw_accept;
                @(negedge clk_core);
                if (accepted_snapshot) begin
                    base += 4;
                    if (base < raw_count) drive_packet(base, raw_count);
                    else begin raw_valid = 0; raw_last = 0; end
                end
            end
            do @(posedge clk_core); while (!token_done_accept);
            if (total_groups - group_start != expected_groups)
                $fatal(1, "M205 performance-probe group mismatch");
            $display("M205 PERF tag=%h blocks=%0d groups=%0d header_to_done_cycles=%0d raw_backpressure=%0d",
                tag, blocks, expected_groups, ($time-start_time)/3,
                total_raw_backpressure-backpressure_start);
            @(negedge clk_core);
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (group_accept) total_groups++;
            if (raw_valid && !raw_ready) total_raw_backpressure++;
            if (protocol_error) $fatal(1, "M205 performance-probe protocol error");
        end
    end

    initial begin
        rst_core = 1; header_valid = 0; raw_valid = 0; raw_lane_valid = 0;
        raw_last = 0; group_ready = 1; token_done_ready = 1;
        total_groups = 0; total_raw_backpressure = 0;
        repeat (3) @(posedge clk_core); @(negedge clk_core); rst_core = 0;
        run_token(24'h206001, 2, 0, 1, 6);
        run_token(24'h206002, 1, 1, 2, 2);
        run_token(24'h206003, 4, 1, 3, 16);
        run_token(24'h206004, 8, 0, 4, 80);
        run_token(24'h206005, 2, 2, 5, 0);
        $display("PASS M205 continuous-source throughput probe groups=%0d raw_backpressure=%0d sealed=false headline=false",
            total_groups, total_raw_backpressure);
        $finish;
    end
    initial begin #1000000; $fatal(1, "M205 performance-probe watchdog"); end
endmodule

`default_nettype wire
