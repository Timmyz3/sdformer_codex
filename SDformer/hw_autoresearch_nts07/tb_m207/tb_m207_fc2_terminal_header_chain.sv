`timescale 1ns/1ps
`default_nettype none

module tb_m207_fc2_terminal_header_chain;
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
    int first_groups, second_groups, done_count, chain_count;

    m207_fc2_raw4_to_paired_window_terminal_collapse_frontend dut (.*);

    function automatic logic [95:0] pattern(input int beat, input int seed);
        logic [95:0] value;
        begin
            value = 0;
            value[((beat*3+seed+1)%12)*8+((beat+seed)%8)] = 1;
            value[((beat*7+seed+2)%12)*8+((beat*5+seed+3)%8)] = 1;
            if ((beat+seed)%3 == 0)
                value[((beat*11+seed+5)%12)*8
                    +((beat*3+seed+6)%8)] = 1;
            return value;
        end
    endfunction

    task automatic set_header(input logic [23:0] tag);
        begin
            header_tag = tag; header_raw_beat_count = 8;
            header_window_depth = 4; header_output_blocks = 2;
            header_valid = 1;
        end
    endtask

    task automatic drive_dense_packet(
        input int base, input int seed, input logic is_last
    );
        begin
            raw_lane_valid = 4'b1111; raw_last = is_last;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = base + lane;
                raw_bitmap[lane] = pattern(base + lane, seed);
            end
            raw_valid = 1;
        end
    endtask

    task automatic send_two_dense_packets(input int seed);
        begin
            @(negedge clk_core); drive_dense_packet(0, seed, 0);
            do @(posedge clk_core); while (!raw_accept);
            @(negedge clk_core); drive_dense_packet(4, seed, 1);
            do @(posedge clk_core); while (!raw_accept);
            @(negedge clk_core); raw_valid = 0; raw_last = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (protocol_error) $fatal(1, "M207 chain protocol error");
            if (group_accept) begin
                if (group_tag == 24'h207101) first_groups++;
                else if (group_tag == 24'h207102) second_groups++;
                else $fatal(1, "M207 chain unknown group tag");
            end
            if (token_done_accept) begin
                if (token_done_descriptor_count != 8
                        || !token_done_had_event)
                    $fatal(1, "M207 chain done metadata mismatch");
                if (done_count == 0) begin
                    if (token_done_tag != 24'h207101 || !header_accept
                            || header_tag != 24'h207102)
                        $fatal(1, "M207 terminal header did not chain");
                    chain_count++;
                end else if (token_done_tag != 24'h207102)
                    $fatal(1, "M207 second done tag mismatch");
                done_count++;
            end
        end
    end

    initial begin
        rst_core = 1; header_valid = 0; raw_valid = 0;
        raw_lane_valid = 0; raw_last = 0; group_ready = 1;
        token_done_ready = 1; first_groups = 0; second_groups = 0;
        done_count = 0; chain_count = 0;
        repeat (3) @(posedge clk_core); @(negedge clk_core); rst_core = 0;

        set_header(24'h207101);
        do @(posedge clk_core); while (!header_accept);
        @(negedge clk_core); header_valid = 0;
        send_two_dense_packets(1);

        // Present the next token while the first pair is still replaying.
        // It must be accepted atomically with the terminal group/done beat.
        @(negedge clk_core); set_header(24'h207102);
        do @(posedge clk_core); while (!header_accept);
        if (!token_done_accept)
            $fatal(1, "M207 next header accepted outside terminal done");
        @(negedge clk_core); header_valid = 0;
        send_two_dense_packets(4);
        do @(posedge clk_core); while (done_count != 2);
        @(negedge clk_core);
        if (first_groups != 6 || second_groups != 6 || chain_count != 1)
            $fatal(1, "M207 chained group/done census mismatch");
        $display("PASS M207 terminal header chain VCS first_groups=%0d second_groups=%0d done=%0d chains=%0d complete_fc2=false system_speedup=false headline=false",
            first_groups, second_groups, done_count, chain_count);
        $finish;
    end
    initial begin #1000000; $fatal(1, "M207 chain watchdog"); end
endmodule

`default_nettype wire
