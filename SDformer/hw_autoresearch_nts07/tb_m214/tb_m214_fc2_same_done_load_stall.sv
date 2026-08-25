`timescale 1ns/1ps
`default_nettype none

// Directed causal test for the M214-only path.  The final nonzero descriptor
// arrives in the last raw packet, so M212's terminal hint closes a lone stage-1
// window.  M214 must load its first group on the accepted done-fence edge and
// then hold that group exactly while the consumer stalls.
module tb_m214_fc2_same_done_load_stall;
    logic clk_core = 0, rst_core;
    always #1.5 clk_core = ~clk_core;

    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_ready, raw_accept, raw_last;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
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
    int same_done_loads, group_stalls, groups, done_count;

    m214_fc2_raw4_to_same_done_load_frontend dut (.*);

    task automatic send_raw_packet(input int base, input logic last,
            input logic event_in_lane3);
        logic accepted;
        begin
            @(negedge clk_core);
            raw_valid = 1;
            raw_last = last;
            raw_lane_valid = 4'b1111;
            for (int lane = 0; lane < 4; lane++) begin
                raw_beat_index[lane] = base + lane;
                raw_bitmap[lane] = 0;
            end
            if (event_in_lane3)
                raw_bitmap[3][0] = 1;
            do begin
                @(posedge clk_core);
                accepted = raw_accept;
            end while (!accepted);
            @(negedge clk_core);
            raw_valid = 0;
            raw_last = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (protocol_error)
                $fatal(1, "M214 same-done stall protocol error");
            if (dut.paired_sink.same_cycle_done_load)
                same_done_loads++;
            if (group_valid && !group_ready)
                group_stalls++;
            if (group_accept) begin
                if (group_tag != 24'h214001
                        || group_output_block != groups
                        || group_source_count != 1
                        || group_bank_valid != 8'b00000001
                        || group_source_channel[0] != 12'd672)
                    $fatal(1, "M214 same-done group identity mismatch");
                groups++;
            end
            if (token_done_accept) begin
                if (token_done_tag != 24'h214001
                        || token_done_descriptor_count != 1
                        || !token_done_had_event)
                    $fatal(1, "M214 same-done completion mismatch");
                done_count++;
            end
        end
    end

    initial begin
        rst_core = 1;
        header_valid = 0;
        raw_valid = 0;
        raw_lane_valid = 0;
        raw_last = 0;
        group_ready = 0;
        token_done_ready = 1;
        same_done_loads = 0;
        group_stalls = 0;
        groups = 0;
        done_count = 0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;
        header_tag = 24'h214001;
        header_raw_beat_count = 8;
        header_window_depth = 4;
        header_output_blocks = 2;
        header_valid = 1;
        do @(posedge clk_core); while (!header_accept);
        @(negedge clk_core);
        header_valid = 0;

        send_raw_packet(0, 0, 0);
        send_raw_packet(4, 1, 1);

        do @(posedge clk_core);
        while (!dut.paired_sink.same_cycle_done_load);
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        group_ready = 1;
        do @(posedge clk_core); while (!token_done_accept);
        @(negedge clk_core);

        if (same_done_loads != 1 || groups != 2 || done_count != 1
                || group_stalls < 2)
            $fatal(1, "M214 same-done causal coverage mismatch");
        $display("PASS M214 same-cycle done-fence load stall VCS same_done_loads=%0d groups=%0d done=%0d group_stalls=%0d identity_mismatches=0 complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            same_done_loads, groups, done_count, group_stalls);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "M214 same-done stall watchdog");
    end
endmodule

`default_nettype wire
