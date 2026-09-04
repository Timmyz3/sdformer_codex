`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_source_streamer;
    localparam int TILE_BITS = 64;
    localparam int TAG_W = 16;
    localparam int INDEX_W = $clog2(TILE_BITS);
    localparam int COUNT_W = $clog2(TILE_BITS + 1);
    localparam int COMMANDS = 5000;

    logic clk_core = 1'b0;
    logic rst_core;
    logic command_valid, command_ready;
    logic [TAG_W-1:0] command_tag;
    logic command_use_motion;
    logic [TILE_BITS-1:0] command_current_bits, command_previous_bits;
    logic source_valid, source_ready;
    logic [TAG_W-1:0] source_tag;
    logic [INDEX_W-1:0] source_index;
    logic source_negative, source_use_motion, source_last;
    logic done_valid, done_ready;
    logic [TAG_W-1:0] done_tag;
    logic done_use_motion;
    logic [COUNT_W-1:0] done_source_count;
    logic [31:0] perf_commands, perf_local_commands, perf_motion_commands;
    logic [31:0] perf_sources, perf_positive_sources, perf_negative_sources;

    integer expected_index [0:TILE_BITS-1];
    bit expected_negative [0:TILE_BITS-1];
    integer expected_count;
    integer expected_head;
    integer total_sources;
    integer total_positive;
    integer total_negative;
    integer total_local;
    integer total_motion;
    integer cycles;

    always #5 clk_core = ~clk_core;

    qfit_dual_line_source_streamer #(
        .TILE_BITS(TILE_BITS), .TAG_W(TAG_W)
    ) dut (.*);

    qfit_dual_line_source_streamer_assertions #(
        .TAG_W(TAG_W), .INDEX_W(INDEX_W), .COUNT_W(COUNT_W)
    ) sva (.*);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycles <= cycles + 1;
            if (cycles > 1000000)
                $fatal(1, "source streamer timeout");
        end
    end

    task automatic prepare_expected(
        input bit use_motion,
        input logic [TILE_BITS-1:0] current_bits,
        input logic [TILE_BITS-1:0] previous_bits
    );
        logic [TILE_BITS-1:0] selected;
        begin
            selected = use_motion ? current_bits ^ previous_bits : current_bits;
            expected_count = 0;
            expected_head = 0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++) begin
                if (selected[bit_index]) begin
                    expected_index[expected_count] = bit_index;
                    expected_negative[expected_count] =
                        use_motion && previous_bits[bit_index] && !current_bits[bit_index];
                    expected_count++;
                end
            end
        end
    endtask

    task automatic run_command(input int command_number);
        bit use_motion;
        logic [TILE_BITS-1:0] current_bits;
        logic [TILE_BITS-1:0] previous_bits;
        begin
            use_motion = command_number[0];
            current_bits = {$urandom, $urandom};
            previous_bits = {$urandom, $urandom};
            if (command_number % 101 == 0) begin
                current_bits = '0;
                previous_bits = use_motion ? '0 : {$urandom, $urandom};
            end
            if (command_number % 137 == 0 && use_motion)
                previous_bits = current_bits;
            prepare_expected(use_motion, current_bits, previous_bits);

            @(negedge clk_core);
            command_valid = 1'b1;
            command_tag = TAG_W'(command_number);
            command_use_motion = use_motion;
            command_current_bits = current_bits;
            command_previous_bits = previous_bits;
            while (!command_ready)
                @(negedge clk_core);
            @(negedge clk_core);
            command_valid = 1'b0;

            while (!done_valid) begin
                source_ready = $urandom_range(0, 3) != 0;
                done_ready = $urandom_range(0, 3) != 0;
                @(posedge clk_core);
                if (source_valid && source_ready) begin
                    if (source_tag != TAG_W'(command_number))
                        $fatal(1, "source tag mismatch command=%0d", command_number);
                    if (expected_head >= expected_count)
                        $fatal(1, "unexpected source command=%0d", command_number);
                    if (source_index != INDEX_W'(expected_index[expected_head]))
                        $fatal(1, "source index mismatch command=%0d got=%0d exp=%0d",
                            command_number, source_index, expected_index[expected_head]);
                    if (source_negative != expected_negative[expected_head])
                        $fatal(1, "source sign mismatch command=%0d index=%0d",
                            command_number, source_index);
                    if (source_use_motion != use_motion)
                        $fatal(1, "source mode mismatch command=%0d", command_number);
                    if (source_last != (expected_head == expected_count - 1))
                        $fatal(1, "source last mismatch command=%0d", command_number);
                    total_sources++;
                    if (source_negative)
                        total_negative++;
                    else
                        total_positive++;
                    expected_head++;
                end
                @(negedge clk_core);
            end

            if (expected_head != expected_count)
                $fatal(1, "source count mismatch command=%0d got=%0d exp=%0d",
                    command_number, expected_head, expected_count);
            if (done_tag != TAG_W'(command_number)
                || done_use_motion != use_motion
                || done_source_count != COUNT_W'(expected_count))
                $fatal(1, "done mismatch command=%0d count=%0d/%0d",
                    command_number, done_source_count, expected_count);
            done_ready = 1'b1;
            @(negedge clk_core);
            while (done_valid)
                @(negedge clk_core);
            if (use_motion)
                total_motion++;
            else
                total_local++;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        command_valid = 1'b0;
        command_tag = '0;
        command_use_motion = 1'b0;
        command_current_bits = '0;
        command_previous_bits = '0;
        source_ready = 1'b0;
        done_ready = 1'b0;
        total_sources = 0;
        total_positive = 0;
        total_negative = 0;
        total_local = 0;
        total_motion = 0;
        cycles = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (int command_number = 0; command_number < COMMANDS; command_number++)
            run_command(command_number);

        if (perf_commands != COMMANDS
            || perf_local_commands != total_local
            || perf_motion_commands != total_motion
            || perf_sources != total_sources
            || perf_positive_sources != total_positive
            || perf_negative_sources != total_negative)
            $fatal(1, "performance counter mismatch");
        $display("PASS dual-line source streamer commands=%0d cycles=%0d local=%0d motion=%0d sources=%0d positive=%0d negative=%0d",
            COMMANDS, cycles, total_local, total_motion, total_sources,
            total_positive, total_negative);
        $finish;
    end
endmodule

`default_nettype wire
