`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_tile_executor;
    localparam int TILE_BITS = 32;
    localparam int OUT_LANES = 8;
    localparam int TAG_W = 16;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int INDEX_W = $clog2(TILE_BITS);
    localparam int OUT_W = $clog2(OUT_LANES);
    localparam int COUNT_W = $clog2(TILE_BITS + 1);
    localparam int COMMANDS = 2000;

    logic clk_core = 1'b0;
    logic rst_core;
    logic weight_epoch_clear;
    logic weight_valid, weight_ready;
    logic [INDEX_W-1:0] weight_source;
    logic [OUT_W-1:0] weight_lane;
    logic signed [W_W-1:0] weight_data;
    logic weight_last, weights_loaded;
    logic command_valid, command_ready;
    logic [TAG_W-1:0] command_tag;
    logic command_use_motion;
    logic [TILE_BITS-1:0] command_current_bits, command_previous_bits;
    logic [OUT_LANES*ACC_W-1:0] command_seed_acc;
    logic output_valid, output_ready;
    logic [TAG_W-1:0] output_tag;
    logic output_use_motion;
    logic [COUNT_W-1:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic protocol_error;
    logic [31:0] perf_commands, perf_local_commands, perf_motion_commands;
    logic [31:0] perf_weight_segment_reads, perf_accumulator_updates;
    logic [31:0] perf_positive_sources, perf_negative_sources;

    integer weight_ref [0:TILE_BITS-1][0:OUT_LANES-1];
    integer expected_acc [0:OUT_LANES-1];
    integer base_acc [0:OUT_LANES-1];
    integer expected_sources;
    integer total_sources;
    integer total_positive;
    integer total_negative;
    integer total_local;
    integer total_motion;
    integer cycles;

    always #5 clk_core = ~clk_core;

    qfit_dual_line_tile_executor #(
        .TILE_BITS(TILE_BITS), .OUT_LANES(OUT_LANES),
        .TAG_W(TAG_W), .W_W(W_W), .ACC_W(ACC_W)
    ) dut (.*);

    qfit_dual_line_tile_executor_assertions #(
        .TAG_W(TAG_W), .COUNT_W(COUNT_W),
        .OUTPUT_W(OUT_LANES*ACC_W)
    ) sva (.*);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycles <= cycles + 1;
            if (cycles > 500000)
                $fatal(1, "dual-line tile executor timeout");
        end
    end

    function automatic integer make_weight(input integer source, input integer lane);
        make_weight = ((source * 13 + lane * 7 + 3) % 31) - 15;
    endfunction

    task automatic load_weights;
        begin
            for (int source = 0; source < TILE_BITS; source++) begin
                for (int lane = 0; lane < OUT_LANES; lane++) begin
                    weight_ref[source][lane] = make_weight(source, lane);
                    @(negedge clk_core);
                    weight_valid = 1'b1;
                    weight_source = INDEX_W'(source);
                    weight_lane = OUT_W'(lane);
                    weight_data = W_W'(weight_ref[source][lane]);
                    weight_last = source == TILE_BITS - 1 && lane == OUT_LANES - 1;
                    while (!weight_ready)
                        @(negedge clk_core);
                    @(negedge clk_core);
                    weight_valid = 1'b0;
                end
            end
            if (!weights_loaded || protocol_error)
                $fatal(1, "ordered weight load failed");
        end
    endtask

    task automatic run_command(input integer command_number);
        bit use_motion;
        logic [TILE_BITS-1:0] current_bits;
        logic [TILE_BITS-1:0] previous_bits;
        integer source_count;
        integer negative_count;
        begin
            use_motion = command_number[0];
            current_bits = $urandom;
            previous_bits = $urandom;
            if (command_number % 101 == 0) begin
                current_bits = '0;
                previous_bits = use_motion ? '0 : $urandom;
            end
            if (command_number % 137 == 1 && use_motion)
                previous_bits = current_bits;

            source_count = 0;
            negative_count = 0;
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                base_acc[lane] = (command_number * 5 + lane * 11) % 257 - 128;
                expected_acc[lane] = base_acc[lane];
                for (int source = 0; source < TILE_BITS; source++) begin
                    if (current_bits[source])
                        expected_acc[lane] += weight_ref[source][lane];
                end
            end
            if (use_motion) begin
                for (int lane = 0; lane < OUT_LANES; lane++) begin
                    integer previous_acc;
                    previous_acc = base_acc[lane];
                    for (int source = 0; source < TILE_BITS; source++) begin
                        if (previous_bits[source])
                            previous_acc += weight_ref[source][lane];
                    end
                    command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(previous_acc);
                end
            end else begin
                for (int lane = 0; lane < OUT_LANES; lane++)
                    command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(base_acc[lane]);
            end
            for (int source = 0; source < TILE_BITS; source++) begin
                if (use_motion ? current_bits[source] ^ previous_bits[source]
                               : current_bits[source]) begin
                    source_count++;
                    if (use_motion && previous_bits[source] && !current_bits[source])
                        negative_count++;
                end
            end

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

            output_ready = $urandom_range(0, 1);
            while (!output_valid) begin
                output_ready = $urandom_range(0, 3) != 0;
                @(negedge clk_core);
            end
            if (output_tag != TAG_W'(command_number)
                || output_use_motion != use_motion
                || output_source_count != COUNT_W'(source_count))
                $fatal(1, "output metadata mismatch command=%0d", command_number);
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                if ($signed(output_acc[lane*ACC_W +: ACC_W]) != expected_acc[lane])
                    $fatal(1, "output mismatch command=%0d lane=%0d got=%0d exp=%0d",
                        command_number, lane,
                        $signed(output_acc[lane*ACC_W +: ACC_W]), expected_acc[lane]);
            end
            output_ready = 1'b1;
            @(negedge clk_core);
            while (output_valid)
                @(negedge clk_core);

            total_sources += source_count;
            total_negative += negative_count;
            total_positive += source_count - negative_count;
            if (use_motion)
                total_motion++;
            else
                total_local++;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        weight_epoch_clear = 1'b0;
        weight_valid = 1'b0;
        weight_source = '0;
        weight_lane = '0;
        weight_data = '0;
        weight_last = 1'b0;
        command_valid = 1'b0;
        command_tag = '0;
        command_use_motion = 1'b0;
        command_current_bits = '0;
        command_previous_bits = '0;
        command_seed_acc = '0;
        output_ready = 1'b0;
        total_sources = 0;
        total_positive = 0;
        total_negative = 0;
        total_local = 0;
        total_motion = 0;
        cycles = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        load_weights();
        for (int command_number = 0; command_number < COMMANDS; command_number++)
            run_command(command_number);

        if (protocol_error
            || perf_commands != COMMANDS
            || perf_local_commands != total_local
            || perf_motion_commands != total_motion
            || perf_weight_segment_reads != total_sources
            || perf_accumulator_updates != total_sources * OUT_LANES
            || perf_positive_sources != total_positive
            || perf_negative_sources != total_negative)
            $fatal(1, "final protocol/performance counter mismatch");
        $display("PASS dual-line tile executor commands=%0d cycles=%0d local=%0d motion=%0d sources=%0d positive=%0d negative=%0d accumulator_updates=%0d",
            COMMANDS, cycles, total_local, total_motion, total_sources,
            total_positive, total_negative, perf_accumulator_updates);
        $finish;
    end
endmodule

`default_nettype wire
