`timescale 1ns/1ps
`default_nettype none

module tb_m166_q8_prefolded_rank3_left_atlif_backend;
    localparam int MAX_EXPECTED_BEATS = 1400;

    logic clk_core;
    logic rst_core;
    logic config_valid;
    logic config_ready;
    logic signed [7:0] config_folded_left [0:9][0:2][0:15];
    logic signed [23:0] config_folded_bias [0:9][0:15];
    logic signed [23:0] config_threshold;
    logic config_accept;
    logic rank_valid;
    logic rank_ready;
    logic [15:0] rank_tag;
    logic rank_channel_last;
    logic signed [7:0] rank_data [0:2][0:15];
    logic rank_accept;
    logic event_valid;
    logic event_ready;
    logic [15:0] event_tag;
    logic event_channel_last;
    logic [2:0] event_beat;
    logic [31:0] event_bits;
    logic event_accept;
    logic configured;
    logic protocol_error;
    logic busy;

    logic random_stall_mode;
    logic throughput_measure_mode;
    integer cycle_count;
    integer expected_write;
    integer expected_read;
    integer accepted_tiles;
    integer accepted_beats;
    integer output_stall_cycles;
    integer input_push_release_overlap_cycles;
    integer five_cycle_ii_hits;
    integer previous_tile_start_cycle;
    integer mixed_event_words;
    logic [15:0] expected_tag [0:MAX_EXPECTED_BEATS-1];
    logic expected_last [0:MAX_EXPECTED_BEATS-1];
    logic [2:0] expected_beat [0:MAX_EXPECTED_BEATS-1];
    logic [31:0] expected_bits [0:MAX_EXPECTED_BEATS-1];

    m166_q8_prefolded_rank3_left_atlif_backend dut (.*);

    bind m166_q8_prefolded_rank3_left_atlif_backend
        m166_q8_prefolded_rank3_left_atlif_backend_assertions sva (
            .clk_core(clk_core),
            .rst_core(rst_core),
            .config_valid(config_valid),
            .config_ready(config_ready),
            .config_accept(config_accept),
            .rank_valid(rank_valid),
            .rank_ready(rank_ready),
            .rank_accept(rank_accept),
            .event_valid(event_valid),
            .event_ready(event_ready),
            .event_tag(event_tag),
            .event_channel_last(event_channel_last),
            .event_beat(event_beat),
            .event_bits(event_bits),
            .event_accept(event_accept),
            .configured(configured),
            .protocol_error(protocol_error),
            .busy(busy),
            .service_active_internal(service_active_q),
            .service_phase_internal(service_phase_q),
            .input_count_internal(input_count_q),
            .output_count_internal(output_count_q),
            .input_push_internal(input_push),
            .input_release_internal(input_release),
            .output_push_internal(output_push)
        );

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    task automatic append_reference(
        input logic [15:0] tag_value,
        input logic last_value
    );
        integer reconstructed;
        logic [31:0] packed_word;
        begin
            for (int beat = 0; beat < 5; beat++) begin
                packed_word = '0;
                for (int row = 0; row < 2; row++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        reconstructed = $signed(
                            config_folded_bias[(beat*2)+row][lane]);
                        for (int rank = 0; rank < 3; rank++) begin
                            reconstructed = reconstructed
                                + $signed(rank_data[rank][lane])
                                * $signed(config_folded_left[(beat*2)+row]
                                    [rank][lane]);
                        end
                        packed_word[(row*16)+lane]
                            = reconstructed >= $signed(config_threshold);
                    end
                end
                if (expected_write >= MAX_EXPECTED_BEATS)
                    $fatal(1, "M166 expected scoreboard overflow");
                expected_tag[expected_write] = tag_value;
                expected_last[expected_write] = last_value;
                expected_beat[expected_write] = beat[2:0];
                expected_bits[expected_write] = packed_word;
                expected_write = expected_write + 1;
            end
        end
    endtask

    task automatic load_tile(input integer tile_index, input integer tag_base);
        begin
            rank_tag = tag_base + tile_index;
            rank_channel_last = ((tile_index % 17) == 16);
            for (int rank = 0; rank < 3; rank++) begin
                for (int lane = 0; lane < 16; lane++) begin
                    rank_data[rank][lane]
                        = (($urandom % 129) - 64);
                end
            end
            // Deterministic rails guarantee both sides of the comparator even
            // if a simulator changes the random sequence.
            rank_data[0][0] = 8'sd127;
            rank_data[1][0] = 8'sd127;
            rank_data[2][0] = 8'sd127;
            rank_data[0][1] = -8'sd128;
            rank_data[1][1] = -8'sd128;
            rank_data[2][1] = -8'sd128;
            append_reference(rank_tag, rank_channel_last);
        end
    endtask

    task automatic drive_tiles(input integer tile_count, input integer tag_base);
        integer sent;
        begin
            sent = 0;
            @(negedge clk_core);
            load_tile(sent, tag_base);
            rank_valid = 1'b1;
            while (sent < tile_count) begin
                @(posedge clk_core);
                if (rank_accept) begin
                    sent = sent + 1;
                    if (sent < tile_count) begin
                        @(negedge clk_core);
                        load_tile(sent, tag_base);
                    end else begin
                        @(negedge clk_core);
                        rank_valid = 1'b0;
                    end
                end
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            event_ready <= 1'b0;
        else if (random_stall_mode)
            event_ready <= ($urandom_range(0, 3) != 0);
        else
            event_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            accepted_beats <= 0;
            output_stall_cycles <= 0;
            input_push_release_overlap_cycles <= 0;
            five_cycle_ii_hits <= 0;
            previous_tile_start_cycle <= -1;
            mixed_event_words <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (rank_accept)
                accepted_tiles <= accepted_tiles + 1;
            if (event_valid && !event_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (dut.input_push && dut.input_release)
                input_push_release_overlap_cycles
                    <= input_push_release_overlap_cycles + 1;
            if (dut.output_push && dut.service_phase_q == 0
                    && throughput_measure_mode) begin
                if (previous_tile_start_cycle >= 0) begin
                    if (cycle_count - previous_tile_start_cycle != 5)
                        $fatal(1, "M166 steady service II drift got=%0d",
                            cycle_count - previous_tile_start_cycle);
                    five_cycle_ii_hits <= five_cycle_ii_hits + 1;
                end
                previous_tile_start_cycle <= cycle_count;
            end
            if (event_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M166 unexpected output beat");
                if (event_tag !== expected_tag[expected_read]
                        || event_channel_last !== expected_last[expected_read]
                        || event_beat !== expected_beat[expected_read]
                        || event_bits !== expected_bits[expected_read]) begin
                    $fatal(1, "M166 mismatch index=%0d tag=%h/%h beat=%0d/%0d bits=%h/%h",
                        expected_read, event_tag, expected_tag[expected_read],
                        event_beat, expected_beat[expected_read], event_bits,
                        expected_bits[expected_read]);
                end
                if (event_bits != 0 && event_bits != 32'hffff_ffff)
                    mixed_event_words <= mixed_event_words + 1;
                expected_read <= expected_read + 1;
                accepted_beats <= accepted_beats + 1;
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        config_valid = 1'b0;
        config_threshold = 24'sd0;
        rank_valid = 1'b0;
        rank_tag = '0;
        rank_channel_last = 1'b0;
        event_ready = 1'b0;
        random_stall_mode = 1'b0;
        throughput_measure_mode = 1'b0;
        cycle_count = 0;
        expected_write = 0;
        expected_read = 0;
        accepted_tiles = 0;
        accepted_beats = 0;
        output_stall_cycles = 0;
        input_push_release_overlap_cycles = 0;
        five_cycle_ii_hits = 0;
        previous_tile_start_cycle = -1;
        mixed_event_words = 0;
        for (int time_index = 0; time_index < 10; time_index++) begin
            for (int lane = 0; lane < 16; lane++) begin
                config_folded_bias[time_index][lane]
                    = ((time_index*17 + lane*3) % 61) - 30;
                for (int rank = 0; rank < 3; rank++) begin
                    config_folded_left[time_index][rank][lane]
                        = ((time_index*11 + rank*5 + lane*7) % 17) - 8;
                end
            end
        end
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        config_valid = 1'b1;
        do @(posedge clk_core); while (!config_accept);
        @(negedge clk_core);
        config_valid = 1'b0;
        if (!configured)
            $fatal(1, "M166 configuration did not commit");

        throughput_measure_mode = 1'b1;
        drive_tiles(64, 16'h6600);
        wait (expected_read == expected_write);
        @(negedge clk_core);
        throughput_measure_mode = 1'b0;
        previous_tile_start_cycle = -1;

        random_stall_mode = 1'b1;
        drive_tiles(177, 16'h6700);
        wait (expected_read == expected_write);
        wait (!busy);
        @(negedge clk_core);
        random_stall_mode = 1'b0;

        if (accepted_tiles != 241 || accepted_beats != 1205)
            $fatal(1, "M166 population drift tiles=%0d beats=%0d",
                accepted_tiles, accepted_beats);
        if (five_cycle_ii_hits < 60)
            $fatal(1, "M166 insufficient 5-cycle II hits=%0d", five_cycle_ii_hits);
        if (output_stall_cycles == 0
                || input_push_release_overlap_cycles == 0
                || mixed_event_words == 0)
            $fatal(1, "M166 directed cover counters missing");

        // A simultaneous reconfiguration and rank request is fail-closed.
        @(negedge clk_core);
        config_valid = 1'b1;
        rank_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        config_valid = 1'b0;
        rank_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || config_ready || rank_ready)
            $fatal(1, "M166 fail-close attack did not stick");

        $display("PASS M166 prefolded rank3-left ATLIF backend VCS tiles=241 output_beats=1205 signed_products=115680 product_slots=96 service_cycles_per_tile=5 steady_ii5_hits=%0d input_push_release_overlap_cycles=%0d output_stall_cycles=%0d mixed_event_words=%0d protocol_attacks=1 folded_left_int8=true folded_bias_q24=true threshold_q24=true dense_reconstruction_materialized=false dynamic_bn_coefficient_generation=false epoch_rank_buffer=false fc2=false paft_valid825=false physical_speedup=false system_speedup=false headline=false",
            five_cycle_ii_hits, input_push_release_overlap_cycles,
            output_stall_cycles, mixed_event_words);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M166 watchdog timeout");
    end
endmodule

`default_nettype wire
