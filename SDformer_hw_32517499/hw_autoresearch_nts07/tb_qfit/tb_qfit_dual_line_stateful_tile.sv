`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_stateful_tile;
    localparam int TILE_BITS = 256;
    localparam int OUT_LANES = 16;
    localparam int TAG_W = 24;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int INDEX_W = $clog2(TILE_BITS);
    localparam int OUT_W = $clog2(OUT_LANES);
    localparam int COUNT_W = $clog2(TILE_BITS + 1);
    localparam int PERF_W = 64;
    localparam int COMMANDS = 2100;
    localparam int EPOCH_RACES = 100;

    logic clk_core = 1'b0;
    logic rst_core;
    logic weight_epoch_clear;
    logic weight_valid, weight_ready;
    logic [INDEX_W-1:0] weight_source;
    logic [OUT_W-1:0] weight_lane;
    logic signed [W_W-1:0] weight_data;
    logic weight_last, weights_loaded;
    logic request_valid, request_ready;
    logic [TAG_W-1:0] request_state_key;
    logic [COUNT_W-1:0] request_valid_bits;
    logic [TILE_BITS-1:0] request_current_bits;
    logic [OUT_LANES*ACC_W-1:0] request_local_seed_acc;
    logic request_sequence_boundary, request_force_refresh;
    logic output_valid, output_ready;
    logic [TAG_W-1:0] output_state_key;
    logic output_use_motion, output_force_local;
    logic [COUNT_W-1:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic protocol_error;
    logic [PERF_W-1:0] perf_requests, perf_state_hits, perf_state_misses;
    logic [PERF_W-1:0] perf_local_tiles, perf_motion_tiles;
    logic [PERF_W-1:0] perf_invalid_valid_bits;
    logic [PERF_W-1:0] perf_weight_segment_reads, perf_accumulator_updates;

    integer weight_ref [0:TILE_BITS-1][0:OUT_LANES-1];
    integer request_seed [0:OUT_LANES-1];
    integer expected_acc [0:OUT_LANES-1];
    integer ref_seed [0:OUT_LANES-1];
    logic ref_valid;
    logic [TAG_W-1:0] ref_key;
    integer ref_valid_bits;
    logic [TILE_BITS-1:0] ref_bits;
    integer ref_acc [0:OUT_LANES-1];
    integer total_sources;
    integer total_updates;
    integer total_local;
    integer total_motion;
    integer total_hits;
    integer total_misses;
    integer invalid_requests;
    integer sequence_boundaries;
    integer forced_refreshes;
    integer key_changes;
    integer seed_changes;
    integer shape_changes;
    integer cycles;
    integer active_key;
    integer epoch_clear_races;
    integer weight_write_races;

    always #5 clk_core = ~clk_core;

    qfit_dual_line_stateful_tile_top #(
        .TILE_BITS(TILE_BITS), .OUT_LANES(OUT_LANES),
        .TAG_W(TAG_W), .W_W(W_W), .ACC_W(ACC_W), .PERF_W(PERF_W)
    ) dut (.*);

    qfit_dual_line_stateful_tile_assertions #(
        .TAG_W(TAG_W), .COUNT_W(COUNT_W),
        .OUTPUT_W(OUT_LANES*ACC_W), .PERF_W(PERF_W)
    ) sva (.*);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycles <= cycles + 1;
            if (cycles > 1000000)
                $fatal(1, "stateful tile timeout");
        end
    end

    function automatic integer make_weight(input integer source, input integer lane);
        make_weight = ((source * 17 + lane * 11 + 5) % 63) - 31;
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
                    if (!weight_ready)
                        $fatal(1, "weight loader unexpectedly backpressured");
                end
            end
            @(negedge clk_core);
            weight_valid = 1'b0;
            if (!weights_loaded || protocol_error)
                $fatal(1, "stateful tile ordered weight load failed");
        end
    endtask

    task automatic race_weight_write_request;
        integer requests_before;
        begin
            requests_before = perf_requests;
            @(negedge clk_core);
            request_valid = 1'b1;
            weight_valid = 1'b1;
            weight_source = INDEX_W'(TILE_BITS - 1);
            weight_lane = OUT_W'(OUT_LANES - 1);
            weight_data = W_W'(weight_ref[TILE_BITS - 1][OUT_LANES - 1] + 1);
            weight_last = 1'b1;
            #1;
            if (request_ready || weight_ready)
                $fatal(1, "request or post-load weight write admitted during closed epoch");
            @(negedge clk_core);
            request_valid = 1'b0;
            weight_valid = 1'b0;
            if (perf_requests != requests_before
                || $signed(dut.u_executor.weight_q[TILE_BITS - 1][OUT_LANES - 1])
                    != weight_ref[TILE_BITS - 1][OUT_LANES - 1])
                $fatal(1, "post-load write race changed request accounting or weight state");
            weight_write_races++;
        end
    endtask

    task automatic race_epoch_clear_request;
        integer requests_before;
        begin
            requests_before = perf_requests;
            @(negedge clk_core);
            request_valid = 1'b1;
            weight_epoch_clear = 1'b1;
            #1;
            if (request_ready)
                $fatal(1, "request admitted during weight epoch clear");
            @(negedge clk_core);
            request_valid = 1'b0;
            weight_epoch_clear = 1'b0;
            if (perf_requests != requests_before || weights_loaded)
                $fatal(1, "epoch-clear race changed request accounting or retained weights");
            ref_valid = 1'b0;
            epoch_clear_races++;
            load_weights();
        end
    endtask

    task automatic run_request(input integer command_number);
        logic [TILE_BITS-1:0] current_bits;
        logic [TILE_BITS-1:0] masked_current;
        logic [TILE_BITS-1:0] previous_bits;
        integer valid_bits;
        integer effective_valid_bits;
        integer current_count;
        integer transition_count;
        integer source_count;
        bit invalid_valid;
        bit identity_match;
        bit expected_motion;
        bit boundary;
        bit refresh;
        bit seed_equal;
        begin
            boundary = command_number % 19 == 0;
            refresh = command_number % 17 == 0;
            invalid_valid = command_number % 20 == 0;
            if (command_number % 16 == 0) begin
                active_key++;
                key_changes++;
            end
            valid_bits = ref_valid ? ref_valid_bits : TILE_BITS;
            if (command_number % 16 == 3) begin
                valid_bits = valid_bits == TILE_BITS ? 128 : TILE_BITS;
                shape_changes++;
            end
            if (invalid_valid)
                valid_bits = TILE_BITS + 1;
            effective_valid_bits = invalid_valid ? TILE_BITS : valid_bits;

            current_bits = '0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++) begin
                case (command_number % 3)
                    0: current_bits[bit_index] = $urandom_range(0, 99) < 7;
                    1: current_bits[bit_index] = $urandom_range(0, 99) < 30;
                    default: current_bits[bit_index] = $urandom_range(0, 99) < 70;
                endcase
            end
            if (command_number % 101 == 0)
                current_bits = '0;
            masked_current = '0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++)
                if (bit_index < effective_valid_bits)
                    masked_current[bit_index] = current_bits[bit_index];

            seed_equal = ref_valid;
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                request_seed[lane] = ref_valid
                    ? ref_seed[lane]
                    : ((active_key * 13 + lane * 7) % 257 - 128);
            end
            if (command_number % 14 == 0) begin
                request_seed[command_number % OUT_LANES] += 1;
                seed_changes++;
            end
            if (ref_valid)
                for (int lane = 0; lane < OUT_LANES; lane++)
                    if (request_seed[lane] != ref_seed[lane])
                        seed_equal = 1'b0;

            identity_match = ref_valid
                && ref_key == TAG_W'(active_key)
                && ref_valid_bits == valid_bits
                && seed_equal
                && !invalid_valid;
            previous_bits = identity_match ? ref_bits : '0;
            current_count = 0;
            transition_count = 0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++) begin
                current_count += masked_current[bit_index];
                transition_count += masked_current[bit_index] ^ previous_bits[bit_index];
            end
            expected_motion = identity_match && !boundary && !refresh
                && transition_count < current_count;
            source_count = expected_motion ? transition_count : current_count;

            for (int lane = 0; lane < OUT_LANES; lane++) begin
                expected_acc[lane] = request_seed[lane];
                for (int source = 0; source < TILE_BITS; source++)
                    if (masked_current[source])
                        expected_acc[lane] += weight_ref[source][lane];
                request_local_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(request_seed[lane]);
            end

            request_state_key = TAG_W'(active_key);
            request_valid_bits = COUNT_W'(valid_bits);
            request_current_bits = current_bits;
            request_sequence_boundary = boundary;
            request_force_refresh = refresh;
            @(negedge clk_core);
            request_valid = 1'b1;
            while (!request_ready)
                @(negedge clk_core);
            @(negedge clk_core);
            request_valid = 1'b0;

            output_ready = 1'b0;
            while (!output_valid) begin
                output_ready = $urandom_range(0, 3) != 0;
                @(negedge clk_core);
            end
            if (output_state_key != TAG_W'(active_key)
                || output_use_motion != expected_motion
                || output_force_local == expected_motion
                || output_source_count != COUNT_W'(source_count))
                $fatal(1, "stateful metadata mismatch command=%0d motion=%0b/%0b sources=%0d/%0d",
                    command_number, output_use_motion, expected_motion,
                    output_source_count, source_count);
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                if ($signed(output_acc[lane*ACC_W +: ACC_W]) != expected_acc[lane])
                    $fatal(1, "stateful Acc32 mismatch command=%0d lane=%0d got=%0d exp=%0d",
                        command_number, lane,
                        $signed(output_acc[lane*ACC_W +: ACC_W]), expected_acc[lane]);
            end
            output_ready = 1'b1;
            @(negedge clk_core);
            while (output_valid)
                @(negedge clk_core);

            if (identity_match)
                total_hits++;
            else
                total_misses++;
            if (expected_motion)
                total_motion++;
            else
                total_local++;
            if (invalid_valid)
                invalid_requests++;
            if (boundary)
                sequence_boundaries++;
            if (refresh)
                forced_refreshes++;
            total_sources += source_count;
            total_updates += source_count * OUT_LANES;

            ref_valid = 1'b1;
            ref_key = TAG_W'(active_key);
            ref_valid_bits = effective_valid_bits;
            ref_bits = masked_current;
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                ref_seed[lane] = request_seed[lane];
                ref_acc[lane] = expected_acc[lane];
            end
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
        request_valid = 1'b0;
        request_state_key = '0;
        request_valid_bits = '0;
        request_current_bits = '0;
        request_local_seed_acc = '0;
        request_sequence_boundary = 1'b0;
        request_force_refresh = 1'b0;
        output_ready = 1'b0;
        ref_valid = 1'b0;
        ref_key = '0;
        ref_valid_bits = 0;
        ref_bits = '0;
        active_key = 1;
        total_sources = 0;
        total_updates = 0;
        total_local = 0;
        total_motion = 0;
        total_hits = 0;
        total_misses = 0;
        invalid_requests = 0;
        sequence_boundaries = 0;
        forced_refreshes = 0;
        key_changes = 0;
        seed_changes = 0;
        shape_changes = 0;
        epoch_clear_races = 0;
        weight_write_races = 0;
        cycles = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        load_weights();
        for (int race = 0; race < EPOCH_RACES; race++) begin
            // Keep these setup requests legal; protocol_error must remain low
            // across each clear/reload race.
            run_request(10001 + race * 20);
            race_weight_write_request();
            race_epoch_clear_request();
        end
        for (int command_number = 0; command_number < COMMANDS; command_number++)
            run_request(command_number);

        if (!protocol_error
            || invalid_requests < 100
            || sequence_boundaries < 100
            || forced_refreshes < 100
            || key_changes < 100
            || seed_changes < 100
            || shape_changes < 100
            || epoch_clear_races != EPOCH_RACES
            || weight_write_races != EPOCH_RACES
            || perf_requests != COMMANDS + EPOCH_RACES
            || perf_state_hits != total_hits
            || perf_state_misses != total_misses
            || perf_local_tiles != total_local
            || perf_motion_tiles != total_motion
            || perf_invalid_valid_bits != invalid_requests
            || perf_weight_segment_reads != total_sources
            || perf_accumulator_updates != total_updates)
            $fatal(1, "stateful final gate/counter mismatch");
        $display("PASS stateful dual-line tile commands=%0d cycles=%0d local=%0d motion=%0d hits=%0d misses=%0d sources=%0d updates=%0d invalid=%0d boundary=%0d refresh=%0d key_change=%0d seed_change=%0d shape_change=%0d epoch_clear_races=%0d weight_write_races=%0d",
            COMMANDS + EPOCH_RACES, cycles, total_local, total_motion, total_hits, total_misses,
            total_sources, total_updates, invalid_requests, sequence_boundaries,
            forced_refreshes, key_changes, seed_changes, shape_changes,
            epoch_clear_races, weight_write_races);
        $finish;
    end
endmodule

`default_nettype wire
