`timescale 1ns/1ps
`default_nettype none

// M1808 public-port-only mapped-gate activity workload for the admitted M1454
// Fixed-T10 C3 component.  Configuration and one warmup tile execute before
// the measured window.  The window contains eight dense tiles, legal sink
// backpressure, complete result drainage and one legal context release.
// Inputs are deterministic directed vectors, not checkpoint captures.  The
// result/tag/beat/retire scoreboard is independent of DUT hierarchy.
module tb_m1808_c3_m1454_fixed_t10_mapped_energy;
    localparam integer TAG_W = 48;
    localparam integer MEASURE_TILES = 8;
    localparam integer TOTAL_TILES = 1 + MEASURE_TILES;

    logic clk_core, rst_core;
    logic config_valid, config_ready, config_accept, config_last;
    logic [255:0] config_data;
    logic raw_valid, raw_ready, raw_accept, raw_last;
    logic [255:0] raw_data;
    logic [TAG_W-1:0] raw_tag;
    logic result_valid, result_ready, result_accept;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [47:0] result_valid_bits, result_data;
    logic release_valid, release_ready, release_accept;
    logic tile_done_valid, context_retire_valid, config_loaded;
    logic protocol_error, busy;
    logic [TAG_W-1:0] tile_done_tag;
    logic [31:0] context_retire_cycles;
    logic stage1_issue, stage2_issue, product_push, product_replace;
    logic fifo_push, fifo_pop;
    logic [4:0] result_fifo_occupancy;
    logic [1:0] raw_bank_occupancy, intermediate_bank_occupancy;
    logic [31:0] debug_config_beats, debug_raw_beats, debug_tiles_loaded;
    logic [31:0] debug_stage1_issues, debug_stage1_done;
    logic [31:0] debug_stage2_issues, debug_stage2_done;
    logic [31:0] debug_product_pushes, debug_result_departures;
    logic [31:0] debug_product_replacements, debug_context_cycles;

    logic [1279:0] fixed_config;
    logic [TAG_W-1:0] expected_tag [0:63];
    logic [2:0] expected_beat [0:63];
    logic [47:0] expected_data [0:63];
    integer expected_read, expected_write, mismatch_count;
    integer public_xz_count, result_stall_cycles, raw_stall_cycles;
    integer tile_done_count, measurement_cycles, global_cycles;
    integer base_raw_beats, base_tiles, base_issues, base_done;
    integer base_pushes, base_departures;
    integer measurement_result_base, measurement_tile_done_base;
    logic measurement_open;
    time first_config_accept_time;
    logic sampled_result_accept, sampled_tile_done_valid;
    logic [TAG_W-1:0] sampled_result_tag, sampled_tile_done_tag;
    logic [2:0] sampled_result_beat;
    logic [47:0] sampled_result_valid_bits, sampled_result_data;
    integer post_reset_settle_cycles;
    logic full_public_check_enabled;

    m518_matched_fixed_t10_atlif dut (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic integer weight_value(
        input integer row, input integer time_index
    );
        integer selector;
        begin
            selector = (row * 7 + time_index * 3) % 17;
            weight_value = selector - 8;
        end
    endfunction

    function automatic integer bias_value(input integer row);
        begin
            bias_value = (row * 43) - 193;
        end
    endfunction

    function automatic integer raw_value(
        input integer tile_seed, input integer time_index, input integer lane
    );
        integer selector;
        begin
            selector = (tile_seed * 11 + time_index * 5 + lane * 3) % 31;
            case (selector)
                0: raw_value = -128;
                1: raw_value = 127;
                2: raw_value = -65;
                3: raw_value = 64;
                default: raw_value = selector - 15;
            endcase
        end
    endfunction

    function automatic longint signed sat_q24(input longint signed value);
        begin
            if (value > 8388607) sat_q24 = 8388607;
            else if (value < -8388608) sat_q24 = -8388608;
            else sat_q24 = value;
        end
    endfunction

    task automatic build_config;
        integer row, time_index, value;
        begin
            fixed_config = '0;
            for (row = 0; row < 10; row = row + 1) begin
                for (time_index = 0; time_index < 10;
                        time_index = time_index + 1) begin
                    value = weight_value(row, time_index);
                    fixed_config[((row*10+time_index)*8)+:8] = value[7:0];
                end
                value = bias_value(row);
                fixed_config[800+(row*24)+:24] = value[23:0];
            end
            fixed_config[1040+:24] = 24'd13;
        end
    endtask

    task automatic build_payload(
        input integer tile_seed, output logic [1279:0] payload
    );
        integer time_index, lane, value;
        begin
            payload = '0;
            for (time_index = 0; time_index < 10;
                    time_index = time_index + 1)
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    value = raw_value(tile_seed, time_index, lane);
                    payload[((time_index*16+lane)*8)+:8] = value[7:0];
                end
        end
    endtask

    task automatic enqueue_expected(
        input integer tile_seed, input logic [TAG_W-1:0] tag_value
    );
        integer beat, pair, row, lane, time_index;
        longint signed total;
        logic [47:0] packed_result;
        begin
            for (beat = 0; beat < 5; beat = beat + 1) begin
                packed_result = '0;
                for (pair = 0; pair < 2; pair = pair + 1) begin
                    row = beat * 2 + pair;
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        total = bias_value(row);
                        for (time_index = 0; time_index < 10;
                                time_index = time_index + 1)
                            total = total + weight_value(row, time_index)
                                * raw_value(tile_seed, time_index, lane);
                        total = sat_q24(total);
                        packed_result[pair*16+lane] = total >= 13;
                    end
                end
                expected_tag[expected_write] = tag_value;
                expected_beat[expected_write] = beat[2:0];
                expected_data[expected_write] = packed_result;
                expected_write = expected_write + 1;
            end
        end
    endtask

    task automatic send_config;
        integer beat, watchdog;
        begin
            config_valid = 1'b1;
            for (beat = 0; beat < 5; beat = beat + 1) begin
                config_data = fixed_config[beat*256+:256];
                config_last = (beat == 4);
                watchdog = 0;
                do begin
                    @(posedge clk_core);
                    watchdog = watchdog + 1;
                    if (watchdog > 100) $fatal(1, "M1808 config timeout");
                end while (!config_accept);
                if (beat == 0) first_config_accept_time = $time;
                if (beat != 4) @(negedge clk_core);
            end
            @(negedge clk_core);
            config_valid = 1'b0;
            config_data = '0;
            config_last = 1'b0;
        end
    endtask

    task automatic send_tile(
        input integer tile_seed, input logic [TAG_W-1:0] tag_value
    );
        integer beat, watchdog;
        logic [1279:0] payload;
        begin
            build_payload(tile_seed, payload);
            enqueue_expected(tile_seed, tag_value);
            raw_valid = 1'b1;
            for (beat = 0; beat < 5; beat = beat + 1) begin
                raw_data = payload[beat*256+:256];
                raw_last = (beat == 4);
                raw_tag = tag_value;
                watchdog = 0;
                do begin
                    @(posedge clk_core);
                    watchdog = watchdog + 1;
                    if (watchdog > 1000) $fatal(1, "M1808 raw timeout");
                end while (!raw_accept);
                if (beat != 4) @(negedge clk_core);
            end
            @(negedge clk_core);
            raw_valid = 1'b0;
            raw_data = '0;
            raw_last = 1'b0;
            raw_tag = '0;
        end
    endtask

    // Legal public result backpressure is part of the measured component
    // workload.  It is deterministic and guarantees non-vacuous stall cover.
    always @(negedge clk_core) begin
        if (rst_core) result_ready <= 1'b0;
        else if (measurement_open)
            result_ready <= !((measurement_cycles % 11) == 3
                || (measurement_cycles % 11) == 4);
        else result_ready <= 1'b1;
    end

    // No hierarchical read is used: every checked signal is a public port.
    //
    // M1808 keeps the architectural/control gate immediate on every
    // post-reset edge.  Only the eleven debug counters receive the already
    // budgeted three-cycle quiet-settling window diagnosed by M1807.  The
    // third edge must prove every debug counter binary and exactly zero;
    // after that boundary the original full aggregate gate is restored.
    always @(posedge clk_core) begin
        // Capture the architectural handshake before the mapped sequential
        // state advances; post-edge public-X checking follows after settling.
        sampled_result_accept = result_accept;
        sampled_result_tag = result_tag;
        sampled_result_beat = result_beat;
        sampled_result_valid_bits = result_valid_bits;
        sampled_result_data = result_data;
        sampled_tile_done_valid = tile_done_valid;
        sampled_tile_done_tag = tile_done_tag;
        global_cycles = global_cycles + 1;
        if (rst_core) begin
            post_reset_settle_cycles = 0;
            full_public_check_enabled = 1'b0;
        end else begin
            #0.2;
            if ($isunknown({config_ready, config_accept, raw_ready, raw_accept,
                    result_valid, result_accept, result_tag, result_beat,
                    result_valid_bits, result_data, release_ready,
                    release_accept, tile_done_valid, tile_done_tag,
                    context_retire_valid, context_retire_cycles, config_loaded,
                    protocol_error, busy, stage1_issue, stage2_issue,
                    product_push, product_replace, fifo_push, fifo_pop,
                    result_fifo_occupancy, raw_bank_occupancy,
                    intermediate_bank_occupancy})) begin
                public_xz_count = public_xz_count + 1;
                $fatal(1,
                    "M1808 architectural/control output contains X/Z cycle=%0d",
                    global_cycles);
            end

            if (!full_public_check_enabled) begin
                post_reset_settle_cycles = post_reset_settle_cycles + 1;
                if (config_accept || raw_accept || result_valid
                        || result_accept || release_accept || tile_done_valid
                        || context_retire_valid || config_loaded
                        || protocol_error || busy || stage1_issue
                        || stage2_issue || product_push || product_replace
                        || fifo_push || fifo_pop
                        || result_fifo_occupancy != 0
                        || raw_bank_occupancy != 0
                        || intermediate_bank_occupancy != 0)
                    $fatal(1,
                        "M1808 activity during reset-settling cycle=%0d",
                        post_reset_settle_cycles);
                if (post_reset_settle_cycles == 3) begin
                    if ($isunknown({debug_config_beats, debug_raw_beats,
                            debug_tiles_loaded, debug_stage1_issues,
                            debug_stage1_done, debug_stage2_issues,
                            debug_stage2_done, debug_product_pushes,
                            debug_result_departures,
                            debug_product_replacements,
                            debug_context_cycles}))
                        $fatal(1,
                            "M1808 debug counter X/Z at settling boundary");
                    if ({debug_config_beats, debug_raw_beats,
                            debug_tiles_loaded, debug_stage1_issues,
                            debug_stage1_done, debug_stage2_issues,
                            debug_stage2_done, debug_product_pushes,
                            debug_result_departures,
                            debug_product_replacements,
                            debug_context_cycles} != 0)
                        $fatal(1,
                            "M1808 debug counter nonzero at settling boundary");
                    full_public_check_enabled = 1'b1;
                    $display(
                        "M1808_RESET_SETTLING_GATE cycles=3 debug=11 binary=1 zero=1");
                end else if (post_reset_settle_cycles > 3) begin
                    $fatal(1, "M1808 reset-settling gate failed to close");
                end
            end else if ($isunknown({config_ready, config_accept,
                    raw_ready, raw_accept, result_valid, result_accept,
                    result_tag, result_beat, result_valid_bits, result_data,
                    release_ready, release_accept, tile_done_valid,
                    tile_done_tag, context_retire_valid,
                    context_retire_cycles, config_loaded, protocol_error,
                    busy, stage1_issue, stage2_issue, product_push,
                    product_replace, fifo_push, fifo_pop,
                    result_fifo_occupancy, raw_bank_occupancy,
                    intermediate_bank_occupancy, debug_config_beats,
                    debug_raw_beats, debug_tiles_loaded, debug_stage1_issues,
                    debug_stage1_done, debug_stage2_issues,
                    debug_stage2_done, debug_product_pushes,
                    debug_result_departures, debug_product_replacements,
                    debug_context_cycles})) begin
                public_xz_count = public_xz_count + 1;
                $fatal(1,
                    "M1808 full public output contains X/Z cycle=%0d",
                    global_cycles);
            end

            if (protocol_error) $fatal(1, "M1808 protocol_error asserted");
            if (measurement_open) begin
                measurement_cycles = measurement_cycles + 1;
                if (result_valid && !result_ready)
                    result_stall_cycles = result_stall_cycles + 1;
                if (raw_valid && !raw_ready)
                    raw_stall_cycles = raw_stall_cycles + 1;
            end
            if (sampled_result_accept) begin
                if ($isunknown({sampled_result_tag, sampled_result_beat,
                        sampled_result_valid_bits, sampled_result_data}))
                    $fatal(1, "M1808 accepted result contains X/Z");
                if (expected_read >= expected_write)
                    $fatal(1, "M1808 unexpected result");
                if (sampled_result_tag !== expected_tag[expected_read]
                        || sampled_result_beat !== expected_beat[expected_read]
                        || sampled_result_valid_bits !== 48'h0000ffffffff
                        || sampled_result_data !== expected_data[expected_read]) begin
                    mismatch_count = mismatch_count + 1;
                    $fatal(1, "M1808 result mismatch index=%0d", expected_read);
                end
                expected_read = expected_read + 1;
            end
            if (sampled_tile_done_valid) begin
                if ($isunknown(sampled_tile_done_tag))
                    $fatal(1, "M1808 tile-done tag contains X/Z");
                tile_done_count = tile_done_count + 1;
            end
        end
    end

    initial begin
        integer tile, watchdog, expected_retire;
        rst_core = 1'b1;
        config_valid = 1'b0; config_data = '0; config_last = 1'b0;
        raw_valid = 1'b0; raw_data = '0; raw_last = 1'b0; raw_tag = '0;
        result_ready = 1'b0;
        release_valid = 1'b0;
        expected_read = 0; expected_write = 0; mismatch_count = 0;
        public_xz_count = 0; result_stall_cycles = 0; raw_stall_cycles = 0;
        tile_done_count = 0; measurement_cycles = 0; global_cycles = 0;
        measurement_open = 1'b0; first_config_accept_time = 0;
        post_reset_settle_cycles = 0;
        full_public_check_enabled = 1'b0;
        build_config();
        repeat (8) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        if (!full_public_check_enabled || post_reset_settle_cycles != 3)
            $fatal(1, "M1808 reset-settling boundary did not close");

        // Configuration and one full tile warm the mapped state outside SAIF.
        send_config();
        send_tile(100, 48'h1790_0000_0000);
        watchdog = 0;
        while (expected_read != 5) begin
            @(posedge clk_core); watchdog = watchdog + 1;
            if (watchdog > 1000) $fatal(1, "M1808 warmup drain timeout");
        end
        if (!config_loaded || protocol_error || debug_config_beats != 5
                || debug_raw_beats != 5 || debug_tiles_loaded != 1
                || debug_stage1_issues != 17 || debug_stage1_done != 1
                || debug_product_pushes != 5 || debug_result_departures != 5)
            $fatal(1, "M1808 warmup public counters drift");

        base_raw_beats = debug_raw_beats;
        base_tiles = debug_tiles_loaded;
        base_issues = debug_stage1_issues;
        base_done = debug_stage1_done;
        base_pushes = debug_product_pushes;
        base_departures = debug_result_departures;
        measurement_result_base = expected_read;
        measurement_tile_done_base = tile_done_count;

        @(negedge clk_core);
        measurement_open = 1'b1;
        measurement_cycles = 0;
        $display("M1808_SAIF_WINDOW_START tiles=%0d", MEASURE_TILES);
        if ($test$plusargs("M1808_UCLI_SAIF")) $stop;

        for (tile = 0; tile < MEASURE_TILES; tile = tile + 1)
            send_tile(200+tile, 48'h1790_1000_0000+tile);

        watchdog = 0;
        while (expected_read != 5*TOTAL_TILES) begin
            @(posedge clk_core); watchdog = watchdog + 1;
            if (watchdog > 5000) $fatal(1, "M1808 measured drain timeout");
        end
        @(negedge clk_core); result_ready = 1'b1;
        release_valid = 1'b1;
        watchdog = 0;
        do begin
            @(posedge clk_core); watchdog = watchdog + 1;
            if (watchdog > 1000) $fatal(1, "M1808 release timeout");
        end while (!release_accept);
        #0.2;
        expected_retire = (($time-first_config_accept_time)/3)+1;
        if (!context_retire_valid || context_retire_cycles != expected_retire)
            $fatal(1, "M1808 retire mismatch got=%0d want=%0d valid=%0b",
                context_retire_cycles, expected_retire, context_retire_valid);
        @(negedge clk_core); release_valid = 1'b0;

        if (mismatch_count != 0 || public_xz_count != 0
                || expected_read-measurement_result_base != 5*MEASURE_TILES
                || tile_done_count-measurement_tile_done_base != MEASURE_TILES
                || debug_raw_beats-base_raw_beats != 5*MEASURE_TILES
                || debug_tiles_loaded-base_tiles != MEASURE_TILES
                || debug_stage1_issues-base_issues != 17*MEASURE_TILES
                || debug_stage1_done-base_done != MEASURE_TILES
                || debug_product_pushes-base_pushes != 5*MEASURE_TILES
                || debug_result_departures-base_departures != 5*MEASURE_TILES
                || debug_stage2_issues != 0 || debug_stage2_done != 0
                || debug_product_replacements != 0
                || result_stall_cycles == 0 || raw_stall_cycles == 0
                || protocol_error || config_loaded || busy)
            $fatal(1, "M1808 conservation/coverage failure read=%0d stalls=%0d/%0d",
                expected_read, result_stall_cycles, raw_stall_cycles);

        measurement_open = 1'b0;
        $display("M1808_PUBLIC_RESULT_CHECK tiles=%0d beats=%0d mismatches=%0d xz=%0d",
            MEASURE_TILES, 5*MEASURE_TILES, mismatch_count, public_xz_count);
        $display("M1808_PUBLIC_COUNTER_DELTAS raw_beats=%0d tiles=%0d issues=%0d done=%0d pushes=%0d departures=%0d",
            debug_raw_beats-base_raw_beats, debug_tiles_loaded-base_tiles,
            debug_stage1_issues-base_issues, debug_stage1_done-base_done,
            debug_product_pushes-base_pushes,
            debug_result_departures-base_departures);
        $display("M1808_PUBLIC_COVERAGE result_stall_cycles=%0d raw_stall_cycles=%0d retire_cycles=%0d",
            result_stall_cycles, raw_stall_cycles, context_retire_cycles);
        $display("PASS_M1808_C3_M1454_FIXED_T10_MAPPED_DIRECTED_COMPONENT_ACTIVITY");
        $display("M1808_SAIF_WINDOW_STOP cycles=%0d", measurement_cycles);
        if ($test$plusargs("M1808_UCLI_SAIF")) $stop;
        $finish;
    end
endmodule

`default_nettype wire


