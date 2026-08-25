`timescale 1ns/1ps
`default_nettype none

module tb_m127_independent_hammer;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;

    logic clk_core, rst_core;
    logic weight_fill_valid;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic row_valid;
    logic [2:0] row_block;
    logic [8:0] row_offset;
    logic [15:0] row_source_mask, row_negate_mask;
    logic update_ready;

    logic d_fill_ready, d_fill_accept, d_row_ready, d_row_accept;
    logic d_update_valid, d_update_accept, d_row_done;
    logic [2:0] d_update_block;
    logic [8:0] d_update_row;
    logic [1823:0] d_update_delta;
    logic [15:0] d_update_selected_mask;
    logic [15:0] d_remaining, d_cache_valid;
    logic d_resident_valid, d_pipe_valid, d_protocol_error, d_busy;
    logic [2:0] d_resident_block;

    logic r_fill_ready, r_fill_accept, r_row_ready, r_row_accept;
    logic r_update_valid, r_update_accept, r_row_done;
    logic [2:0] r_update_block;
    logic [8:0] r_update_row;
    logic [1823:0] r_update_delta;
    logic [15:0] r_update_selected_mask;
    logic [15:0] r_remaining, r_cache_valid;
    logic r_resident_valid, r_protocol_error, r_busy;
    logic [2:0] r_resident_block;

    int cycle_count;
    int cycle_exact_checks;
    int rows_checked, updates_checked, sources_checked, lanes_checked;
    int canonical_checks, stall_cycles, max_stall_burst;
    int consecutive_intra_row_pairs, four_group_ii1_rows;
    int tail_k1, tail_k2, tail_k3, tail_k4;
    int plus512_checks, minus512_checks;
    int first_group_checks, first_group_absolute_latency_sum;
    int first_group_additional_cycles_sum;
    int inter_row_single_k4_updates;
    int inter_row_single_k4_min_interval, inter_row_single_k4_max_interval;
    int previous_single_k4_cycle;
    int cache_transition_checks, cache_attacks, block_attacks;
    int fill_sequence_attacks, reset_isolation_checks;
    int pipeline_stall_flush_checks;
    int unsigned prng_state;
    bit compare_enable;

    m127_block_phased_pipelined_k4_row_fold dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(d_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(d_fill_accept),
        .row_valid(row_valid), .row_ready(d_row_ready),
        .row_block(row_block), .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(d_row_accept),
        .update_valid(d_update_valid), .update_ready(update_ready),
        .update_block(d_update_block), .update_row(d_update_row),
        .update_delta(d_update_delta),
        .update_selected_mask(d_update_selected_mask),
        .update_accept(d_update_accept), .row_done(d_row_done),
        .observed_remaining_mask(d_remaining),
        .observed_cache_valid(d_cache_valid),
        .observed_resident_block_valid(d_resident_valid),
        .observed_resident_block(d_resident_block),
        .observed_pair_pipeline_valid(d_pipe_valid),
        .protocol_error(d_protocol_error), .busy(d_busy)
    );

    m125_block_phased_k4_row_fold reference (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(r_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(r_fill_accept),
        .row_valid(row_valid), .row_ready(r_row_ready),
        .row_block(row_block), .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(r_row_accept),
        .update_valid(r_update_valid), .update_ready(update_ready),
        .update_block(r_update_block), .update_row(r_update_row),
        .update_delta(r_update_delta),
        .update_selected_mask(r_update_selected_mask),
        .update_accept(r_update_accept), .row_done(r_row_done),
        .observed_remaining_mask(r_remaining),
        .observed_cache_valid(r_cache_valid),
        .observed_resident_block_valid(r_resident_valid),
        .observed_resident_block(r_resident_block),
        .protocol_error(r_protocol_error), .busy(r_busy)
    );

    m127_block_phased_pipelined_k4_row_fold_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(d_fill_ready),
        .weight_fill_accept(d_fill_accept),
        .row_valid(row_valid), .row_ready(d_row_ready),
        .row_accept(d_row_accept),
        .update_valid(d_update_valid), .update_ready(update_ready),
        .update_accept(d_update_accept),
        .update_block(d_update_block), .update_row(d_update_row),
        .update_delta(d_update_delta),
        .update_selected_mask(d_update_selected_mask),
        .observed_remaining_mask(d_remaining),
        .observed_pair_pipeline_valid(d_pipe_valid),
        .row_done(d_row_done), .protocol_error(d_protocol_error)
    );

    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            if (compare_enable) begin
                cycle_exact_checks = cycle_exact_checks + 1;
                if ({d_fill_ready, d_fill_accept, d_row_ready, d_row_accept,
                     d_update_valid, d_update_accept, d_row_done,
                     d_remaining, d_cache_valid, d_resident_valid,
                     d_resident_block, d_protocol_error, d_busy}
                    !==
                    {r_fill_ready, r_fill_accept, r_row_ready, r_row_accept,
                     r_update_valid, r_update_accept, r_row_done,
                     r_remaining, r_cache_valid, r_resident_valid,
                     r_resident_block, r_protocol_error, r_busy})
                    $fatal(1, "M127/M125 accepted-cycle control mismatch cycle=%0d",
                           cycle_count);
                if (d_update_valid
                        && {d_update_block, d_update_row,
                            d_update_selected_mask, d_update_delta}
                           !==
                           {r_update_block, r_update_row,
                            r_update_selected_mask, r_update_delta})
                    begin
                        $display("M127 payload block=%0d row=%0d sel=%h lane0=%0d lane1=%0d",
                                 d_update_block, d_update_row,
                                 d_update_selected_mask,
                                 $signed(d_update_delta[0 +: ACC_BITS]),
                                 $signed(d_update_delta[ACC_BITS +: ACC_BITS]));
                        $display("M125 payload block=%0d row=%0d sel=%h lane0=%0d lane1=%0d",
                                 r_update_block, r_update_row,
                                 r_update_selected_mask,
                                 $signed(r_update_delta[0 +: ACC_BITS]),
                                 $signed(r_update_delta[ACC_BITS +: ACC_BITS]));
                        $fatal(1, "M127/M125 valid-payload mismatch cycle=%0d",
                               cycle_count);
                    end
                if (!d_protocol_error && d_pipe_valid !== d_update_valid)
                    $fatal(1, "M127 pair pipeline visibility mismatch");
            end
        end
    end

    function automatic integer signed model_weight(
        input int block_id,
        input int source,
        input int lane
    );
        int raw;
        begin
            if (lane == 0 && source < 4)
                model_weight = -128;
            else if (lane == 1 && source < 4)
                model_weight = 127;
            else begin
                raw = (block_id * 71 + source * 43 + lane * 29
                       + source * lane * 3) & 8'hff;
                model_weight = raw - 128;
            end
        end
    endfunction

    function automatic logic [15:0] oracle_lowest4(input logic [15:0] mask);
        int picked;
        begin
            oracle_lowest4 = 0;
            picked = 0;
            for (int source = 0; source < SOURCES; source++) begin
                if (mask[source] && picked < 4) begin
                    oracle_lowest4[source] = 1'b1;
                    picked = picked + 1;
                end
            end
        end
    endfunction

    task automatic drive_idle;
        begin
            weight_fill_valid = 0;
            weight_fill_block = 0;
            weight_fill_source = 0;
            weight_fill_beat = 0;
            weight_fill_data = 0;
            row_valid = 0;
            row_block = 0;
            row_offset = 0;
            row_source_mask = 0;
            row_negate_mask = 0;
        end
    endtask

    task automatic clean_reset;
        begin
            @(negedge clk_core);
            compare_enable = 0;
            drive_idle();
            update_ready = 0;
            rst_core = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            compare_enable = 1;
            update_ready = 1;
            repeat (2) @(posedge clk_core);
            if (d_protocol_error || d_resident_valid || d_cache_valid != 0
                    || d_remaining != 0 || d_pipe_valid || d_busy)
                $fatal(1, "M127 reset did not clear state");
        end
    endtask

    task automatic fill_source(input int block_id, input int source);
        logic [255:0] payload;
        int value;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = 0;
                for (int item = 0; item < 32; item++) begin
                    value = model_weight(block_id, source, beat * 32 + item);
                    payload[item * 8 +: 8] = value[7:0];
                end
                @(negedge clk_core);
                weight_fill_valid = 1;
                weight_fill_block = block_id[2:0];
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                do @(posedge clk_core); while (!d_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 0;
        end
    endtask

    task automatic fill_block(input int block_id);
        begin
            for (int source = 0; source < SOURCES; source++)
                fill_source(block_id, source);
            if (!d_resident_valid || d_resident_block !== block_id[2:0]
                    || d_cache_valid !== 16'hffff)
                $fatal(1, "M127 full cache identity mismatch block=%0d", block_id);
        end
    endtask

    task automatic check_row(
        input int block_id,
        input int row_id,
        input logic [15:0] initial_mask,
        input logic [15:0] negate_mask,
        input int initial_stall,
        input bit random_stalls,
        input bit record_single_k4
    );
        logic [15:0] remaining, consumed, expected_selected;
        logic [1823:0] stalled_delta;
        logic [15:0] stalled_selected;
        logic [2:0] stalled_block;
        logic [8:0] stalled_row;
        int expected_value, update_count, stall_left, stall_run;
        int watchdog, row_accept_cycle, first_update_cycle;
        int previous_update_cycle, selected_count;
        bit stall_snapshot_valid, first_seen;
        begin
            remaining = initial_mask;
            consumed = 0;
            update_count = 0;
            stall_left = initial_stall;
            stall_run = 0;
            watchdog = 0;
            stall_snapshot_valid = 0;
            first_seen = 0;
            previous_update_cycle = -1;

            @(negedge clk_core);
            update_ready = 1;
            row_valid = 1;
            row_block = block_id[2:0];
            row_offset = row_id[8:0];
            row_source_mask = initial_mask;
            row_negate_mask = negate_mask;
            do @(posedge clk_core); while (!d_row_accept);
            row_accept_cycle = cycle_count;
            @(negedge clk_core);
            row_valid = 0;
            if (random_stalls && stall_left < 0) begin
                prng_state = prng_state * 32'd1664525 + 32'd1013904223;
                stall_left = prng_state[5:0];
            end

            while (!d_row_done) begin
                if (stall_left > 0)
                    update_ready = 0;
                else
                    update_ready = 1;
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 2000)
                    $fatal(1, "M127 independent row watchdog row=%0d", row_id);
                if (d_update_valid) begin
                    expected_selected = oracle_lowest4(remaining);
                    if (d_update_selected_mask !== expected_selected)
                        $fatal(1, "M127 canonical group mismatch row=%0d got=%h expected=%h",
                               row_id, d_update_selected_mask, expected_selected);
                    if ((d_update_selected_mask & consumed) != 0)
                        $fatal(1, "M127 duplicate source row=%0d", row_id);
                    if (d_update_block !== block_id[2:0]
                            || d_update_row !== row_id[8:0])
                        $fatal(1, "M127 block/row identity mismatch row=%0d", row_id);
                    if (!update_ready) begin
                        stall_cycles = stall_cycles + 1;
                        stall_run = stall_run + 1;
                        if (stall_run > max_stall_burst)
                            max_stall_burst = stall_run;
                        if (!stall_snapshot_valid) begin
                            stalled_delta = d_update_delta;
                            stalled_selected = d_update_selected_mask;
                            stalled_block = d_update_block;
                            stalled_row = d_update_row;
                            stall_snapshot_valid = 1;
                        end else if ({d_update_delta, d_update_selected_mask,
                                      d_update_block, d_update_row}
                                     !==
                                     {stalled_delta, stalled_selected,
                                      stalled_block, stalled_row})
                            $fatal(1, "M127 pipeline changed under long stall");
                        stall_left = stall_left - 1;
                    end
                end
                if (d_update_accept) begin
                    canonical_checks = canonical_checks + 1;
                    selected_count = $countones(d_update_selected_mask);
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_value = 0;
                        for (int source = 0; source < SOURCES; source++) begin
                            if (expected_selected[source]) begin
                                expected_value = expected_value
                                    + (negate_mask[source]
                                       ? -model_weight(block_id, source, lane)
                                       : model_weight(block_id, source, lane));
                            end
                        end
                        if ($signed(d_update_delta[lane * ACC_BITS +: ACC_BITS])
                                !== expected_value)
                            $fatal(1, "M127 numeric mismatch row=%0d lane=%0d got=%0d expected=%0d",
                                   row_id, lane,
                                   $signed(d_update_delta[lane * ACC_BITS +: ACC_BITS]),
                                   expected_value);
                    end
                    if (!first_seen) begin
                        first_seen = 1;
                        first_update_cycle = cycle_count;
                        if (initial_stall == 0 && !random_stalls) begin
                            if (first_update_cycle - row_accept_cycle != 1)
                                $fatal(1, "M127 no-stall first-group latency is not one cycle");
                            first_group_checks = first_group_checks + 1;
                            first_group_absolute_latency_sum
                                = first_group_absolute_latency_sum + 1;
                        end
                    end
                    if (previous_update_cycle >= 0
                            && cycle_count == previous_update_cycle + 1)
                        consecutive_intra_row_pairs
                            = consecutive_intra_row_pairs + 1;
                    previous_update_cycle = cycle_count;
                    if (row_id == 100
                            && d_update_selected_mask == 16'h000f) begin
                        if ($signed(d_update_delta[0 +: ACC_BITS]) !== -512)
                            $fatal(1, "M127 -512 boundary mismatch");
                        minus512_checks = minus512_checks + 1;
                    end
                    if (row_id == 101
                            && d_update_selected_mask == 16'h000f) begin
                        if ($signed(d_update_delta[0 +: ACC_BITS]) !== 512)
                            $fatal(1, "M127 +512 boundary mismatch");
                        plus512_checks = plus512_checks + 1;
                    end
                    if ($countones(remaining) <= 4) begin
                        case (selected_count)
                            1: tail_k1 = tail_k1 + 1;
                            2: tail_k2 = tail_k2 + 1;
                            3: tail_k3 = tail_k3 + 1;
                            4: tail_k4 = tail_k4 + 1;
                            default: $fatal(1, "M127 invalid tail size");
                        endcase
                    end
                    if (stall_snapshot_valid
                            && {d_update_delta, d_update_selected_mask,
                                d_update_block, d_update_row}
                               !==
                               {stalled_delta, stalled_selected,
                                stalled_block, stalled_row})
                        $fatal(1, "M127 stalled payload changed at release");
                    consumed = consumed | d_update_selected_mask;
                    remaining = remaining & ~d_update_selected_mask;
                    sources_checked = sources_checked + selected_count;
                    lanes_checked = lanes_checked + LANES;
                    updates_checked = updates_checked + 1;
                    update_count = update_count + 1;
                    stall_snapshot_valid = 0;
                    stall_run = 0;
                    if (random_stalls && remaining != 0) begin
                        prng_state = prng_state * 32'd1664525 + 32'd1013904223;
                        stall_left = prng_state[5:0];
                    end else begin
                        stall_left = 0;
                    end
                end
                @(negedge clk_core);
            end
            update_ready = 1;
            if (consumed !== initial_mask || remaining != 0
                    || d_remaining != 0
                    || update_count != (($countones(initial_mask) + 3) / 4))
                $fatal(1, "M127 source conservation mismatch row=%0d", row_id);
            if ($countones(initial_mask) == 16 && initial_stall == 0
                    && !random_stalls && update_count == 4) begin
                if (previous_update_cycle - first_update_cycle != 3)
                    $fatal(1, "M127 four-group stream is not II1");
                four_group_ii1_rows = four_group_ii1_rows + 1;
            end
            if (record_single_k4) begin
                if (update_count != 1 || $countones(initial_mask) != 4)
                    $fatal(1, "M127 inter-row probe geometry mismatch");
                if (previous_single_k4_cycle >= 0) begin
                    int interval;
                    interval = first_update_cycle - previous_single_k4_cycle;
                    if (interval < inter_row_single_k4_min_interval)
                        inter_row_single_k4_min_interval = interval;
                    if (interval > inter_row_single_k4_max_interval)
                        inter_row_single_k4_max_interval = interval;
                end
                previous_single_k4_cycle = first_update_cycle;
                inter_row_single_k4_updates = inter_row_single_k4_updates + 1;
            end
            rows_checked = rows_checked + 1;
        end
    endtask

    task automatic probe_cross_row_single_k4(
        input int block_id,
        input int first_row_id
    );
        logic [15:0] mask;
        logic [15:0] negate;
        int row_accept_cycle, update_cycle, expected_value;
        begin
            update_ready = 1;
            for (int item = 0; item < 4; item++) begin
                case (item)
                    0: begin mask = 16'h000f; negate = 16'h0005; end
                    1: begin mask = 16'h00f0; negate = 16'h0050; end
                    2: begin mask = 16'h0f00; negate = 16'h0a00; end
                    default: begin mask = 16'hf000; negate = 16'h5000; end
                endcase
                @(negedge clk_core);
                row_valid = 1;
                row_block = block_id[2:0];
                row_offset = (first_row_id + item) & 9'h1ff;
                row_source_mask = mask;
                row_negate_mask = negate;
                do @(posedge clk_core); while (!d_row_accept);
                row_accept_cycle = cycle_count;
                @(negedge clk_core);
                row_valid = 0;
                do @(posedge clk_core); while (!d_update_accept);
                update_cycle = cycle_count;
                if (d_update_selected_mask !== mask
                        || d_update_block !== block_id[2:0]
                        || d_update_row !== ((first_row_id + item) & 9'h1ff))
                    $fatal(1, "M127 cross-row K4 identity mismatch item=%0d", item);
                for (int lane = 0; lane < LANES; lane++) begin
                    expected_value = 0;
                    for (int source = 0; source < SOURCES; source++)
                        if (mask[source])
                            expected_value = expected_value
                                + (negate[source]
                                   ? -model_weight(block_id, source, lane)
                                   : model_weight(block_id, source, lane));
                    if ($signed(d_update_delta[lane * ACC_BITS +: ACC_BITS])
                            !== expected_value)
                        $fatal(1, "M127 cross-row K4 numeric mismatch");
                end
                first_group_checks = first_group_checks + 1;
                if (update_cycle - row_accept_cycle != 1)
                    $fatal(1, "M127 cross-row first-group latency is not one cycle");
                first_group_absolute_latency_sum
                    = first_group_absolute_latency_sum + 1;
                if (previous_single_k4_cycle >= 0) begin
                    int interval;
                    interval = update_cycle - previous_single_k4_cycle;
                    if (interval < inter_row_single_k4_min_interval)
                        inter_row_single_k4_min_interval = interval;
                    if (interval > inter_row_single_k4_max_interval)
                        inter_row_single_k4_max_interval = interval;
                end
                previous_single_k4_cycle = update_cycle;
                inter_row_single_k4_updates = inter_row_single_k4_updates + 1;
                rows_checked = rows_checked + 1;
                updates_checked = updates_checked + 1;
                sources_checked = sources_checked + 4;
                lanes_checked = lanes_checked + LANES;
                canonical_checks = canonical_checks + 1;
                tail_k4 = tail_k4 + 1;
                #0.1;
                if (!d_row_done || d_remaining != 0)
                    $fatal(1, "M127 cross-row K4 completion mismatch");
            end
        end
    endtask

    task automatic expect_fault(input int kind);
        begin
            @(negedge clk_core);
            if (kind == 0) begin
                row_valid = 1;
                row_block = d_resident_block;
                row_offset = 9;
                row_source_mask = 16'h0002;
                row_negate_mask = 0;
                cache_attacks = cache_attacks + 1;
            end else if (kind == 1) begin
                row_valid = 1;
                row_block = d_resident_block + 1'b1;
                row_offset = 10;
                row_source_mask = 16'h0001;
                row_negate_mask = 0;
                block_attacks = block_attacks + 1;
            end else begin
                weight_fill_valid = 1;
                weight_fill_block = d_resident_block;
                weight_fill_source = 0;
                weight_fill_beat = 2;
                weight_fill_data = 0;
                fill_sequence_attacks = fill_sequence_attacks + 1;
            end
            #0.1;
            if (!d_protocol_error || d_fill_accept || d_row_accept
                    || d_update_valid)
                $fatal(1, "M127 attack kind=%0d not fail closed", kind);
            @(posedge clk_core);
            @(negedge clk_core);
            drive_idle();
            repeat (2) @(posedge clk_core);
            if (!d_protocol_error)
                $fatal(1, "M127 attack kind=%0d not sticky", kind);
        end
    endtask

    task automatic probe_reset_request_isolation;
        begin
            clean_reset();
            @(negedge clk_core);
            compare_enable = 0;
            rst_core = 1;
            weight_fill_valid = 1;
            weight_fill_block = 2;
            weight_fill_source = 1;
            weight_fill_beat = 0;
            row_valid = 1;
            row_block = 2;
            row_source_mask = 16'h0001;
            update_ready = 1;
            #0.1;
            if (d_fill_ready || d_fill_accept || d_row_ready || d_row_accept
                    || d_update_valid || d_update_accept || d_row_done
                    || d_protocol_error)
                $fatal(1, "M127 reset request isolation counterexample");
            repeat (2) @(posedge clk_core);
            reset_isolation_checks = reset_isolation_checks + 1;
            @(negedge clk_core);
            drive_idle();
            rst_core = 0;
            compare_enable = 1;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic probe_reset_stalled_pipeline;
        begin
            clean_reset();
            fill_block(6);
            @(negedge clk_core);
            update_ready = 0;
            row_valid = 1;
            row_block = 6;
            row_offset = 333;
            row_source_mask = 16'hffff;
            row_negate_mask = 16'ha55a;
            do @(posedge clk_core); while (!d_row_accept);
            @(negedge clk_core);
            row_valid = 0;
            #0.1;
            if (!d_update_valid || !d_pipe_valid)
                $fatal(1, "M127 stalled pipeline was not populated");
            compare_enable = 0;
            rst_core = 1;
            weight_fill_valid = 1;
            row_valid = 1;
            update_ready = 1;
            #0.1;
            if (d_fill_ready || d_fill_accept || d_row_ready || d_row_accept
                    || d_update_valid || d_update_accept || d_row_done
                    || d_protocol_error)
                $fatal(1, "M127 reset failed to hide populated pipeline");
            repeat (2) @(posedge clk_core);
            if (d_cache_valid != 0 || d_remaining != 0 || d_pipe_valid
                    || d_busy)
                $fatal(1, "M127 reset failed to clear pipeline/cache state");
            reset_isolation_checks = reset_isolation_checks + 1;
            pipeline_stall_flush_checks = pipeline_stall_flush_checks + 1;
            @(negedge clk_core);
            drive_idle();
            rst_core = 0;
            compare_enable = 1;
            update_ready = 1;
            repeat (2) @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        update_ready = 0;
        compare_enable = 0;
        cycle_count = 0;
        cycle_exact_checks = 0;
        rows_checked = 0;
        updates_checked = 0;
        sources_checked = 0;
        lanes_checked = 0;
        canonical_checks = 0;
        stall_cycles = 0;
        max_stall_burst = 0;
        consecutive_intra_row_pairs = 0;
        four_group_ii1_rows = 0;
        tail_k1 = 0;
        tail_k2 = 0;
        tail_k3 = 0;
        tail_k4 = 0;
        plus512_checks = 0;
        minus512_checks = 0;
        first_group_checks = 0;
        first_group_absolute_latency_sum = 0;
        first_group_additional_cycles_sum = 0;
        inter_row_single_k4_updates = 0;
        inter_row_single_k4_min_interval = 999999;
        inter_row_single_k4_max_interval = 0;
        previous_single_k4_cycle = -1;
        cache_transition_checks = 0;
        cache_attacks = 0;
        block_attacks = 0;
        fill_sequence_attacks = 0;
        reset_isolation_checks = 0;
        pipeline_stall_flush_checks = 0;
        prng_state = 32'h1275a17e;
        drive_idle();

        probe_reset_request_isolation();
        fill_block(3);

        check_row(3, 100, 16'h000f, 16'h0000, 0, 0, 0);
        check_row(3, 101, 16'h000f, 16'h000f, 0, 0, 0);
        check_row(3, 110, 16'h8000, 16'h8000, 0, 0, 0);
        check_row(3, 111, 16'h8001, 16'h8000, 0, 0, 0);
        check_row(3, 112, 16'h4210, 16'h4010, 0, 0, 0);
        check_row(3, 113, 16'h8421, 16'h8020, 0, 0, 0);
        check_row(3, 120, 16'h001f, 16'h0015, 0, 0, 0);
        check_row(3, 121, 16'h003f, 16'h002a, 0, 0, 0);
        check_row(3, 122, 16'h007f, 16'h0055, 0, 0, 0);
        check_row(3, 123, 16'h00ff, 16'h00aa, 0, 0, 0);
        check_row(3, 130, 16'hffff, 16'ha55a, 0, 0, 0);
        check_row(3, 131, 16'h0000, 16'h0000, 0, 0, 0);

        probe_cross_row_single_k4(3, 140);

        for (int row = 0; row < 64; row++) begin
            logic [15:0] random_mask;
            logic [15:0] random_negate;
            prng_state = prng_state * 32'd1664525 + 32'd1013904223;
            random_mask = prng_state[15:0];
            if (random_mask == 0)
                random_mask = 16'h0001;
            prng_state = prng_state * 32'd1664525 + 32'd1013904223;
            random_negate = random_mask & prng_state[15:0];
            check_row(3, 200 + row, random_mask, random_negate,
                      row == 0 ? 63 : -1, 1, 0);
        end

        fill_source(4, 0);
        if (!d_resident_valid || d_resident_block != 4
                || d_cache_valid != 16'h0001)
            $fatal(1, "M127 block transition cache invalidation failed");
        cache_transition_checks = cache_transition_checks + 1;
        check_row(4, 300, 16'h0001, 16'h0000, 7, 0, 0);
        expect_fault(0);
        clean_reset();
        fill_source(4, 0);
        expect_fault(1);
        clean_reset();
        fill_source(4, 0);
        expect_fault(2);

        probe_reset_stalled_pipeline();

        if (rows_checked != 81 || updates_checked < 100
                || sources_checked < 400 || lanes_checked != updates_checked * 96
                || canonical_checks != updates_checked
                || tail_k1 == 0 || tail_k2 == 0 || tail_k3 == 0 || tail_k4 == 0
                || four_group_ii1_rows == 0
                || consecutive_intra_row_pairs < 3
                || max_stall_burst < 63
                || plus512_checks != 1 || minus512_checks != 1
                || inter_row_single_k4_updates != 4
                || inter_row_single_k4_min_interval != 2
                || inter_row_single_k4_max_interval != 2
                || cache_transition_checks != 1 || cache_attacks != 1
                || block_attacks != 1 || fill_sequence_attacks != 1
                || reset_isolation_checks != 2
                || pipeline_stall_flush_checks != 1
                || cycle_exact_checks < 1000)
            $fatal(1, "M127 independent aggregate mismatch rows=%0d updates=%0d sources=%0d lanes=%0d stalls=%0d maxstall=%0d tails=%0d/%0d/%0d/%0d ii1rows=%0d inter=%0d/%0d exact=%0d",
                   rows_checked, updates_checked, sources_checked,
                   lanes_checked, stall_cycles, max_stall_burst,
                   tail_k1, tail_k2, tail_k3, tail_k4,
                   four_group_ii1_rows, inter_row_single_k4_min_interval,
                   inter_row_single_k4_max_interval, cycle_exact_checks);

        $display("PASS M127 independent hammer rows=%0d updates=%0d sources=%0d lanes=%0d canonical=%0d stall_cycles=%0d max_stall_burst=%0d tail_k1=%0d tail_k2=%0d tail_k3=%0d tail_k4=%0d intra_row_ii1_pairs=%0d four_group_ii1_rows=%0d inter_row_single_k4_updates=%0d inter_row_single_k4_min_interval=%0d inter_row_single_k4_max_interval=%0d plus512=%0d minus512=%0d cycle_exact_checks=%0d first_group_checks=%0d first_group_absolute_latency_cycles=1 first_group_additional_cycles_vs_m125=0 cache_transition_checks=%0d cache_attacks=%0d block_attacks=%0d fill_sequence_attacks=%0d reset_isolation_checks=%0d pipeline_stall_flush_checks=%0d pair_sum_payload_bits=1920 full_elastic_stage_bits_at_least=1950 canonical_groups=true valid_numeric_equivalence=true reset_isolation=true intra_row_update_ii1=true cross_row_single_group_ii1=false dc_frequency_improvement=false physical_speedup=false system_speedup=false headline=false",
                 rows_checked, updates_checked, sources_checked,
                 lanes_checked, canonical_checks, stall_cycles,
                 max_stall_burst, tail_k1, tail_k2, tail_k3, tail_k4,
                 consecutive_intra_row_pairs, four_group_ii1_rows,
                 inter_row_single_k4_updates,
                 inter_row_single_k4_min_interval,
                 inter_row_single_k4_max_interval,
                 plus512_checks, minus512_checks, cycle_exact_checks,
                 first_group_checks, cache_transition_checks,
                 cache_attacks, block_attacks, fill_sequence_attacks,
                 reset_isolation_checks, pipeline_stall_flush_checks);
        $finish;
    end
endmodule

`default_nettype wire
