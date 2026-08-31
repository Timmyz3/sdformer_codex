module tb_m348_exact_q128_signed_residual_matcher;
    localparam integer TOTAL = 3000;
    localparam integer DEPTH = 4096;

    logic core_clk = 1'b0;
    logic reset_n = 1'b0;
    logic cfg_valid;
    logic cfg_ready;
    logic [2:0] cfg_group;
    logic [255:0] cfg_patterns_flat;
    logic cfg_commit;
    logic cfg_active;
    logic cfg_protocol_error;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_original_pattern;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_original_pattern;
    logic [15:0] out_best_center;
    logic [6:0] out_best_center_id;
    logic [4:0] out_best_distance;
    logic [4:0] out_population;
    logic out_use_pwp;
    logic out_fallback_bit_sparse;
    logic [15:0] out_plus_mask;
    logic [15:0] out_minus_mask;

    logic [15:0] patterns [0:127];
    logic [15:0] expected_original [0:DEPTH-1];
    logic [15:0] expected_center [0:DEPTH-1];
    logic [6:0] expected_id [0:DEPTH-1];
    logic [4:0] expected_distance [0:DEPTH-1];
    logic [4:0] expected_population [0:DEPTH-1];
    logic expected_use [0:DEPTH-1];
    logic [15:0] expected_plus [0:DEPTH-1];
    logic [15:0] expected_minus [0:DEPTH-1];
    integer expected_accept_cycle [0:DEPTH-1];

    integer cycle_count;
    integer generated;
    integer written;
    integer read_count;
    integer mismatch_count;
    integer use_count;
    integer fallback_count;
    integer mixed_signed_count;
    integer exact_count;
    integer tie_count;
    integer stall_count;
    integer protocol_attack_count;
    integer accept_run;
    integer retire_run;
    integer max_accept_run;
    integer max_retire_run;
    integer latency_min;
    integer latency_max;
    logic input_accepted_last;

    always #1.5 core_clk = ~core_clk;

    function automatic integer popcount16(input logic [15:0] value);
        integer bit_index;
        begin
            popcount16 = 0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                popcount16 = popcount16 + value[bit_index];
        end
    endfunction

    task automatic load_group(input integer group_index,
                              input logic commit_value);
        integer lane;
        begin
            @(negedge core_clk);
            cfg_patterns_flat = '0;
            for (lane = 0; lane < 16; lane = lane + 1)
                cfg_patterns_flat[lane * 16 +: 16] =
                    patterns[group_index * 16 + lane];
            cfg_group = group_index[2:0];
            cfg_commit = commit_value;
            cfg_valid = 1'b1;
            while (!cfg_ready)
                @(negedge core_clk);
            @(posedge core_clk);
            @(negedge core_clk);
            cfg_valid = 1'b0;
            cfg_commit = 1'b0;
        end
    endtask

    task automatic configure_patterns;
        integer group_index;
        begin
            for (group_index = 0; group_index < 8;
                    group_index = group_index + 1)
                load_group(group_index, group_index == 7);
            @(posedge core_clk);
            if (!cfg_active || cfg_protocol_error)
                $fatal(1, "M348 configuration activation failure");
        end
    endtask

    task automatic make_transaction(input integer transaction_index);
        begin
            case (transaction_index)
                0: in_original_pattern = patterns[37];
                1: in_original_pattern = 16'h0000;
                2: in_original_pattern = 16'h0001;
                3: in_original_pattern = 16'h0001;
                4: in_original_pattern = 16'h00fd;
                5: in_original_pattern = 16'h7ffe;
                default: begin
                    in_original_pattern = $urandom;
                    if ((transaction_index % 11) == 0)
                        in_original_pattern = patterns[transaction_index % 128];
                    else if ((transaction_index % 13) == 0)
                        in_original_pattern =
                            patterns[transaction_index % 128] ^
                            (16'h0001 << (transaction_index % 16));
                end
            endcase
        end
    endtask

    m348_exact_q128_signed_residual_matcher u_dut (.*);
    m348_exact_q128_signed_residual_matcher_assertions u_sva (.*);

    integer center_index;
    integer distance;
    integer best_distance;
    integer best_index;
    integer population;
    integer observed_latency;
    logic [15:0] best_center;
    logic use_pwp;
    logic [15:0] plus_mask;
    logic [15:0] minus_mask;
    always @(posedge core_clk) begin
        cycle_count = cycle_count + 1;
        if (reset_n) begin
            input_accepted_last <= in_valid && in_ready;
            if (in_valid && in_ready) begin
                best_index = 0;
                best_center = patterns[0];
                best_distance = popcount16(in_original_pattern ^ best_center);
                for (center_index = 1; center_index < 128;
                        center_index = center_index + 1) begin
                    distance = popcount16(
                        in_original_pattern ^ patterns[center_index]);
                    if (distance < best_distance) begin
                        best_distance = distance;
                        best_index = center_index;
                        best_center = patterns[center_index];
                    end else if (distance == best_distance) begin
                        tie_count = tie_count + 1;
                    end
                end
                population = popcount16(in_original_pattern);
                use_pwp = 1 + best_distance < population;
                plus_mask = use_pwp ?
                    (in_original_pattern & ~best_center) : in_original_pattern;
                minus_mask = use_pwp ?
                    (best_center & ~in_original_pattern) : 16'h0000;
                expected_original[written] = in_original_pattern;
                expected_center[written] = best_center;
                expected_id[written] = best_index[6:0];
                expected_distance[written] = best_distance[4:0];
                expected_population[written] = population[4:0];
                expected_use[written] = use_pwp;
                expected_plus[written] = plus_mask;
                expected_minus[written] = minus_mask;
                expected_accept_cycle[written] = cycle_count;
                written = written + 1;
                accept_run = accept_run + 1;
                if (accept_run > max_accept_run)
                    max_accept_run = accept_run;
            end else begin
                accept_run = 0;
            end

            if (out_valid && !out_ready)
                stall_count = stall_count + 1;
            if (out_valid && out_ready) begin
                if (read_count >= written)
                    $fatal(1, "M348 output without expected transaction");
                if (out_original_pattern !== expected_original[read_count] ||
                        out_best_center !== expected_center[read_count] ||
                        out_best_center_id !== expected_id[read_count] ||
                        out_best_distance !== expected_distance[read_count] ||
                        out_population !== expected_population[read_count] ||
                        out_use_pwp !== expected_use[read_count] ||
                        out_fallback_bit_sparse !== !expected_use[read_count] ||
                        out_plus_mask !== expected_plus[read_count] ||
                        out_minus_mask !== expected_minus[read_count]) begin
                    mismatch_count = mismatch_count + 1;
                    $fatal(1, "M348 numerical/order mismatch at %0d", read_count);
                end
                if (out_use_pwp) use_count = use_count + 1;
                else fallback_count = fallback_count + 1;
                if (out_use_pwp && out_plus_mask != 0 && out_minus_mask != 0)
                    mixed_signed_count = mixed_signed_count + 1;
                if (out_use_pwp && out_best_distance == 0)
                    exact_count = exact_count + 1;
                observed_latency = cycle_count - expected_accept_cycle[read_count];
                if (observed_latency < latency_min) latency_min = observed_latency;
                if (observed_latency > latency_max) latency_max = observed_latency;
                read_count = read_count + 1;
                retire_run = retire_run + 1;
                if (retire_run > max_retire_run)
                    max_retire_run = retire_run;
            end else begin
                retire_run = 0;
            end
        end else begin
            input_accepted_last <= 1'b0;
        end
    end

    always @(negedge core_clk) begin
        if (!reset_n) begin
            in_valid <= 1'b0;
            out_ready <= 1'b0;
        end else begin
            if (read_count < 400)
                out_ready <= 1'b1;
            else
                out_ready <= ($urandom_range(0, 4) != 0);
            if (input_accepted_last)
                in_valid <= 1'b0;
            if (cfg_active && (!in_valid || input_accepted_last) &&
                    generated < TOTAL &&
                    (generated < 512 || $urandom_range(0, 4) != 0)) begin
                make_transaction(generated);
                in_valid <= 1'b1;
                generated = generated + 1;
            end
        end
    end

    integer pattern_index;
    integer watchdog;
    initial begin
        cfg_valid = 0;
        cfg_group = 0;
        cfg_patterns_flat = 0;
        cfg_commit = 0;
        in_valid = 0;
        in_original_pattern = 0;
        out_ready = 0;
        cycle_count = 0;
        generated = 0;
        written = 0;
        read_count = 0;
        mismatch_count = 0;
        use_count = 0;
        fallback_count = 0;
        mixed_signed_count = 0;
        exact_count = 0;
        tie_count = 0;
        stall_count = 0;
        protocol_attack_count = 0;
        accept_run = 0;
        retire_run = 0;
        max_accept_run = 0;
        max_retire_run = 0;
        latency_min = 1 << 30;
        latency_max = 0;
        input_accepted_last = 0;
        for (pattern_index = 0; pattern_index < 128;
                pattern_index = pattern_index + 1)
            patterns[pattern_index] =
                16'h2101 + pattern_index * 16'h01f3;
        patterns[5] = 16'h0003;
        patterns[7] = 16'h00fe;
        patterns[9] = 16'h0005;
        patterns[37] = 16'h5a3c;

        repeat (5) @(posedge core_clk);
        reset_n = 1'b1;

        // Fail-closed attack: group one before group zero.
        cfg_group = 3'd1;
        cfg_patterns_flat = '0;
        cfg_commit = 1'b0;
        cfg_valid = 1'b1;
        @(posedge core_clk);
        @(negedge core_clk);
        cfg_valid = 1'b0;
        protocol_attack_count = protocol_attack_count + 1;
        @(posedge core_clk);
        if (!cfg_protocol_error || cfg_active)
            $fatal(1, "M348 protocol attack did not fail closed");

        reset_n = 1'b0;
        repeat (3) @(posedge core_clk);
        reset_n = 1'b1;
        configure_patterns();

        watchdog = 0;
        while (read_count < TOTAL && watchdog < 100000) begin
            @(posedge core_clk);
            watchdog = watchdog + 1;
        end
        @(negedge core_clk);
        in_valid = 1'b0;
        out_ready = 1'b1;
        if (read_count != TOTAL || written != TOTAL ||
                mismatch_count != 0 || use_count == 0 || fallback_count == 0 ||
                mixed_signed_count == 0 || exact_count == 0 ||
                tie_count == 0 || stall_count == 0 ||
                protocol_attack_count != 1 ||
                max_accept_run < 256 || max_retire_run < 128)
            $fatal(1, "M348 coverage/termination failure");
        $display("PASS M348 q128 exact signed matcher transactions=%0d use=%0d fallback=%0d mixed=%0d exact=%0d ties=%0d stalls=%0d protocol_attacks=%0d max_accept_run=%0d max_retire_run=%0d latency_min=%0d latency_max=%0d mismatches=%0d ii1=true center_id=true signed_residual=true exact_fallback=true system_speedup=false headline=false",
                 read_count, use_count, fallback_count, mixed_signed_count,
                 exact_count, tie_count, stall_count, protocol_attack_count,
                 max_accept_run, max_retire_run, latency_min, latency_max,
                 mismatch_count);
        $finish;
    end

endmodule
