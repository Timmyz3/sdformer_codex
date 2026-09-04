`timescale 1ns/1ps
`default_nettype none

module tb_m276_m235_full220800_protocol_ii;
    localparam int TAG_BITS = 24;
    localparam int VECTOR_COUNT = 220800;
    localparam int STALL_VECTOR = 110399;
    localparam int STALL_TAG = STALL_VECTOR + 1;
    localparam int INTRINSIC_LATENCY = 8;
    localparam int INTRINSIC_II = 9;

    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] request_tag;
    logic [21:0] variance_plus_epsilon_uq6p16;
    logic signed [17:0] mean_sq3p14;
    logic signed [15:0] gamma_sq1p14, beta_sq1p14;
    logic result_valid, result_ready, result_accept;
    logic [TAG_BITS-1:0] result_tag;
    logic [19:0] invstd_uq4p16;
    logic signed [19:0] alpha_sq3p16, offset_sq3p16;
    logic protocol_error, busy;
    logic [3:0] debug_state;
    logic [31:0] debug_request_count, debug_result_count;

    integer vector_variance [0:VECTOR_COUNT-1];
    integer vector_mean [0:VECTOR_COUNT-1];
    integer vector_gamma [0:VECTOR_COUNT-1];
    integer vector_beta [0:VECTOR_COUNT-1];
    integer vector_even_exp [0:VECTOR_COUNT-1];
    integer vector_mantissa [0:VECTOR_COUNT-1];
    integer vector_lut [0:VECTOR_COUNT-1];
    integer vector_invstd [0:VECTOR_COUNT-1];
    integer vector_alpha [0:VECTOR_COUNT-1];
    integer vector_offset [0:VECTOR_COUNT-1];
    integer lut_hits [0:63];

    integer cycle_count, corpus_requests, corpus_results, output_mismatches;
    integer request_backpressure_cycles, backpressured_requests;
    integer intrinsic_ii_samples, intrinsic_ii_min, intrinsic_ii_max;
    integer max_first_result_latency, result_stall_cycles;
    integer protocol_attacks, attack_setup_requests;
    integer last_corpus_accept_cycle, last_corpus_accept_tag;
    integer inflight_accept_cycle;
    bit inflight_first_result_seen;
    integer actual_offset_negative, actual_offset_positive, actual_offset_zero;

    m235_dynamic_bn_segmented_lut_newton_coefficient_engine dut (.*);
    m235_dynamic_bn_segmented_lut_newton_coefficient_engine_assertions base_sva (.*);
    m276_m235_full220800_protocol_ii_assertions protocol_sva (.*);

    task automatic drive_corpus_vector(input integer vector_id);
        begin
            request_tag = vector_id + 1;
            variance_plus_epsilon_uq6p16 = vector_variance[vector_id];
            mean_sq3p14 = vector_mean[vector_id];
            gamma_sq1p14 = vector_gamma[vector_id];
            beta_sq1p14 = vector_beta[vector_id];
            request_valid = 1'b1;
        end
    endtask

    always @(posedge clk_core) begin : scoreboard
        integer result_index, interval, latency;
        if (rst_core) begin
            cycle_count = 0;
            corpus_requests = 0;
            corpus_results = 0;
            output_mismatches = 0;
            request_backpressure_cycles = 0;
            intrinsic_ii_samples = 0;
            intrinsic_ii_min = 32'h7fffffff;
            intrinsic_ii_max = 0;
            max_first_result_latency = 0;
            result_stall_cycles = 0;
            attack_setup_requests = 0;
            last_corpus_accept_cycle = -1;
            last_corpus_accept_tag = -1;
            inflight_accept_cycle = -1;
            inflight_first_result_seen = 1'b0;
            actual_offset_negative = 0;
            actual_offset_positive = 0;
            actual_offset_zero = 0;
        end else begin
            cycle_count = cycle_count + 1;

            if (request_valid && !request_ready && !protocol_error)
                request_backpressure_cycles = request_backpressure_cycles + 1;

            if (request_accept) begin
                inflight_accept_cycle = cycle_count;
                inflight_first_result_seen = 1'b0;
                if (request_tag >= 1 && request_tag <= VECTOR_COUNT) begin
                    if (request_tag !== corpus_requests + 1)
                        $fatal(1, "M276 request sequence drift expected=%0d got=%0d",
                               corpus_requests + 1, request_tag);
                    if (last_corpus_accept_cycle >= 0 &&
                            last_corpus_accept_tag != STALL_TAG) begin
                        interval = cycle_count - last_corpus_accept_cycle;
                        intrinsic_ii_samples = intrinsic_ii_samples + 1;
                        if (interval < intrinsic_ii_min)
                            intrinsic_ii_min = interval;
                        if (interval > intrinsic_ii_max)
                            intrinsic_ii_max = interval;
                        if (interval != INTRINSIC_II)
                            $fatal(1, "M276 intrinsic II drift tag=%0d interval=%0d",
                                   request_tag, interval);
                    end
                    last_corpus_accept_cycle = cycle_count;
                    last_corpus_accept_tag = request_tag;
                    corpus_requests = corpus_requests + 1;
                end else begin
                    attack_setup_requests = attack_setup_requests + 1;
                end
            end

            if (result_valid && !inflight_first_result_seen) begin
                latency = cycle_count - inflight_accept_cycle;
                inflight_first_result_seen = 1'b1;
                if (result_tag >= 1 && result_tag <= VECTOR_COUNT) begin
                    if (latency > max_first_result_latency)
                        max_first_result_latency = latency;
                    if (latency != INTRINSIC_LATENCY)
                        $fatal(1, "M276 first-result latency drift tag=%0d latency=%0d",
                               result_tag, latency);
                end
            end

            if (result_valid && !result_ready && !protocol_error &&
                    result_tag >= 1 && result_tag <= VECTOR_COUNT)
                result_stall_cycles = result_stall_cycles + 1;

            if (result_accept) begin
                if (!(result_tag >= 1 && result_tag <= VECTOR_COUNT))
                    $fatal(1, "M276 unexpected non-corpus result accept tag=%0d", result_tag);
                result_index = result_tag - 1;
                if (invstd_uq4p16 !== vector_invstd[result_index][19:0] ||
                        alpha_sq3p16 !== vector_alpha[result_index][19:0] ||
                        offset_sq3p16 !== vector_offset[result_index][19:0]) begin
                    output_mismatches = output_mismatches + 1;
                    $fatal(1, "M276/M235 mismatch tag=%0d inv=%0d/%0d alpha=%0d/%0d offset=%0d/%0d",
                           result_tag, vector_invstd[result_index], invstd_uq4p16,
                           vector_alpha[result_index], $signed(alpha_sq3p16),
                           vector_offset[result_index], $signed(offset_sq3p16));
                end
                if ($signed(offset_sq3p16) < 0)
                    actual_offset_negative = actual_offset_negative + 1;
                else if ($signed(offset_sq3p16) > 0)
                    actual_offset_positive = actual_offset_positive + 1;
                else
                    actual_offset_zero = actual_offset_zero + 1;
                corpus_results = corpus_results + 1;
            end
        end
    end

    task automatic illegal_zero_with_pending_result;
        integer requests_before, results_before;
        begin
            @(negedge clk_core);
            request_tag = 24'h276bad;
            variance_plus_epsilon_uq6p16 = vector_variance[0];
            mean_sq3p14 = vector_mean[0];
            gamma_sq1p14 = vector_gamma[0];
            beta_sq1p14 = vector_beta[0];
            request_valid = 1'b1;
            result_ready = 1'b0;
            do @(posedge clk_core); while (!request_accept);
            @(negedge clk_core);
            request_valid = 1'b0;
            wait (result_valid);
            requests_before = debug_request_count;
            results_before = debug_result_count;
            @(negedge clk_core);
            request_valid = 1'b1;
            variance_plus_epsilon_uq6p16 = 0;
            result_ready = 1'b1;
            #0.1;
            if (!protocol_error || request_accept || result_accept ||
                    request_ready || result_valid)
                $fatal(1, "M276 illegal pending-result fault atomicity failed");
            @(posedge clk_core);
            #0.2;
            if (debug_request_count != requests_before ||
                    debug_result_count != results_before || !protocol_error)
                $fatal(1, "M276 illegal pending-result state commit");
            protocol_attacks = protocol_attacks + 1;
            @(negedge clk_core);
            request_valid = 1'b0;
        end
    endtask

    initial begin
        #30000000;
        $fatal(1, "M276/M235 watchdog");
    end

    initial begin : load_drive_and_check
        integer fd, scan_status, vector_id, flat_index, row_count;
        integer variance_q, mean_q, gamma_q, beta_q;
        integer even_exp, mantissa_q, lut_index, inv_q, alpha_q, offset_q;
        integer exp_mask, tail_mask, waited_cycles, lut_bin;
        reg [2047:0] line;

        rst_core = 1'b1;
        request_valid = 1'b0;
        result_ready = 1'b0;
        request_tag = '0;
        variance_plus_epsilon_uq6p16 = '0;
        mean_sq3p14 = '0;
        gamma_sq1p14 = '0;
        beta_sq1p14 = '0;
        backpressured_requests = 0;
        protocol_attacks = 0;
        exp_mask = 0;
        tail_mask = 0;
        for (lut_bin = 0; lut_bin < 64; lut_bin = lut_bin + 1)
            lut_hits[lut_bin] = 0;

        fd = $fopen("results/m245_m235_full220800_vectors_r1_20260825/m245_m235_full220800_vectors.csv", "r");
        if (fd == 0)
            $fatal(1, "M276/M235 frozen full vector file missing");
        scan_status = $fgets(line, fd);
        row_count = 0;
        while (!$feof(fd)) begin
            scan_status = $fscanf(fd,
                "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                vector_id, flat_index, variance_q, mean_q, gamma_q, beta_q,
                even_exp, mantissa_q, lut_index, inv_q, alpha_q, offset_q);
            if (scan_status == 12) begin
                if (row_count >= VECTOR_COUNT || vector_id != row_count ||
                        flat_index != row_count)
                    $fatal(1, "M276 vector identity drift row=%0d vector=%0d flat=%0d",
                           row_count, vector_id, flat_index);
                if (variance_q <= 0 || lut_index < 0 || lut_index > 63)
                    $fatal(1, "M276 illegal frozen input row=%0d", row_count);
                vector_variance[row_count] = variance_q;
                vector_mean[row_count] = mean_q;
                vector_gamma[row_count] = gamma_q;
                vector_beta[row_count] = beta_q;
                vector_even_exp[row_count] = even_exp;
                vector_mantissa[row_count] = mantissa_q;
                vector_lut[row_count] = lut_index;
                vector_invstd[row_count] = inv_q;
                vector_alpha[row_count] = alpha_q;
                vector_offset[row_count] = offset_q;
                lut_hits[lut_index] = lut_hits[lut_index] + 1;
                case (even_exp)
                    -6: exp_mask = exp_mask | 1;
                    -4: exp_mask = exp_mask | 2;
                    -2: exp_mask = exp_mask | 4;
                     0: exp_mask = exp_mask | 8;
                     2: exp_mask = exp_mask | 16;
                     4: exp_mask = exp_mask | 32;
                    default: $fatal(1, "M276 unexpected even exponent row=%0d exp=%0d",
                                    row_count, even_exp);
                endcase
                case (row_count)
                    175162: tail_mask = tail_mask | 1;
                    175604: tail_mask = tail_mask | 2;
                    176110: tail_mask = tail_mask | 4;
                    182167: tail_mask = tail_mask | 8;
                    190728: tail_mask = tail_mask | 16;
                    219956: tail_mask = tail_mask | 32;
                    default: begin end
                endcase
                row_count = row_count + 1;
            end else if (!$feof(fd)) begin
                $fatal(1, "M276 vector parse failed status=%0d row=%0d",
                       scan_status, row_count);
            end
        end
        $fclose(fd);
        if (row_count != VECTOR_COUNT || exp_mask != 63 || tail_mask != 63)
            $fatal(1, "M276 population coverage failure rows=%0d exp_mask=%0h tail_mask=%0h",
                   row_count, exp_mask, tail_mask);
        for (lut_bin = 0; lut_bin < 64; lut_bin = lut_bin + 1)
            if (lut_hits[lut_bin] == 0)
                $fatal(1, "M276 LUT bin absent bin=%0d", lut_bin);

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        result_ready = 1'b1;
        drive_corpus_vector(0);

        for (vector_id = 0; vector_id < VECTOR_COUNT; vector_id = vector_id + 1) begin
            waited_cycles = 0;
            do begin
                @(posedge clk_core);
                if (!request_accept)
                    waited_cycles = waited_cycles + 1;
            end while (!request_accept);
            if (waited_cycles > 0)
                backpressured_requests = backpressured_requests + 1;

            if (vector_id == STALL_VECTOR) begin
                fork
                    begin : one_result_backpressure_window
                        @(negedge clk_core);
                        result_ready = 1'b0;
                        wait (result_valid && result_tag == STALL_TAG);
                        repeat (5) @(posedge clk_core);
                        @(negedge clk_core);
                        result_ready = 1'b1;
                    end
                join_none
            end

            @(negedge clk_core);
            if (vector_id + 1 < VECTOR_COUNT)
                drive_corpus_vector(vector_id + 1);
            else
                request_valid = 1'b0;
        end

        wait (corpus_results == VECTOR_COUNT);
        @(negedge clk_core);
        if (corpus_requests != VECTOR_COUNT || corpus_results != VECTOR_COUNT ||
                output_mismatches != 0 || debug_request_count != VECTOR_COUNT ||
                debug_result_count != VECTOR_COUNT ||
                backpressured_requests != VECTOR_COUNT - 1 ||
                intrinsic_ii_samples != VECTOR_COUNT - 2 ||
                intrinsic_ii_min != INTRINSIC_II ||
                intrinsic_ii_max != INTRINSIC_II ||
                max_first_result_latency != INTRINSIC_LATENCY ||
                result_stall_cycles != 5 ||
                actual_offset_negative != 117431 ||
                actual_offset_positive != 103369 ||
                actual_offset_zero != 0)
            $fatal(1, "M276 closure failure req=%0d res=%0d mismatch=%0d dbg=%0d/%0d bp_req=%0d ii_n=%0d ii=%0d/%0d lat=%0d stall=%0d offset_sign=%0d/%0d/%0d",
                   corpus_requests, corpus_results, output_mismatches,
                   debug_request_count, debug_result_count,
                   backpressured_requests, intrinsic_ii_samples,
                   intrinsic_ii_min, intrinsic_ii_max,
                   max_first_result_latency, result_stall_cycles,
                   actual_offset_negative, actual_offset_positive,
                   actual_offset_zero);

        illegal_zero_with_pending_result();
        if (protocol_attacks != 1 || attack_setup_requests != 1)
            $fatal(1, "M276 protocol coverage missing attacks=%0d setup=%0d",
                   protocol_attacks, attack_setup_requests);

        $display("PASS M276 M235 full220800 protocol_ii corpus_vectors=220800 corpus_requests=220800 corpus_results=220800 mismatches=0 first_result_latency=8 intrinsic_ii_min=9 intrinsic_ii_max=9 intrinsic_ii_samples=220798 backpressured_requests=220799 request_backpressure_cycles=%0d result_stalls=5 illegal_pending_attacks=1 attack_setup_requests=1 lut_bins=64 even_exponents=6 tail_extrema=6 unchanged_production_rtl=true new_speedup=false moment_finalizer=false event_equivalence=false full_bn=false system_speedup=false headline=false",
                 request_backpressure_cycles);
        $finish;
    end
endmodule

`default_nettype wire
