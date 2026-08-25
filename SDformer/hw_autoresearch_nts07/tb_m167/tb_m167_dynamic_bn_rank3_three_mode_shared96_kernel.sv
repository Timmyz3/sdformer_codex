`timescale 1ns/1ps
`default_nettype none

module tb_m167_dynamic_bn_rank3_three_mode_shared96_kernel;
    localparam int MAX_RESULTS = 400;

    logic clk_core;
    logic rst_core;
    logic issue_valid;
    logic issue_ready;
    logic [1:0] issue_mode;
    logic [15:0] issue_tag;
    logic signed [7:0] front_data [0:1][0:15];
    logic signed [7:0] front_right_factor [0:2][0:1];
    logic signed [7:0] back_rank_data [0:2][0:15];
    logic signed [7:0] back_folded_left [0:1][0:2][0:15];
    logic signed [23:0] back_folded_bias [0:1][0:15];
    logic signed [23:0] back_threshold;
    logic signed [7:0] prefold_a [0:95];
    logic signed [7:0] prefold_b [0:95];
    logic issue_accept;
    logic result_valid;
    logic result_ready;
    logic [1:0] result_mode;
    logic [15:0] result_tag;
    logic signed [16:0] front_projection_delta [0:2][0:15];
    logic signed [8:0] front_moment_sum_delta [0:15];
    logic [16:0] front_moment_sumsq_delta [0:15];
    logic [31:0] back_event_bits;
    logic signed [23:0] back_event_amplitude;
    logic signed [15:0] prefold_product [0:95];
    logic result_accept;
    logic [95:0] main_product_active_mask;
    logic [31:0] square_product_active_mask;
    logic protocol_error;
    logic busy;

    logic random_stall_mode;
    logic force_stall_mode;
    logic throughput_mode;
    integer expected_write;
    integer expected_read;
    integer issue_count;
    integer result_count;
    integer front_count;
    integer back_count;
    integer prefold_count;
    integer output_stall_cycles;
    integer same_cycle_replace_count;
    integer consecutive_issue_hits;
    integer previous_issue_cycle;
    integer cycle_count;
    integer amplitude_checks;

    logic [1:0] expected_mode [0:MAX_RESULTS-1];
    logic [15:0] expected_tag [0:MAX_RESULTS-1];
    logic signed [16:0] expected_projection
        [0:MAX_RESULTS-1][0:2][0:15];
    logic signed [8:0] expected_sum [0:MAX_RESULTS-1][0:15];
    logic [16:0] expected_sumsq [0:MAX_RESULTS-1][0:15];
    logic [31:0] expected_events [0:MAX_RESULTS-1];
    logic signed [23:0] expected_amplitude [0:MAX_RESULTS-1];
    logic signed [15:0] expected_prefold [0:MAX_RESULTS-1][0:95];

    m167_dynamic_bn_rank3_three_mode_shared96_kernel dut (.*);

    bind m167_dynamic_bn_rank3_three_mode_shared96_kernel
        m167_dynamic_bn_rank3_three_mode_shared96_kernel_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    task automatic clear_expected_slot(input integer slot);
        begin
            expected_events[slot] = '0;
            expected_amplitude[slot] = '0;
            for (int rank = 0; rank < 3; rank++)
                for (int lane = 0; lane < 16; lane++)
                    expected_projection[slot][rank][lane] = '0;
            for (int lane = 0; lane < 16; lane++) begin
                expected_sum[slot][lane] = '0;
                expected_sumsq[slot][lane] = '0;
            end
            for (int product_slot = 0; product_slot < 96; product_slot++)
                expected_prefold[slot][product_slot] = '0;
        end
    endtask

    task automatic load_issue(input integer index, input integer tag_base);
        integer slot;
        integer reconstructed;
        logic [31:0] packed_word;
        begin
            slot = expected_write;
            if (slot >= MAX_RESULTS)
                $fatal(1, "M167 expected scoreboard overflow");
            issue_mode = index % 3;
            issue_tag = tag_base + index;
            clear_expected_slot(slot);
            for (int row = 0; row < 2; row++) begin
                for (int lane = 0; lane < 16; lane++) begin
                    front_data[row][lane] = (($urandom % 129) - 64);
                    back_folded_bias[row][lane]
                        = ((index*19 + row*13 + lane*7) % 257) - 128;
                    for (int rank = 0; rank < 3; rank++)
                        back_folded_left[row][rank][lane]
                            = ((index*5 + row*11 + rank*3 + lane) % 17) - 8;
                end
            end
            for (int rank = 0; rank < 3; rank++) begin
                for (int row = 0; row < 2; row++)
                    front_right_factor[rank][row]
                        = ((index*7 + rank*5 + row*3) % 19) - 9;
                for (int lane = 0; lane < 16; lane++)
                    back_rank_data[rank][lane] = (($urandom % 129) - 64);
            end
            back_threshold = ((index % 31) + 1);
            for (int product_slot = 0; product_slot < 96; product_slot++) begin
                prefold_a[product_slot] = (($urandom % 129) - 64);
                prefold_b[product_slot] = (($urandom % 129) - 64);
            end

            expected_mode[slot] = issue_mode;
            expected_tag[slot] = issue_tag;
            case (issue_mode)
                0: begin
                    for (int rank = 0; rank < 3; rank++) begin
                        for (int lane = 0; lane < 16; lane++) begin
                            expected_projection[slot][rank][lane]
                                = $signed(front_data[0][lane])
                                    * $signed(front_right_factor[rank][0])
                                + $signed(front_data[1][lane])
                                    * $signed(front_right_factor[rank][1]);
                        end
                    end
                    for (int lane = 0; lane < 16; lane++) begin
                        expected_sum[slot][lane]
                            = $signed(front_data[0][lane])
                                + $signed(front_data[1][lane]);
                        expected_sumsq[slot][lane]
                            = $signed(front_data[0][lane])
                                * $signed(front_data[0][lane])
                            + $signed(front_data[1][lane])
                                * $signed(front_data[1][lane]);
                    end
                end
                1: begin
                    packed_word = '0;
                    for (int row = 0; row < 2; row++) begin
                        for (int lane = 0; lane < 16; lane++) begin
                            reconstructed = $signed(back_folded_bias[row][lane]);
                            for (int rank = 0; rank < 3; rank++)
                                reconstructed = reconstructed
                                    + $signed(back_rank_data[rank][lane])
                                    * $signed(back_folded_left[row][rank][lane]);
                            packed_word[(row*16)+lane]
                                = reconstructed >= $signed(back_threshold);
                        end
                    end
                    expected_events[slot] = packed_word;
                    expected_amplitude[slot] = back_threshold;
                end
                2: begin
                    for (int product_slot = 0; product_slot < 96;
                            product_slot++) begin
                        expected_prefold[slot][product_slot]
                            = $signed(prefold_a[product_slot])
                                * $signed(prefold_b[product_slot]);
                    end
                end
                default: $fatal(1, "M167 legal driver generated mode3");
            endcase
            expected_write = expected_write + 1;
        end
    endtask

    task automatic drive_issues(input integer count, input integer tag_base);
        integer sent;
        begin
            sent = 0;
            @(negedge clk_core);
            load_issue(sent, tag_base);
            issue_valid = 1'b1;
            while (sent < count) begin
                @(posedge clk_core);
                if (issue_accept) begin
                    sent = sent + 1;
                    if (sent < count) begin
                        @(negedge clk_core);
                        load_issue(sent, tag_base);
                    end else begin
                        @(negedge clk_core);
                        issue_valid = 1'b0;
                    end
                end
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core || force_stall_mode)
            result_ready <= 1'b0;
        else if (random_stall_mode)
            result_ready <= ($urandom_range(0, 3) != 0);
        else
            result_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            issue_count <= 0;
            result_count <= 0;
            front_count <= 0;
            back_count <= 0;
            prefold_count <= 0;
            output_stall_cycles <= 0;
            same_cycle_replace_count <= 0;
            consecutive_issue_hits <= 0;
            previous_issue_cycle <= -1;
            amplitude_checks <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (issue_accept) begin
                issue_count <= issue_count + 1;
                case (issue_mode)
                    0: front_count <= front_count + 1;
                    1: back_count <= back_count + 1;
                    2: prefold_count <= prefold_count + 1;
                    default: $fatal(1, "M167 accepted illegal mode");
                endcase
                if (throughput_mode) begin
                    if (previous_issue_cycle >= 0) begin
                        if (cycle_count - previous_issue_cycle != 1)
                            $fatal(1, "M167 issue II drift=%0d",
                                cycle_count - previous_issue_cycle);
                        consecutive_issue_hits <= consecutive_issue_hits + 1;
                    end
                    previous_issue_cycle <= cycle_count;
                end
            end
            if (result_valid && !result_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (result_accept && issue_accept)
                same_cycle_replace_count <= same_cycle_replace_count + 1;
            if (result_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M167 unexpected result");
                if (result_mode !== expected_mode[expected_read]
                        || result_tag !== expected_tag[expected_read])
                    $fatal(1, "M167 header mismatch index=%0d", expected_read);
                case (result_mode)
                    0: begin
                        for (int rank = 0; rank < 3; rank++)
                            for (int lane = 0; lane < 16; lane++)
                                if (front_projection_delta[rank][lane]
                                        !== expected_projection[expected_read]
                                            [rank][lane])
                                    $fatal(1, "M167 front projection mismatch");
                        for (int lane = 0; lane < 16; lane++) begin
                            if (front_moment_sum_delta[lane]
                                    !== expected_sum[expected_read][lane]
                                    || front_moment_sumsq_delta[lane]
                                    !== expected_sumsq[expected_read][lane])
                                $fatal(1, "M167 front moment mismatch");
                        end
                    end
                    1: begin
                        if (back_event_bits !== expected_events[expected_read]
                                || back_event_amplitude
                                    !== expected_amplitude[expected_read])
                            $fatal(1, "M167 back event/amplitude mismatch");
                        amplitude_checks <= amplitude_checks + 1;
                    end
                    2: begin
                        for (int product_slot = 0; product_slot < 96;
                                product_slot++)
                            if (prefold_product[product_slot]
                                    !== expected_prefold[expected_read]
                                        [product_slot])
                                $fatal(1, "M167 prefold mismatch slot=%0d",
                                    product_slot);
                    end
                    default: $fatal(1, "M167 result mode3");
                endcase
                expected_read <= expected_read + 1;
                result_count <= result_count + 1;
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        issue_valid = 1'b0;
        issue_mode = '0;
        issue_tag = '0;
        result_ready = 1'b0;
        random_stall_mode = 1'b0;
        force_stall_mode = 1'b0;
        throughput_mode = 1'b0;
        expected_write = 0;
        expected_read = 0;
        cycle_count = 0;
        issue_count = 0;
        result_count = 0;
        front_count = 0;
        back_count = 0;
        prefold_count = 0;
        output_stall_cycles = 0;
        same_cycle_replace_count = 0;
        consecutive_issue_hits = 0;
        previous_issue_cycle = -1;
        amplitude_checks = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        throughput_mode = 1'b1;
        drive_issues(90, 16'h7100);
        wait (expected_read == expected_write);
        @(negedge clk_core);
        throughput_mode = 1'b0;
        previous_issue_cycle = -1;

        random_stall_mode = 1'b1;
        drive_issues(270, 16'h7200);
        wait (expected_read == expected_write);
        wait (!busy);
        @(negedge clk_core);
        random_stall_mode = 1'b0;

        // Preserve one already accepted legal result across a younger mode-3
        // protocol attack.  The attack must close later issues, not erase work.
        force_stall_mode = 1'b1;
        @(negedge clk_core);
        load_issue(0, 16'h73f0);
        issue_valid = 1'b1;
        do @(posedge clk_core); while (!issue_accept);
        @(negedge clk_core);
        issue_mode = 2'd3;
        issue_tag = 16'h73ff;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 1'b0;
        force_stall_mode = 1'b0;
        wait (expected_read == expected_write);
        repeat (2) @(posedge clk_core);

        if (!protocol_error || issue_ready)
            $fatal(1, "M167 protocol attack did not fail closed");
        if (issue_count != 361 || result_count != 361
                || front_count != 121 || back_count != 120
                || prefold_count != 120 || amplitude_checks != 120)
            $fatal(1, "M167 population drift issues=%0d results=%0d modes=%0d/%0d/%0d amplitudes=%0d",
                issue_count, result_count, front_count, back_count,
                prefold_count, amplitude_checks);
        if (consecutive_issue_hits < 89 || output_stall_cycles == 0
                || same_cycle_replace_count == 0)
            $fatal(1, "M167 recurrence/cover counters missing");

        $display("PASS M167 dynamic-BN rank3 three-mode shared96 kernel VCS issues=361 results=361 front_issues=121 back_issues=120 prefold_issues=120 main_signed_int8_product_slots=96 front_square_lanes=32 front_products=11616 back_products=11520 prefold_products=11520 front_squares=3872 consecutive_issue_ii1_hits=%0d same_cycle_result_replace=%0d output_stall_cycles=%0d amplitude_sideband_checks=120 protocol_attacks=1 full_front_tile_issues=5 full_back_tile_issues=5 prefold_products_per_group=640 prefold_issue_cycles_per_group=7 full_rank3_products_per_tile=960 dense_t10_products_per_tile=1600 dense_capacity_lower_bound_cycles=17 conditional_capacity_cycle_boundary=1.7 shared_full_controller=false paft_valid825=false physical_speedup=false system_speedup=false headline=false",
            consecutive_issue_hits, same_cycle_replace_count,
            output_stall_cycles);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M167 watchdog timeout");
    end
endmodule

`default_nettype wire
