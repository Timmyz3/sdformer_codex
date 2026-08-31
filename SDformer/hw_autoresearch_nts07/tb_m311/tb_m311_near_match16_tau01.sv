module tb_m311_near_match16_tau01;
    localparam integer TOTAL = 2200;
    localparam integer DEPTH = 4096;

    logic core_clk = 1'b0;
    logic reset_n = 1'b0;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_pattern;
    logic [255:0] in_centers_flat;
    logic [1:0] in_tau;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_original_pattern;
    logic [15:0] out_selected_pattern;
    logic [4:0] out_selected_distance;
    logic [4:0] out_population;
    logic [1:0] out_tau;
    logic out_snapped;
    logic out_exact_hit;
    logic out_positive_distance;
    logic input_accepted_last;

    logic [15:0] expected_original [0:DEPTH-1];
    logic [15:0] expected_selected [0:DEPTH-1];
    logic [4:0] expected_distance [0:DEPTH-1];
    logic [4:0] expected_population [0:DEPTH-1];
    logic [1:0] expected_tau [0:DEPTH-1];
    logic expected_snapped [0:DEPTH-1];
    logic expected_exact [0:DEPTH-1];
    logic expected_positive [0:DEPTH-1];
    integer write_index;
    integer read_index;
    integer generated;
    integer exact_count;
    integer positive_count;
    integer rejected_count;
    integer guard_count;
    integer stall_count;
    integer tie_count;

    always #1.5 core_clk = ~core_clk;

    function automatic integer popcount16(input logic [15:0] value);
        integer index;
        begin
            popcount16 = 0;
            for (index = 0; index < 16; index = index + 1)
                popcount16 = popcount16 + value[index];
        end
    endfunction

    task automatic make_transaction(input integer transaction_index);
        integer center;
        integer bit_index;
        logic [15:0] base;
        begin
            in_tau = transaction_index[0];
            in_pattern = $urandom;
            for (center = 0; center < 16; center = center + 1)
                in_centers_flat[center * 16 +: 16] =
                    16'h0800 + center * 16'h0311 + transaction_index * 16'h0013;
            case (transaction_index)
                0: begin
                    in_tau = 0;
                    in_pattern = 16'h00f3;
                    in_centers_flat[5 * 16 +: 16] = 16'h00f3;
                end
                1: begin
                    in_tau = 1;
                    in_pattern = 16'h00f3;
                    in_centers_flat[5 * 16 +: 16] = 16'h00f2;
                end
                2: begin
                    in_tau = 0;
                    in_pattern = 16'h00f3;
                    in_centers_flat[5 * 16 +: 16] = 16'h00f2;
                end
                3: begin
                    in_tau = 1;
                    in_pattern = 16'h00f3;
                    in_centers_flat[5 * 16 +: 16] = 16'h00f0;
                end
                4: begin
                    in_tau = 1;
                    in_pattern = 16'h0000;
                    in_centers_flat[5 * 16 +: 16] = 16'h0000;
                end
                5: begin
                    in_tau = 1;
                    in_pattern = 16'h0001;
                    in_centers_flat[5 * 16 +: 16] = 16'h0001;
                end
                6: begin
                    in_tau = 1;
                    in_pattern = 16'h0003;
                    in_centers_flat[3 * 16 +: 16] = 16'h0007;
                    in_centers_flat[9 * 16 +: 16] = 16'h000b;
                    tie_count = tie_count + 1;
                end
                default: begin
                    if ((transaction_index % 7) == 0) begin
                        center = transaction_index % 16;
                        base = in_centers_flat[center * 16 +: 16];
                        if (popcount16(base) < 2)
                            base = base | 16'h0003;
                        in_centers_flat[center * 16 +: 16] = base;
                        bit_index = (transaction_index / 7) % 16;
                        in_pattern = base ^ (16'h0001 << bit_index);
                        in_tau = 1;
                    end
                end
            endcase
        end
    endtask

    m311_near_match16_tau01 u_dut (.*);
    m311_near_match16_tau01_assertions u_sva (.*);

    integer center_index;
    integer distance;
    integer best_distance;
    integer population;
    logic [15:0] center_value;
    logic [15:0] best_center;
    logic snap;
    always @(posedge core_clk) begin
        if (reset_n) begin
            input_accepted_last <= in_valid && in_ready;
            if (in_valid && in_ready) begin
                best_center = in_centers_flat[15:0];
                best_distance = popcount16(in_pattern ^ best_center);
                for (center_index = 1; center_index < 16;
                        center_index = center_index + 1) begin
                    center_value = in_centers_flat[center_index * 16 +: 16];
                    distance = popcount16(in_pattern ^ center_value);
                    if ((distance < best_distance) ||
                            ((distance == best_distance) &&
                             (center_value < best_center))) begin
                        best_distance = distance;
                        best_center = center_value;
                    end
                end
                population = popcount16(in_pattern);
                snap = population >= 2 && best_distance <= in_tau;
                expected_original[write_index] = in_pattern;
                expected_selected[write_index] = snap ? best_center : in_pattern;
                expected_distance[write_index] = best_distance;
                expected_population[write_index] = population;
                expected_tau[write_index] = in_tau;
                expected_snapped[write_index] = snap;
                expected_exact[write_index] = snap && best_distance == 0;
                expected_positive[write_index] = snap && best_distance != 0;
                write_index = write_index + 1;
            end
            if (out_valid && !out_ready)
                stall_count = stall_count + 1;
            if (out_valid && out_ready) begin
                if (read_index >= write_index)
                    $fatal(1, "M311 output without queued input");
                if (out_original_pattern !== expected_original[read_index] ||
                        out_selected_pattern !== expected_selected[read_index] ||
                        out_selected_distance !== expected_distance[read_index] ||
                        out_population !== expected_population[read_index] ||
                        out_tau !== expected_tau[read_index] ||
                        out_snapped !== expected_snapped[read_index] ||
                        out_exact_hit !== expected_exact[read_index] ||
                        out_positive_distance !== expected_positive[read_index])
                    $fatal(1, "M311 numerical/order mismatch at %0d", read_index);
                if (out_exact_hit) exact_count = exact_count + 1;
                if (out_positive_distance) positive_count = positive_count + 1;
                if (!out_snapped && out_population >= 2)
                    rejected_count = rejected_count + 1;
                if (!out_snapped && out_population < 2)
                    guard_count = guard_count + 1;
                read_index = read_index + 1;
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
            out_ready <= ($urandom_range(0, 3) != 0);
            if (input_accepted_last)
                in_valid <= 1'b0;
            if ((!in_valid || input_accepted_last) && generated < TOTAL &&
                    (generated < 16 || $urandom_range(0, 3) != 0)) begin
                make_transaction(generated);
                in_valid <= 1'b1;
                generated = generated + 1;
            end
        end
    end

    integer watchdog;
    initial begin
        in_valid = 0;
        out_ready = 0;
        in_pattern = 0;
        in_centers_flat = 0;
        in_tau = 0;
        write_index = 0;
        read_index = 0;
        generated = 0;
        exact_count = 0;
        positive_count = 0;
        rejected_count = 0;
        guard_count = 0;
        stall_count = 0;
        tie_count = 0;
        input_accepted_last = 0;
        repeat (5) @(posedge core_clk);
        reset_n = 1;
        watchdog = 0;
        while (read_index < TOTAL && watchdog < 30000) begin
            @(posedge core_clk);
            watchdog = watchdog + 1;
        end
        @(negedge core_clk);
        $display("M311_TERMINATION generated=%0d written=%0d read=%0d exact=%0d positive=%0d rejected=%0d guard=%0d stalls=%0d ties=%0d watchdog=%0d",
                 generated, write_index, read_index, exact_count, positive_count,
                 rejected_count, guard_count, stall_count, tie_count, watchdog);
        if (read_index != TOTAL || write_index != TOTAL ||
                exact_count == 0 || positive_count == 0 ||
                rejected_count == 0 || guard_count == 0 ||
                stall_count == 0 || tie_count == 0)
            $fatal(1, "M311 coverage/termination failure");
        $display("PASS M311 near-match16 tau01 VCS transactions=%0d exact=%0d positive=%0d rejected=%0d guard=%0d stalls=%0d ties=%0d mismatches=0 ii1=true tau0_subset=true executable_sram=false system_speedup=false headline=false",
                 read_index, exact_count, positive_count, rejected_count,
                 guard_count, stall_count, tie_count);
        $finish;
    end

endmodule
