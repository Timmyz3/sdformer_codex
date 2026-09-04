`timescale 1ns/1ps
`default_nettype none

module tb_m111_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int BANKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 24;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int COMMIT_VECTORS = BANKS * WIN_ROWS;
    localparam integer signed S24_MAX = 8388607;
    localparam integer signed S24_MIN = -8388608;

    logic clk_core, rst_core;
    logic window_start_valid, window_start_ready, window_start_accept;
    logic update_valid, update_ready, update_accept;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [VECTOR_BITS-1:0] update_delta;
    logic window_end_valid, window_end_ready, window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [VECTOR_BITS-1:0] commit_data;
    logic commit_last, window_done;
    logic [7:0] mem_rd_en;
    logic [8:0] mem_rd_addr [0:7];
    logic [VECTOR_BITS-1:0] mem_rd_data [0:7];
    logic [7:0] mem_wr_en;
    logic [8:0] mem_wr_addr [0:7];
    logic [VECTOR_BITS-1:0] mem_wr_data [0:7];
    logic protocol_error, window_active, busy;

    logic [VECTOR_BITS-1:0] memory [0:BANKS-1][0:WIN_ROWS-1];
    integer signed reference [0:BANKS-1][0:WIN_ROWS-1][0:LANES-1];
    bit reference_valid [0:BANKS-1][0:WIN_ROWS-1];
    bit commit_read_seen [0:BANKS-1][0:WIN_ROWS-1];

    bit pending_write;
    bit pending_overflow;
    int pending_bank;
    int pending_row;
    logic [VECTOR_BITS-1:0] pending_data;
    logic [VECTOR_BITS-1:0] calculated_data;
    bit calculated_overflow;
    integer signed base_value;
    integer signed delta_value;
    longint signed sum_value;

    bit allow_fault;
    bit automatic_commit_ready;
    int cycle_count;
    int current_commit_index;
    int current_commit_reads;
    int current_expected_valid_rows;
    int windows_completed;
    int positive_updates;
    int positive_writes;
    int positive_ii1_pairs;
    int positive_overlap_cycles;
    int positive_commit_stalls;
    int positive_commit_vectors;
    int positive_lane_checks;
    int lazy_clear_stale_zero_checks;
    int same_address_attacks;
    int overflow_attacks;
    bit prior_update_accept;

    m111_w384_signed24_accumulator_frontend dut (.*);

    m111_independent_assertions independent_sva (.*);

    always #1 clk_core = ~clk_core;

    function automatic logic [VECTOR_BITS-1:0] make_pattern(input int seed);
        logic [VECTOR_BITS-1:0] payload;
        integer signed value;
        begin
            payload = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                value = ((seed * 97 + lane * 53) % 4001) - 2000;
                payload[lane * ACC_BITS +: ACC_BITS] = value[ACC_BITS-1:0];
            end
            return payload;
        end
    endfunction

    function automatic logic [VECTOR_BITS-1:0] make_boundary_base;
        logic [VECTOR_BITS-1:0] payload;
        begin
            payload = make_pattern(701);
            payload[0 * ACC_BITS +: ACC_BITS] = 24'h7fffff;
            payload[1 * ACC_BITS +: ACC_BITS] = 24'h800000;
            payload[2 * ACC_BITS +: ACC_BITS] = 24'hffffff;
            payload[3 * ACC_BITS +: ACC_BITS] = 24'h000001;
            return payload;
        end
    endfunction

    function automatic logic [VECTOR_BITS-1:0] make_boundary_adjust;
        logic [VECTOR_BITS-1:0] payload;
        begin
            payload = '0;
            payload[0 * ACC_BITS +: ACC_BITS] = -24'sd7;
            payload[1 * ACC_BITS +: ACC_BITS] = 24'sd8;
            payload[2 * ACC_BITS +: ACC_BITS] = 24'sd1;
            payload[3 * ACC_BITS +: ACC_BITS] = -24'sd1;
            return payload;
        end
    endfunction

    always @(negedge clk_core) begin
        if (rst_core)
            commit_ready = 1'b0;
        else if (automatic_commit_ready)
            commit_ready = ((cycle_count % 7) != 2)
                         && ((cycle_count % 19) != 5)
                         && ((cycle_count % 43) != 11);
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count = 0;
            pending_write = 1'b0;
            pending_overflow = 1'b0;
            prior_update_accept = 1'b0;
        end else begin
            cycle_count = cycle_count + 1;

            if ($countones(mem_rd_en) > 1 || $countones(mem_wr_en) > 1)
                $fatal(1, "M111 independent global port multiplicity violation");

            if (pending_overflow) begin
                if (|mem_wr_en)
                    $fatal(1, "M111 independent overflow wrote memory");
                if (!protocol_error)
                    $fatal(1, "M111 independent overflow did not raise protocol_error");
            end else if (pending_write) begin
                if ($countones(mem_wr_en) != 1 || !mem_wr_en[pending_bank])
                    $fatal(1, "M111 independent missing buffered write bank=%0d row=%0d", pending_bank, pending_row);
                if (mem_wr_addr[pending_bank] !== pending_row[8:0]
                        || mem_wr_data[pending_bank] !== pending_data)
                    $fatal(1, "M111 independent buffered write mismatch bank=%0d row=%0d", pending_bank, pending_row);
                positive_writes = positive_writes + (!allow_fault);
            end else if ((|mem_wr_en) && !allow_fault) begin
                $fatal(1, "M111 independent unexpected positive-path write");
            end

            pending_write = 1'b0;
            pending_overflow = 1'b0;

            if (window_start_accept) begin
                for (int bank = 0; bank < BANKS; bank++) begin
                    for (int row = 0; row < WIN_ROWS; row++) begin
                        reference_valid[bank][row] = 1'b0;
                        commit_read_seen[bank][row] = 1'b0;
                    end
                end
                current_commit_index = 0;
                current_commit_reads = 0;
            end

            if (update_accept) begin
                if ($countones(mem_rd_en) != 1 || !mem_rd_en[update_block]
                        || mem_rd_addr[update_block] !== update_row)
                    $fatal(1, "M111 independent accepted update read command mismatch");
                calculated_data = '0;
                calculated_overflow = 1'b0;
                for (int lane = 0; lane < LANES; lane++) begin
                    delta_value = $signed(update_delta[lane * ACC_BITS +: ACC_BITS]);
                    base_value = reference_valid[update_block][update_row]
                               ? reference[update_block][update_row][lane] : 0;
                    sum_value = base_value + delta_value;
                    if (sum_value > S24_MAX || sum_value < S24_MIN)
                        calculated_overflow = 1'b1;
                    calculated_data[lane * ACC_BITS +: ACC_BITS]
                        = sum_value[ACC_BITS-1:0];
                end
                pending_bank = update_block;
                pending_row = update_row;
                pending_data = calculated_data;
                pending_overflow = calculated_overflow;
                pending_write = !calculated_overflow;
                if (!calculated_overflow) begin
                    for (int lane = 0; lane < LANES; lane++)
                        reference[update_block][update_row][lane]
                            = $signed(calculated_data[lane * ACC_BITS +: ACC_BITS]);
                    reference_valid[update_block][update_row] = 1'b1;
                end
                if (!allow_fault)
                    positive_updates = positive_updates + 1;
                if (!allow_fault && prior_update_accept)
                    positive_ii1_pairs = positive_ii1_pairs + 1;
            end
            prior_update_accept = update_accept;

            if (!window_active && busy && (|mem_rd_en)) begin
                for (int bank = 0; bank < BANKS; bank++) begin
                    if (mem_rd_en[bank]) begin
                        if (!reference_valid[bank][mem_rd_addr[bank]])
                            $fatal(1, "M111 independent commit read invalid row bank=%0d row=%0d", bank, mem_rd_addr[bank]);
                        if (commit_read_seen[bank][mem_rd_addr[bank]])
                            $fatal(1, "M111 independent duplicate commit read bank=%0d row=%0d", bank, mem_rd_addr[bank]);
                        commit_read_seen[bank][mem_rd_addr[bank]] = 1'b1;
                        current_commit_reads = current_commit_reads + 1;
                    end
                end
            end

            if (!allow_fault && (|mem_rd_en) && (|mem_wr_en))
                positive_overlap_cycles = positive_overlap_cycles + 1;
            if (!allow_fault && commit_valid && !commit_ready)
                positive_commit_stalls = positive_commit_stalls + 1;

            if (!allow_fault && commit_valid && commit_ready) begin
                int expected_bank;
                int expected_row;
                expected_bank = current_commit_index / WIN_ROWS;
                expected_row = current_commit_index % WIN_ROWS;
                if (commit_block !== expected_bank[2:0]
                        || commit_row !== expected_row[8:0])
                    $fatal(1, "M111 independent commit order expected=%0d/%0d got=%0d/%0d",
                           expected_bank, expected_row, commit_block, commit_row);
                if (commit_last !== (current_commit_index == COMMIT_VECTORS-1))
                    $fatal(1, "M111 independent commit_last mismatch index=%0d", current_commit_index);
                for (int lane = 0; lane < LANES; lane++) begin
                    integer signed expected_value;
                    expected_value = reference_valid[expected_bank][expected_row]
                                   ? reference[expected_bank][expected_row][lane] : 0;
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS]) !== expected_value)
                        $fatal(1, "M111 independent numeric mismatch bank=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_bank, expected_row, lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]), expected_value);
                end
                if (windows_completed == 1 && expected_bank == 0 && expected_row == 7) begin
                    if (commit_data !== '0)
                        $fatal(1, "M111 independent lazy-clear leaked stale SRAM row");
                    lazy_clear_stale_zero_checks = lazy_clear_stale_zero_checks + 1;
                end
                current_commit_index = current_commit_index + 1;
                positive_commit_vectors = positive_commit_vectors + 1;
                positive_lane_checks = positive_lane_checks + LANES;
            end

            if (window_done && !allow_fault) begin
                if (current_commit_index != COMMIT_VECTORS)
                    $fatal(1, "M111 independent incomplete commit count=%0d", current_commit_index);
                current_expected_valid_rows = 0;
                for (int bank = 0; bank < BANKS; bank++) begin
                    for (int row = 0; row < WIN_ROWS; row++) begin
                        if (reference_valid[bank][row]) begin
                            current_expected_valid_rows = current_expected_valid_rows + 1;
                            if (!commit_read_seen[bank][row])
                                $fatal(1, "M111 independent missed valid commit read bank=%0d row=%0d", bank, row);
                        end else if (commit_read_seen[bank][row]) begin
                            $fatal(1, "M111 independent read invalid commit row bank=%0d row=%0d", bank, row);
                        end
                    end
                end
                if (current_commit_reads != current_expected_valid_rows)
                    $fatal(1, "M111 independent commit read conservation reads=%0d expected=%0d",
                           current_commit_reads, current_expected_valid_rows);
                windows_completed = windows_completed + 1;
            end

            if (!allow_fault && protocol_error)
                $fatal(1, "M111 independent unexpected positive protocol error cycle=%0d", cycle_count);
        end

        for (int bank = 0; bank < BANKS; bank++) begin
            if (mem_rd_en[bank])
                mem_rd_data[bank] <= memory[bank][mem_rd_addr[bank]];
            if (mem_wr_en[bank])
                memory[bank][mem_wr_addr[bank]] <= mem_wr_data[bank];
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            update_valid = 1'b0;
            window_end_valid = 1'b0;
            commit_ready = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            while (!window_start_ready) @(negedge clk_core);
            window_start_valid = 1'b1;
            @(posedge clk_core);
            if (!window_start_accept)
                $fatal(1, "M111 independent start not accepted");
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic drive_update(
        input int block,
        input int row,
        input logic [VECTOR_BITS-1:0] payload
    );
        begin
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = block[2:0];
            update_row = row[8:0];
            update_delta = payload;
            @(posedge clk_core);
            if (!update_accept)
                $fatal(1, "M111 independent legal update not accepted block=%0d row=%0d", block, row);
        end
    endtask

    task automatic stop_updates_and_drain;
        begin
            @(negedge clk_core);
            update_valid = 1'b0;
            while (!window_end_ready) @(negedge clk_core);
        end
    endtask

    task automatic end_window;
        begin
            window_end_valid = 1'b1;
            @(posedge clk_core);
            if (!window_end_accept)
                $fatal(1, "M111 independent end not accepted");
            @(negedge clk_core);
            window_end_valid = 1'b0;
        end
    endtask

    task automatic wait_done;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (!window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 8000)
                    $fatal(1, "M111 independent commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic run_same_address_attack;
        logic [VECTOR_BITS-1:0] first_payload;
        logic [VECTOR_BITS-1:0] second_payload;
        begin
            allow_fault = 1'b1;
            reset_dut();
            start_window();
            first_payload = '0;
            first_payload[0 +: ACC_BITS] = 24'sd7;
            first_payload[ACC_BITS +: ACC_BITS] = -24'sd13;
            drive_update(3, 91, first_payload);
            @(negedge clk_core);
            second_payload = '0;
            second_payload[0 +: ACC_BITS] = 24'sd9;
            update_valid = 1'b1;
            update_block = 3;
            update_row = 91;
            update_delta = second_payload;
            #0.1;
            if (update_ready || update_accept || !protocol_error
                    || !mem_wr_en[3] || mem_wr_addr[3] != 91
                    || mem_wr_data[3] !== first_payload)
                $fatal(1, "M111 independent same-address fail-close/preserve failure");
            @(posedge clk_core);
            same_address_attacks = same_address_attacks + 1;
            @(negedge clk_core);
            update_valid = 1'b0;
            if ($signed(memory[3][91][0 +: ACC_BITS]) !== 7
                    || $signed(memory[3][91][ACC_BITS +: ACC_BITS]) !== -13)
                $fatal(1, "M111 independent older write was not preserved");
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "M111 independent same-address fault not sticky");
        end
    endtask

    task automatic run_overflow_attack(input bit positive);
        logic [VECTOR_BITS-1:0] extreme;
        logic [VECTOR_BITS-1:0] one_step;
        begin
            allow_fault = 1'b1;
            reset_dut();
            start_window();
            extreme = '0;
            one_step = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                extreme[lane * ACC_BITS +: ACC_BITS]
                    = positive ? 24'h7fffff : 24'h800000;
                one_step[lane * ACC_BITS +: ACC_BITS]
                    = positive ? 24'sd1 : -24'sd1;
            end
            drive_update(0, positive ? 33 : 34, extreme);
            drive_update(1, positive ? 71 : 72, '0);
            drive_update(0, positive ? 33 : 34, one_step);
            @(negedge clk_core);
            update_valid = 1'b0;
            #0.1;
            if (!protocol_error || (|mem_wr_en))
                $fatal(1, "M111 independent signed24 overflow suppression failure positive=%0d", positive);
            @(posedge clk_core);
            overflow_attacks = overflow_attacks + 1;
            @(negedge clk_core);
            if ($signed(memory[0][positive ? 33 : 34][0 +: ACC_BITS])
                    !== (positive ? S24_MAX : S24_MIN))
                $fatal(1, "M111 independent overflow corrupted older value positive=%0d", positive);
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "M111 independent overflow fault not sticky positive=%0d", positive);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start_valid = 1'b0;
        update_valid = 1'b0;
        update_block = '0;
        update_row = '0;
        update_delta = '0;
        window_end_valid = 1'b0;
        commit_ready = 1'b0;
        allow_fault = 1'b0;
        automatic_commit_ready = 1'b1;
        cycle_count = 0;
        current_commit_index = 0;
        current_commit_reads = 0;
        windows_completed = 0;
        positive_updates = 0;
        positive_writes = 0;
        positive_ii1_pairs = 0;
        positive_overlap_cycles = 0;
        positive_commit_stalls = 0;
        positive_commit_vectors = 0;
        positive_lane_checks = 0;
        lazy_clear_stale_zero_checks = 0;
        same_address_attacks = 0;
        overflow_attacks = 0;
        prior_update_accept = 1'b0;
        pending_write = 1'b0;
        pending_overflow = 1'b0;
        for (int bank = 0; bank < BANKS; bank++) begin
            mem_rd_data[bank] = 'x;
            for (int row = 0; row < WIN_ROWS; row++) begin
                memory[bank][row] = 'x;
                reference_valid[bank][row] = 1'b0;
                commit_read_seen[bank][row] = 1'b0;
                for (int lane = 0; lane < LANES; lane++)
                    reference[bank][row][lane] = 0;
            end
        end

        reset_dut();
        start_window();
        drive_update(0, 7, make_boundary_base());
        drive_update(1, 11, make_pattern(17));
        drive_update(0, 7, make_boundary_adjust());
        for (int index = 0; index < 125; index++)
            drive_update((index * 5 + 2) % BANKS,
                         (index * 37 + 19) % WIN_ROWS,
                         make_pattern(100 + index));
        stop_updates_and_drain();
        end_window();
        wait_done();

        // A second window proves lazy-valid clear: SRAM still holds (0,7),
        // but the row is not touched and must commit zero without a macro read.
        start_window();
        for (int index = 0; index < 41; index++)
            drive_update((index * 3 + 1) % BANKS,
                         (index * 41 + 23) % WIN_ROWS,
                         make_pattern(800 + index));
        stop_updates_and_drain();
        end_window();
        wait_done();

        if (windows_completed != 2 || positive_updates != 169
                || positive_writes != 169
                || positive_commit_vectors != 2 * COMMIT_VECTORS
                || positive_lane_checks != 2 * COMMIT_VECTORS * LANES
                || positive_ii1_pairs < 165 || positive_overlap_cycles == 0
                || positive_commit_stalls == 0
                || lazy_clear_stale_zero_checks != 1)
            $fatal(1, "M111 independent positive conservation failure windows=%0d updates=%0d writes=%0d commits=%0d lanes=%0d ii1=%0d overlap=%0d stalls=%0d stale=%0d",
                   windows_completed, positive_updates, positive_writes,
                   positive_commit_vectors, positive_lane_checks,
                   positive_ii1_pairs, positive_overlap_cycles,
                   positive_commit_stalls, lazy_clear_stale_zero_checks);

        run_same_address_attack();
        run_overflow_attack(1'b1);
        run_overflow_attack(1'b0);

        if (same_address_attacks != 1 || overflow_attacks != 2)
            $fatal(1, "M111 independent attack coverage mismatch");

        $display("PASS M111 INDEPENDENT HAMMER commercial_vcs=true windows=2 updates=169 writes=169 commits=6144 lane_checks=589824 lazy_clear_stale_zero=1 full_boundary_rmw=true non_same_address_ii1=true dual_port_overlap=%0d commit_stalls=%0d same_address_preserve_attacks=1 positive_overflow=1 negative_overflow=1 global_ports=8x1R1W logical_accumulator_bytes=884736 valid_bits=3072 behavioral_macro=true foundry_macro=false m109_r2_projection=2.53546204172554 scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 positive_overlap_cycles, positive_commit_stalls);
        $finish;
    end
endmodule

`default_nettype wire
