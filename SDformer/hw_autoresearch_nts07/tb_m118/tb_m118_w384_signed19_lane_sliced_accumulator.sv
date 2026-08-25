`timescale 1ns/1ps
`default_nettype none

module tb_m118_w384_signed19_lane_sliced_accumulator;
    localparam int WIN_ROWS = 384;
    localparam int BANKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int COMMIT_VECTORS = BANKS * WIN_ROWS;

    logic clk_core, rst_core;
    logic window_start_valid, window_start_ready, window_start_accept;
    logic update_valid, update_ready;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [VECTOR_BITS-1:0] update_delta;
    logic update_accept;
    logic window_end_valid, window_end_ready, window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [VECTOR_BITS-1:0] commit_data;
    logic commit_last, window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [ACC_BITS-1:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [ACC_BITS-1:0] lane_mem_wr_data [0:LANES-1];
    logic protocol_error, window_active, busy;

    logic [ACC_BITS-1:0] memory [0:LANES-1][0:BANKS*WIN_ROWS-1];
    integer signed reference [0:7][0:WIN_ROWS-1][0:LANES-1];
    int cycle_count;
    int accepted_updates;
    int consecutive_update_pairs;
    int memory_writes;
    int read_write_overlap_cycles;
    int commit_accepts;
    int commit_stalls;
    int completed_windows;
    int positive_ii1_pairs;
    int positive_read_write_overlap;
    int positive_commit_stalls;
    int expected_commit_block;
    int expected_commit_row;
    bit previous_update_accept;
    bit positive_phase;

    m118_w384_signed19_lane_sliced_accumulator_adapter dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_accept(update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .protocol_error(protocol_error),
        .window_active(window_active),
        .busy(busy)
    );

    m118_w384_signed19_lane_sliced_accumulator_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_accept(update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .protocol_error(protocol_error),
        .window_active(window_active),
        .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane]
                    <= memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                memory[lane][lane_mem_wr_addr]
                    <= lane_mem_wr_data[lane];
        end
        if (lane_mem_wr_en)
            memory_writes <= memory_writes + 1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            commit_ready <= 1'b0;
            previous_update_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            commit_ready <= ((cycle_count % 13) != 4)
                         && ((cycle_count % 31) != 9);
            if (update_accept) begin
                accepted_updates <= accepted_updates + 1;
                if (previous_update_accept)
                    consecutive_update_pairs <= consecutive_update_pairs + 1;
            end
            previous_update_accept <= update_accept;
            if (lane_mem_rd_en && lane_mem_wr_en)
                read_write_overlap_cycles <= read_write_overlap_cycles + 1;
            if (commit_valid && !commit_ready)
                commit_stalls <= commit_stalls + 1;
            if (positive_phase && protocol_error)
                $fatal(1, "M118 unexpected protocol_error cycle=%0d", cycle_count);

            if (positive_phase && commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M118 commit order mismatch expected=%0d/%0d got=%0d/%0d",
                           expected_commit_block, expected_commit_row,
                           commit_block, commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M118 numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row, lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                end
                if (commit_last !== (expected_commit_block == BANKS-1
                                     && expected_commit_row == WIN_ROWS-1))
                    $fatal(1, "M118 commit_last mismatch");
                commit_accepts <= commit_accepts + 1;
                if (expected_commit_row == WIN_ROWS-1) begin
                    expected_commit_row <= 0;
                    if (expected_commit_block == BANKS-1)
                        expected_commit_block <= 0;
                    else
                        expected_commit_block <= expected_commit_block + 1;
                end else begin
                    expected_commit_row <= expected_commit_row + 1;
                end
            end
            if (window_done)
                completed_windows <= completed_windows + 1;
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            update_valid = 1'b0;
            window_end_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic clear_reference;
        begin
            for (int bank = 0; bank < BANKS; bank++)
                for (int row = 0; row < WIN_ROWS; row++)
                    for (int lane = 0; lane < LANES; lane++)
                        reference[bank][row][lane] = 0;
        end
    endtask

    task automatic start_window;
        begin
            clear_reference();
            @(negedge clk_core);
            window_start_valid = 1'b1;
            do @(posedge clk_core); while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic send_update(
        input int index,
        input int block,
        input int row,
        input bit last_in_stream
    );
        integer signed value;
        logic [VECTOR_BITS-1:0] payload;
        begin
            payload = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                value = ((index * 3 + lane * 5) % 31) - 15;
                payload[lane * ACC_BITS +: ACC_BITS] = value[ACC_BITS-1:0];
            end
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = block[2:0];
            update_row = row[8:0];
            update_delta = payload;
            do @(posedge clk_core); while (!update_accept);
            for (int lane = 0; lane < LANES; lane++) begin
                value = ((index * 3 + lane * 5) % 31) - 15;
                reference[block][row][lane]
                    = reference[block][row][lane] + value;
            end
            if (last_in_stream) begin
                @(negedge clk_core);
                update_valid = 1'b0;
            end
        end
    endtask

    task automatic end_window;
        begin
            @(negedge clk_core);
            window_end_valid = 1'b1;
            do @(posedge clk_core); while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 1'b0;
        end
    endtask

    task automatic wait_window_done;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (!window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 10000)
                    $fatal(1, "M118 positive watchdog timeout completed=%0d commits=%0d valid=%0d ready=%0d busy=%0d active=%0d issue_active=%0d pipe=%0d issue=%0d/%0d out=%0d/%0d",
                           completed_windows, commit_accepts, commit_valid,
                           commit_ready, busy, window_active,
                           dut.core.commit_active_q, dut.core.commit_pipe_valid_q,
                           dut.core.commit_issue_block_q,
                           dut.core.commit_issue_row_q,
                           commit_block, commit_row);
            end
            @(posedge clk_core);
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
        cycle_count = 0;
        accepted_updates = 0;
        consecutive_update_pairs = 0;
        memory_writes = 0;
        read_write_overlap_cycles = 0;
        commit_accepts = 0;
        commit_stalls = 0;
        completed_windows = 0;
        positive_ii1_pairs = 0;
        positive_read_write_overlap = 0;
        positive_commit_stalls = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        previous_update_accept = 1'b0;
        positive_phase = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < BANKS * WIN_ROWS; address++)
                memory[lane][address] = 'x;
        end

        reset_dut();
        start_window();
        for (int index = 0; index < 1024; index++)
            send_update(index, index % BANKS, (index * 13) % WIN_ROWS,
                        index == 1023);
        end_window();
        wait_window_done();

        start_window();
        for (int index = 0; index < 32; index++)
            send_update(2000 + index, (index * 3) % BANKS,
                        (index * 29) % WIN_ROWS, index == 31);
        end_window();
        wait_window_done();
        repeat (3) @(posedge clk_core);
        if (accepted_updates != 1056 || memory_writes != 1056
                || completed_windows != 2
                || commit_accepts != 2 * COMMIT_VECTORS
                || consecutive_update_pairs < 1052
                || read_write_overlap_cycles == 0 || commit_stalls == 0)
            $fatal(1, "M118 positive coverage/conservation mismatch updates=%0d writes=%0d windows=%0d commit=%0d ii1=%0d overlap=%0d stalls=%0d",
                   accepted_updates, memory_writes, completed_windows,
                   commit_accepts, consecutive_update_pairs,
                   read_write_overlap_cycles, commit_stalls);
        positive_ii1_pairs = consecutive_update_pairs;
        positive_read_write_overlap = read_write_overlap_cycles;
        positive_commit_stalls = commit_stalls;

        // Consecutive same-address RMW is an explicit macro hazard and must
        // reject the second request without erasing the older accepted write.
        positive_phase = 1'b0;
        reset_dut();
        @(negedge clk_core);
        window_start_valid = 1'b1;
        do @(posedge clk_core); while (!window_start_accept);
        @(negedge clk_core);
        window_start_valid = 1'b0;
        update_valid = 1'b1;
        update_block = 0;
        update_row = 0;
        update_delta = '0;
        update_delta[ACC_BITS-1:0] = 19'sd7;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_valid = 1'b1;
        update_block = 0;
        update_row = 0;
        update_delta[ACC_BITS-1:0] = 19'sd9;
        @(posedge clk_core);
        if (update_ready || update_accept || !protocol_error || !lane_mem_wr_en)
            $fatal(1, "M118 same-address hazard did not preserve prior write and fail closed");
        @(negedge clk_core);
        update_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M118 same-address fault not sticky");

        // Independently reach the signed19 positive overflow guard with legal address
        // spacing: max at (0,0), a different address, then +1 at (0,0).
        reset_dut();
        @(negedge clk_core);
        window_start_valid = 1'b1;
        do @(posedge clk_core); while (!window_start_accept);
        @(negedge clk_core);
        window_start_valid = 1'b0;
        update_valid = 1'b1;
        update_block = 0;
        update_row = 0;
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++)
            update_delta[lane * ACC_BITS +: ACC_BITS] = 19'sh3ffff;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_block = 1;
        update_row = 1;
        update_delta = '0;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_block = 0;
        update_row = 0;
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++)
            update_delta[lane * ACC_BITS +: ACC_BITS] = 19'sd1;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_valid = 1'b0;
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M118 signed19 overflow was not suppressed fail-closed");
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M118 overflow fault not sticky");

        // Exercise the opposite signed endpoint as a separate fail-closed
        // attack: minimum at (0,0), a different address, then -1 at (0,0).
        reset_dut();
        @(negedge clk_core);
        window_start_valid = 1'b1;
        do @(posedge clk_core); while (!window_start_accept);
        @(negedge clk_core);
        window_start_valid = 1'b0;
        update_valid = 1'b1;
        update_block = 0;
        update_row = 0;
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++)
            update_delta[lane * ACC_BITS +: ACC_BITS] = 19'sh40000;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_block = 1;
        update_row = 1;
        update_delta = '0;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_block = 0;
        update_row = 0;
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++)
            update_delta[lane * ACC_BITS +: ACC_BITS] = 19'sh7ffff;
        do @(posedge clk_core); while (!update_accept);
        @(negedge clk_core);
        update_valid = 1'b0;
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M118 signed19 underflow was not suppressed fail-closed");
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M118 underflow fault not sticky");

        $display("PASS M118 W384 lane-sliced accumulator VCS windows=2 updates=1056 vector_lane_checks=%0d commits=%0d lazy_valid_clears=2 positive_memory_writes=1056 ii1_pairs=%0d read_write_overlap=%0d commit_stalls=%0d same_address_attacks=1 overflow_attacks=2 lanes=96 vector_bits=1824 accumulator_bytes=700416 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=19 behavioral_macro=true overflow_guard=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 2 * COMMIT_VECTORS * LANES, commit_accepts,
                 positive_ii1_pairs, positive_read_write_overlap,
                 positive_commit_stalls);
        $finish;
    end
endmodule

`default_nettype wire
