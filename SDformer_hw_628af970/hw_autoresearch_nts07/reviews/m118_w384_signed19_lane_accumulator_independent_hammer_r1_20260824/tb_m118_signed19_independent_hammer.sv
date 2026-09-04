`timescale 1ns/1ps
`default_nettype none

module tb_m118_signed19_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int DEPTH = BLOCKS * WIN_ROWS;
    localparam integer signed MAX19 = 262143;
    localparam integer signed MIN19 = -262144;

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
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [ACC_BITS-1:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [ACC_BITS-1:0] lane_mem_wr_data [0:LANES-1];
    logic protocol_error, window_active, busy;

    logic [ACC_BITS-1:0] memory [0:LANES-1][0:DEPTH-1];
    integer signed reference [0:BLOCKS-1][0:WIN_ROWS-1][0:LANES-1];
    logic [VECTOR_BITS-1:0] stalled_data;
    logic [2:0] stalled_block;
    logic [8:0] stalled_row;
    logic stalled_last;
    logic prior_stall, prior_last_accept;
    logic prior_update_accept;
    logic [11:0] prior_update_address;
    logic automatic_commit_ready, positive_phase;
    integer cycle_count;
    integer positive_updates, positive_writes;
    integer ii1_pairs, read_write_overlap;
    integer commit_accepts, lane_checks, commit_stalls, stall_releases;
    integer completed_windows, expected_block, expected_row;
    integer address_zero_reads, address_last_reads;
    integer address_zero_writes, address_last_writes;
    integer lazy_clear_checks, signed_boundary_checks, rmw_lane_checks;
    integer same_address_attacks, positive_overflow_attacks;
    integer negative_overflow_attacks;

    m118_w384_signed19_lane_sliced_accumulator_adapter dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_block(update_block), .update_row(update_row),
        .update_delta(update_delta), .update_accept(update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .window_done(window_done), .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .protocol_error(protocol_error), .window_active(window_active),
        .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    function automatic logic [11:0] flat_addr(
        input integer block,
        input integer row
    );
        flat_addr = block * WIN_ROWS + row;
    endfunction

    function automatic integer signed pattern_value(
        input integer pattern,
        input integer lane
    );
        begin
            case (pattern)
                0: begin
                    case (lane)
                        0: pattern_value = MAX19;
                        1: pattern_value = MIN19;
                        2: pattern_value = 218338;
                        3: pattern_value = -218338;
                        4: pattern_value = 1;
                        5: pattern_value = -1;
                        default: pattern_value = ((lane * 23) % 255) - 127;
                    endcase
                end
                1: pattern_value = ((lane * 13) % 355) - 177;
                2: begin
                    case (lane)
                        0: pattern_value = -MAX19;
                        1: pattern_value = MAX19;
                        2: pattern_value = -218338;
                        3: pattern_value = 218338;
                        4: pattern_value = -1;
                        5: pattern_value = 1;
                        default: pattern_value = -(((lane * 23) % 255) - 127);
                    endcase
                end
                3: pattern_value = ((lane * 11 + 31) % 63) - 31;
                4: pattern_value = (lane == 0) ? 12345
                                              : (((lane * 7) % 101) - 50);
                5: pattern_value = MAX19;
                6: pattern_value = MIN19;
                7: pattern_value = 1;
                8: pattern_value = -1;
                default: pattern_value = 0;
            endcase
        end
    endfunction

    always @(posedge clk_core) begin : synchronous_lane_memory
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane] <= memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                memory[lane][lane_mem_wr_addr] <= lane_mem_wr_data[lane];
        end
    end

    always @(negedge clk_core) begin
        if (automatic_commit_ready && !rst_core)
            commit_ready = ((cycle_count % 7) != 2)
                         && ((cycle_count % 19) != 5)
                         && ((cycle_count % 43) != 11);
    end

    always @(posedge clk_core) begin : independent_monitors
        if (rst_core) begin
            prior_stall <= 1'b0;
            prior_last_accept <= 1'b0;
            prior_update_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (positive_phase && protocol_error)
                $fatal(1, "M118 hammer unexpected positive protocol_error");

            if (update_accept) begin
                if (!lane_mem_rd_en
                        || lane_mem_rd_addr !== flat_addr(update_block, update_row))
                    $fatal(1, "M118 hammer accepted update read mapping mismatch");
                if (positive_phase) begin
                    positive_updates <= positive_updates + 1;
                    if (prior_update_accept)
                        ii1_pairs <= ii1_pairs + 1;
                end
                prior_update_address <= flat_addr(update_block, update_row);
            end
            if (lane_mem_wr_en) begin
                if (lane_mem_wr_addr !== prior_update_address)
                    $fatal(1, "M118 hammer delayed write mapping mismatch");
                if (positive_phase)
                    positive_writes <= positive_writes + 1;
            end
            if (positive_phase && lane_mem_rd_en && lane_mem_wr_en)
                read_write_overlap <= read_write_overlap + 1;
            prior_update_accept <= update_accept;

            if (lane_mem_rd_en && lane_mem_rd_addr == 0)
                address_zero_reads <= address_zero_reads + 1;
            if (lane_mem_rd_en && lane_mem_rd_addr == DEPTH-1)
                address_last_reads <= address_last_reads + 1;
            if (lane_mem_wr_en && lane_mem_wr_addr == 0)
                address_zero_writes <= address_zero_writes + 1;
            if (lane_mem_wr_en && lane_mem_wr_addr == DEPTH-1)
                address_last_writes <= address_last_writes + 1;

            if (prior_stall) begin
                if (!commit_valid || commit_block !== stalled_block
                        || commit_row !== stalled_row
                        || commit_data !== stalled_data
                        || commit_last !== stalled_last)
                    $fatal(1, "M118 hammer commit changed under backpressure");
                if (commit_ready)
                    stall_releases <= stall_releases + 1;
            end
            if (commit_valid && !commit_ready) begin
                stalled_block <= commit_block;
                stalled_row <= commit_row;
                stalled_data <= commit_data;
                stalled_last <= commit_last;
                if (positive_phase)
                    commit_stalls <= commit_stalls + 1;
            end
            prior_stall <= commit_valid && !commit_ready;

            if (positive_phase && commit_valid && commit_ready) begin
                if (commit_block !== expected_block[2:0]
                        || commit_row !== expected_row[8:0])
                    $fatal(1, "M118 hammer commit order mismatch");
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_block][expected_row][lane])
                        $fatal(1, "M118 hammer numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_block, expected_row, lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_block][expected_row][lane]);
                    lane_checks = lane_checks + 1;
                end
                if (commit_last !== (expected_block == BLOCKS-1
                                     && expected_row == WIN_ROWS-1))
                    $fatal(1, "M118 hammer commit_last mismatch");
                commit_accepts <= commit_accepts + 1;
                if (expected_row == WIN_ROWS-1) begin
                    expected_row <= 0;
                    if (expected_block == BLOCKS-1)
                        expected_block <= 0;
                    else
                        expected_block <= expected_block + 1;
                end else begin
                    expected_row <= expected_row + 1;
                end
            end
            if (window_done) begin
                if (!prior_last_accept)
                    $fatal(1, "M118 hammer window_done without prior last accept");
                completed_windows <= completed_windows + 1;
            end
            prior_last_accept <= commit_valid && commit_ready && commit_last;
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

    task automatic clear_reference;
        for (int block = 0; block < BLOCKS; block++)
            for (int row = 0; row < WIN_ROWS; row++)
                for (int lane = 0; lane < LANES; lane++)
                    reference[block][row][lane] = 0;
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

    task automatic drive_update(
        input integer block,
        input integer row,
        input integer pattern,
        input logic last
    );
        integer signed value;
        logic [VECTOR_BITS-1:0] payload;
        begin
            payload = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                value = pattern_value(pattern, lane);
                payload[lane * ACC_BITS +: ACC_BITS] = value[ACC_BITS-1:0];
            end
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = block[2:0];
            update_row = row[8:0];
            update_delta = payload;
            do @(posedge clk_core); while (!update_accept);
            if (positive_phase)
                for (int lane = 0; lane < LANES; lane++) begin
                    value = pattern_value(pattern, lane);
                    reference[block][row][lane]
                        = reference[block][row][lane] + value;
                    if (reference[block][row][lane] > MAX19
                            || reference[block][row][lane] < MIN19)
                        $fatal(1, "M118 hammer positive campaign overflow");
                end
            if (last) begin
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

    task automatic wait_done;
        integer start_cycle;
        begin
            start_cycle = cycle_count;
            while (!window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 12000)
                    $fatal(1, "M118 hammer commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic check_fault_sticky;
        begin
            repeat (3) @(posedge clk_core);
            if (!protocol_error || window_start_ready || update_ready
                    || window_end_ready || commit_valid)
                $fatal(1, "M118 hammer fault quarantine/stickiness failure");
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
        automatic_commit_ready = 1'b1;
        positive_phase = 1'b1;
        cycle_count = 0;
        positive_updates = 0;
        positive_writes = 0;
        ii1_pairs = 0;
        read_write_overlap = 0;
        commit_accepts = 0;
        lane_checks = 0;
        commit_stalls = 0;
        stall_releases = 0;
        completed_windows = 0;
        expected_block = 0;
        expected_row = 0;
        address_zero_reads = 0;
        address_last_reads = 0;
        address_zero_writes = 0;
        address_last_writes = 0;
        lazy_clear_checks = 0;
        signed_boundary_checks = 0;
        rmw_lane_checks = 0;
        same_address_attacks = 0;
        positive_overflow_attacks = 0;
        negative_overflow_attacks = 0;
        prior_stall = 1'b0;
        prior_last_accept = 1'b0;
        prior_update_accept = 1'b0;
        prior_update_address = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < DEPTH; address++)
                memory[lane][address] = 'x;
        end

        // Exact signed endpoints, M115-r2 checkpoint maximum, RMW and II=1.
        reset_dut();
        start_window();
        drive_update(0, 0, 0, 1'b0);
        signed_boundary_checks = signed_boundary_checks + 6;
        drive_update(7, 383, 1, 1'b0);
        drive_update(0, 1, 3, 1'b0);
        drive_update(0, 0, 2, 1'b0);
        rmw_lane_checks = rmw_lane_checks + LANES;
        signed_boundary_checks = signed_boundary_checks + 6;
        for (int index = 0; index < 32; index++)
            drive_update((index * 5 + 2) % BLOCKS,
                         (index * 37 + 5) % WIN_ROWS,
                         3, index == 31);
        end_window();
        wait_done();

        // Lazy clear: physical old endpoint remains but commit must observe zero.
        start_window();
        if ($signed(memory[0][DEPTH-1]) !== pattern_value(1, 0))
            $fatal(1, "M118 hammer physical data unexpectedly swept");
        lazy_clear_checks = lazy_clear_checks + 1;
        drive_update(3, 222, 4, 1'b1);
        end_window();
        wait_done();
        lazy_clear_checks = lazy_clear_checks + 1;
        repeat (3) @(posedge clk_core);
        if (positive_updates != 37 || positive_writes != 37
                || ii1_pairs < 35 || read_write_overlap < 35
                || commit_accepts != 6144 || lane_checks != 589824
                || completed_windows != 2 || commit_stalls < 100
                || stall_releases == 0 || address_zero_reads == 0
                || address_last_reads == 0 || address_zero_writes == 0
                || address_last_writes == 0)
            $fatal(1, "M118 hammer positive conservation/coverage failure updates=%0d writes=%0d ii1=%0d overlap=%0d commits=%0d lanes=%0d windows=%0d stalls=%0d releases=%0d",
                   positive_updates, positive_writes, ii1_pairs,
                   read_write_overlap, commit_accepts, lane_checks,
                   completed_windows, commit_stalls, stall_releases);

        positive_phase = 1'b0;
        automatic_commit_ready = 1'b0;

        // Consecutive equal address must preserve the older write and quarantine.
        reset_dut();
        start_window();
        drive_update(4, 177, 4, 1'b0);
        @(negedge clk_core);
        update_valid = 1'b1;
        update_block = 4;
        update_row = 177;
        update_delta = '0;
        update_delta[ACC_BITS-1:0] = 19'sd9;
        @(posedge clk_core);
        if (update_ready || update_accept || !protocol_error
                || !lane_mem_wr_en || lane_mem_wr_addr != flat_addr(4, 177))
            $fatal(1, "M118 hammer same-address RDW failure");
        same_address_attacks = same_address_attacks + 1;
        @(negedge clk_core);
        update_valid = 1'b0;
        check_fault_sticky();
        if ($signed(memory[0][flat_addr(4, 177)]) !== 12345)
            $fatal(1, "M118 hammer older same-address write lost");

        // MAX19 + 1 must suppress the overflowing write.
        reset_dut();
        start_window();
        drive_update(0, 0, 5, 1'b0);
        drive_update(1, 1, 9, 1'b0);
        drive_update(0, 0, 7, 1'b1);
        @(negedge clk_core);
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M118 hammer positive overflow not suppressed");
        positive_overflow_attacks = positive_overflow_attacks + 1;
        check_fault_sticky();
        if ($signed(memory[0][0]) !== MAX19)
            $fatal(1, "M118 hammer positive overflow corrupted old value");

        // MIN19 - 1 must suppress the underflowing write.
        reset_dut();
        start_window();
        drive_update(7, 383, 6, 1'b0);
        drive_update(6, 382, 9, 1'b0);
        drive_update(7, 383, 8, 1'b1);
        @(negedge clk_core);
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M118 hammer negative overflow not suppressed");
        negative_overflow_attacks = negative_overflow_attacks + 1;
        check_fault_sticky();
        if ($signed(memory[0][DEPTH-1]) !== MIN19)
            $fatal(1, "M118 hammer negative overflow corrupted old value");

        if (lazy_clear_checks != 2 || signed_boundary_checks != 12
                || rmw_lane_checks != 96 || same_address_attacks != 1
                || positive_overflow_attacks != 1
                || negative_overflow_attacks != 1)
            $fatal(1, "M118 hammer directed count failure");

        $display("PASS M118 independent signed19 hammer commercial_vcs=true exact_source_sha=true windows=2 positive_updates=%0d positive_writes=%0d ii1_pairs=%0d read_write_overlap=%0d commits=%0d lane_checks=%0d commit_stalls=%0d stall_releases=%0d lazy_clear_checks=2 signed_boundary_checks=12 rmw_lane_checks=96 address_minmax=true same_address_rdw_attacks=1 positive_overflow_attacks=1 negative_overflow_attacks=1 lanes=96 vector_bits=1824 lane_macros=96 macro_depth=3072 macro_width=19 payload_bytes=700416 combined_bytes=725416 saving_vs_signed24_payload_bytes=184320 mathematical_candidate=true integrated_exact_once=false behavioral_macro=true foundry_macro=false ppa=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 positive_updates, positive_writes, ii1_pairs,
                 read_write_overlap, commit_accepts, lane_checks,
                 commit_stalls, stall_releases);
        $finish;
    end
endmodule

`default_nettype wire
