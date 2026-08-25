`timescale 1ns/1ps
`default_nettype none

module tb_m116_signed20_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 20;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int DEPTH = BLOCKS * WIN_ROWS;
    localparam int COMMIT_VECTORS = DEPTH;
    localparam integer signed MAX20 = 524287;
    localparam integer signed MIN20 = -524288;

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
    logic [VECTOR_BITS-1:0] stalled_commit_payload;
    logic [2:0] stalled_commit_block;
    logic [8:0] stalled_commit_row;
    logic stalled_commit_last;
    logic prior_commit_stall;
    logic prior_commit_last_accept;
    logic prior_update_accept;
    logic [11:0] prior_update_address;
    logic automatic_commit_ready;
    logic positive_phase;
    integer cycle_count;
    integer positive_updates;
    integer positive_writes;
    integer ii1_pairs;
    integer read_write_overlap;
    integer commit_accepts;
    integer commit_stalls;
    integer commit_stall_releases;
    integer lane_checks;
    integer completed_windows;
    integer expected_commit_block;
    integer expected_commit_row;
    integer address_zero_reads;
    integer address_last_reads;
    integer address_zero_writes;
    integer address_last_writes;
    integer lazy_clear_checks;
    integer signed_boundary_checks;
    integer rmw_checks;
    integer same_address_attacks;
    integer positive_overflow_attacks;
    integer negative_overflow_attacks;

    m116_w384_signed20_lane_sliced_accumulator_adapter dut (
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
                        0: pattern_value = MAX20;
                        1: pattern_value = MIN20;
                        2: pattern_value = 436676;
                        3: pattern_value = -436676;
                        4: pattern_value = 1;
                        5: pattern_value = -1;
                        default: pattern_value = ((lane * 29) % 255) - 127;
                    endcase
                end
                1: pattern_value = ((lane * 17) % 511) - 255;
                2: begin
                    case (lane)
                        0: pattern_value = -MAX20;
                        1: pattern_value = MAX20;
                        2: pattern_value = -436676;
                        3: pattern_value = 436676;
                        4: pattern_value = -1;
                        5: pattern_value = 1;
                        default: pattern_value = -(((lane * 29) % 255) - 127);
                    endcase
                end
                3: pattern_value = ((lane * 11 + 37) % 63) - 31;
                4: pattern_value = (lane == 0) ? 12345
                                              : (((lane * 7) % 101) - 50);
                5: pattern_value = MAX20;
                6: pattern_value = MIN20;
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
            prior_commit_stall <= 1'b0;
            prior_commit_last_accept <= 1'b0;
            prior_update_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (positive_phase && protocol_error)
                $fatal(1, "M116 hammer unexpected positive protocol_error cycle=%0d", cycle_count);

            if (update_accept) begin
                if (!lane_mem_rd_en
                        || lane_mem_rd_addr !== flat_addr(update_block, update_row))
                    $fatal(1, "M116 hammer accepted update read mapping mismatch");
                if (positive_phase) begin
                    positive_updates <= positive_updates + 1;
                    if (prior_update_accept)
                        ii1_pairs <= ii1_pairs + 1;
                end
                prior_update_address <= flat_addr(update_block, update_row);
            end
            if (lane_mem_wr_en) begin
                if (lane_mem_wr_addr !== prior_update_address)
                    $fatal(1, "M116 hammer delayed write mapping mismatch got=%0d expected=%0d",
                           lane_mem_wr_addr, prior_update_address);
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

            if (prior_commit_stall) begin
                if (!commit_valid
                        || commit_block !== stalled_commit_block
                        || commit_row !== stalled_commit_row
                        || commit_data !== stalled_commit_payload
                        || commit_last !== stalled_commit_last)
                    $fatal(1, "M116 hammer commit payload/order changed under stall");
                if (commit_ready)
                    commit_stall_releases <= commit_stall_releases + 1;
            end
            if (commit_valid && !commit_ready) begin
                stalled_commit_block <= commit_block;
                stalled_commit_row <= commit_row;
                stalled_commit_payload <= commit_data;
                stalled_commit_last <= commit_last;
                if (positive_phase)
                    commit_stalls <= commit_stalls + 1;
            end
            prior_commit_stall <= commit_valid && !commit_ready;

            if (positive_phase && commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M116 hammer commit order mismatch expected=%0d/%0d got=%0d/%0d",
                           expected_commit_block, expected_commit_row,
                           commit_block, commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M116 hammer numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row, lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                    lane_checks = lane_checks + 1;
                end
                if (commit_last !== (expected_commit_block == BLOCKS-1
                                     && expected_commit_row == WIN_ROWS-1))
                    $fatal(1, "M116 hammer commit_last mismatch");
                commit_accepts <= commit_accepts + 1;
                if (expected_commit_row == WIN_ROWS-1) begin
                    expected_commit_row <= 0;
                    if (expected_commit_block == BLOCKS-1)
                        expected_commit_block <= 0;
                    else
                        expected_commit_block <= expected_commit_block + 1;
                end else begin
                    expected_commit_row <= expected_commit_row + 1;
                end
            end
            if (window_done) begin
                if (!prior_commit_last_accept)
                    $fatal(1, "M116 hammer window_done not after last commit accept");
                completed_windows <= completed_windows + 1;
            end
            prior_commit_last_accept <= commit_valid && commit_ready
                                      && commit_last;
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
        begin
            for (int block = 0; block < BLOCKS; block++)
                for (int row = 0; row < WIN_ROWS; row++)
                    for (int lane = 0; lane < LANES; lane++)
                        reference[block][row][lane] = 0;
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
                    if (reference[block][row][lane] > MAX20
                            || reference[block][row][lane] < MIN20)
                        $fatal(1, "M116 hammer generated positive overflow");
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
                    $fatal(1, "M116 hammer commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic check_fault_sticky;
        begin
            repeat (3) @(posedge clk_core);
            if (!protocol_error || window_start_ready || update_ready
                    || window_end_ready || commit_valid)
                $fatal(1, "M116 hammer fault quarantine/stickiness failure");
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
        commit_stalls = 0;
        commit_stall_releases = 0;
        lane_checks = 0;
        completed_windows = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        address_zero_reads = 0;
        address_last_reads = 0;
        address_zero_writes = 0;
        address_last_writes = 0;
        lazy_clear_checks = 0;
        signed_boundary_checks = 0;
        rmw_checks = 0;
        same_address_attacks = 0;
        positive_overflow_attacks = 0;
        negative_overflow_attacks = 0;
        prior_commit_stall = 1'b0;
        prior_commit_last_accept = 1'b0;
        prior_update_accept = 1'b0;
        prior_update_address = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < DEPTH; address++)
                memory[lane][address] = 'x;
        end

        // Window 0: signed endpoints, checkpoint bound, RMW, address endpoints,
        // and a long non-conflicting II=1 update stream.
        reset_dut();
        start_window();
        drive_update(0, 0, 0, 1'b0);
        signed_boundary_checks = signed_boundary_checks + 6;
        drive_update(7, 383, 1, 1'b0);
        drive_update(0, 1, 3, 1'b0);
        drive_update(0, 0, 2, 1'b0);
        rmw_checks = rmw_checks + LANES;
        signed_boundary_checks = signed_boundary_checks + 6;
        for (int index = 0; index < 64; index++)
            drive_update((index * 5 + 2) % BLOCKS,
                         (index * 37 + 5) % WIN_ROWS, 3, index == 63);
        end_window();
        wait_done();

        // Window 1: data array must retain old bits while the valid bitmap
        // logically clears every unwritten vector.
        start_window();
        if ($signed(memory[0][DEPTH-1])
                !== pattern_value(1, 0))
            $fatal(1, "M116 hammer data array was swept instead of lazy-cleared");
        lazy_clear_checks = lazy_clear_checks + 1;
        drive_update(3, 222, 4, 1'b1);
        end_window();
        wait_done();
        lazy_clear_checks = lazy_clear_checks + 1;
        repeat (3) @(posedge clk_core);

        if (positive_updates != 69 || positive_writes != 69
                || ii1_pairs < 67 || read_write_overlap < 67
                || commit_accepts != 2 * COMMIT_VECTORS
                || lane_checks != 2 * COMMIT_VECTORS * LANES
                || completed_windows != 2 || commit_stalls < 100
                || commit_stall_releases == 0
                || address_zero_reads == 0 || address_last_reads == 0
                || address_zero_writes == 0 || address_last_writes == 0)
            $fatal(1, "M116 hammer positive conservation/coverage failure updates=%0d writes=%0d ii1=%0d overlap=%0d commits=%0d lanes=%0d windows=%0d stalls=%0d releases=%0d addr0=%0d/%0d addrlast=%0d/%0d",
                   positive_updates, positive_writes, ii1_pairs,
                   read_write_overlap, commit_accepts, lane_checks,
                   completed_windows, commit_stalls, commit_stall_releases,
                   address_zero_reads, address_zero_writes,
                   address_last_reads, address_last_writes);

        positive_phase = 1'b0;
        automatic_commit_ready = 1'b0;

        // A consecutive same-address request must be rejected while the older
        // accepted write still reaches the external lane memories.
        reset_dut();
        start_window();
        drive_update(4, 177, 4, 1'b0);
        @(negedge clk_core);
        update_valid = 1'b1;
        update_block = 4;
        update_row = 177;
        update_delta = '0;
        update_delta[ACC_BITS-1:0] = 20'sd9;
        @(posedge clk_core);
        if (update_ready || update_accept || !protocol_error
                || !lane_mem_wr_en || lane_mem_wr_addr != flat_addr(4, 177))
            $fatal(1, "M116 hammer same-address RDW fail-closed/preserved-write failure");
        same_address_attacks = same_address_attacks + 1;
        @(negedge clk_core);
        update_valid = 1'b0;
        check_fault_sticky();
        if ($signed(memory[0][flat_addr(4, 177)]) !== 12345)
            $fatal(1, "M116 hammer older accepted write lost on RDW fault");

        // Positive overflow: MAX20 + 1 after a legal spacer address.
        reset_dut();
        start_window();
        drive_update(0, 0, 5, 1'b0);
        drive_update(1, 1, 9, 1'b0);
        drive_update(0, 0, 7, 1'b1);
        @(negedge clk_core);
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M116 hammer positive overflow not suppressed");
        positive_overflow_attacks = positive_overflow_attacks + 1;
        check_fault_sticky();
        if ($signed(memory[0][0]) !== MAX20)
            $fatal(1, "M116 hammer positive overflow corrupted prior data");

        // Negative overflow: MIN20 - 1 after a legal spacer address.
        reset_dut();
        start_window();
        drive_update(7, 383, 6, 1'b0);
        drive_update(6, 382, 9, 1'b0);
        drive_update(7, 383, 8, 1'b1);
        @(negedge clk_core);
        if (!protocol_error || lane_mem_wr_en)
            $fatal(1, "M116 hammer negative overflow not suppressed");
        negative_overflow_attacks = negative_overflow_attacks + 1;
        check_fault_sticky();
        if ($signed(memory[0][DEPTH-1]) !== MIN20)
            $fatal(1, "M116 hammer negative overflow corrupted prior data");

        if (lazy_clear_checks != 2 || signed_boundary_checks != 12
                || rmw_checks != 96 || same_address_attacks != 1
                || positive_overflow_attacks != 1
                || negative_overflow_attacks != 1)
            $fatal(1, "M116 hammer directed counter failure");

        $display("PASS M116 independent signed20 hammer commercial_vcs=true exact_source_sha=true windows=2 positive_updates=%0d positive_writes=%0d ii1_pairs=%0d read_write_overlap=%0d commits=%0d lane_checks=%0d commit_stalls=%0d stall_releases=%0d lazy_clear_checks=2 signed_boundary_checks=12 rmw_lane_checks=96 address_minmax=true same_address_rdw_attacks=1 positive_overflow_attacks=1 negative_overflow_attacks=1 lanes=96 vector_bits=1920 lane_macros=96 macro_depth=3072 macro_width=20 payload_bytes=737280 saving_vs_signed24_bytes=147456 checkpoint_only=true behavioral_macro=true foundry_macro=false ppa=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 positive_updates, positive_writes, ii1_pairs,
                 read_write_overlap, commit_accepts, lane_checks,
                 commit_stalls, commit_stall_releases);
        $finish;
    end
endmodule

`default_nettype wire
