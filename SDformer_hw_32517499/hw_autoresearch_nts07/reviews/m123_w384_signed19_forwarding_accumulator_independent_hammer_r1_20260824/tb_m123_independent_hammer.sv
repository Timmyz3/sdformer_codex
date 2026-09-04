`timescale 1ns/1ps
`default_nettype none

module tb_m123_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int DEPTH = BLOCKS * WIN_ROWS;
    localparam int MAX19 = 262143;
    localparam int MIN19 = -262144;

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
    logic [ACC_BITS-1:0] macro_delay_stage [0:LANES-1];
    integer signed model [0:BLOCKS-1][0:WIN_ROWS-1][0:LANES-1];
    logic [ACC_BITS-1:0] expected_write_data [0:LANES-1];
    logic [11:0] expected_write_addr;
    bit expected_write_valid;
    bit positive_monitor;
    bit previous_accept;
    logic [2:0] previous_block;
    logic [8:0] previous_row;
    integer expected_commit_block;
    integer expected_commit_row;
    integer cycle_count;
    integer total_update_accepts;
    integer total_writes;
    integer positive_update_accepts;
    integer positive_writes;
    integer positive_write_data_checks;
    integer positive_commits;
    integer positive_commit_lane_checks;
    integer positive_commit_stalls;
    integer same_address_pairs;
    integer same_address_reads_suppressed;
    integer nonforward_reads;
    integer different_bank_same_row_checks;
    integer new_invalid_row_checks;
    integer existing_row_checks;
    integer aaa_chains;
    integer aba_chains;
    integer overflow_attacks;
    integer invalid_row_attacks;
    integer reset_edges_with_write_enable;
    integer reset_edges_with_accept;
    integer reset_physical_writes;
    integer successful_windows;
    bit macro_delay2;

    m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter dut (
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
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .protocol_error(protocol_error),
        .window_active(window_active), .busy(busy)
    );

    m123_w384_signed19_forwarding_lane_sliced_accumulator_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_accept(update_accept), .update_block(update_block),
        .update_row(update_row), .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .window_done(window_done), .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .protocol_error(protocol_error), .window_active(window_active),
        .busy(busy)
    );

    function automatic logic [11:0] flat_addr(
        input integer block,
        input integer row
    );
        flat_addr = block * WIN_ROWS + row;
    endfunction

    always #1 clk_core = ~clk_core;

    // Deliberately strict one-cycle synchronous 1R1W lane-macro model.  The
    // read bus is poisoned whenever no read is issued, so a forwarded update
    // cannot accidentally consume a held or asynchronous macro value.
    always @(posedge clk_core) begin
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en) begin
                macro_delay_stage[lane] <= memory[lane][lane_mem_rd_addr];
                lane_mem_rd_data[lane] <= macro_delay2
                    ? macro_delay_stage[lane]
                    : memory[lane][lane_mem_rd_addr];
            end
            else if (window_active)
                lane_mem_rd_data[lane] <= 'x;
            if (lane_mem_wr_en) begin
                memory[lane][lane_mem_wr_addr] <= lane_mem_wr_data[lane];
                if (rst_core)
                    reset_physical_writes = reset_physical_writes + (lane == 0);
            end
        end
    end

    always_comb begin
        commit_ready = !rst_core && ((cycle_count % 7) != 2)
                     && ((cycle_count % 17) != 5);
    end

    // Independent transaction scoreboard.  It proves exact accepted-to-write
    // conservation and checks that every write is the pending signed sum,
    // rather than relying on the production next-cycle SVA alone.
    always @(posedge clk_core) begin : independent_scoreboard
        integer signed delta_value;
        integer signed sum_value;
        cycle_count = cycle_count + 1;
        if (update_accept)
            total_update_accepts = total_update_accepts + 1;
        if (lane_mem_wr_en)
            total_writes = total_writes + 1;
        if (rst_core) begin
            if (lane_mem_wr_en)
                reset_edges_with_write_enable = reset_edges_with_write_enable + 1;
            if (window_start_accept || update_accept || window_end_accept)
                reset_edges_with_accept = reset_edges_with_accept + 1;
            expected_write_valid = 1'b0;
            previous_accept = 1'b0;
        end else if (positive_monitor) begin
            if (expected_write_valid) begin
                if (!lane_mem_wr_en)
                    $fatal(1, "M123 hammer accepted positive update did not write");
                if (lane_mem_wr_addr !== expected_write_addr)
                    $fatal(1, "M123 hammer write address mismatch got=%0d expected=%0d",
                           lane_mem_wr_addr, expected_write_addr);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (lane_mem_wr_data[lane] !== expected_write_data[lane])
                        $fatal(1, "M123 hammer forwarded/pending sum mismatch addr=%0d lane=%0d got=%0d expected=%0d",
                               expected_write_addr, lane,
                               $signed(lane_mem_wr_data[lane]),
                               $signed(expected_write_data[lane]));
                    positive_write_data_checks = positive_write_data_checks + 1;
                end
                positive_writes = positive_writes + 1;
            end else if (lane_mem_wr_en) begin
                $fatal(1, "M123 hammer write has no prior positive accept addr=%0d",
                       lane_mem_wr_addr);
            end
            expected_write_valid = 1'b0;

            if (window_start_accept) begin
                for (int block = 0; block < BLOCKS; block++)
                    for (int row = 0; row < WIN_ROWS; row++)
                        for (int lane = 0; lane < LANES; lane++)
                            model[block][row][lane] = 0;
                expected_commit_block = 0;
                expected_commit_row = 0;
            end

            if (update_accept) begin
                positive_update_accepts = positive_update_accepts + 1;
                if (previous_accept
                        && update_block == previous_block
                        && update_row == previous_row) begin
                    same_address_pairs = same_address_pairs + 1;
                    if (lane_mem_rd_en)
                        $fatal(1, "M123 hammer same-address accept issued macro read");
                    same_address_reads_suppressed
                        = same_address_reads_suppressed + 1;
                end else begin
                    if (!lane_mem_rd_en
                            || lane_mem_rd_addr !== flat_addr(update_block,
                                                             update_row))
                        $fatal(1, "M123 hammer non-forward accept lacks exact macro read");
                    nonforward_reads = nonforward_reads + 1;
                end
                expected_write_addr = flat_addr(update_block, update_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    delta_value = $signed(
                        update_delta[lane * ACC_BITS +: ACC_BITS]);
                    sum_value = model[update_block][update_row][lane]
                              + delta_value;
                    if (sum_value < MIN19 || sum_value > MAX19)
                        $fatal(1, "M123 hammer unexpected positive-phase overflow");
                    model[update_block][update_row][lane] = sum_value;
                    expected_write_data[lane] = sum_value[ACC_BITS-1:0];
                end
                expected_write_valid = 1'b1;
            end

            if (commit_valid && !commit_ready)
                positive_commit_stalls = positive_commit_stalls + 1;
            if (commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M123 hammer commit order mismatch got=%0d/%0d expected=%0d/%0d",
                           commit_block, commit_row,
                           expected_commit_block, expected_commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== model[expected_commit_block]
                                     [expected_commit_row][lane])
                        $fatal(1, "M123 hammer commit numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row,
                               lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]),
                               model[expected_commit_block]
                                    [expected_commit_row][lane]);
                    positive_commit_lane_checks
                        = positive_commit_lane_checks + 1;
                end
                positive_commits = positive_commits + 1;
                if (commit_last !== (expected_commit_block == BLOCKS-1
                                     && expected_commit_row == WIN_ROWS-1))
                    $fatal(1, "M123 hammer commit_last shape mismatch");
                if (expected_commit_row == WIN_ROWS-1) begin
                    expected_commit_row = 0;
                    if (expected_commit_block == BLOCKS-1)
                        expected_commit_block = 0;
                    else
                        expected_commit_block = expected_commit_block + 1;
                end else begin
                    expected_commit_row = expected_commit_row + 1;
                end
            end
            if (window_done)
                successful_windows = successful_windows + 1;

            previous_accept = update_accept;
            if (update_accept) begin
                previous_block = update_block;
                previous_row = update_row;
            end
        end else begin
            expected_write_valid = 1'b0;
            previous_accept = 1'b0;
        end
    end

    task automatic clear_requests;
        begin
            window_start_valid = 1'b0;
            update_valid = 1'b0;
            update_block = '0;
            update_row = '0;
            update_delta = '0;
            window_end_valid = 1'b0;
        end
    endtask

    task automatic reset_clean;
        begin
            @(negedge clk_core);
            clear_requests();
            rst_core = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            window_start_valid = 1'b1;
            do @(posedge clk_core); while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic send_update_pattern(
        input integer block,
        input integer row,
        input integer mode,
        input integer scalar,
        input bit last
    );
        integer signed value;
        logic [VECTOR_BITS-1:0] payload;
        begin
            payload = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                case (mode)
                    0: value = scalar;
                    1: value = (lane % 2 == 0) ? scalar : -scalar;
                    2: value = ((lane * 11 + scalar) % 63) - 31;
                    default: value = 0;
                endcase
                payload[lane * ACC_BITS +: ACC_BITS]
                    = value[ACC_BITS-1:0];
            end
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = block[2:0];
            update_row = row[8:0];
            update_delta = payload;
            do @(posedge clk_core); while (!update_accept);
            if (last) begin
                @(negedge clk_core);
                update_valid = 1'b0;
            end
        end
    endtask

    task automatic end_window_and_wait;
        integer watchdog;
        begin
            @(negedge clk_core);
            window_end_valid = 1'b1;
            do @(posedge clk_core); while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 1'b0;
            watchdog = 0;
            while (!window_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 10000)
                    $fatal(1, "M123 hammer commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic overflow_attack(input bit positive_direction);
        integer base_accepts;
        integer base_writes;
        begin
            positive_monitor = 1'b0;
            reset_clean();
            start_window();
            base_accepts = total_update_accepts;
            base_writes = total_writes;
            send_update_pattern(0, 77, 0,
                                positive_direction ? MAX19 : MIN19, 1'b0);
            send_update_pattern(0, 77, 0,
                                positive_direction ? 1 : -1, 1'b1);
            @(posedge clk_core);
            if (!protocol_error || lane_mem_wr_en)
                $fatal(1, "M123 hammer forwarded overflow did not fail closed dir=%0d",
                       positive_direction);
            repeat (2) @(posedge clk_core);
            if (!protocol_error
                    || total_update_accepts - base_accepts != 2
                    || total_writes - base_writes != 1)
                $fatal(1, "M123 hammer overflow conservation mismatch dir=%0d accepts=%0d writes=%0d fault=%0d",
                       positive_direction,
                       total_update_accepts - base_accepts,
                       total_writes - base_writes, protocol_error);
            overflow_attacks = overflow_attacks + 1;
        end
    endtask

    initial begin : campaign
        integer base_accepts;
        integer base_writes;
        integer reset_write_base;
        integer reset_accept_base;
        clk_core = 1'b0;
        rst_core = 1'b1;
        clear_requests();
        positive_monitor = 1'b0;
        expected_write_valid = 1'b0;
        previous_accept = 1'b0;
        previous_block = '0;
        previous_row = '0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        cycle_count = 0;
        total_update_accepts = 0;
        total_writes = 0;
        positive_update_accepts = 0;
        positive_writes = 0;
        positive_write_data_checks = 0;
        positive_commits = 0;
        positive_commit_lane_checks = 0;
        positive_commit_stalls = 0;
        same_address_pairs = 0;
        same_address_reads_suppressed = 0;
        nonforward_reads = 0;
        different_bank_same_row_checks = 0;
        new_invalid_row_checks = 0;
        existing_row_checks = 0;
        aaa_chains = 0;
        aba_chains = 0;
        overflow_attacks = 0;
        invalid_row_attacks = 0;
        reset_edges_with_write_enable = 0;
        reset_edges_with_accept = 0;
        reset_physical_writes = 0;
        successful_windows = 0;
        macro_delay2 = $test$plusargs("MACRO_DELAY2");
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            macro_delay_stage[lane] = 'x;
            for (int address = 0; address < DEPTH; address++)
                memory[lane][address]
                    = (19'sd12345 + lane * 7 + address * 3)
                      & 19'h7ffff;
        end

        reset_clean();
        positive_monitor = 1'b1;
        start_window();
        base_accepts = positive_update_accepts;
        base_writes = positive_writes;

        // Original M120 two-event failure, now on a logically invalid/new row.
        send_update_pattern(0, 10, 0, 7, 1'b0);
        send_update_pattern(0, 10, 0, -2, 1'b0);
        new_invalid_row_checks = new_invalid_row_checks + 1;

        // Existing-row A-B-A followed by A-A-A forwarding.
        send_update_pattern(1, 20, 0, 100, 1'b0); // A seed
        send_update_pattern(2, 30, 0, -4, 1'b0);  // B
        send_update_pattern(1, 20, 0, 7, 1'b0);   // A from macro
        send_update_pattern(1, 20, 0, -20, 1'b0); // A forward
        send_update_pattern(1, 20, 0, 3, 1'b0);   // A forward
        existing_row_checks = existing_row_checks + 1;
        aaa_chains = aaa_chains + 1;
        aba_chains = aba_chains + 1;

        // Same row number in different banks must not forward.
        send_update_pattern(3, 40, 0, 11, 1'b0);
        send_update_pattern(4, 40, 0, -9, 1'b0);
        different_bank_same_row_checks = different_bank_same_row_checks + 1;

        // Legal signed19 endpoints and mixed-lane positive/negative deltas.
        send_update_pattern(5, 50, 0, MAX19, 1'b0);
        send_update_pattern(5, 50, 0, -1, 1'b0);
        send_update_pattern(6, 60, 0, MIN19, 1'b0);
        send_update_pattern(6, 60, 0, 1, 1'b0);
        send_update_pattern(7, 383, 1, 17, 1'b0);
        send_update_pattern(7, 383, 1, -17, 1'b1);
        end_window_and_wait();
        if (positive_update_accepts - base_accepts != 15
                || positive_writes - base_writes != 15)
            $fatal(1, "M123 hammer main conservation mismatch accepts=%0d writes=%0d",
                   positive_update_accepts - base_accepts,
                   positive_writes - base_writes);

        // Overflow must also fail closed on the forwarding path, not only the
        // production A-B-A/macro-read path.
        overflow_attack(1'b1);
        overflow_attack(1'b0);

        // Out-of-range row is rejected before read/write and becomes sticky.
        positive_monitor = 1'b0;
        reset_clean();
        start_window();
        base_accepts = total_update_accepts;
        base_writes = total_writes;
        @(negedge clk_core);
        update_valid = 1'b1;
        update_block = 3;
        update_row = 9'd384;
        update_delta = '0;
        @(posedge clk_core);
        if (!protocol_error || update_accept || lane_mem_rd_en || lane_mem_wr_en)
            $fatal(1, "M123 hammer invalid row did not reject fail closed");
        @(negedge clk_core);
        update_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || total_update_accepts != base_accepts
                || total_writes != base_writes)
            $fatal(1, "M123 hammer invalid-row sticky/conservation mismatch");
        invalid_row_attacks = invalid_row_attacks + 1;

        // Reset boundary characterization: an accepted pending update exposes
        // a write enable on the reset edge, and accepts are not externally
        // quiesced while synchronous reset is high. This is recorded, not
        // treated as recovery, because the production contract excludes it.
        reset_clean();
        start_window();
        send_update_pattern(4, 88, 0, 9, 1'b0);
        reset_write_base = reset_edges_with_write_enable;
        reset_accept_base = reset_edges_with_accept;
        @(negedge clk_core);
        update_valid = 1'b0;
        rst_core = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        window_start_valid = 1'b1;
        @(posedge clk_core);
        if (!window_start_accept)
            $fatal(1, "M123 hammer expected reset-visible phantom accept absent");
        @(negedge clk_core);
        window_start_valid = 1'b0;
        repeat (1) @(posedge clk_core);
        if (reset_edges_with_write_enable - reset_write_base != 1
                || reset_edges_with_accept - reset_accept_base < 1
                || reset_physical_writes < 1)
            $fatal(1, "M123 hammer reset boundary observation drift writes=%0d accepts=%0d physical=%0d",
                   reset_edges_with_write_enable - reset_write_base,
                   reset_edges_with_accept - reset_accept_base,
                   reset_physical_writes);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);

        // Clean operation resumes, but the pre-reset accepted +9 is lost by
        // contract. Lazy-valid must ignore its physical residue and commit +3.
        positive_monitor = 1'b1;
        start_window();
        send_update_pattern(4, 88, 0, 3, 1'b1);
        end_window_and_wait();

        repeat (3) @(posedge clk_core);
        if (protocol_error || successful_windows != 2
                || positive_update_accepts != 16
                || positive_writes != 16
                || positive_write_data_checks != 16 * LANES
                || positive_commits != 2 * DEPTH
                || positive_commit_lane_checks != 2 * DEPTH * LANES
                || positive_commit_stalls == 0
                || same_address_pairs != 6
                || same_address_reads_suppressed != 6
                || nonforward_reads != 10
                || different_bank_same_row_checks != 1
                || new_invalid_row_checks != 1
                || existing_row_checks != 1
                || aaa_chains != 1 || aba_chains != 1
                || overflow_attacks != 2 || invalid_row_attacks != 1)
            $fatal(1, "M123 hammer final mismatch windows=%0d accepts=%0d writes=%0d write_lanes=%0d commits=%0d commit_lanes=%0d stalls=%0d same=%0d suppressed=%0d reads=%0d ovf=%0d invalid=%0d fault=%0d",
                   successful_windows, positive_update_accepts,
                   positive_writes, positive_write_data_checks,
                   positive_commits, positive_commit_lane_checks,
                   positive_commit_stalls, same_address_pairs,
                   same_address_reads_suppressed, nonforward_reads,
                   overflow_attacks, invalid_row_attacks, protocol_error);

        $display("PASS M123 independent hammer commercial_vcs=true positive_windows=2 positive_updates=16 positive_writes=16 positive_write_lane_checks=1536 commits=6144 commit_lane_checks=589824 same_address_pairs=6 same_address_reads_suppressed=6 original_m120_two_event_closed=true aaa_chains=1 aba_chains=1 new_invalid_row=1 existing_row=1 different_bank_same_row=1 mixed_sign_delta=1 signed19_nonoverflow_boundaries=2 forwarded_overflow_attacks=2 invalid_row_attacks=1 one_cycle_sync_macro_poisoned_no_read=true pending_sum_data_exact=true end_commit_full_numeric=true commit_stalls=%0d reset_edge_write_enable=%0d reset_edge_accept=%0d reset_physical_writes=%0d reset_quiescence=false reset_recovery=false foundry_macro=false physical_speedup=false system_speedup=false headline=false",
                 positive_commit_stalls, reset_edges_with_write_enable,
                 reset_edges_with_accept, reset_physical_writes);
        $finish;
    end
endmodule

`default_nettype wire
