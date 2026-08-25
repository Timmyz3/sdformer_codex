`timescale 1ns/1ps
`default_nettype none

module tb_m126_independent_hammer;
    localparam int BLOCKS = 8;
    localparam int ROWS = 384;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;

    logic clk_core, rst_core;
    logic window_start_valid, window_start_ready, window_start_accept;
    logic weight_fill_valid, weight_fill_ready;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic weight_fill_accept;
    logic row_valid, row_ready;
    logic [2:0] row_block;
    logic [8:0] row_offset;
    logic [15:0] row_source_mask, row_negate_mask;
    logic row_accept, row_done;
    logic window_end_valid, window_end_ready, window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [1823:0] commit_data;
    logic commit_last, window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [18:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [18:0] lane_mem_wr_data [0:LANES-1];
    logic observed_fold_update_accept;
    logic observed_accumulator_update_accept;
    logic [2:0] observed_fold_update_block;
    logic [8:0] observed_fold_update_row;
    logic [1823:0] observed_fold_update_delta;
    logic [15:0] observed_fold_selected_mask;
    logic [15:0] observed_fold_remaining_mask;
    logic [15:0] observed_cache_valid;
    logic [2:0] observed_resident_block;
    logic observed_resident_block_valid;
    logic fold_protocol_error, accumulator_protocol_error;
    logic protocol_error, window_active, busy;

    logic [18:0] lane_memory [0:LANES-1][0:BLOCKS*ROWS-1];
    integer signed reference [0:BLOCKS-1][0:ROWS-1][0:LANES-1];
    integer signed pending_write_data [0:LANES-1];
    bit pending_write_valid;
    int pending_write_addr;
    bit strict_positive;
    bit manual_commit_ready;
    bit manual_commit_value;

    int cycle_count;
    int start_accepts;
    int fill_accepts;
    int row_accepts;
    int row_dones;
    int fold_accepts;
    int accumulator_accepts;
    int selected_sources;
    int lane_writes;
    int write_lane_checks;
    int commit_accepts;
    int commit_lane_checks;
    int commit_stalls;
    int full_k4_updates;
    int tail_updates;
    int forwarding_pairs;
    int same_row_replays;
    int plus512_checks;
    int minus512_checks;
    int block_transition_checks;
    int reset_high_cycles;
    int reset_handshake_violations;
    int reset_pending_internal_write_visible;
    int reset_edge_suppressed_writes;
    int expected_commit_block;
    int expected_commit_row;
    bit previous_fold_accept;
    logic [2:0] previous_fold_block;
    logic [8:0] previous_fold_row;

    m126_block_phased_k4_forwarding_accumulator_island dut (.*);

    m126_block_phased_k4_forwarding_accumulator_island_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid), .row_ready(row_ready),
        .row_accept(row_accept), .row_done(row_done),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .window_done(window_done), .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_wr_en(lane_mem_wr_en),
        .observed_fold_update_accept(observed_fold_update_accept),
        .observed_accumulator_update_accept(observed_accumulator_update_accept),
        .observed_fold_selected_mask(observed_fold_selected_mask),
        .observed_fold_remaining_mask(observed_fold_remaining_mask),
        .protocol_error(protocol_error)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed model_weight(
        input int block_id,
        input int source,
        input int lane
    );
        int raw;
        begin
            if (block_id == 0 && lane == 0 && source < 4)
                model_weight = -128;
            else if (block_id == 0 && lane == 1 && source < 4)
                model_weight = 127;
            else begin
                raw = (block_id * 61 + source * 37 + lane * 29) & 8'hff;
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

    always @(negedge clk_core) begin
        if (!manual_commit_ready)
            commit_ready = ((cycle_count % 5) != 1)
                         && ((cycle_count % 13) != 8);
        else
            commit_ready = manual_commit_value;
    end

    always @(posedge clk_core) begin : independent_scoreboard
        int flat_addr;
        if (rst_core) begin
            pending_write_valid = 1'b0;
            previous_fold_accept = 1'b0;
            if (window_start_ready || window_start_accept
                    || weight_fill_ready || weight_fill_accept
                    || row_ready || row_accept || row_done
                    || window_end_ready || window_end_accept
                    || commit_valid || window_done || lane_mem_rd_en
                    || lane_mem_wr_en || observed_fold_update_accept
                    || observed_accumulator_update_accept || protocol_error)
                reset_handshake_violations = reset_handshake_violations + 1;
            reset_high_cycles = reset_high_cycles + 1;
        end else begin
            cycle_count = cycle_count + 1;
            if (strict_positive && lane_mem_wr_en !== pending_write_valid)
                $fatal(1, "M126 write conservation mismatch write=%0b pending=%0b cycle=%0d",
                       lane_mem_wr_en, pending_write_valid, cycle_count);
            if (lane_mem_wr_en) begin
                if (strict_positive && lane_mem_wr_addr !== pending_write_addr[11:0])
                    $fatal(1, "M126 write address mismatch got=%0d expected=%0d",
                           lane_mem_wr_addr, pending_write_addr);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (strict_positive
                            && $signed(lane_mem_wr_data[lane])
                               !== pending_write_data[lane])
                        $fatal(1, "M126 write data mismatch addr=%0d lane=%0d got=%0d expected=%0d",
                               lane_mem_wr_addr, lane,
                               $signed(lane_mem_wr_data[lane]),
                               pending_write_data[lane]);
                    lane_memory[lane][lane_mem_wr_addr]
                        <= lane_mem_wr_data[lane];
                end
                lane_writes = lane_writes + 1;
                write_lane_checks = write_lane_checks + LANES;
            end
            if (lane_mem_rd_en) begin
                for (int lane = 0; lane < LANES; lane++)
                    lane_mem_rd_data[lane]
                        <= lane_memory[lane][lane_mem_rd_addr];
            end

            if (observed_fold_update_accept
                    !== observed_accumulator_update_accept)
                $fatal(1, "M126 fold/accumulator accept divergence");
            if (observed_fold_update_accept) begin
                flat_addr = observed_fold_update_block * ROWS
                          + observed_fold_update_row;
                if (previous_fold_accept
                        && previous_fold_block == observed_fold_update_block
                        && previous_fold_row == observed_fold_update_row) begin
                    if (lane_mem_rd_en)
                        $fatal(1, "M126 same-address forwarding issued macro read");
                    forwarding_pairs = forwarding_pairs + 1;
                end
                for (int lane = 0; lane < LANES; lane++) begin
                    reference[observed_fold_update_block]
                             [observed_fold_update_row][lane]
                        = reference[observed_fold_update_block]
                                   [observed_fold_update_row][lane]
                        + $signed(observed_fold_update_delta[
                            lane * ACC_BITS +: ACC_BITS]);
                    pending_write_data[lane]
                        = reference[observed_fold_update_block]
                                   [observed_fold_update_row][lane];
                end
                pending_write_addr = flat_addr;
                pending_write_valid = 1'b1;
                fold_accepts = fold_accepts + 1;
                accumulator_accepts = accumulator_accepts + 1;
                selected_sources = selected_sources
                                 + $countones(observed_fold_selected_mask);
                if ($countones(observed_fold_selected_mask) == 4)
                    full_k4_updates = full_k4_updates + 1;
                else
                    tail_updates = tail_updates + 1;
                previous_fold_block = observed_fold_update_block;
                previous_fold_row = observed_fold_update_row;
            end else begin
                pending_write_valid = 1'b0;
            end
            previous_fold_accept = observed_fold_update_accept;

            if (window_start_accept)
                start_accepts = start_accepts + 1;
            if (weight_fill_accept)
                fill_accepts = fill_accepts + 1;
            if (row_accept)
                row_accepts = row_accepts + 1;
            if (row_done)
                row_dones = row_dones + 1;
            if (commit_valid && !commit_ready)
                commit_stalls = commit_stalls + 1;
            if (commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M126 commit identity mismatch got=(%0d,%0d) expected=(%0d,%0d)",
                           commit_block, commit_row,
                           expected_commit_block, expected_commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M126 commit data mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row,
                               lane,
                               $signed(commit_data[lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                end
                commit_accepts = commit_accepts + 1;
                commit_lane_checks = commit_lane_checks + LANES;
                if (expected_commit_row == ROWS-1) begin
                    expected_commit_row = 0;
                    expected_commit_block = expected_commit_block == BLOCKS-1
                                          ? 0 : expected_commit_block + 1;
                end else begin
                    expected_commit_row = expected_commit_row + 1;
                end
            end
            if (strict_positive && protocol_error)
                $fatal(1, "M126 unexpected positive protocol error fold=%0b accumulator=%0b",
                       fold_protocol_error, accumulator_protocol_error);
        end
    end

    task automatic drive_idle;
        begin
            window_start_valid = 0;
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
            window_end_valid = 0;
        end
    endtask

    task automatic clean_reset;
        begin
            @(negedge clk_core);
            drive_idle();
            rst_core = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            repeat (2) @(posedge clk_core);
            if (protocol_error || window_active || busy)
                $fatal(1, "M126 reset recovery failure");
        end
    endtask

    task automatic start_window;
        int watchdog;
        begin
            @(negedge clk_core);
            window_start_valid = 1;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 20)
                    $fatal(1, "M126 independent start watchdog");
            end while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 0;
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
                do @(posedge clk_core); while (!weight_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 0;
        end
    endtask

    task automatic fill_block(input int block_id);
        begin
            for (int source = 0; source < SOURCES; source++)
                fill_source(block_id, source);
            if (!observed_resident_block_valid
                    || observed_resident_block !== block_id[2:0]
                    || observed_cache_valid !== 16'hffff)
                $fatal(1, "M126 cache identity mismatch block=%0d valid=%h",
                       block_id, observed_cache_valid);
        end
    endtask

    task automatic send_row(
        input int block_id,
        input int row_id,
        input logic [15:0] mask,
        input logic [15:0] negate
    );
        logic [15:0] remaining;
        logic [15:0] consumed;
        logic [15:0] expected_selected;
        int expected_delta;
        int watchdog;
        begin
            remaining = mask;
            consumed = 0;
            @(negedge clk_core);
            row_valid = 1;
            row_block = block_id[2:0];
            row_offset = row_id[8:0];
            row_source_mask = mask;
            row_negate_mask = negate;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 0;
            watchdog = 0;
            while (!row_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 100)
                    $fatal(1, "M126 independent row watchdog block=%0d row=%0d",
                           block_id, row_id);
                if (observed_fold_update_accept) begin
                    expected_selected = oracle_lowest4(remaining);
                    if (observed_fold_selected_mask !== expected_selected
                            || observed_fold_update_block !== block_id[2:0]
                            || observed_fold_update_row !== row_id[8:0])
                        $fatal(1, "M126 lowest4/identity mismatch block=%0d row=%0d got=%h expected=%h",
                               block_id, row_id,
                               observed_fold_selected_mask, expected_selected);
                    if ((observed_fold_selected_mask & consumed) != 0)
                        $fatal(1, "M126 duplicate selected source");
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_delta = 0;
                        for (int source = 0; source < SOURCES; source++) begin
                            if (expected_selected[source])
                                expected_delta = expected_delta
                                    + (negate[source]
                                       ? -model_weight(block_id, source, lane)
                                       : model_weight(block_id, source, lane));
                        end
                        if ($signed(observed_fold_update_delta[
                                lane * ACC_BITS +: ACC_BITS]) !== expected_delta)
                            $fatal(1, "M126 fold delta mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                                   block_id, row_id, lane,
                                   $signed(observed_fold_update_delta[
                                       lane * ACC_BITS +: ACC_BITS]),
                                   expected_delta);
                    end
                    if (block_id == 0 && row_id == 20
                            && expected_selected == 16'h000f) begin
                        if ($signed(observed_fold_update_delta[0 +: ACC_BITS]) != 512)
                            $fatal(1, "M126 +512 boundary mismatch");
                        plus512_checks = plus512_checks + 1;
                    end
                    if (block_id == 0 && row_id == 21
                            && expected_selected == 16'h000f) begin
                        if ($signed(observed_fold_update_delta[0 +: ACC_BITS]) != -512)
                            $fatal(1, "M126 -512 boundary mismatch");
                        minus512_checks = minus512_checks + 1;
                    end
                    consumed = consumed | observed_fold_selected_mask;
                    remaining = remaining & ~observed_fold_selected_mask;
                end
            end
            if (consumed !== mask || remaining != 0
                    || observed_fold_remaining_mask != 0)
                $fatal(1, "M126 row source conservation mismatch mask=%h consumed=%h remaining=%h",
                       mask, consumed, remaining);
            @(posedge clk_core);
        end
    endtask

    task automatic end_and_commit;
        int watchdog;
        begin
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            window_end_valid = 1;
            do @(posedge clk_core); while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 0;
            watchdog = 0;
            while (!window_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 10000)
                    $fatal(1, "M126 independent commit watchdog commits=%0d",
                           commit_accepts);
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic reset_isolation_attack;
        int writes_before;
        int watchdog;
        begin
            strict_positive = 0;
            clean_reset();
            start_window();
            fill_source(0, 0);
            @(negedge clk_core);
            row_valid = 1;
            row_block = 0;
            row_offset = 7;
            row_source_mask = 16'h0001;
            row_negate_mask = 0;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 0;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 20)
                    $fatal(1, "M126 reset attack update watchdog");
            end while (!observed_fold_update_accept);
            writes_before = lane_writes;
            @(negedge clk_core);
            if (!dut.internal_lane_mem_wr_en || !lane_mem_wr_en)
                $fatal(1, "M126 reset write attack was not sensitized");
            reset_pending_internal_write_visible
                = reset_pending_internal_write_visible + 1;
            rst_core = 1;
            window_start_valid = 1;
            weight_fill_valid = 1;
            row_valid = 1;
            window_end_valid = 1;
            manual_commit_ready = 1;
            manual_commit_value = 1;
            #0.1;
            if (!dut.internal_lane_mem_wr_en)
                $fatal(1, "M126 internal pending write disappeared before reset edge");
            if (window_start_ready || window_start_accept
                    || weight_fill_ready || weight_fill_accept
                    || row_ready || row_accept || row_done
                    || window_end_ready || window_end_accept
                    || commit_valid || window_done || lane_mem_rd_en
                    || lane_mem_wr_en || observed_fold_update_accept
                    || observed_accumulator_update_accept || protocol_error)
                $fatal(1, "M126 reset combinational isolation failure");
            repeat (2) @(posedge clk_core);
            if (lane_writes != writes_before)
                $fatal(1, "M126 reset leaked physical write before=%0d after=%0d",
                       writes_before, lane_writes);
            reset_edge_suppressed_writes = reset_edge_suppressed_writes + 1;
            @(negedge clk_core);
            drive_idle();
            rst_core = 0;
            manual_commit_value = 0;
            repeat (3) @(posedge clk_core);
            manual_commit_ready = 0;
            if (protocol_error || window_active || busy)
                $fatal(1, "M126 reset attack recovery failure");
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        manual_commit_ready = 1;
        manual_commit_value = 0;
        strict_positive = 0;
        cycle_count = 0;
        start_accepts = 0;
        fill_accepts = 0;
        row_accepts = 0;
        row_dones = 0;
        fold_accepts = 0;
        accumulator_accepts = 0;
        selected_sources = 0;
        lane_writes = 0;
        write_lane_checks = 0;
        commit_accepts = 0;
        commit_lane_checks = 0;
        commit_stalls = 0;
        full_k4_updates = 0;
        tail_updates = 0;
        forwarding_pairs = 0;
        same_row_replays = 0;
        plus512_checks = 0;
        minus512_checks = 0;
        block_transition_checks = 0;
        reset_high_cycles = 0;
        reset_handshake_violations = 0;
        reset_pending_internal_write_visible = 0;
        reset_edge_suppressed_writes = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        previous_fold_accept = 0;
        previous_fold_block = 0;
        previous_fold_row = 0;
        pending_write_valid = 0;
        pending_write_addr = 0;
        drive_idle();
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 0;
            pending_write_data[lane] = 0;
            for (int address = 0; address < BLOCKS*ROWS; address++)
                lane_memory[lane][address] = 0;
        end
        for (int block_id = 0; block_id < BLOCKS; block_id++)
            for (int row_id = 0; row_id < ROWS; row_id++)
                for (int lane = 0; lane < LANES; lane++)
                    reference[block_id][row_id][lane] = 0;

        clean_reset();
        strict_positive = 1;
        manual_commit_ready = 0;
        start_window();
        fill_block(0);
        send_row(0, 10, 16'hffff, 16'ha55a);
        send_row(0, 10, 16'h000f, 16'h0005);
        same_row_replays = same_row_replays + 1;
        send_row(0, 11, 16'h0007, 16'h0005);
        send_row(0, 12, 16'h0003, 16'h0002);
        send_row(0, 13, 16'h0001, 16'h0001);
        send_row(0, 14, 16'h0000, 16'h0000);
        send_row(0, 20, 16'h000f, 16'h000f);
        send_row(0, 21, 16'h000f, 16'h0000);
        send_row(0, 383, 16'h8421, 16'h8020);

        fill_source(1, 0);
        if (observed_resident_block != 1
                || observed_cache_valid != 16'h0001)
            $fatal(1, "M126 block transition invalidation failure valid=%h block=%0d",
                   observed_cache_valid, observed_resident_block);
        block_transition_checks = block_transition_checks + 1;
        for (int source = 1; source < SOURCES; source++)
            fill_source(1, source);
        send_row(1, 5, 16'hffff, 16'h5aa5);
        send_row(1, 5, 16'h0f0f, 16'h0505);
        same_row_replays = same_row_replays + 1;
        send_row(1, 6, 16'h00e0, 16'h00a0);

        end_and_commit();
        if (start_accepts != 1 || fill_accepts != 96
                || row_accepts != 12 || row_dones != 12
                || fold_accepts != 18 || accumulator_accepts != 18
                || selected_sources != 65 || lane_writes != 18
                || write_lane_checks != 1728
                || commit_accepts != BLOCKS*ROWS
                || commit_lane_checks != BLOCKS*ROWS*LANES
                || full_k4_updates != 14 || tail_updates != 4
                || forwarding_pairs < 7 || same_row_replays != 2
                || plus512_checks != 1 || minus512_checks != 1
                || block_transition_checks != 1 || commit_stalls == 0
                || protocol_error)
            $fatal(1, "M126 independent positive aggregate mismatch starts=%0d fills=%0d rows=%0d done=%0d folds=%0d acc=%0d sources=%0d writes=%0d write_lanes=%0d commits=%0d commit_lanes=%0d full=%0d tail=%0d forward=%0d stalls=%0d",
                   start_accepts, fill_accepts, row_accepts, row_dones,
                   fold_accepts, accumulator_accepts, selected_sources,
                   lane_writes, write_lane_checks, commit_accepts,
                   commit_lane_checks, full_k4_updates, tail_updates,
                   forwarding_pairs, commit_stalls);

        reset_isolation_attack();
        if (reset_handshake_violations != 0
                || reset_pending_internal_write_visible != 1
                || reset_edge_suppressed_writes != 1)
            $fatal(1, "M126 reset isolation aggregate mismatch violations=%0d internal_pending=%0d suppressed=%0d",
                   reset_handshake_violations,
                   reset_pending_internal_write_visible,
                   reset_edge_suppressed_writes);

        $display("PASS M126 independent hammer positive_rows=12 positive_folds=18 positive_accumulator_accepts=18 selected_sources=65 lane_writes=18 write_lane_checks=1728 forwarding_pairs=%0d full_k4=14 tails=4 same_row_replays=2 block_transition_checks=1 plus512=1 minus512=1 commits=3072 commit_lane_checks=294912 commit_stalls=%0d reset_high_cycles=%0d reset_handshake_violations=0 reset_pending_internal_write_visible=1 reset_edge_suppressed_writes=1 source_update_write_commit_conservation=true reset_isolation=true directed_ratio_3p385476=traffic_only projection_3p1725369=cycle_only physical_speedup=false system_speedup=false",
                 forwarding_pairs, commit_stalls, reset_high_cycles);
        $finish;
    end
endmodule

`default_nettype wire
