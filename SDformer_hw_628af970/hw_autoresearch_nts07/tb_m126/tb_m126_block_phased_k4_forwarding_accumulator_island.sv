`timescale 1ns/1ps
`default_nettype none

module tb_m126_block_phased_k4_forwarding_accumulator_island;
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

    int cycle_count;
    int start_accepts;
    int fill_accepts;
    int row_accepts;
    int row_dones;
    int fold_update_accepts;
    int selected_sources;
    int full_k4_updates;
    int tail_updates;
    int same_row_update_pairs;
    int lane_memory_writes;
    int lane_read_write_overlap;
    int commit_accepts;
    int commit_lane_checks;
    int commit_stalls;
    int plus512_checks;
    int reset_attacks;
    int positive_fold_updates;
    int positive_selected_sources;
    int positive_tail_updates;
    int positive_lane_writes;
    int expected_commit_block;
    int expected_commit_row;
    bit previous_fold_update_accept;
    logic [2:0] previous_fold_update_block;
    logic [8:0] previous_fold_update_row;
    bit positive_phase;

    m126_block_phased_k4_forwarding_accumulator_island dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid),
        .row_ready(row_ready),
        .row_block(row_block),
        .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(row_accept),
        .row_done(row_done),
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
        .observed_fold_update_accept(observed_fold_update_accept),
        .observed_accumulator_update_accept(
            observed_accumulator_update_accept),
        .observed_fold_update_block(observed_fold_update_block),
        .observed_fold_update_row(observed_fold_update_row),
        .observed_fold_update_delta(observed_fold_update_delta),
        .observed_fold_selected_mask(observed_fold_selected_mask),
        .observed_fold_remaining_mask(observed_fold_remaining_mask),
        .observed_cache_valid(observed_cache_valid),
        .observed_resident_block(observed_resident_block),
        .observed_resident_block_valid(observed_resident_block_valid),
        .fold_protocol_error(fold_protocol_error),
        .accumulator_protocol_error(accumulator_protocol_error),
        .protocol_error(protocol_error),
        .window_active(window_active),
        .busy(busy)
    );

    m126_block_phased_k4_forwarding_accumulator_island_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid),
        .row_ready(row_ready),
        .row_accept(row_accept),
        .row_done(row_done),
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
        .lane_mem_wr_en(lane_mem_wr_en),
        .observed_fold_update_accept(observed_fold_update_accept),
        .observed_accumulator_update_accept(
            observed_accumulator_update_accept),
        .observed_fold_selected_mask(observed_fold_selected_mask),
        .observed_fold_remaining_mask(observed_fold_remaining_mask),
        .protocol_error(protocol_error)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input int block,
        input int source,
        input int lane
    );
        if (block == 0 && lane == 0 && source < 4)
            weight_value = -128;
        else
            weight_value = ((block * 19 + source * 37 + lane * 29)
                            & 8'hff) - 128;
    endfunction

    function automatic logic [15:0] expected_select4(input logic [15:0] mask);
        logic [15:0] remaining;
        logic found;
        begin
            expected_select4 = '0;
            remaining = mask;
            for (int pick = 0; pick < 4; pick++) begin
                found = 1'b0;
                for (int source = 0; source < SOURCES; source++) begin
                    if (!found && remaining[source]) begin
                        expected_select4[source] = 1'b1;
                        remaining[source] = 1'b0;
                        found = 1'b1;
                    end
                end
            end
        end
    endfunction

    function automatic logic [15:0] row_mask_for(
        input int block,
        input int row
    );
        logic [31:0] mixed;
        logic [15:0] candidate;
        begin
            if (row == 0)
                candidate = 16'h0000;
            else if (block == 0 && row == 1)
                candidate = 16'h000f;
            else if ((row % 23) == 2)
                candidate = 16'hffff;
            else if ((row % 29) == 3)
                candidate = 16'h0001 << ((row + block) % SOURCES);
            else begin
                mixed = (block + 1) * 32'h45d9f3b
                      ^ (row + 7) * 32'h9e3779b9;
                candidate = mixed[15:0]
                          ^ {mixed[23:16], mixed[31:24]};
                if (candidate == 0)
                    candidate = 16'h0001 << ((row + block) % SOURCES);
            end
            row_mask_for = candidate;
        end
    endfunction

    function automatic logic [15:0] negate_mask_for(
        input int block,
        input int row,
        input logic [15:0] mask
    );
        logic [31:0] mixed;
        begin
            if (block == 0 && row == 1)
                negate_mask_for = 16'h000f;
            else begin
                mixed = (block + 3) * 32'h27d4eb2d
                      ^ (row + 11) * 32'h165667b1;
                negate_mask_for = mask
                                & (mixed[15:0] ^ mixed[31:16]);
            end
        end
    endfunction

    always @(posedge clk_core) begin : lane_memory_model
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane]
                    <= lane_memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                lane_memory[lane][lane_mem_wr_addr]
                    <= lane_mem_wr_data[lane];
        end
        if (lane_mem_wr_en)
            lane_memory_writes <= lane_memory_writes + 1;
        if (lane_mem_rd_en && lane_mem_wr_en)
            lane_read_write_overlap <= lane_read_write_overlap + 1;
    end

    always @(posedge clk_core) begin : scoreboard
        if (rst_core) begin
            previous_fold_update_accept <= 1'b0;
            previous_fold_update_block <= '0;
            previous_fold_update_row <= '0;
            commit_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            commit_ready <= ((cycle_count % 11) != 3)
                         && ((cycle_count % 37) != 8);
            if (window_start_accept)
                start_accepts <= start_accepts + 1;
            if (weight_fill_accept)
                fill_accepts <= fill_accepts + 1;
            if (row_accept)
                row_accepts <= row_accepts + 1;
            if (row_done)
                row_dones <= row_dones + 1;
            if (observed_fold_update_accept) begin
                fold_update_accepts <= fold_update_accepts + 1;
                selected_sources <= selected_sources
                                    + $countones(observed_fold_selected_mask);
                if ($countones(observed_fold_selected_mask) == 4)
                    full_k4_updates <= full_k4_updates + 1;
                else
                    tail_updates <= tail_updates + 1;
                if (previous_fold_update_accept
                        && previous_fold_update_block
                           == observed_fold_update_block
                        && previous_fold_update_row
                           == observed_fold_update_row)
                    same_row_update_pairs <= same_row_update_pairs + 1;
                previous_fold_update_block <= observed_fold_update_block;
                previous_fold_update_row <= observed_fold_update_row;
            end
            previous_fold_update_accept <= observed_fold_update_accept;

            if (commit_valid && !commit_ready)
                commit_stalls <= commit_stalls + 1;
            if (commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M126 commit order mismatch got=(%0d,%0d) expected=(%0d,%0d)",
                           commit_block, commit_row,
                           expected_commit_block, expected_commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M126 commit mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row,
                               lane,
                               $signed(commit_data[
                                   lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                end
                commit_accepts <= commit_accepts + 1;
                commit_lane_checks <= commit_lane_checks + LANES;
                if (expected_commit_row == ROWS-1) begin
                    expected_commit_row <= 0;
                    if (expected_commit_block == BLOCKS-1)
                        expected_commit_block <= 0;
                    else
                        expected_commit_block <= expected_commit_block + 1;
                end else begin
                    expected_commit_row <= expected_commit_row + 1;
                end
            end
            if (positive_phase && protocol_error)
                $fatal(1, "M126 unexpected protocol_error cycle=%0d fold=%0b accumulator=%0b",
                       cycle_count, fold_protocol_error,
                       accumulator_protocol_error);
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            weight_fill_valid = 1'b0;
            row_valid = 1'b0;
            window_end_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic start_window;
        int watchdog;
        begin
            @(negedge clk_core);
            window_start_valid = 1'b1;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 20)
                    $fatal(1, "M126 start watchdog");
            end while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic fill_one_source(input int block, input int source);
        logic [255:0] payload;
        integer signed value;
        int watchdog;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = '0;
                for (int item = 0; item < 32; item++) begin
                    value = weight_value(block, source, beat * 32 + item);
                    payload[item * 8 +: 8] = value[7:0];
                end
                @(negedge clk_core);
                weight_fill_valid = 1'b1;
                weight_fill_block = block[2:0];
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                watchdog = 0;
                do begin
                    @(posedge clk_core);
                    watchdog = watchdog + 1;
                    if (watchdog > 20)
                        $fatal(1, "M126 fill watchdog block=%0d source=%0d beat=%0d",
                               block, source, beat);
                end while (!weight_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 1'b0;
        end
    endtask

    task automatic send_and_check_row(
        input int block,
        input int row,
        input logic [15:0] mask,
        input logic [15:0] negate
    );
        logic [15:0] remaining;
        logic [15:0] expected_selected;
        integer signed expected_delta;
        int watchdog;
        begin
            for (int lane = 0; lane < LANES; lane++) begin
                for (int source = 0; source < SOURCES; source++) begin
                    if (mask[source]) begin
                        expected_delta = weight_value(block, source, lane);
                        if (negate[source])
                            expected_delta = -expected_delta;
                        reference[block][row][lane]
                            = reference[block][row][lane] + expected_delta;
                    end
                end
            end

            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = block[2:0];
            row_offset = row[8:0];
            row_source_mask = mask;
            row_negate_mask = negate;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 30)
                    $fatal(1, "M126 row accept watchdog block=%0d row=%0d",
                           block, row);
            end while (!row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;

            remaining = mask;
            watchdog = 0;
            while (!row_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (observed_fold_update_accept) begin
                    expected_selected = expected_select4(remaining);
                    if (observed_fold_selected_mask !== expected_selected
                            || observed_fold_update_block !== block[2:0]
                            || observed_fold_update_row !== row[8:0])
                        $fatal(1, "M126 fold identity mismatch block=%0d row=%0d got_mask=%h expected=%h",
                               block, row, observed_fold_selected_mask,
                               expected_selected);
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_delta = 0;
                        for (int source = 0; source < SOURCES; source++) begin
                            if (expected_selected[source])
                                expected_delta = expected_delta
                                    + (negate[source]
                                       ? -weight_value(block, source, lane)
                                       : weight_value(block, source, lane));
                        end
                        if ($signed(observed_fold_update_delta[
                                lane * ACC_BITS +: ACC_BITS])
                                !== expected_delta)
                            $fatal(1, "M126 fold numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                                   block, row, lane,
                                   $signed(observed_fold_update_delta[
                                       lane * ACC_BITS +: ACC_BITS]),
                                   expected_delta);
                    end
                    if (block == 0 && row == 1
                            && expected_selected == 16'h000f) begin
                        if ($signed(observed_fold_update_delta[0 +: ACC_BITS])
                                != 512)
                            $fatal(1, "M126 +512 signed11 boundary failed");
                        plus512_checks = plus512_checks + 1;
                    end
                    remaining = remaining & ~expected_selected;
                end
                if (watchdog > 100)
                    $fatal(1, "M126 row completion watchdog block=%0d row=%0d remaining=%h",
                           block, row, remaining);
            end
            if (remaining != 0 || observed_fold_remaining_mask != 0)
                $fatal(1, "M126 source conservation mismatch block=%0d row=%0d remaining=%h dut=%h",
                       block, row, remaining,
                       observed_fold_remaining_mask);
        end
    endtask

    task automatic end_window_and_commit;
        int watchdog;
        begin
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            window_end_valid = 1'b1;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 30)
                    $fatal(1, "M126 end watchdog");
            end while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 1'b0;
            watchdog = 0;
            while (!window_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 10000)
                    $fatal(1, "M126 commit watchdog commits=%0d",
                           commit_accepts);
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic reset_edge_attack;
        int writes_before;
        int watchdog;
        begin
            start_window();
            fill_one_source(0, 0);
            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = 0;
            row_offset = 7;
            row_source_mask = 16'h0001;
            row_negate_mask = 16'h0000;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;
            watchdog = 0;
            do begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 20)
                    $fatal(1, "M126 reset attack update watchdog");
            end while (!observed_fold_update_accept);
            writes_before = lane_memory_writes;
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b1;
            weight_fill_valid = 1'b1;
            row_valid = 1'b1;
            window_end_valid = 1'b1;
            commit_ready = 1'b1;
            #0.1;
            if (window_start_ready || window_start_accept
                    || weight_fill_ready || weight_fill_accept
                    || row_ready || row_accept || row_done
                    || window_end_ready || window_end_accept
                    || commit_valid || window_done
                    || lane_mem_rd_en || lane_mem_wr_en
                    || observed_fold_update_accept || protocol_error)
                $fatal(1, "M126 reset isolation combinational failure");
            repeat (2) @(posedge clk_core);
            if (lane_memory_writes != writes_before)
                $fatal(1, "M126 reset edge leaked physical write before=%0d after=%0d",
                       writes_before, lane_memory_writes);
            reset_attacks = reset_attacks + 1;
            @(negedge clk_core);
            window_start_valid = 1'b0;
            weight_fill_valid = 1'b0;
            row_valid = 1'b0;
            window_end_valid = 1'b0;
            rst_core = 1'b0;
            repeat (3) @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start_valid = 1'b0;
        weight_fill_valid = 1'b0;
        weight_fill_block = '0;
        weight_fill_source = '0;
        weight_fill_beat = '0;
        weight_fill_data = '0;
        row_valid = 1'b0;
        row_block = '0;
        row_offset = '0;
        row_source_mask = '0;
        row_negate_mask = '0;
        window_end_valid = 1'b0;
        commit_ready = 1'b0;
        cycle_count = 0;
        start_accepts = 0;
        fill_accepts = 0;
        row_accepts = 0;
        row_dones = 0;
        fold_update_accepts = 0;
        selected_sources = 0;
        full_k4_updates = 0;
        tail_updates = 0;
        same_row_update_pairs = 0;
        lane_memory_writes = 0;
        lane_read_write_overlap = 0;
        commit_accepts = 0;
        commit_lane_checks = 0;
        commit_stalls = 0;
        plus512_checks = 0;
        reset_attacks = 0;
        positive_fold_updates = 0;
        positive_selected_sources = 0;
        positive_tail_updates = 0;
        positive_lane_writes = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        previous_fold_update_accept = 1'b0;
        previous_fold_update_block = '0;
        previous_fold_update_row = '0;
        positive_phase = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = '0;
            for (int address = 0; address < BLOCKS*ROWS; address++)
                lane_memory[lane][address] = '0;
        end
        for (int block = 0; block < BLOCKS; block++)
            for (int row = 0; row < ROWS; row++)
                for (int lane = 0; lane < LANES; lane++)
                    reference[block][row][lane] = 0;

        reset_dut();
        start_window();
        for (int block = 0; block < BLOCKS; block++) begin
            for (int source = 0; source < SOURCES; source++)
                fill_one_source(block, source);
            if (!observed_resident_block_valid
                    || observed_resident_block !== block[2:0]
                    || observed_cache_valid !== 16'hffff)
                $fatal(1, "M126 cache identity mismatch block=%0d valid=%h",
                       block, observed_cache_valid);
            for (int row = 0; row < ROWS; row++) begin
                logic [15:0] mask;
                logic [15:0] negate;
                mask = row_mask_for(block, row);
                negate = negate_mask_for(block, row, mask);
                send_and_check_row(block, row, mask, negate);
            end
        end
        end_window_and_commit();
        if (start_accepts != 1 || fill_accepts != BLOCKS*SOURCES*3
                || row_accepts != BLOCKS*ROWS
                || row_dones != BLOCKS*ROWS
                || fold_update_accepts == 0
                || selected_sources == 0
                || full_k4_updates == 0 || tail_updates == 0
                || same_row_update_pairs == 0
                || lane_memory_writes != fold_update_accepts
                || commit_accepts != BLOCKS*ROWS
                || commit_lane_checks != BLOCKS*ROWS*LANES
                || commit_stalls == 0 || plus512_checks != 1
                || protocol_error)
            $fatal(1, "M126 positive conservation mismatch starts=%0d fills=%0d rows=%0d done=%0d updates=%0d selected=%0d full=%0d tail=%0d same=%0d writes=%0d commits=%0d lanes=%0d stalls=%0d plus512=%0d",
                   start_accepts, fill_accepts, row_accepts, row_dones,
                   fold_update_accepts, selected_sources,
                   full_k4_updates, tail_updates, same_row_update_pairs,
                   lane_memory_writes, commit_accepts,
                   commit_lane_checks, commit_stalls, plus512_checks);

        positive_fold_updates = fold_update_accepts;
        positive_selected_sources = selected_sources;
        positive_tail_updates = tail_updates;
        positive_lane_writes = lane_memory_writes;

        positive_phase = 1'b0;
        reset_edge_attack();
        if (reset_attacks != 1)
            $fatal(1, "M126 reset attack coverage missing");

        $display("PASS M126 K4 fold plus forwarding accumulator VCS starts=%0d fills=%0d rows=%0d row_done=%0d fold_updates=%0d selected_sources=%0d full_k4_updates=%0d tail_updates=%0d same_row_update_pairs=%0d lane_writes=%0d rw_overlap=%0d commits=%0d commit_lane_checks=%0d commit_stalls=%0d plus512_checks=%0d reset_attacks=%0d positive_fold_updates=%0d positive_selected_sources=%0d positive_tail_updates=%0d positive_lane_writes=%0d reset_pending_updates=1 reset_suppressed_writes=1 blocks=8 rows_per_block=384 lanes=96 cache_bytes=1536 fold_bits=11 accumulator_bits=19 m125_m123_integrated=true reset_isolation=true functional_directed_update_compression=3.385476385476 heldout_fixed8_service_projection=3.1725369008459166 projection_only=true foundry_weight_macro=false foundry_accumulator_macro=false physical_speedup=false system_speedup=false headline=false",
                 start_accepts, fill_accepts, row_accepts, row_dones,
                 fold_update_accepts, selected_sources,
                 full_k4_updates, tail_updates, same_row_update_pairs,
                 lane_memory_writes, lane_read_write_overlap,
                 commit_accepts, commit_lane_checks, commit_stalls,
                 plus512_checks, reset_attacks,
                 positive_fold_updates, positive_selected_sources,
                 positive_tail_updates, positive_lane_writes);
        $finish;
    end
endmodule

`default_nettype wire
