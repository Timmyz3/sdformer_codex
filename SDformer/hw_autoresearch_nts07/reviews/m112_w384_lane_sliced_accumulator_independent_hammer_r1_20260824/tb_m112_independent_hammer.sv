`timescale 1ns/1ps
`default_nettype none

module tb_m112_independent_hammer;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int LANES = 96;
    localparam int ACC_BITS = 24;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int DEPTH = BLOCKS * WIN_ROWS;
    localparam int COMMIT_VECTORS = DEPTH;
    localparam integer signed S24_MAX = 8388607;
    localparam integer signed S24_MIN = -8388608;

    logic clk_core, rst_core;
    logic window_start_valid, update_valid, window_end_valid;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [VECTOR_BITS-1:0] update_delta;
    logic commit_ready;

    logic d_window_start_ready, d_window_start_accept;
    logic d_update_ready, d_update_accept;
    logic d_window_end_ready, d_window_end_accept;
    logic d_commit_valid;
    logic [2:0] d_commit_block;
    logic [8:0] d_commit_row;
    logic [VECTOR_BITS-1:0] d_commit_data;
    logic d_commit_last, d_window_done;
    logic d_lane_mem_rd_en;
    logic [11:0] d_lane_mem_rd_addr;
    logic [ACC_BITS-1:0] d_lane_mem_rd_data [0:LANES-1];
    logic d_lane_mem_wr_en;
    logic [11:0] d_lane_mem_wr_addr;
    logic [ACC_BITS-1:0] d_lane_mem_wr_data [0:LANES-1];
    logic d_protocol_error, d_window_active, d_busy;

    logic r_window_start_ready, r_window_start_accept;
    logic r_update_ready, r_update_accept;
    logic r_window_end_ready, r_window_end_accept;
    logic r_commit_valid;
    logic [2:0] r_commit_block;
    logic [8:0] r_commit_row;
    logic [VECTOR_BITS-1:0] r_commit_data;
    logic r_commit_last, r_window_done;
    logic [BLOCKS-1:0] r_mem_rd_en;
    logic [8:0] r_mem_rd_addr [0:BLOCKS-1];
    logic [VECTOR_BITS-1:0] r_mem_rd_data [0:BLOCKS-1];
    logic [BLOCKS-1:0] r_mem_wr_en;
    logic [8:0] r_mem_wr_addr [0:BLOCKS-1];
    logic [VECTOR_BITS-1:0] r_mem_wr_data [0:BLOCKS-1];
    logic r_protocol_error, r_window_active, r_busy;

    logic [ACC_BITS-1:0] lane_memory [0:LANES-1][0:DEPTH-1];
    logic [VECTOR_BITS-1:0] reference_memory [0:BLOCKS-1]
                                                   [0:WIN_ROWS-1];
    integer signed software_value [0:DEPTH-1][0:LANES-1];
    logic software_valid [0:DEPTH-1];
    logic commit_read_seen [0:DEPTH-1];

    logic allow_fault;
    logic automatic_commit_ready;
    integer cycle_count;
    integer current_commit_index;
    integer current_commit_reads;
    integer windows_completed;
    integer positive_updates;
    integer positive_writes;
    integer positive_ii1_pairs;
    integer positive_overlap_cycles;
    integer positive_commit_stalls;
    integer positive_commit_vectors;
    integer positive_lane_checks;
    integer lazy_stale_zero_checks;
    integer exact_flat_read_checks;
    integer exact_flat_write_checks;
    integer lane_write_checks;
    integer same_address_attacks;
    integer range_attacks;
    integer overflow_attacks;
    integer collision_attacks;
    logic prior_update_accept;

    m112_w384_lane_sliced_accumulator_adapter dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(d_window_start_ready),
        .window_start_accept(d_window_start_accept),
        .update_valid(update_valid), .update_ready(d_update_ready),
        .update_block(update_block), .update_row(update_row),
        .update_delta(update_delta), .update_accept(d_update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(d_window_end_ready),
        .window_end_accept(d_window_end_accept),
        .commit_valid(d_commit_valid), .commit_ready(commit_ready),
        .commit_block(d_commit_block), .commit_row(d_commit_row),
        .commit_data(d_commit_data), .commit_last(d_commit_last),
        .window_done(d_window_done),
        .lane_mem_rd_en(d_lane_mem_rd_en),
        .lane_mem_rd_addr(d_lane_mem_rd_addr),
        .lane_mem_rd_data(d_lane_mem_rd_data),
        .lane_mem_wr_en(d_lane_mem_wr_en),
        .lane_mem_wr_addr(d_lane_mem_wr_addr),
        .lane_mem_wr_data(d_lane_mem_wr_data),
        .protocol_error(d_protocol_error),
        .window_active(d_window_active), .busy(d_busy)
    );

    // A frozen M111 instance is used only as an independent command/data
    // reference for the wrapper reshape.  A separate integer scoreboard below
    // still checks signed24 results, lazy-valid behavior and commit ordering.
    m111_w384_signed24_accumulator_frontend reference_core (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(r_window_start_ready),
        .window_start_accept(r_window_start_accept),
        .update_valid(update_valid), .update_ready(r_update_ready),
        .update_block(update_block), .update_row(update_row),
        .update_delta(update_delta), .update_accept(r_update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(r_window_end_ready),
        .window_end_accept(r_window_end_accept),
        .commit_valid(r_commit_valid), .commit_ready(commit_ready),
        .commit_block(r_commit_block), .commit_row(r_commit_row),
        .commit_data(r_commit_data), .commit_last(r_commit_last),
        .window_done(r_window_done),
        .mem_rd_en(r_mem_rd_en), .mem_rd_addr(r_mem_rd_addr),
        .mem_rd_data(r_mem_rd_data), .mem_wr_en(r_mem_wr_en),
        .mem_wr_addr(r_mem_wr_addr), .mem_wr_data(r_mem_wr_data),
        .protocol_error(r_protocol_error),
        .window_active(r_window_active), .busy(r_busy)
    );

    m112_independent_assertions independent_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(d_window_start_ready),
        .window_start_accept(d_window_start_accept),
        .update_valid(update_valid), .update_ready(d_update_ready),
        .update_accept(d_update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(d_window_end_ready),
        .window_end_accept(d_window_end_accept),
        .commit_valid(d_commit_valid), .commit_ready(commit_ready),
        .commit_block(d_commit_block), .commit_row(d_commit_row),
        .commit_data(d_commit_data), .commit_last(d_commit_last),
        .window_done(d_window_done),
        .lane_mem_rd_en(d_lane_mem_rd_en),
        .lane_mem_rd_addr(d_lane_mem_rd_addr),
        .lane_mem_wr_en(d_lane_mem_wr_en),
        .lane_mem_wr_addr(d_lane_mem_wr_addr),
        .protocol_error(d_protocol_error),
        .window_active(d_window_active), .busy(d_busy)
    );

    always #1.5 clk_core = ~clk_core;

    initial begin
        #3000000;
        $fatal(1, "M112 independent watchdog cycles=%0d commit=%0d fault=%0d",
               cycle_count, current_commit_index, d_protocol_error);
    end

    function automatic integer flatten(input integer block,
                                       input integer row);
        flatten = block * WIN_ROWS + row;
    endfunction

    function automatic logic [VECTOR_BITS-1:0] make_pattern(input integer seed);
        logic [VECTOR_BITS-1:0] payload;
        integer signed value;
        begin
            payload = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                value = ((seed * 97 + lane * 53) % 4001) - 2000;
                payload[lane * ACC_BITS +: ACC_BITS]
                    = value[ACC_BITS-1:0];
            end
            return payload;
        end
    endfunction

    always @(negedge clk_core) begin
        if (rst_core)
            commit_ready = 1'b0;
        else if (automatic_commit_ready)
            commit_ready = ((cycle_count % 5) != 1)
                         && ((cycle_count % 17) != 6)
                         && ((cycle_count % 37) != 15);
    end

    always @(posedge clk_core) begin : memory_and_scoreboard
        integer selected_block;
        integer expected_flat;
        integer signed base_value;
        integer signed delta_value;
        longint signed sum_value;

        if (rst_core) begin
            cycle_count = 0;
            prior_update_accept = 1'b0;
        end else begin
            cycle_count = cycle_count + 1;

            // External behavior of the reshaped wrapper must be cycle-for-cycle
            // identical to the frozen M111 core on identical memories.
            if ({d_window_start_ready, d_window_start_accept,
                 d_update_ready, d_update_accept,
                 d_window_end_ready, d_window_end_accept,
                 d_commit_valid, d_commit_block, d_commit_row,
                 d_commit_data, d_commit_last, d_window_done,
                 d_protocol_error, d_window_active, d_busy}
                !==
                {r_window_start_ready, r_window_start_accept,
                 r_update_ready, r_update_accept,
                 r_window_end_ready, r_window_end_accept,
                 r_commit_valid, r_commit_block, r_commit_row,
                 r_commit_data, r_commit_last, r_window_done,
                 r_protocol_error, r_window_active, r_busy})
                $fatal(1, "M112 wrapper/reference external mismatch cycle=%0d",
                       cycle_count);

            if ($countones(r_mem_rd_en) > 1
                    || $countones(r_mem_wr_en) > 1)
                $fatal(1, "M112 reference core port multiplicity drift");
            if (d_lane_mem_rd_en !== (|r_mem_rd_en)
                    || d_lane_mem_wr_en !== (|r_mem_wr_en))
                $fatal(1, "M112 wrapper enable reshape mismatch");

            if (d_lane_mem_rd_en) begin
                selected_block = -1;
                for (int block = 0; block < BLOCKS; block++)
                    if (r_mem_rd_en[block]) selected_block = block;
                expected_flat = flatten(selected_block,
                                        r_mem_rd_addr[selected_block]);
                if (d_lane_mem_rd_addr !== expected_flat[11:0]
                        || d_lane_mem_rd_addr >= DEPTH)
                    $fatal(1, "M112 exact flattened read mismatch block=%0d row=%0d got=%0d expected=%0d",
                           selected_block, r_mem_rd_addr[selected_block],
                           d_lane_mem_rd_addr, expected_flat);
                exact_flat_read_checks = exact_flat_read_checks + 1;
            end

            if (d_lane_mem_wr_en) begin
                selected_block = -1;
                for (int block = 0; block < BLOCKS; block++)
                    if (r_mem_wr_en[block]) selected_block = block;
                expected_flat = flatten(selected_block,
                                        r_mem_wr_addr[selected_block]);
                if (d_lane_mem_wr_addr !== expected_flat[11:0]
                        || d_lane_mem_wr_addr >= DEPTH)
                    $fatal(1, "M112 exact flattened write mismatch block=%0d row=%0d got=%0d expected=%0d",
                           selected_block, r_mem_wr_addr[selected_block],
                           d_lane_mem_wr_addr, expected_flat);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (d_lane_mem_wr_data[lane]
                            !== r_mem_wr_data[selected_block]
                                          [lane * ACC_BITS +: ACC_BITS])
                        $fatal(1, "M112 lane write slice mismatch flat=%0d lane=%0d",
                               expected_flat, lane);
                    lane_write_checks = lane_write_checks + 1;
                end
                exact_flat_write_checks = exact_flat_write_checks + 1;
                if (!allow_fault)
                    positive_writes = positive_writes + 1;
            end

            if (d_window_start_accept) begin
                for (int address = 0; address < DEPTH; address++) begin
                    software_valid[address] = 1'b0;
                    commit_read_seen[address] = 1'b0;
                end
                current_commit_index = 0;
                current_commit_reads = 0;
            end

            if (d_update_accept && !allow_fault) begin
                expected_flat = flatten(update_block, update_row);
                if (!d_lane_mem_rd_en
                        || d_lane_mem_rd_addr !== expected_flat[11:0])
                    $fatal(1, "M112 accepted update exact read mismatch flat=%0d",
                           expected_flat);
                for (int lane = 0; lane < LANES; lane++) begin
                    delta_value = $signed(
                        update_delta[lane * ACC_BITS +: ACC_BITS]);
                    base_value = software_valid[expected_flat]
                               ? software_value[expected_flat][lane] : 0;
                    sum_value = base_value + delta_value;
                    if (sum_value > S24_MAX || sum_value < S24_MIN)
                        $fatal(1, "M112 unexpected positive overflow flat=%0d lane=%0d",
                               expected_flat, lane);
                    software_value[expected_flat][lane] = sum_value;
                end
                software_valid[expected_flat] = 1'b1;
                positive_updates = positive_updates + 1;
                if (prior_update_accept)
                    positive_ii1_pairs = positive_ii1_pairs + 1;
            end
            prior_update_accept = d_update_accept && !allow_fault;

            if (!allow_fault && d_lane_mem_rd_en
                    && !d_window_active && d_busy) begin
                expected_flat = d_lane_mem_rd_addr;
                if (!software_valid[expected_flat]
                        || commit_read_seen[expected_flat])
                    $fatal(1, "M112 commit read conservation failure flat=%0d valid=%0d seen=%0d",
                           expected_flat, software_valid[expected_flat],
                           commit_read_seen[expected_flat]);
                commit_read_seen[expected_flat] = 1'b1;
                current_commit_reads = current_commit_reads + 1;
            end

            if (!allow_fault && d_lane_mem_rd_en && d_lane_mem_wr_en)
                positive_overlap_cycles = positive_overlap_cycles + 1;
            if (!allow_fault && d_commit_valid && !commit_ready)
                positive_commit_stalls = positive_commit_stalls + 1;

            if (!allow_fault && d_commit_valid && commit_ready) begin
                integer expected_block;
                integer expected_row;
                integer signed expected_value;
                expected_block = current_commit_index / WIN_ROWS;
                expected_row = current_commit_index % WIN_ROWS;
                expected_flat = current_commit_index;
                if (d_commit_block !== expected_block[2:0]
                        || d_commit_row !== expected_row[8:0]
                        || d_commit_last
                           !== (current_commit_index == COMMIT_VECTORS - 1))
                    $fatal(1, "M112 commit order/last mismatch index=%0d",
                           current_commit_index);
                for (int lane = 0; lane < LANES; lane++) begin
                    expected_value = software_valid[expected_flat]
                                   ? software_value[expected_flat][lane] : 0;
                    if ($signed(d_commit_data[
                            lane * ACC_BITS +: ACC_BITS]) !== expected_value)
                        $fatal(1, "M112 signed result mismatch flat=%0d lane=%0d got=%0d expected=%0d",
                               expected_flat, lane,
                               $signed(d_commit_data[
                                   lane * ACC_BITS +: ACC_BITS]),
                               expected_value);
                    positive_lane_checks = positive_lane_checks + 1;
                end
                if (windows_completed == 1 && expected_flat == 3071) begin
                    if (d_commit_data !== '0
                            || lane_memory[0][3071] == '0)
                        $fatal(1, "M112 lazy valid leaked or stale-memory witness missing");
                    lazy_stale_zero_checks = lazy_stale_zero_checks + 1;
                end
                current_commit_index = current_commit_index + 1;
                positive_commit_vectors = positive_commit_vectors + 1;
            end

            if (d_window_done && !allow_fault) begin
                integer expected_valid_rows;
                expected_valid_rows = 0;
                if (current_commit_index != COMMIT_VECTORS)
                    $fatal(1, "M112 incomplete commit count=%0d",
                           current_commit_index);
                for (int address = 0; address < DEPTH; address++) begin
                    if (software_valid[address]) begin
                        expected_valid_rows = expected_valid_rows + 1;
                        if (!commit_read_seen[address])
                            $fatal(1, "M112 valid commit row not read flat=%0d",
                                   address);
                    end else if (commit_read_seen[address]) begin
                        $fatal(1, "M112 invalid commit row read flat=%0d",
                               address);
                    end
                end
                if (current_commit_reads != expected_valid_rows)
                    $fatal(1, "M112 commit read count mismatch got=%0d expected=%0d",
                           current_commit_reads, expected_valid_rows);
                windows_completed = windows_completed + 1;
            end

            if (!allow_fault && d_protocol_error)
                $fatal(1, "M112 unexpected positive protocol error cycle=%0d",
                       cycle_count);
        end

        for (int lane = 0; lane < LANES; lane++) begin
            if (d_lane_mem_rd_en)
                d_lane_mem_rd_data[lane]
                    <= lane_memory[lane][d_lane_mem_rd_addr];
            if (d_lane_mem_wr_en)
                lane_memory[lane][d_lane_mem_wr_addr]
                    <= d_lane_mem_wr_data[lane];
        end
        for (int block = 0; block < BLOCKS; block++) begin
            if (r_mem_rd_en[block])
                r_mem_rd_data[block]
                    <= reference_memory[block][r_mem_rd_addr[block]];
            if (r_mem_wr_en[block])
                reference_memory[block][r_mem_wr_addr[block]]
                    <= r_mem_wr_data[block];
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
            update_block = 0;
            update_row = 0;
            update_delta = 0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
            if (d_protocol_error || r_protocol_error)
                $fatal(1, "M112 reset recovery failed");
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            window_start_valid = 1'b1;
            @(posedge clk_core);
            if (!d_window_start_accept || !r_window_start_accept)
                $fatal(1, "M112 start not accepted");
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic drive_flat_update(
        input integer address,
        input logic [VECTOR_BITS-1:0] payload
    );
        integer block;
        integer row;
        begin
            block = address / WIN_ROWS;
            row = address % WIN_ROWS;
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = block[2:0];
            update_row = row[8:0];
            update_delta = payload;
            #0.1;
            if (!d_update_ready || !d_lane_mem_rd_en
                    || d_lane_mem_rd_addr !== address[11:0])
                $fatal(1, "M112 legal flattened update not ready address=%0d",
                       address);
            @(posedge clk_core);
            if (!d_update_accept || !r_update_accept)
                $fatal(1, "M112 legal update not accepted address=%0d",
                       address);
        end
    endtask

    task automatic stop_updates_and_drain;
        begin
            @(negedge clk_core);
            update_valid = 1'b0;
            while (!d_window_end_ready || !r_window_end_ready)
                @(negedge clk_core);
        end
    endtask

    task automatic end_window;
        begin
            window_end_valid = 1'b1;
            @(posedge clk_core);
            if (!d_window_end_accept || !r_window_end_accept)
                $fatal(1, "M112 end not accepted");
            @(negedge clk_core);
            window_end_valid = 1'b0;
        end
    endtask

    task automatic wait_done;
        integer start_cycle;
        begin
            start_cycle = cycle_count;
            while (!d_window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 10000)
                    $fatal(1, "M112 commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic prove_sticky;
        begin
            @(negedge clk_core);
            window_start_valid = 1'b0;
            update_valid = 1'b0;
            window_end_valid = 1'b0;
            repeat (3) begin
                @(posedge clk_core); #0.1;
                if (!d_protocol_error || !r_protocol_error
                        || d_window_start_ready || d_update_ready
                        || d_window_end_ready || d_commit_valid)
                    $fatal(1, "M112 fault not sticky/quarantined");
            end
        end
    endtask

    task automatic run_same_address_attack;
        logic [VECTOR_BITS-1:0] first_payload;
        begin
            allow_fault = 1'b1;
            reset_dut();
            start_window();
            first_payload = make_pattern(9001);
            drive_flat_update(3071, first_payload);
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = 7;
            update_row = 383;
            update_delta = make_pattern(9002);
            #0.1;
            if (d_update_ready || d_update_accept || !d_protocol_error
                    || !d_lane_mem_wr_en || d_lane_mem_wr_addr != 3071)
                $fatal(1, "M112 same-address preserve/fail-close failure");
            for (int lane = 0; lane < LANES; lane++)
                if (d_lane_mem_wr_data[lane]
                        !== first_payload[lane * ACC_BITS +: ACC_BITS])
                    $fatal(1, "M112 same-address older lane write lost lane=%0d",
                           lane);
            @(posedge clk_core);
            same_address_attacks = same_address_attacks + 1;
            prove_sticky();
        end
    endtask

    task automatic run_range_attack;
        begin
            allow_fault = 1'b1;
            reset_dut();
            start_window();
            @(negedge clk_core);
            update_valid = 1'b1;
            update_block = 0;
            update_row = 384;
            update_delta = make_pattern(9100);
            #0.1;
            if (d_update_ready || d_update_accept || !d_protocol_error
                    || d_lane_mem_rd_en || d_lane_mem_wr_en)
                $fatal(1, "M112 row384 range attack did not fail closed");
            @(posedge clk_core);
            range_attacks = range_attacks + 1;
            prove_sticky();
        end
    endtask

    task automatic run_overflow_attack(input logic positive);
        logic [VECTOR_BITS-1:0] extreme;
        logic [VECTOR_BITS-1:0] step;
        integer target;
        begin
            allow_fault = 1'b1;
            reset_dut();
            start_window();
            extreme = '0;
            step = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                extreme[lane * ACC_BITS +: ACC_BITS]
                    = positive ? 24'h7fffff : 24'h800000;
                step[lane * ACC_BITS +: ACC_BITS]
                    = positive ? 24'sd1 : -24'sd1;
            end
            target = positive ? 383 : 2688;
            drive_flat_update(target, extreme);
            drive_flat_update(1536, '0);
            drive_flat_update(target, step);
            @(negedge clk_core);
            update_valid = 1'b0;
            #0.1;
            if (!d_protocol_error || !r_protocol_error
                    || d_lane_mem_wr_en || (|r_mem_wr_en))
                $fatal(1, "M112 overflow suppression failure positive=%0d",
                       positive);
            @(posedge clk_core);
            overflow_attacks = overflow_attacks + 1;
            prove_sticky();
        end
    endtask

    task automatic run_collision_attack;
        begin
            allow_fault = 1'b1;
            reset_dut();
            @(negedge clk_core);
            window_start_valid = 1'b1;
            update_valid = 1'b1;
            update_block = 0;
            update_row = 0;
            update_delta = 0;
            #0.1;
            if (!d_protocol_error || d_window_start_accept
                    || d_update_accept || d_lane_mem_rd_en)
                $fatal(1, "M112 start/update collision did not fail closed");
            @(posedge clk_core);
            collision_attacks = collision_attacks + 1;
            prove_sticky();
        end
    endtask

    initial begin : test
        logic [VECTOR_BITS-1:0] initial_word;
        integer address;
        integer block;
        integer row;
        integer raw;

        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start_valid = 1'b0;
        update_valid = 1'b0;
        update_block = 0;
        update_row = 0;
        update_delta = 0;
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
        lazy_stale_zero_checks = 0;
        exact_flat_read_checks = 0;
        exact_flat_write_checks = 0;
        lane_write_checks = 0;
        same_address_attacks = 0;
        range_attacks = 0;
        overflow_attacks = 0;
        collision_attacks = 0;
        prior_update_accept = 1'b0;

        // Initialize both physical organizations identically with nonzero
        // stale data so lazy-valid zeroing is observable, not vacuous.
        for (address = 0; address < DEPTH; address++) begin
            block = address / WIN_ROWS;
            row = address % WIN_ROWS;
            initial_word = '0;
            software_valid[address] = 1'b0;
            commit_read_seen[address] = 1'b0;
            for (int lane = 0; lane < LANES; lane++) begin
                raw = ((address * 17 + lane * 131 + 1) % 8388607) + 1;
                lane_memory[lane][address] = raw[ACC_BITS-1:0];
                initial_word[lane * ACC_BITS +: ACC_BITS]
                    = raw[ACC_BITS-1:0];
                software_value[address][lane] = 0;
                d_lane_mem_rd_data[lane] = 0;
            end
            reference_memory[block][row] = initial_word;
        end
        for (block = 0; block < BLOCKS; block++)
            r_mem_rd_data[block] = 0;

        reset_dut();
        start_window();
        // Exact upper/lower flatten endpoints plus a safe one-gap RMW.
        drive_flat_update(3071, make_pattern(1));
        drive_flat_update(0, make_pattern(2));
        drive_flat_update(3071, make_pattern(3));
        // A reverse/permuted sweep exercises every block and lane slice while
        // retaining changed-address II=1.
        for (int index = 0; index < 253; index++) begin
            address = 3071 - (((index + 1) * 127) % DEPTH);
            drive_flat_update(address, make_pattern(100 + index));
        end
        stop_updates_and_drain();
        end_window();
        wait_done();

        // Empty second window leaves nonzero data in all physical memories but
        // clears only 3072 valid bits; every committed vector must be zero and
        // no lane macro read may be issued.
        start_window();
        stop_updates_and_drain();
        end_window();
        wait_done();

        if (windows_completed != 2 || positive_updates != 256
                || positive_writes != 256
                || positive_ii1_pairs != 255
                || positive_overlap_cycles != 255
                || positive_commit_vectors != 2 * COMMIT_VECTORS
                || positive_lane_checks != 2 * COMMIT_VECTORS * LANES
                || positive_commit_stalls == 0
                || lazy_stale_zero_checks != 1)
            $fatal(1, "M112 positive conservation windows=%0d updates=%0d writes=%0d ii1=%0d overlap=%0d commits=%0d lanes=%0d stalls=%0d lazy=%0d",
                   windows_completed, positive_updates, positive_writes,
                   positive_ii1_pairs, positive_overlap_cycles,
                   positive_commit_vectors, positive_lane_checks,
                   positive_commit_stalls, lazy_stale_zero_checks);

        run_same_address_attack();
        run_range_attack();
        run_overflow_attack(1'b1);
        run_overflow_attack(1'b0);
        run_collision_attack();

        if (same_address_attacks != 1 || range_attacks != 1
                || overflow_attacks != 2 || collision_attacks != 1)
            $fatal(1, "M112 attack coverage mismatch");

        $display("PASS M112 INDEPENDENT HAMMER commercial_vcs=true wrapper_cycle_equivalent_m111=true windows=2 reverse_updates=256 writes=256 ii1_pairs=255 read_write_overlap=255 commits=6144 lane_result_checks=589824 exact_flat_read_checks=%0d exact_flat_write_checks=%0d lane_write_slice_checks=%0d flat_zero=true flat_3071=true lazy_stale_zero=1 commit_stalls=%0d same_address_preserve=1 row384_range=1 positive_overflow=1 negative_overflow=1 collision=1 lane_macros=96 macro_depth=3072 macro_width=24 logical_accumulator_bytes=884736 valid_bits=3072 behavioral_sync_1r1w=true foundry_macro=false m109_r2_2p535_is_projection=true scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 exact_flat_read_checks, exact_flat_write_checks,
                 lane_write_checks, positive_commit_stalls);
        $finish;
    end
endmodule

`default_nettype wire
