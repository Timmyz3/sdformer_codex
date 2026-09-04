`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dynamic_bn_banked_moment_scheduler;
    localparam int IN_W = 32;
    localparam int TAG_W = 48;
    localparam int MAX_POP = 9;
    localparam int MAX_TILES = 4;
    localparam int COUNT_W = $clog2(MAX_POP + 1);
    localparam int TILE_W = $clog2(MAX_TILES);
    localparam int ACTIVE_W = $clog2(MAX_TILES + 1);
    localparam int GROWTH_W = $clog2(MAX_POP);
    localparam int SUM_W = IN_W + GROWTH_W;
    localparam int SQUARE_W = (2*IN_W)-1;
    localparam int SUMSQ_W = SQUARE_W + GROWTH_W;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic operator_start_valid = 1'b0;
    logic operator_start_ready;
    logic [COUNT_W-1:0] operator_reduction_population = '0;
    logic [ACTIVE_W-1:0] operator_active_lane_tiles = '0;
    logic [TAG_W-1:0] operator_start_tag = '0;
    logic operator_start_legal;
    logic operator_active;
    logic [COUNT_W-1:0] active_reduction_population;
    logic [ACTIVE_W-1:0] active_lane_tiles;
    logic [TAG_W-1:0] active_tag;
    logic packet_valid = 1'b0;
    logic packet_ready;
    logic [TILE_W-1:0] packet_lane_tile_id = '0;
    logic packet_first = 1'b0;
    logic packet_last = 1'b0;
    logic [(96*IN_W)-1:0] packet_values = '0;
    logic packet_legal;
    logic [COUNT_W-1:0] packet_accepted_count;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [TAG_W-1:0] result_tag;
    logic [TILE_W-1:0] result_lane_tile_id;
    logic [2:0] result_slice_id;
    logic [COUNT_W-1:0] result_count;
    logic [(16*SUM_W)-1:0] result_sum;
    logic [(16*SUMSQ_W)-1:0] result_sumsq;
    logic operator_done;
    logic [TAG_W-1:0] operator_done_tag;
    logic protocol_error;
    logic [2:0] fifo_level;
    logic [2:0] serializer_slice;

    logic signed [SUM_W-1:0] reference_sum [0:MAX_TILES-1][0:95];
    logic [SUMSQ_W-1:0] reference_sumsq [0:MAX_TILES-1][0:95];
    bit result_seen [0:MAX_TILES-1][0:5];
    logic signed [IN_W-1:0] stimulus [0:95];

    integer legal_packets = 0;
    integer illegal_packets = 0;
    integer results_checked = 0;
    integer done_count = 0;
    integer fifo_full_cycles = 0;
    integer output_stalls = 0;
    integer resets_midflight = 0;
    integer full_swaps = 0;
    integer directed_full_swaps = 0;
    integer directed_illegal_full_cancels = 0;
    integer directed_pending_result_cancels = 0;
    integer expected_results = 0;
    integer active_expected_pop = 0;
    integer active_expected_tiles = 0;
    logic [TAG_W-1:0] active_expected_tag = '0;
    integer force_stall_budget = 0;
    bit directed_full_swap_window = 1'b0;
    bit directed_illegal_full_window = 1'b0;
    bit directed_pending_result_window = 1'b0;
    bit manual_result_ready_mode = 1'b0;

    always #5 clk_core = ~clk_core;

    qfit_dynamic_bn_banked_moment_scheduler #(
        .IN_W(IN_W), .TAG_W(TAG_W),
        .MAX_REDUCTION_POPULATION(MAX_POP), .MAX_LANE_TILES(MAX_TILES)
    ) dut (.*);

    function automatic logic [SQUARE_W-1:0] reference_square(
        input logic signed [IN_W-1:0] value
    );
        logic signed [(2*IN_W)-1:0] wide_value;
        logic signed [(4*IN_W)-1:0] wide_product;
        begin
            wide_value = {{IN_W{value[IN_W-1]}}, value};
            wide_product = wide_value * wide_value;
            if (|wide_product[(4*IN_W)-1:SQUARE_W])
                $fatal(1, "M21 reference square exceeded exact width");
            reference_square = wide_product[SQUARE_W-1:0];
        end
    endfunction

    task automatic clear_scoreboard;
        for (int tile = 0; tile < MAX_TILES; tile++) begin
            for (int lane = 0; lane < 96; lane++) begin
                reference_sum[tile][lane] = '0;
                reference_sumsq[tile][lane] = '0;
            end
            for (int slice = 0; slice < 6; slice++)
                result_seen[tile][slice] = 1'b0;
        end
    endtask

    task automatic fill_packet(input int tile, input int beat, input int mode);
        for (int lane = 0; lane < 96; lane++) begin
            case (mode)
                1: stimulus[lane] = (lane[0])
                    ? {1'b1, {(IN_W-1){1'b0}}}
                    : {1'b0, {(IN_W-1){1'b1}}};
                2: stimulus[lane] = $signed((tile+1)*1000 + beat*97 + lane - 48);
                default: begin
                    stimulus[lane] = $urandom;
                    if (((tile*11 + beat*7 + lane) % 37) == 0)
                        stimulus[lane] = {1'b1, {(IN_W-1){1'b0}}};
                    else if (((tile*13 + beat*5 + lane) % 41) == 0)
                        stimulus[lane] = {1'b0, {(IN_W-1){1'b1}}};
                end
            endcase
            packet_values[(lane*IN_W) +: IN_W] = stimulus[lane];
        end
    endtask

    task automatic update_reference(input int tile, input bit first_value);
        logic signed [SUM_W-1:0] value_extended;
        logic [SUMSQ_W-1:0] square_extended;
        for (int lane = 0; lane < 96; lane++) begin
            value_extended = {{(SUM_W-IN_W){stimulus[lane][IN_W-1]}}, stimulus[lane]};
            square_extended = {{(SUMSQ_W-SQUARE_W){1'b0}},
                               reference_square(stimulus[lane])};
            if (first_value) begin
                reference_sum[tile][lane] = value_extended;
                reference_sumsq[tile][lane] = square_extended;
            end else begin
                reference_sum[tile][lane] = reference_sum[tile][lane] + value_extended;
                reference_sumsq[tile][lane] = reference_sumsq[tile][lane] + square_extended;
            end
        end
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        operator_start_valid = 1'b0;
        packet_valid = 1'b0;
        result_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        #1;
        if (!operator_start_ready || operator_active || protocol_error
            || result_valid || fifo_level != 0)
            $fatal(1, "M21 reset did not restore the idle contract");
    endtask

    task automatic start_operator(
        input int population, input int tiles, input logic [TAG_W-1:0] tag
    );
        @(negedge clk_core);
        operator_start_valid = 1'b1;
        operator_reduction_population = population[COUNT_W-1:0];
        operator_active_lane_tiles = tiles[ACTIVE_W-1:0];
        operator_start_tag = tag;
        #1;
        if (!operator_start_ready || !operator_start_legal)
            $fatal(1, "legal M21 operator start rejected pop=%0d tiles=%0d",
                   population, tiles);
        @(posedge clk_core);
        @(negedge clk_core);
        operator_start_valid = 1'b0;
        active_expected_pop = population;
        active_expected_tiles = tiles;
        active_expected_tag = tag;
        expected_results = tiles * 6;
        force_stall_budget = 4;
    endtask

    task automatic send_packet(
        input int tile, input int beat, input int population, input int mode
    );
        bit accepted;
        fill_packet(tile, beat, mode);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = tile[TILE_W-1:0];
        packet_first = beat == 0;
        packet_last = beat == population-1;
        accepted = 1'b0;
        while (!accepted) begin
            #1;
            if (packet_ready) begin
                if (!packet_legal)
                    $fatal(1, "legal M21 packet rejected tile=%0d beat=%0d", tile, beat);
                @(posedge clk_core);
                update_reference(tile, beat == 0);
                legal_packets = legal_packets + 1;
                accepted = 1'b1;
            end else begin
                @(negedge clk_core);
            end
        end
        @(negedge clk_core);
        packet_valid = 1'b0;
        packet_first = 1'b0;
        packet_last = 1'b0;
    endtask

    task automatic run_operator(
        input int population, input int tiles,
        input logic [TAG_W-1:0] tag, input int mode
    );
        int old_done;
        clear_scoreboard();
        old_done = done_count;
        start_operator(population, tiles, tag);
        for (int beat = 0; beat < population; beat++) begin
            for (int order = 0; order < tiles; order++) begin
                int tile;
                tile = (beat*2 + order) % tiles;
                send_packet(tile, beat, population, mode);
            end
        end
        wait (done_count == old_done + 1);
        @(negedge clk_core);
        if (operator_active || result_valid || fifo_level != 0 || protocol_error)
            $fatal(1, "M21 operator did not drain cleanly");
        for (int tile = 0; tile < tiles; tile++)
            for (int slice = 0; slice < 6; slice++)
                if (!result_seen[tile][slice])
                    $fatal(1, "missing M21 result tile=%0d slice=%0d", tile, slice);
    endtask

    task automatic run_directed_full_swap_operator;
        int old_done;
        clear_scoreboard();
        old_done = done_count;
        start_operator(5, 1, 48'h21f001);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = '0;
        for (int beat = 0; beat < 5; beat++) begin
            fill_packet(0, beat, 2);
            packet_first = beat == 0;
            packet_last = beat == 4;
            #1;
            while (!packet_ready) begin
                @(negedge clk_core);
                #1;
            end
            if (!packet_legal)
                $fatal(1, "directed full-swap legal packet rejected beat=%0d", beat);
            if (beat == 4) begin
                if (fifo_level != 4 || serializer_slice != 5
                    || !dut.dequeue_candidate)
                    $fatal(1, "directed M21 full-swap candidate was not established");
                directed_full_swap_window = 1'b1;
            end
            @(posedge clk_core);
            update_reference(0, beat == 0);
            legal_packets = legal_packets + 1;
            if (beat != 4)
                @(negedge clk_core);
        end
        @(negedge clk_core);
        packet_valid = 1'b0;
        packet_first = 1'b0;
        packet_last = 1'b0;
        directed_full_swap_window = 1'b0;
        wait (done_count == old_done + 1);
        @(negedge clk_core);
        if (operator_active || result_valid || fifo_level != 0 || protocol_error)
            $fatal(1, "directed full-swap operator did not drain cleanly");
        for (int slice = 0; slice < 6; slice++)
            if (!result_seen[0][slice])
                $fatal(1, "directed full-swap result missing slice=%0d", slice);
    endtask

    task automatic expect_illegal_full_swap_cancellation;
        logic [1:0] saved_read_ptr;
        logic [1:0] saved_write_ptr;
        apply_reset();
        clear_scoreboard();
        start_operator(6, 1, 48'h21f002);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = '0;
        for (int beat = 0; beat < 4; beat++) begin
            fill_packet(0, beat, 2);
            packet_first = beat == 0;
            packet_last = 1'b0;
            #1;
            if (!packet_ready || !packet_legal)
                $fatal(1, "directed illegal-full setup rejected beat=%0d", beat);
            @(posedge clk_core);
            update_reference(0, beat == 0);
            legal_packets = legal_packets + 1;
            @(negedge clk_core);
        end

        fill_packet(0, 4, 2);
        packet_first = 1'b1; // repeated first is illegal at accepted_count=4
        packet_last = 1'b0;
        #1;
        while (!packet_ready) begin
            @(negedge clk_core);
            #1;
        end
        if (fifo_level != 4 || serializer_slice != 5
            || !dut.dequeue_candidate || packet_legal)
            $fatal(1, "illegal full-swap cancellation condition not established");
        saved_read_ptr = dut.fifo_read_ptr_q;
        saved_write_ptr = dut.fifo_write_ptr_q;
        directed_illegal_full_window = 1'b1;
        @(posedge clk_core);
        #1;
        if (!protocol_error || fifo_level != 4 || serializer_slice != 5
            || dut.fifo_read_ptr_q != saved_read_ptr
            || dut.fifo_write_ptr_q != saved_write_ptr)
            $fatal(1, "illegal full offer mutated FIFO/serializer state");
        illegal_packets = illegal_packets + 1;
        @(negedge clk_core);
        directed_illegal_full_window = 1'b0;
        packet_valid = 1'b0;
    endtask

    task automatic expect_pending_result_illegal_cancellation;
        logic [4:0] saved_results_retired;
        logic [1:0] saved_read_ptr;
        logic [1:0] saved_write_ptr;
        logic [2:0] saved_slice;
        logic [COUNT_W-1:0] saved_result_count;
        integer saved_results_checked;

        manual_result_ready_mode = 1'b1;
        apply_reset();
        clear_scoreboard();
        start_operator(1, 2, 48'h21f003);
        result_ready = 1'b0;
        send_packet(0, 0, 1, 2);
        while (!dut.result_valid_q)
            @(negedge clk_core);
        #1;
        if (!result_valid || result_slice_id != 0 || result_count != 1)
            $fatal(1, "pending M21 result was not established");

        saved_results_retired = dut.results_retired_q;
        saved_read_ptr = dut.fifo_read_ptr_q;
        saved_write_ptr = dut.fifo_write_ptr_q;
        saved_slice = serializer_slice;
        saved_result_count = result_count;
        saved_results_checked = results_checked;
        result_ready = 1'b1;
        packet_valid = 1'b1;
        packet_lane_tile_id = '0;
        packet_first = 1'b1;
        packet_last = 1'b1;
        fill_packet(0, 1, 2);
        #1;
        if (!packet_ready || packet_legal || !dut.illegal_packet_fire
            || result_valid || dut.result_fire)
            $fatal(1, "illegal+pending-result collision was not suppressed combinationally");
        directed_pending_result_window = 1'b1;
        @(posedge clk_core);
        #1;
        if (!protocol_error || result_valid || dut.result_fire
            || !dut.result_valid_q
            || dut.results_retired_q != saved_results_retired
            || dut.fifo_read_ptr_q != saved_read_ptr
            || dut.fifo_write_ptr_q != saved_write_ptr
            || serializer_slice != saved_slice
            || result_count != saved_result_count
            || results_checked != saved_results_checked)
            $fatal(1, "illegal pending result mutated retirement/FIFO/result state");
        illegal_packets = illegal_packets + 1;
        @(negedge clk_core);
        packet_valid = 1'b0;
        directed_pending_result_window = 1'b0;
        repeat (3) begin
            @(posedge clk_core);
            #1;
            if (result_valid || dut.result_fire
                || dut.results_retired_q != saved_results_retired
                || results_checked != saved_results_checked)
                $fatal(1, "sticky M21 protocol_error leaked a result");
        end
        apply_reset();
        #1;
        if (!operator_start_ready || result_valid || protocol_error)
            $fatal(1, "reset did not recover pending-result cancellation");
        manual_result_ready_mode = 1'b0;
    endtask

    task automatic expect_illegal_start;
        apply_reset();
        @(negedge clk_core);
        operator_start_valid = 1'b1;
        operator_reduction_population = '0;
        operator_active_lane_tiles = 1;
        operator_start_tag = 48'hbad001;
        #1;
        if (!operator_start_ready || operator_start_legal)
            $fatal(1, "zero-population M21 start was not rejected");
        @(posedge clk_core);
        #1;
        if (!protocol_error || operator_start_ready || packet_ready)
            $fatal(1, "illegal M21 start did not fail closed");
        @(negedge clk_core);
        operator_start_valid = 1'b0;
        @(posedge clk_core);
        #1;
        if (!protocol_error)
            $fatal(1, "M21 start protocol_error was not sticky");
    endtask

    task automatic expect_illegal_packet(
        input bit first_value, input bit last_value, input int tile
    );
        apply_reset();
        clear_scoreboard();
        start_operator(3, 2, 48'hbad100 + illegal_packets);
        fill_packet(tile, 0, 2);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = tile[TILE_W-1:0];
        packet_first = first_value;
        packet_last = last_value;
        #1;
        if (!packet_ready || packet_legal)
            $fatal(1, "illegal M21 packet was not classified at enqueue");
        @(posedge clk_core);
        #1;
        if (!protocol_error || packet_ready)
            $fatal(1, "illegal M21 packet did not fail closed");
        illegal_packets = illegal_packets + 1;
        @(negedge clk_core);
        packet_valid = 1'b0;
        @(posedge clk_core);
        #1;
        if (!protocol_error)
            $fatal(1, "M21 packet protocol_error was not sticky");
    endtask

    task automatic expect_active_illegal_packet(
        input bit first_value, input bit last_value
    );
        apply_reset();
        clear_scoreboard();
        start_operator(2, 1, 48'hbad200 + illegal_packets);
        send_packet(0, 0, 2, 2);
        fill_packet(0, 1, 2);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = '0;
        packet_first = first_value;
        packet_last = last_value;
        while (!packet_ready)
            @(negedge clk_core);
        #1;
        if (packet_legal)
            $fatal(1, "active illegal M21 packet was not rejected");
        @(posedge clk_core);
        #1;
        if (!protocol_error || packet_ready)
            $fatal(1, "active illegal M21 packet did not fail closed");
        illegal_packets = illegal_packets + 1;
        @(negedge clk_core);
        packet_valid = 1'b0;
    endtask

    always @(negedge clk_core) begin
        if (!rst_core && !manual_result_ready_mode) begin
            if (result_valid) begin
                if (force_stall_budget > 0) begin
                    result_ready = 1'b0;
                    force_stall_budget = force_stall_budget - 1;
                end else begin
                    result_ready = $urandom_range(0, 1);
                end
            end else begin
                result_ready = $urandom_range(0, 1);
            end
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (fifo_level == 4)
                fifo_full_cycles <= fifo_full_cycles + 1;
            if (result_valid && !result_ready)
                output_stalls <= output_stalls + 1;
            if (result_valid && result_ready) begin
                int tile;
                int slice;
                tile = result_lane_tile_id;
                slice = result_slice_id;
                if (result_tag !== active_expected_tag
                    || result_count != active_expected_pop)
                    $fatal(1, "M21 result metadata mismatch");
                if (tile >= active_expected_tiles || slice >= 6
                    || result_seen[tile][slice])
                    $fatal(1, "M21 result identity invalid/duplicate tile=%0d slice=%0d",
                           tile, slice);
                for (int lane = 0; lane < 16; lane++) begin
                    int channel;
                    channel = slice*16 + lane;
                    if ($signed(result_sum[(lane*SUM_W) +: SUM_W])
                        !== reference_sum[tile][channel])
                        $fatal(1, "M21 sum mismatch tile=%0d channel=%0d", tile, channel);
                    if (result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W]
                        !== reference_sumsq[tile][channel])
                        $fatal(1, "M21 sumsq mismatch tile=%0d channel=%0d", tile, channel);
                end
                result_seen[tile][slice] <= 1'b1;
                results_checked <= results_checked + 1;
            end
            if (operator_done) begin
                if (operator_done_tag !== active_expected_tag)
                    $fatal(1, "M21 done tag mismatch");
                done_count <= done_count + 1;
            end
            if (fifo_level == 4 && dut.enqueue_fire && dut.dequeue_fire) begin
                full_swaps <= full_swaps + 1;
                if (directed_full_swap_window)
                    directed_full_swaps <= directed_full_swaps + 1;
            end
            if (directed_illegal_full_window && fifo_level == 4
                && packet_valid && packet_ready && !packet_legal
                && dut.dequeue_candidate && !dut.dequeue_fire
                && !dut.process_slice)
                directed_illegal_full_cancels
                    <= directed_illegal_full_cancels + 1;
            if (directed_pending_result_window
                && dut.result_valid_q && result_ready
                && dut.illegal_packet_fire && !result_valid
                && !dut.result_fire)
                directed_pending_result_cancels
                    <= directed_pending_result_cancels + 1;
        end
    end

    initial begin
        $display("SIMULATOR=Synopsys VCS");
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M21 test requires bound SVA runtime");
`endif
        clear_scoreboard();
        apply_reset();

        run_directed_full_swap_operator();

        // Interleaved tiles, random gaps induced by FIFO backpressure, signed
        // extrema, full declared population, and repeated operators prove
        // first-packet overwrite of state left resident by earlier operators.
        run_operator(5, 3, 48'h210001, 0);
        run_operator(1, 4, 48'h210002, 1);
        run_operator(MAX_POP, 2, 48'h210003, 1);

        // Reset is the explicit cancellation path for queued/partial work.
        clear_scoreboard();
        start_operator(3, 1, 48'h210004);
        send_packet(0, 0, 3, 2);
        repeat (2) @(posedge clk_core);
        resets_midflight = resets_midflight + 1;
        apply_reset();
        run_operator(2, 1, 48'h210005, 2);

        expect_illegal_full_swap_cancellation();
        expect_pending_result_illegal_cancellation();

        expect_illegal_packet(1'b0, 1'b0, 0); // missing first
        expect_illegal_packet(1'b1, 1'b1, 0); // early last
        expect_illegal_packet(1'b1, 1'b0, 2); // tile outside active set
        expect_active_illegal_packet(1'b1, 1'b1); // repeated first
        expect_active_illegal_packet(1'b0, 1'b0); // missing required last
        expect_illegal_start();

        if (legal_packets != 52 || illegal_packets != 7
            || results_checked != 66 || done_count != 5
            || fifo_full_cycles <= 0 || output_stalls <= 0
            || resets_midflight != 1 || full_swaps <= 0
            || directed_full_swaps != 1
            || directed_illegal_full_cancels != 1
            || directed_pending_result_cancels != 1)
            $fatal(1, "M21 coverage drift packets=%0d illegal=%0d results=%0d done=%0d fifo_full=%0d stalls=%0d resets=%0d full_swaps=%0d directed_swaps=%0d illegal_full_cancels=%0d pending_result_cancels=%0d",
                   legal_packets, illegal_packets, results_checked, done_count,
                   fifo_full_cycles, output_stalls, resets_midflight,
                   full_swaps, directed_full_swaps,
                   directed_illegal_full_cancels,
                   directed_pending_result_cancels);
        $display("M21_RESULT legal_packets=%0d illegal_packets=%0d results=%0d done=%0d fifo_full_cycles=%0d result_stalls=%0d resets_midflight=%0d directed_full_swaps=%0d directed_illegal_full_cancels=%0d directed_pending_result_cancels=%0d",
                 legal_packets, illegal_packets, results_checked, done_count,
                 fifo_full_cycles, output_stalls, resets_midflight,
                 directed_full_swaps, directed_illegal_full_cancels,
                 directed_pending_result_cancels);
        $display("PASS: Synopsys VCS M21 banked raw-moment scheduler reference miter");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "M21 simulation timeout");
    end
endmodule

`default_nettype wire
