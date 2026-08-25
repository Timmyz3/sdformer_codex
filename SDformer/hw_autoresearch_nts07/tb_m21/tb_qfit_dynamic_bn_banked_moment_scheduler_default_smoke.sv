`timescale 1ns/1ps
`default_nettype none

// Default-parameter dynamic smoke: this is not elaboration-only.  It executes
// population=1 across all sixteen resident lane tiles, retires 16*6 results,
// and checks every output lane against an independent arithmetic reference.
module tb_qfit_dynamic_bn_banked_moment_scheduler_default_smoke;
    localparam int IN_W = 32;
    localparam int TAG_W = 48;
    localparam int MAX_POP = 4194304;
    localparam int MAX_TILES = 16;
    localparam int COUNT_W = $clog2(MAX_POP + 1);
    localparam int TILE_W = $clog2(MAX_TILES);
    localparam int ACTIVE_W = $clog2(MAX_TILES + 1);
    localparam int SUM_W = IN_W + $clog2(MAX_POP);
    localparam int SUMSQ_W = (2*IN_W)-1 + $clog2(MAX_POP);
    localparam logic [TAG_W-1:0] DYNAMIC_TAG = 48'hdef021;

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

    bit result_seen [0:MAX_TILES-1][0:5];
    integer dynamic_results = 0;
    integer dynamic_done = 0;
    integer packet_backpressure_cycles = 0;
    integer result_stall_cycles = 0;
    integer directed_packet_backpressure_cycles = 0;
    integer ready_cycle = 0;
    bit dynamic_scoreboard_active = 1'b0;
    bit automatic_result_ready = 1'b0;

    always #5 clk_core = ~clk_core;

    date_m21_banked_moment_scheduler_dc_top dut (.*);

    function automatic integer reference_value(
        input integer tile, input integer channel
    );
        reference_value = (tile * 1000) + channel - 48;
    endfunction

    task automatic clear_seen;
        for (int tile = 0; tile < MAX_TILES; tile++)
            for (int slice = 0; slice < 6; slice++)
                result_seen[tile][slice] = 1'b0;
    endtask

    task automatic fill_tile_packet(input integer tile);
        for (int channel = 0; channel < 96; channel++)
            packet_values[(channel*IN_W) +: IN_W]
                = $signed(reference_value(tile, channel));
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        operator_start_valid = 1'b0;
        packet_valid = 1'b0;
        result_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        #1;
        if (!operator_start_ready || operator_active || protocol_error
            || result_valid || fifo_level != 0)
            $fatal(1, "M21 default reset contract failed");
    endtask

    task automatic start_operator(input integer population, input integer tiles);
        @(negedge clk_core);
        operator_start_valid = 1'b1;
        operator_reduction_population = population[COUNT_W-1:0];
        operator_active_lane_tiles = tiles[ACTIVE_W-1:0];
        operator_start_tag = DYNAMIC_TAG;
        #1;
        if (!operator_start_ready || !operator_start_legal)
            $fatal(1, "M21 default legal start rejected pop=%0d tiles=%0d",
                   population, tiles);
        @(posedge clk_core);
        @(negedge clk_core);
        operator_start_valid = 1'b0;
    endtask

    task automatic send_tile_packet(
        input integer tile, input bit first_value, input bit last_value
    );
        bit accepted;
        integer wait_cycles;
        fill_tile_packet(tile);
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_lane_tile_id = tile[TILE_W-1:0];
        packet_first = first_value;
        packet_last = last_value;
        accepted = 1'b0;
        wait_cycles = 0;
        while (!accepted) begin
            #1;
            if (packet_ready) begin
                if (!packet_legal)
                    $fatal(1, "M21 default legal packet rejected tile=%0d", tile);
                @(posedge clk_core);
                accepted = 1'b1;
            end else begin
                packet_backpressure_cycles = packet_backpressure_cycles + 1;
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 4096)
                    $fatal(1, "M21 default packet backpressure timeout tile=%0d", tile);
                @(negedge clk_core);
            end
        end
        @(negedge clk_core);
        packet_valid = 1'b0;
        packet_first = 1'b0;
        packet_last = 1'b0;
    endtask

    task automatic send_directed_full_fifo_prefix;
        integer wait_cycles;

        // Four consecutive packets fill the FIFO while tile0's slice0 result
        // is deliberately held.  The fifth packet is then kept valid for
        // exactly three known full/not-ready cycles before result retirement
        // resumes.  This makes packet backpressure phase-independent.
        automatic_result_ready = 1'b0;
        result_ready = 1'b0;
        @(negedge clk_core);
        packet_valid = 1'b1;
        packet_first = 1'b1;
        packet_last = 1'b1;
        for (int tile = 0; tile < 4; tile++) begin
            fill_tile_packet(tile);
            packet_lane_tile_id = tile[TILE_W-1:0];
            #1;
            if (!packet_ready || !packet_legal)
                $fatal(1, "M21 default directed FIFO fill rejected tile=%0d", tile);
            @(posedge clk_core);
            if (tile != 3)
                @(negedge clk_core);
        end

        @(negedge clk_core);
        fill_tile_packet(4);
        packet_lane_tile_id = 4;
        #1;
        for (int hold = 0; hold < 3; hold++) begin
            if (fifo_level != 4 || packet_ready || !packet_valid
                || !result_valid || result_ready)
                $fatal(1, "M21 default directed packet backpressure not established hold=%0d",
                       hold);
            directed_packet_backpressure_cycles
                = directed_packet_backpressure_cycles + 1;
            packet_backpressure_cycles = packet_backpressure_cycles + 1;
            @(posedge clk_core);
            @(negedge clk_core);
            #1;
        end

        automatic_result_ready = 1'b1;
        result_ready = 1'b1;
        wait_cycles = 0;
        while (!packet_ready) begin
            packet_backpressure_cycles = packet_backpressure_cycles + 1;
            wait_cycles = wait_cycles + 1;
            if (wait_cycles > 4096)
                $fatal(1, "M21 default directed fifth-packet timeout");
            @(negedge clk_core);
            #1;
        end
        if (!packet_legal)
            $fatal(1, "M21 default directed fifth packet became illegal");
        @(posedge clk_core);
        @(negedge clk_core);
        packet_valid = 1'b0;
        packet_first = 1'b0;
        packet_last = 1'b0;
    endtask

    task automatic wait_dynamic_done_bounded;
        integer wait_cycles;
        wait_cycles = 0;
        while (dynamic_done == 0 && wait_cycles < 4096) begin
            @(posedge clk_core);
            wait_cycles = wait_cycles + 1;
        end
        if (dynamic_done != 1)
            $fatal(1, "M21 default dynamic done timeout");
    endtask

    always @(negedge clk_core) begin
        if (!rst_core && automatic_result_ready) begin
            // Deterministic output backpressure exercises registered result
            // retirement while guaranteeing forward progress.
            ready_cycle = ready_cycle + 1;
            result_ready = (ready_cycle % 4) != 0;
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core && result_valid && !result_ready)
            result_stall_cycles <= result_stall_cycles + 1;
        if (!rst_core && result_valid && result_ready && dynamic_scoreboard_active) begin
            int tile;
            int slice;
            tile = result_lane_tile_id;
            slice = result_slice_id;
            if (result_tag != DYNAMIC_TAG || result_count != 1
                || tile >= MAX_TILES || slice >= 6
                || result_seen[tile][slice])
                $fatal(1, "M21 default result metadata invalid tile=%0d slice=%0d",
                       tile, slice);
            for (int lane = 0; lane < 16; lane++) begin
                int value;
                int channel;
                logic [SUMSQ_W-1:0] expected_square;
                channel = slice*16 + lane;
                value = reference_value(tile, channel);
                expected_square = value * value;
                if ($signed(result_sum[(lane*SUM_W) +: SUM_W]) != value)
                    $fatal(1, "M21 default sum mismatch tile=%0d channel=%0d",
                           tile, channel);
                if (result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W] != expected_square)
                    $fatal(1, "M21 default sumsq mismatch tile=%0d channel=%0d",
                           tile, channel);
            end
            result_seen[tile][slice] <= 1'b1;
            dynamic_results <= dynamic_results + 1;
        end
        if (!rst_core && operator_done && dynamic_scoreboard_active) begin
            if (operator_done_tag != DYNAMIC_TAG)
                $fatal(1, "M21 default done tag mismatch");
            dynamic_done <= dynamic_done + 1;
        end
    end

    initial begin
        if (COUNT_W != 23 || ACTIVE_W != 5 || SUM_W != 54 || SUMSQ_W != 85)
            $fatal(1, "M21 default derived-width contract drifted");
        clear_seen();

        // Representable population above MAX must be rejected fail closed.
        apply_reset();
        @(negedge clk_core);
        operator_start_valid = 1'b1;
        operator_reduction_population = MAX_POP + 1;
        operator_active_lane_tiles = 1;
        operator_start_tag = 48'hbad021;
        #1;
        if (!operator_start_ready || operator_start_legal)
            $fatal(1, "M21 default above-MAX start was not rejected");
        @(posedge clk_core);
        #1;
        if (!protocol_error || operator_start_ready)
            $fatal(1, "M21 default above-MAX rejection did not latch");

        // Reset cancels queued partial work and first in the following
        // operator must overwrite the unreset raw-moment banks.
        apply_reset();
        start_operator(2, 1);
        send_tile_packet(0, 1'b1, 1'b0);
        apply_reset();

        clear_seen();
        dynamic_scoreboard_active = 1'b1;
        start_operator(1, MAX_TILES);
        send_directed_full_fifo_prefix();
        for (int tile = 5; tile < MAX_TILES; tile++)
            send_tile_packet(tile, 1'b1, 1'b1);
        wait_dynamic_done_bounded();
        @(negedge clk_core);
        automatic_result_ready = 1'b0;
        result_ready = 1'b0;
        if (dynamic_results != 96 || operator_active || result_valid
            || fifo_level != 0 || protocol_error)
            $fatal(1, "M21 default dynamic run did not drain exactly");
        for (int tile = 0; tile < MAX_TILES; tile++)
            for (int slice = 0; slice < 6; slice++)
                if (!result_seen[tile][slice])
                    $fatal(1, "M21 default missing tile=%0d slice=%0d", tile, slice);
        if (packet_backpressure_cycles <= 0 || result_stall_cycles <= 0)
            $fatal(1, "M21 default dynamic backpressure coverage is empty");
        if (directed_packet_backpressure_cycles != 3)
            $fatal(1, "M21 default directed packet backpressure count drifted");

        $display("M21_DEFAULT_SMOKE max_population=%0d lane_tiles=%0d fifo_depth=4 packet_lanes=96 slices=6 count_w=%0d sum_w=%0d sumsq_w=%0d dynamic_population=1 dynamic_results=96 dynamic_done=1 above_max=PASS midflight_reset=PASS",
                 MAX_POP, MAX_TILES, COUNT_W, SUM_W, SUMSQ_W);
        $display("M21_DEFAULT_DYNAMIC packet_backpressure_cycles=%0d directed_packet_backpressure_cycles=3 result_stall_cycles=%0d tile_slice_results=96 arithmetic_lanes_checked=1536",
                 packet_backpressure_cycles, result_stall_cycles);
        $display("PASS: Synopsys VCS M21 default-parameter dynamic 16-tile smoke");
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M21 default dynamic smoke timeout");
    end
endmodule

`default_nettype wire
