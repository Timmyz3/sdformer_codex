`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_rank3_resident_stream_core;
    localparam int TAG_W = 48;
    localparam int T = 10;
    localparam int RANK = 3;
    localparam int LANES = 16;
    localparam int ACC_W = 24;
    localparam int THROUGHPUT_TILES = 24;
    localparam int TILES = 28;
    localparam int DIRECTED_SHIFT = 8;
    localparam integer signed DIRECTED_THRESHOLD = 12345;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic parameter_valid = 1'b0;
    logic parameter_ready;
    logic [(RANK*T*8)-1:0] parameter_right_factor = '0;
    logic [(T*RANK*8)-1:0] parameter_left_factor = '0;
    logic [(T*ACC_W)-1:0] parameter_bias_by_row = '0;
    logic signed [ACC_W-1:0] parameter_threshold = '0;
    logic [4:0] parameter_requant_shift = 5'd3;
    logic parameter_loaded;
    logic parameter_release_valid = 1'b0;
    logic parameter_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [2:0] input_beat = '0;
    logic [255:0] input_values = '0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [31:0] result_bits;
    logic done;
    logic [TAG_W-1:0] done_tag;
    logic protocol_error;
    logic busy;
    logic arithmetic_active;
    logic stage_select;
    logic [2:0] phase_cycle;
    logic [95:0] multiplier_active_mask;
    logic [4:0] result_fifo_occupancy;

    integer signed cfg_right [0:RANK*T-1];
    integer signed cfg_left [0:T*RANK-1];
    integer signed cfg_bias [0:T-1];
    integer signed tile_x [0:TILES-1][0:T*LANES-1];
    integer signed reference_intermediate [0:TILES-1][0:RANK*LANES-1];
    bit reference_bits [0:TILES-1][0:T*LANES-1];

    integer cycle_count = 0;
    integer input_beats = 0;
    integer output_beats = 0;
    integer tiles_done = 0;
    integer arithmetic_cycles = 0;
    integer stage1_cycles = 0;
    integer stage2_cycles = 0;
    integer output_stall_cycles = 0;
    integer parameter_fires = 0;
    integer parameter_release_fires = 0;
    integer tile_starts = 0;
    integer previous_tile_start = -1;
    integer maximum_fifo_occupancy = 0;
    integer fifo_credit_wait_cycles = 0;
    integer tie_cases = 0;
    integer positive_intermediate_saturations = 0;
    integer negative_intermediate_saturations = 0;
    integer positive_output_saturations = 0;
    integer negative_output_saturations = 0;
    integer threshold_equal_cases = 0;
    integer threshold_below_cases = 0;
    integer intermediate_tiles_checked = 0;
    integer early_release_wait_cycles = 0;
    integer release_input_collision_checks = 0;
    integer midpacket_release_checks = 0;
    integer midpacket_release_wait_cycles = 0;
    logic check_ii = 1'b1;
    logic force_sink_stall = 1'b0;

    always #5 clk_core = ~clk_core;

    qfit_atlif_rank3_resident_stream_core dut (.*);

    function automatic integer signed rne_q8(
        input longint signed value, input integer shift
    );
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded;
        begin
            magnitude = value < 0 ? -value : value;
            quotient = magnitude >> shift;
            remainder = shift == 0 ? 0 : magnitude & ((64'd1 << shift)-1);
            half = shift == 0 ? 0 : 64'd1 << (shift-1);
            if (shift != 0
                && (remainder > half || (remainder == half && quotient[0])))
                quotient = quotient + 1;
            rounded = value < 0 ? -$signed(quotient) : $signed(quotient);
            if (rounded > 127)
                rne_q8 = 127;
            else if (rounded < -128)
                rne_q8 = -128;
            else
                rne_q8 = rounded;
        end
    endfunction

    function automatic integer signed clamp_q24(input longint signed value);
        if (value > 8388607)
            clamp_q24 = 8388607;
        else if (value < -8388608)
            clamp_q24 = -8388608;
        else
            clamp_q24 = value;
    endfunction

    task automatic compute_reference_range(
        input int first_tile,
        input int last_tile,
        input int shift,
        input integer signed threshold,
        input bit count_directed_coverage
    );
        integer signed intermediate [0:RANK*LANES-1];
        longint signed accumulator;
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded_unclamped;
        integer signed saturated;
        for (int tile = first_tile; tile < last_tile; tile++) begin
            for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    accumulator = 0;
                    for (int row = 0; row < T; row++)
                        accumulator += tile_x[tile][row*LANES+lane]
                            * cfg_right[rank_index*T+row];
                    magnitude = accumulator < 0 ? -accumulator : accumulator;
                    quotient = magnitude >> shift;
                    remainder = shift == 0 ? 0
                        : magnitude & ((64'd1 << shift)-1);
                    half = shift == 0 ? 0 : 64'd1 << (shift-1);
                    if (count_directed_coverage && shift != 0
                        && remainder == half)
                        tie_cases = tie_cases + 1;
                    if (shift != 0
                        && (remainder > half
                            || (remainder == half && quotient[0])))
                        quotient = quotient + 1;
                    rounded_unclamped = accumulator < 0
                        ? -$signed(quotient) : $signed(quotient);
                    if (count_directed_coverage && rounded_unclamped > 127)
                        positive_intermediate_saturations
                            = positive_intermediate_saturations + 1;
                    if (count_directed_coverage && rounded_unclamped < -128)
                        negative_intermediate_saturations
                            = negative_intermediate_saturations + 1;
                    intermediate[rank_index*LANES+lane]
                        = rne_q8(accumulator, shift);
                    reference_intermediate[tile][rank_index*LANES+lane]
                        = intermediate[rank_index*LANES+lane];
                end
            end
            for (int row = 0; row < T; row++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    accumulator = cfg_bias[row];
                    for (int rank_index = 0; rank_index < RANK; rank_index++)
                        accumulator += intermediate[rank_index*LANES+lane]
                            * cfg_left[row*RANK+rank_index];
                    if (count_directed_coverage && accumulator > 8388607)
                        positive_output_saturations
                            = positive_output_saturations + 1;
                    if (count_directed_coverage && accumulator < -8388608)
                        negative_output_saturations
                            = negative_output_saturations + 1;
                    saturated = clamp_q24(accumulator);
                    if (count_directed_coverage && saturated == threshold)
                        threshold_equal_cases = threshold_equal_cases + 1;
                    if (count_directed_coverage && saturated == threshold-1)
                        threshold_below_cases = threshold_below_cases + 1;
                    reference_bits[tile][row*LANES+lane]
                        = saturated >= threshold;
                end
            end
        end
    endtask

    task automatic build_normal_stimulus;
        for (int index = 0; index < RANK*T; index++)
            cfg_right[index] = (index*13 + 5) % 15 - 7;
        for (int index = 0; index < T*RANK; index++)
            cfg_left[index] = (index*17 + 3) % 15 - 7;
        for (int row = 0; row < T; row++)
            cfg_bias[row] = row*37 - 160;
        for (int tile = 0; tile < THROUGHPUT_TILES; tile++)
            for (int index = 0; index < T*LANES; index++)
                tile_x[tile][index] = (tile*31 + index*19 + 11) % 16 - 8;
        compute_reference_range(0, THROUGHPUT_TILES, 3, 0, 1'b0);
    endtask

    task automatic build_directed_stimulus;
        for (int index = 0; index < RANK*T; index++)
            cfg_right[index] = 0;
        for (int index = 0; index < T*RANK; index++)
            cfg_left[index] = 0;
        for (int row = 0; row < T; row++)
            cfg_bias[row] = 0;

        // Rank zero forms exact signed RNE ties.  Rank one deliberately
        // overflows signed INT8 after shift eight for both signs.
        for (int row = 0; row < 6; row++)
            cfg_right[row] = 1;
        for (int row = 0; row < 3; row++)
            cfg_right[T+row] = 127;

        // Rows zero/one prove >= threshold packing at equality and one below.
        // Rows two/three drive positive/negative Q24 saturation, respectively.
        cfg_bias[0] = DIRECTED_THRESHOLD;
        cfg_bias[1] = DIRECTED_THRESHOLD-1;
        cfg_bias[2] = 8388607;
        cfg_bias[3] = -8388608;
        cfg_left[(2*RANK)+1] = 127;
        cfg_left[(3*RANK)+1] = 127;
        for (int row = 4; row < T; row++)
            cfg_left[row*RANK] = row[0] ? -1 : 1;

        for (int tile = THROUGHPUT_TILES; tile < TILES; tile++) begin
            for (int index = 0; index < T*LANES; index++)
                tile_x[tile][index] = 0;
            for (int lane = 0; lane < LANES; lane++) begin
                case (lane % 8)
                    0: begin tile_x[tile][lane] = 127;
                             tile_x[tile][LANES+lane] = 1; end
                    1: begin tile_x[tile][lane] = 127;
                             tile_x[tile][LANES+lane] = 127;
                             tile_x[tile][(2*LANES)+lane] = 127;
                             tile_x[tile][(3*LANES)+lane] = 3; end
                    2: tile_x[tile][lane] = -128;
                    3: begin tile_x[tile][lane] = -128;
                             tile_x[tile][LANES+lane] = -128;
                             tile_x[tile][(2*LANES)+lane] = -128; end
                    4: begin tile_x[tile][lane] = 127;
                             tile_x[tile][LANES+lane] = 127;
                             tile_x[tile][(2*LANES)+lane] = 127;
                             tile_x[tile][(3*LANES)+lane] = 127;
                             tile_x[tile][(4*LANES)+lane] = 127;
                             tile_x[tile][(5*LANES)+lane] = 5; end
                    5: begin tile_x[tile][lane] = -128;
                             tile_x[tile][LANES+lane] = -128;
                             tile_x[tile][(2*LANES)+lane] = -128;
                             tile_x[tile][(3*LANES)+lane] = -128;
                             tile_x[tile][(4*LANES)+lane] = -128; end
                    6: begin tile_x[tile][lane] = 127;
                             tile_x[tile][LANES+lane] = 127;
                             tile_x[tile][(2*LANES)+lane] = 2; end
                    default: begin tile_x[tile][lane] = -128;
                                   tile_x[tile][LANES+lane] = -128; end
                endcase
            end
        end
        compute_reference_range(THROUGHPUT_TILES, TILES, DIRECTED_SHIFT,
                                DIRECTED_THRESHOLD, 1'b1);
    endtask

    task automatic load_parameters(
        input int shift,
        input integer signed threshold
    );
        for (int index = 0; index < RANK*T; index++)
            parameter_right_factor[index*8 +: 8] = cfg_right[index][7:0];
        for (int index = 0; index < T*RANK; index++)
            parameter_left_factor[index*8 +: 8] = cfg_left[index][7:0];
        for (int row = 0; row < T; row++)
            parameter_bias_by_row[row*ACC_W +: ACC_W]
                = cfg_bias[row][ACC_W-1:0];
        parameter_threshold = threshold[ACC_W-1:0];
        parameter_requant_shift = shift[4:0];
        @(negedge clk_core);
        parameter_valid = 1'b1;
        do @(posedge clk_core); while (!parameter_ready);
        @(negedge clk_core);
        parameter_valid = 1'b0;
    endtask

    task automatic release_parameters;
        @(negedge clk_core);
        parameter_release_valid = 1'b1;
        input_valid = 1'b1;
        input_beat = 0;
        input_tag = 48'h4d30ffffffff;
        input_values = '0;
        #1;
        if (!parameter_release_ready || input_ready)
            $fatal(1, "M30 release/input arbitration did not prioritize release");
        do @(posedge clk_core); while (!parameter_release_ready);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
        input_valid = 1'b0;
        release_input_collision_checks = release_input_collision_checks + 1;
    endtask

    task automatic send_tile(input int tile);
        for (int beat = 0; beat < 5; beat++) begin
            for (int index = 0; index < 32; index++)
                input_values[index*8 +: 8]
                    = tile_x[tile][beat*32+index][7:0];
            input_tag = 48'h4d3000000000 + tile;
            input_beat = beat[2:0];
            input_valid = 1'b1;
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
        end
        input_valid = 1'b0;
    endtask

    task automatic send_tile_with_midpacket_release(input int tile);
        for (int beat = 0; beat < 5; beat++) begin
            for (int index = 0; index < 32; index++)
                input_values[index*8 +: 8]
                    = tile_x[tile][beat*32+index][7:0];
            input_tag = 48'h4d3000000000 + tile;
            input_beat = beat[2:0];
            input_valid = 1'b1;
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
            if (beat == 2)
                parameter_release_valid = 1'b1;
            if (beat >= 3 && !parameter_release_valid)
                $fatal(1, "M30 midpacket release request was lost");
        end

        // A new beat zero must remain blocked while the held release waits for
        // the completed tile to compute, enqueue, and drain.
        input_tag = 48'h4d30fffffffe;
        input_beat = 0;
        input_values = '0;
        input_valid = 1'b1;
        #1;
        if (input_ready)
            $fatal(1, "M30 accepted a new tile while release was pending");
        while (!parameter_release_ready) begin
            if (input_ready || !parameter_loaded)
                $fatal(1, "M30 midpacket release boundary contract failed");
            midpacket_release_wait_cycles
                = midpacket_release_wait_cycles + 1;
            @(negedge clk_core);
        end
        @(posedge clk_core);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
        input_valid = 1'b0;
        midpacket_release_checks = midpacket_release_checks + 1;
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (parameter_valid && parameter_ready)
                parameter_fires = parameter_fires + 1;
            if (parameter_release_valid && parameter_release_ready) begin
                parameter_release_fires = parameter_release_fires + 1;
                previous_tile_start = -1;
            end
            if (input_valid && input_ready)
                input_beats = input_beats + 1;
            if (arithmetic_active) begin
                arithmetic_cycles = arithmetic_cycles + 1;
                if (stage_select)
                    stage2_cycles = stage2_cycles + 1;
                else
                    stage1_cycles = stage1_cycles + 1;
                if (multiplier_active_mask !== {96{1'b1}})
                    $fatal(1, "M30 active multiplier mask mismatch");
            end
            if (arithmetic_active && !stage_select && phase_cycle == 0) begin
                if (check_ii && previous_tile_start >= 0
                    && cycle_count - previous_tile_start != 10)
                    $fatal(1, "M30 steady II drift got %0d expected 10",
                           cycle_count - previous_tile_start);
                previous_tile_start = cycle_count;
                tile_starts = tile_starts + 1;
            end
            if (result_valid && !result_ready)
                output_stall_cycles = output_stall_cycles + 1;
            if (result_fifo_occupancy > maximum_fifo_occupancy)
                maximum_fifo_occupancy = result_fifo_occupancy;
            if (busy && !arithmetic_active && result_fifo_occupancy >= 12)
                fifo_credit_wait_cycles = fifo_credit_wait_cycles + 1;
            if (result_valid && result_ready) begin
                int tile;
                tile = output_beats / 5;
                if (result_tag !== 48'h4d3000000000 + tile)
                    $fatal(1, "M30 output tag mismatch beat=%0d", output_beats);
                if (result_beat !== (output_beats % 5))
                    $fatal(1, "M30 output beat order mismatch got=%0d expected=%0d",
                           result_beat, output_beats % 5);
                for (int index = 0; index < 32; index++) begin
                    if (result_bits[index]
                        !== reference_bits[tile][result_beat*32+index])
                        $fatal(1, "M30 threshold bit mismatch tile=%0d beat=%0d index=%0d",
                               tile, result_beat, index);
                end
                output_beats = output_beats + 1;
            end
            if (done) begin
                if (done_tag !== 48'h4d3000000000 + tiles_done)
                    $fatal(1, "M30 done order mismatch");
                tiles_done = tiles_done + 1;
            end
        end
    end

    // Sparse bounded stalls exercise FIFO decoupling while retaining more
    // than enough average sink bandwidth for an unstalled II=10 producer.
    always @(negedge clk_core) begin
        if (rst_core)
            result_ready <= 1'b0;
        else
            result_ready <= !force_sink_stall
                && !((cycle_count % 29) inside {[11:13]});
    end

    // Observe stage-1 architectural state before the first stage-2 issue.
    // This prevents threshold choices from masking RNE or INT8 saturation bugs.
    always @(negedge clk_core) begin
        if (!rst_core && arithmetic_active && stage_select
            && phase_cycle == 0) begin
            for (int index = 0; index < RANK*LANES; index++) begin
                if ($signed(dut.intermediate_q[index])
                    !== reference_intermediate[intermediate_tiles_checked][index])
                    $fatal(1, "M30 intermediate mismatch tile=%0d index=%0d got=%0d expected=%0d",
                           intermediate_tiles_checked, index,
                           $signed(dut.intermediate_q[index]),
                           reference_intermediate[intermediate_tiles_checked][index]);
            end
            intermediate_tiles_checked = intermediate_tiles_checked + 1;
        end
    end

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M30 evidence requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M30 evidence requires enabled SVA");
`endif
        build_normal_stimulus();
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        load_parameters(3, 0);
        if (!parameter_loaded)
            @(posedge clk_core);
        for (int tile = 0; tile < THROUGHPUT_TILES; tile++)
            send_tile(tile);
        while (output_beats < THROUGHPUT_TILES*5
               || tiles_done < THROUGHPUT_TILES || busy)
            @(posedge clk_core);
        check_ii = 1'b0;
        release_parameters();
        build_directed_stimulus();
        load_parameters(DIRECTED_SHIFT, DIRECTED_THRESHOLD);
        send_tile_with_midpacket_release(THROUGHPUT_TILES);
        load_parameters(DIRECTED_SHIFT, DIRECTED_THRESHOLD);
        force_sink_stall = 1'b1;
        fork
            begin
                for (int tile = THROUGHPUT_TILES+1; tile < TILES; tile++)
                    send_tile(tile);
            end
            begin
                repeat (100) @(posedge clk_core);
                force_sink_stall = 1'b0;
            end
            begin
                while (!(busy && result_fifo_occupancy >= 12))
                    @(posedge clk_core);
                @(negedge clk_core);
                parameter_release_valid = 1'b1;
                while (!parameter_release_ready) begin
                    if (!parameter_loaded)
                        $fatal(1, "M30 context changed before held release became ready");
                    early_release_wait_cycles = early_release_wait_cycles + 1;
                    @(negedge clk_core);
                end
                @(posedge clk_core);
                @(negedge clk_core);
                parameter_release_valid = 1'b0;
            end
        join
        while (output_beats < TILES*5 || tiles_done < TILES)
            @(posedge clk_core);
        repeat (3) @(posedge clk_core);
        if (protocol_error)
            $fatal(1, "M30 protocol_error asserted");
        if (parameter_fires != 3 || parameter_release_fires != 3
            || input_beats != TILES*5 || release_input_collision_checks != 1
            || midpacket_release_checks != 1
            || midpacket_release_wait_cycles <= 0)
            $fatal(1, "M30 load accounting mismatch parameter=%0d release=%0d input=%0d",
                   parameter_fires, parameter_release_fires, input_beats);
        if (tile_starts != TILES || arithmetic_cycles != TILES*10
            || stage1_cycles != TILES*5 || stage2_cycles != TILES*5)
            $fatal(1, "M30 arithmetic accounting mismatch starts=%0d all=%0d s1=%0d s2=%0d",
                   tile_starts, arithmetic_cycles, stage1_cycles, stage2_cycles);
        if (output_beats != TILES*5 || tiles_done != TILES)
            $fatal(1, "M30 output accounting mismatch");
        if (intermediate_tiles_checked != TILES)
            $fatal(1, "M30 intermediate miter count mismatch got=%0d expected=%0d",
                   intermediate_tiles_checked, TILES);
        if (output_stall_cycles <= 0 || maximum_fifo_occupancy != 15
            || fifo_credit_wait_cycles <= 0 || early_release_wait_cycles <= 0)
            $fatal(1, "M30 FIFO backpressure was not exercised");
        if (tie_cases < 8 || positive_intermediate_saturations == 0
            || negative_intermediate_saturations == 0
            || positive_output_saturations == 0
            || negative_output_saturations == 0
            || threshold_equal_cases == 0 || threshold_below_cases == 0)
            $fatal(1, "M30 directed arithmetic coverage incomplete ties=%0d mid_sat=%0d/%0d out_sat=%0d/%0d threshold=%0d/%0d",
                   tie_cases, positive_intermediate_saturations,
                   negative_intermediate_saturations,
                   positive_output_saturations, negative_output_saturations,
                   threshold_equal_cases, threshold_below_cases);
        $display("M30_PASS tiles=%0d input_beats=%0d output_beats=%0d arithmetic_cycles=%0d first_cohort_ii=10 stalls=%0d max_fifo=%0d fifo_wait=%0d reloads=%0d ties=%0d mid_sat=%0d/%0d out_sat=%0d/%0d threshold_eq_below=%0d/%0d",
                 TILES, input_beats, output_beats, arithmetic_cycles,
                 output_stall_cycles, maximum_fifo_occupancy,
                 fifo_credit_wait_cycles, parameter_fires, tie_cases,
                 positive_intermediate_saturations,
                 negative_intermediate_saturations,
                 positive_output_saturations, negative_output_saturations,
                 threshold_equal_cases, threshold_below_cases);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M30 timeout");
    end
endmodule

`default_nettype wire
