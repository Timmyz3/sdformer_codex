`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_rank3_exact96_core;
    localparam int TAG_W = 48;
    localparam int T = 10;
    localparam int RANK = 3;
    localparam int LANES = 16;
    localparam int ACC_W = 24;
    localparam int REQUANT_SHIFT = 8;
    localparam int RANDOM_TILES = 24;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic request_valid = 1'b0;
    logic request_ready;
    logic request_legal;
    logic [TAG_W-1:0] request_tag = '0;
    logic [(T*LANES*8)-1:0] request_x = '0;
    logic [(RANK*T*8)-1:0] request_right_factor = '0;
    logic [(T*RANK*8)-1:0] request_left_factor = '0;
    logic [(T*LANES*ACC_W)-1:0] request_bias = '0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [(32*ACC_W)-1:0] result_values;
    logic done;
    logic [TAG_W-1:0] done_tag;
    logic protocol_error;
    logic busy;
    logic arithmetic_active;
    logic stage_select;
    logic [2:0] phase_cycle;
    logic [95:0] multiplier_active_mask;

    integer signed stimulus_x [0:(T*LANES)-1];
    integer signed stimulus_right [0:(RANK*T)-1];
    integer signed stimulus_left [0:(T*RANK)-1];
    integer signed stimulus_bias [0:(T*LANES)-1];
    integer signed reference_intermediate [0:(RANK*LANES)-1];
    integer signed reference_output [0:(T*LANES)-1];

    integer cycle_count = 0;
    integer tiles_checked = 0;
    integer beats_checked = 0;
    integer arithmetic_cycles = 0;
    integer stage1_cycles = 0;
    integer stage2_cycles = 0;
    integer output_stall_cycles = 0;
    integer boundary_checks = 0;
    integer tie_cases = 0;
    integer positive_intermediate_saturations = 0;
    integer negative_intermediate_saturations = 0;
    integer positive_output_saturations = 0;
    integer negative_output_saturations = 0;

    always #5 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (arithmetic_active) begin
                arithmetic_cycles = arithmetic_cycles + 1;
                if (stage_select)
                    stage2_cycles = stage2_cycles + 1;
                else
                    stage1_cycles = stage1_cycles + 1;
                if (multiplier_active_mask !== {96{1'b1}})
                    $fatal(1, "M27 active arithmetic did not expose 96 live slots");
            end
            if (result_valid && !result_ready)
                output_stall_cycles = output_stall_cycles + 1;
        end
    end

    qfit_atlif_rank3_exact96_core #(.REQUANT_SHIFT(REQUANT_SHIFT)) dut (.*);

    function automatic integer signed clamp_q24(input longint signed value);
        begin
            if (value > 8388607)
                clamp_q24 = 8388607;
            else if (value < -8388608)
                clamp_q24 = -8388608;
            else
                clamp_q24 = value;
        end
    endfunction

    function automatic integer signed reference_rne_q8(
        input longint signed value, input integer shift
    );
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded;
        begin
            magnitude = value < 0 ? -value : value;
            if (shift == 0) begin
                quotient = magnitude;
            end else begin
                quotient = magnitude >> shift;
                remainder = magnitude & ((64'd1 << shift)-1);
                half = 64'd1 << (shift-1);
                if (remainder > half || (remainder == half && quotient[0]))
                    quotient = quotient + 1;
            end
            rounded = value < 0 ? -$signed(quotient) : $signed(quotient);
            if (rounded > 127)
                reference_rne_q8 = 127;
            else if (rounded < -128)
                reference_rne_q8 = -128;
            else
                reference_rne_q8 = rounded;
        end
    endfunction

    task automatic clear_stimulus;
        for (int index = 0; index < T*LANES; index++) begin
            stimulus_x[index] = 0;
            stimulus_bias[index] = 0;
        end
        for (int index = 0; index < RANK*T; index++)
            stimulus_right[index] = 0;
        for (int index = 0; index < T*RANK; index++)
            stimulus_left[index] = 0;
    endtask

    task automatic build_directed(input int mode, output int shift);
        clear_stimulus();
        shift = REQUANT_SHIFT;
        case (mode)
            0: begin
                // With the frozen shift of eight, form exact signed ties at
                // +/-0.5, +/-1.5, +/-2.5, and exact +/-1.0 in Q8 units.
                for (int rank_index = 0; rank_index < RANK; rank_index++)
                    for (int row = 0; row < 6; row++)
                        stimulus_right[(rank_index*T)+row] = 1;
                for (int lane = 0; lane < LANES; lane++) begin
                    case (lane % 8)
                        0: begin stimulus_x[lane] = 127;
                                 stimulus_x[LANES+lane] = 1; end
                        1: begin stimulus_x[lane] = 127;
                                 stimulus_x[LANES+lane] = 127;
                                 stimulus_x[(2*LANES)+lane] = 127;
                                 stimulus_x[(3*LANES)+lane] = 3; end
                        2: stimulus_x[lane] = -128;
                        3: begin stimulus_x[lane] = -128;
                                 stimulus_x[LANES+lane] = -128;
                                 stimulus_x[(2*LANES)+lane] = -128; end
                        4: begin stimulus_x[lane] = 127;
                                 stimulus_x[LANES+lane] = 127;
                                 stimulus_x[(2*LANES)+lane] = 127;
                                 stimulus_x[(3*LANES)+lane] = 127;
                                 stimulus_x[(4*LANES)+lane] = 127;
                                 stimulus_x[(5*LANES)+lane] = 5; end
                        5: begin stimulus_x[lane] = -128;
                                 stimulus_x[LANES+lane] = -128;
                                 stimulus_x[(2*LANES)+lane] = -128;
                                 stimulus_x[(3*LANES)+lane] = -128;
                                 stimulus_x[(4*LANES)+lane] = -128; end
                        6: begin stimulus_x[lane] = 127;
                                 stimulus_x[LANES+lane] = 127;
                                 stimulus_x[(2*LANES)+lane] = 2; end
                        default: begin stimulus_x[lane] = -128;
                                       stimulus_x[LANES+lane] = -128; end
                    endcase
                end
                for (int row = 0; row < T; row++) begin
                    stimulus_left[row*RANK] = 1;
                    stimulus_left[(row*RANK)+1] = -1;
                    stimulus_left[(row*RANK)+2] = 1;
                end
            end
            1: begin
                // Intermediate positive and negative saturation after shift 8.
                for (int lane = 0; lane < LANES; lane++) begin
                    for (int row = 0; row < 3; row++)
                        stimulus_x[(row*LANES)+lane] = lane[0] ? -128 : 127;
                end
                for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                    for (int row = 0; row < 3; row++)
                        stimulus_right[(rank_index*T)+row] = 127;
                end
                for (int row = 0; row < T; row++)
                    for (int rank_index = 0; rank_index < RANK; rank_index++)
                        stimulus_left[(row*RANK)+rank_index] = rank_index == 1 ? -2 : 1;
            end
            2: begin
                // Stage-2 Q24 upper/lower saturation after dynamic bias addition.
                for (int lane = 0; lane < LANES; lane++)
                    for (int source_row = 0; source_row < 3; source_row++)
                        stimulus_x[(source_row*LANES)+lane] = 127;
                for (int rank_index = 0; rank_index < RANK; rank_index++)
                    for (int source_row = 0; source_row < 3; source_row++)
                        stimulus_right[(rank_index*T)+source_row] = 127;
                for (int row = 0; row < T; row++) begin
                    for (int rank_index = 0; rank_index < RANK; rank_index++)
                        stimulus_left[(row*RANK)+rank_index] = row[0] ? -128 : 127;
                    for (int lane = 0; lane < LANES; lane++)
                        stimulus_bias[(row*LANES)+lane]
                            = row[0] ? -8388608 : 8388607;
                end
            end
            default: begin
                for (int index = 0; index < T*LANES; index++)
                    stimulus_x[index] = (index*29 + 17) % 256 - 128;
                for (int index = 0; index < RANK*T; index++)
                    stimulus_right[index] = (index*13 + 5) % 31 - 15;
                for (int index = 0; index < T*RANK; index++)
                    stimulus_left[index] = (index*19 + 3) % 63 - 31;
                for (int index = 0; index < T*LANES; index++)
                    stimulus_bias[index] = (index*7919) % 200001 - 100000;
            end
        endcase
    endtask

    task automatic build_random(input int tile, output int shift);
        shift = REQUANT_SHIFT;
        for (int index = 0; index < T*LANES; index++) begin
            stimulus_x[index] = $signed($urandom_range(0, 255) - 128);
            stimulus_bias[index] = $signed($urandom_range(0, 4000000) - 2000000);
        end
        for (int index = 0; index < RANK*T; index++)
            stimulus_right[index] = $signed($urandom_range(0, 255) - 128);
        for (int index = 0; index < T*RANK; index++)
            stimulus_left[index] = $signed($urandom_range(0, 255) - 128);
        if ((tile % 5) == 0) begin
            stimulus_bias[tile % (T*LANES)] = 8388607;
            stimulus_bias[(tile*7) % (T*LANES)] = -8388608;
        end
    endtask

    task automatic pack_stimulus;
        for (int index = 0; index < T*LANES; index++) begin
            request_x[(index*8) +: 8] = stimulus_x[index][7:0];
            request_bias[(index*ACC_W) +: ACC_W]
                = stimulus_bias[index][ACC_W-1:0];
        end
        for (int index = 0; index < RANK*T; index++)
            request_right_factor[(index*8) +: 8] = stimulus_right[index][7:0];
        for (int index = 0; index < T*RANK; index++)
            request_left_factor[(index*8) +: 8] = stimulus_left[index][7:0];
    endtask

    task automatic compute_reference(input int shift);
        longint signed accumulator;
        longint unsigned magnitude;
        longint unsigned remainder;
        longint unsigned half;
        longint unsigned quotient;
        longint signed rounded_unclamped;
        integer signed quantized;
        for (int rank_index = 0; rank_index < RANK; rank_index++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                accumulator = 0;
                for (int row = 0; row < T; row++)
                    accumulator = accumulator
                        + stimulus_x[(row*LANES)+lane]
                        * stimulus_right[(rank_index*T)+row];
                if (shift != 0) begin
                    magnitude = accumulator < 0 ? -accumulator : accumulator;
                    quotient = magnitude >> shift;
                    remainder = magnitude & ((64'd1 << shift)-1);
                    half = 64'd1 << (shift-1);
                    if (remainder == half)
                        tie_cases = tie_cases + 1;
                    if (remainder > half || (remainder == half && quotient[0]))
                        quotient = quotient + 1;
                end else begin
                    magnitude = accumulator < 0 ? -accumulator : accumulator;
                    quotient = magnitude;
                end
                rounded_unclamped = accumulator < 0
                    ? -$signed(quotient) : $signed(quotient);
                quantized = reference_rne_q8(accumulator, shift);
                if (rounded_unclamped > 127)
                    positive_intermediate_saturations
                        = positive_intermediate_saturations + 1;
                if (rounded_unclamped < -128)
                    negative_intermediate_saturations
                        = negative_intermediate_saturations + 1;
                reference_intermediate[(rank_index*LANES)+lane] = quantized;
            end
        end
        for (int row = 0; row < T; row++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                accumulator = stimulus_bias[(row*LANES)+lane];
                for (int rank_index = 0; rank_index < RANK; rank_index++)
                    accumulator = accumulator
                        + reference_intermediate[(rank_index*LANES)+lane]
                        * stimulus_left[(row*RANK)+rank_index];
                if (accumulator > 8388607)
                    positive_output_saturations = positive_output_saturations + 1;
                if (accumulator < -8388608)
                    negative_output_saturations = negative_output_saturations + 1;
                reference_output[(row*LANES)+lane] = clamp_q24(accumulator);
            end
        end
    endtask

    task automatic check_beat(input int expected_beat, input logic [TAG_W-1:0] tag);
        integer signed observed;
        int row;
        int lane;
        if (result_tag !== tag || result_beat !== expected_beat[2:0])
            $fatal(1, "M27 result identity mismatch got tag=%h beat=%0d expected=%h/%0d",
                   result_tag, result_beat, tag, expected_beat);
        for (int output_index = 0; output_index < 32; output_index++) begin
            row = (expected_beat*2) + (output_index/LANES);
            lane = output_index % LANES;
            observed = $signed(result_values[(output_index*ACC_W) +: ACC_W]);
            if (observed !== reference_output[(row*LANES)+lane])
                $fatal(1, "M27 arithmetic mismatch beat=%0d row=%0d lane=%0d got=%0d expected=%0d",
                       expected_beat, row, lane, observed,
                       reference_output[(row*LANES)+lane]);
        end
        beats_checked = beats_checked + 1;
    endtask

    task automatic run_tile(
        input int tile_index,
        input bit directed,
        input int mode,
        input bit use_backpressure
    );
        int shift;
        int accept_cycle;
        int expected_beat;
        int stall_budget;
        bit finished;
        logic [TAG_W-1:0] tag;
        if (directed)
            build_directed(mode, shift);
        else
            build_random(tile_index, shift);
        pack_stimulus();
        compute_reference(shift);
        tag = 48'h270000000000 + tile_index;

        @(negedge clk_core);
        if (!request_ready || busy || protocol_error)
            $fatal(1, "M27 core unavailable before tile %0d", tile_index);
        request_tag = tag;
        request_valid = 1'b1;
        result_ready = 1'b1;
        #1;
        if (!request_legal)
            $fatal(1, "M27 legal request rejected shift=%0d", shift);
        @(posedge clk_core);
        #1;
        accept_cycle = cycle_count;
        @(negedge clk_core);
        request_valid = 1'b0;

        expected_beat = 0;
        stall_budget = use_backpressure ? 3 : 0;
        finished = 1'b0;
        while (!finished) begin
            @(negedge clk_core);
            if (use_backpressure && result_valid && result_beat == 1
                && stall_budget > 0) begin
                result_ready = 1'b0;
                stall_budget = stall_budget - 1;
            end else if (use_backpressure && result_valid
                         && (($urandom % 4) == 0)) begin
                result_ready = 1'b0;
            end else begin
                result_ready = 1'b1;
            end
            #1;
            if (result_valid && result_ready) begin
                check_beat(expected_beat, tag);
                if (!use_backpressure) begin
                    if ((cycle_count-accept_cycle) !== (6+expected_beat))
                        $fatal(1, "M27 no-stall beat latency mismatch beat=%0d delta=%0d",
                               expected_beat, cycle_count-accept_cycle);
                end
                expected_beat = expected_beat + 1;
            end
            @(posedge clk_core);
            #1;
            if (done) begin
                if (done_tag !== tag || expected_beat != 5)
                    $fatal(1, "M27 done identity/count mismatch tag=%h beats=%0d",
                           done_tag, expected_beat);
                if (!use_backpressure && (cycle_count-accept_cycle) !== 11)
                    $fatal(1, "M27 no-stall done latency mismatch delta=%0d",
                           cycle_count-accept_cycle);
                finished = 1'b1;
            end
        end
        @(negedge clk_core);
        result_ready = 1'b0;
        if (busy || result_valid || protocol_error || !request_ready)
            $fatal(1, "M27 core did not return cleanly idle after tile %0d", tile_index);
        tiles_checked = tiles_checked + 1;
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        request_valid = 1'b0;
        result_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        #1;
        if (!request_ready || busy || protocol_error || result_valid || done)
            $fatal(1, "M27 reset contract failed");
    endtask

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M27 evidence requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M27 evidence requires enabled SVA");
`endif
        apply_reset();

        for (int mode = 0; mode < 4; mode++)
            run_tile(mode, 1'b1, mode, mode == 3);
        for (int tile = 0; tile < RANDOM_TILES; tile++)
            run_tile(tile+4, 1'b0, 0, (tile % 3) == 0);

        // Give concurrent assertions one sampling edge after the final done
        // pulse so the last completion is represented in cover accounting.
        @(posedge clk_core);
        #1;

        if (tiles_checked != RANDOM_TILES+4 || beats_checked != (RANDOM_TILES+4)*5)
            $fatal(1, "M27 coverage count mismatch tiles=%0d beats=%0d",
                   tiles_checked, beats_checked);
        if (tie_cases < 8 || positive_intermediate_saturations == 0
            || negative_intermediate_saturations == 0
            || positive_output_saturations == 0
            || negative_output_saturations == 0 || output_stall_cycles < 3)
            $fatal(1, "M27 directed arithmetic coverage incomplete ties=%0d mid_sat=%0d/%0d out_sat=%0d/%0d stalls=%0d",
                   tie_cases, positive_intermediate_saturations,
                   negative_intermediate_saturations,
                   positive_output_saturations, negative_output_saturations,
                   output_stall_cycles);

        $display("M27_CYCLE_CONTRACT product_issue_cycles=10 accept_to_first_valid=6 accept_to_valid_beats=6,7,8,9,10 accept_to_done=11 arithmetic_cycles_per_unstalled_tile=10 stage1_cycles=5 stage2_cycles=5 transition_bubbles=0");
        $display("M27_RESULT tiles=%0d random_tiles=%0d beats=%0d arithmetic_cycles=%0d stage1_cycles=%0d stage2_cycles=%0d output_stall_cycles=%0d ties=%0d intermediate_sat_pos=%0d intermediate_sat_neg=%0d output_sat_pos=%0d output_sat_neg=%0d",
                 tiles_checked, RANDOM_TILES, beats_checked, arithmetic_cycles,
                 stage1_cycles, stage2_cycles, output_stall_cycles, tie_cases,
                 positive_intermediate_saturations,
                 negative_intermediate_saturations,
                 positive_output_saturations, negative_output_saturations);
        $display("M27_SCOPE multiplier_slots=96 multiplier_width=8x8 requant_shift=%0d request_load_cycles=not_modeled result_dma_cycles=not_modeled system_speedup=not_claimed", REQUANT_SHIFT);
        $display("PASS: Synopsys VCS M27 rank-3 exact-96 ATLIF factor tile reference miter");
        $finish;
    end
endmodule

`default_nettype wire
