`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_rank3_illegal_shift;
    localparam int TAG_W = 48;
    localparam int ACC_W = 24;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic parameter_valid = 1'b0;
    logic parameter_ready;
    logic [(3*10*8)-1:0] parameter_right_factor = '0;
    logic [(10*3*8)-1:0] parameter_left_factor = '0;
    logic [(10*ACC_W)-1:0] parameter_bias_by_row = '0;
    logic signed [ACC_W-1:0] parameter_threshold = '0;
    logic [4:0] parameter_requant_shift = '0;
    logic parameter_loaded;
    logic parameter_release_valid = 1'b0;
    logic parameter_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [2:0] input_beat = '0;
    logic [255:0] input_values = '0;
    logic result_valid;
    logic result_ready = 1'b1;
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
    integer legal_arithmetic_tiles = 0;

    always #5 clk_core = ~clk_core;
    qfit_atlif_rank3_resident_stream_core dut (.*);

    task automatic reset_dut;
        rst_core = 1'b1;
        parameter_valid = 1'b0;
        parameter_release_valid = 1'b0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
    endtask

    task automatic load_shift(input int shift, input integer signed threshold);
        parameter_requant_shift = shift[4:0];
        parameter_threshold = threshold[ACC_W-1:0];
        parameter_valid = 1'b1;
        if (!parameter_ready)
            $fatal(1, "M30 illegal-shift test did not reach parameter boundary");
        @(posedge clk_core);
        @(negedge clk_core);
        parameter_valid = 1'b0;
        @(posedge clk_core);
    endtask

    task automatic release_shift;
        @(negedge clk_core);
        parameter_release_valid = 1'b1;
        if (!parameter_release_ready)
            $fatal(1, "M30 legal shift context was not releasable");
        @(posedge clk_core);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
        @(posedge clk_core);
        if (parameter_loaded || protocol_error)
            $fatal(1, "M30 legal shift context release failed");
    endtask

    task automatic run_legal_shift(
        input int shift,
        input integer signed source_value,
        input integer signed right_value,
        input integer signed expected_intermediate,
        input integer signed threshold,
        input bit expected_bit,
        input int tile_index
    );
        int checked_beats;
        parameter_right_factor = '0;
        parameter_left_factor = '0;
        parameter_bias_by_row = '0;
        parameter_right_factor[0 +: 8] = right_value[7:0];
        for (int row = 0; row < 10; row++)
            parameter_left_factor[(row*3*8) +: 8] = 8'sd1;
        load_shift(shift, threshold);
        $display("M30_SHIFT_TRACE loaded shift=%0d tile=%0d", shift, tile_index);
        if (protocol_error || !parameter_loaded || !input_ready)
            $fatal(1, "M30 legal requant shift %0d was rejected", shift);

        // Drive the first request away from the sampling edge; otherwise a
        // testbench/DUT active-region race can make beat zero fire twice.
        @(negedge clk_core);
        input_tag = 48'h4d3050000000 + tile_index;
        input_valid = 1'b1;
        for (int beat = 0; beat < 5; beat++) begin
            input_values = '0;
            if (beat == 0)
                for (int lane = 0; lane < 16; lane++)
                    input_values[(lane*8) +: 8] = source_value[7:0];
            input_beat = beat[2:0];
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
            if (protocol_error)
                $fatal(1, "M30 legal shift input protocol failed shift=%0d offered_beat=%0d expected=%0d fill_active=%0b fill_bank=%0b bank_state=%0d/%0d",
                       shift, beat, dut.expected_input_beat_q,
                       dut.fill_active_q, dut.fill_bank_q,
                       dut.bank_state_q[0], dut.bank_state_q[1]);
        end
        input_valid = 1'b0;
        $display("M30_SHIFT_TRACE input_complete shift=%0d tile=%0d", shift, tile_index);

        do @(negedge clk_core); while (!(arithmetic_active && stage_select
                                         && phase_cycle == 0));
        $display("M30_SHIFT_TRACE stage2_start shift=%0d tile=%0d", shift, tile_index);
        for (int lane = 0; lane < 16; lane++) begin
            if ($signed(dut.intermediate_q[lane]) !== expected_intermediate)
                $fatal(1, "M30 legal shift arithmetic mismatch shift=%0d lane=%0d got=%0d expected=%0d",
                       shift, lane, $signed(dut.intermediate_q[lane]),
                       expected_intermediate);
            if ($signed(dut.intermediate_q[16+lane]) !== 0
                || $signed(dut.intermediate_q[32+lane]) !== 0)
                $fatal(1, "M30 legal shift inactive-rank mismatch shift=%0d lane=%0d",
                       shift, lane);
        end

        checked_beats = 0;
        while (checked_beats < 5) begin
            @(negedge clk_core);
            if (result_valid && result_ready) begin
                if (result_tag !== 48'h4d3050000000 + tile_index
                    || result_beat !== checked_beats[2:0]
                    || result_bits !== {32{expected_bit}})
                    $fatal(1, "M30 legal shift output mismatch shift=%0d beat=%0d",
                           shift, checked_beats);
                checked_beats = checked_beats + 1;
            end
        end
        while (busy) @(posedge clk_core);
        $display("M30_SHIFT_TRACE output_complete shift=%0d tile=%0d", shift, tile_index);
        legal_arithmetic_tiles = legal_arithmetic_tiles + 1;
        release_shift();
        $display("M30_SHIFT_TRACE released shift=%0d tile=%0d", shift, tile_index);
    endtask

    task automatic attempt_illegal_shift(input int shift);
        load_shift(shift, 0);
        if (!protocol_error || parameter_loaded || input_ready || arithmetic_active)
            $fatal(1, "M30 illegal requant shift %0d did not fail closed", shift);
    endtask

    initial begin
        reset_dut();
        run_legal_shift(0, 127, 127, 127, 127, 1'b1, 0);
        run_legal_shift(1, 3, 1, 2, 2, 1'b1, 1);
        run_legal_shift(1, -3, 1, -2, -2, 1'b1, 2);
        run_legal_shift(1, -3, 1, -2, -1, 1'b0, 3);
        run_legal_shift(23, 127, 1, 0, 0, 1'b1, 4);
        if (legal_arithmetic_tiles != 5)
            $fatal(1, "M30 legal shift arithmetic coverage incomplete");
        reset_dut(); attempt_illegal_shift(24);
        reset_dut(); attempt_illegal_shift(25);
        reset_dut(); attempt_illegal_shift(31);
        $display("M30_ILLEGAL_SHIFT_PASS legal_arithmetic_tiles=5 shifts=0,1,23 signed_thresholds=positive,negative,boundary illegal_fail_closed=24,25,31");
        $finish;
    end

    initial begin
        #10000;
        $display("M30_SHIFT_TIMEOUT busy=%0b arithmetic=%0b stage=%0b phase=%0d input_ready=%0b protocol_error=%0b fifo=%0d result_valid=%0b beat=%0d",
                 busy, arithmetic_active, stage_select, phase_cycle,
                 input_ready, protocol_error, result_fifo_occupancy,
                 result_valid, result_beat);
        $fatal(1, "M30 illegal-shift timeout");
    end
endmodule

`default_nettype wire
