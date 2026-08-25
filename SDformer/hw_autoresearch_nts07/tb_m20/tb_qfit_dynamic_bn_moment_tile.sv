`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dynamic_bn_moment_tile;
    localparam int LANES = 16;
    localparam int IN_W = 32;
    // 257 deliberately crosses a power-of-two boundary and makes the test
    // exercise every declared accumulator bit with repeated signed extrema.
    localparam int MAX_REDUCTION_POPULATION = 257;
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1);
    localparam int POP_GROWTH_W = $clog2(MAX_REDUCTION_POPULATION);
    localparam int SUM_W = IN_W + POP_GROWTH_W;
    localparam int SQUARE_W = (2 * IN_W) - 1;
    localparam int SUMSQ_W = SQUARE_W + POP_GROWTH_W;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic in_valid = 1'b0;
    logic in_ready;
    logic in_first = 1'b0;
    logic in_last = 1'b0;
    logic [COUNT_W-1:0] reduction_population = '0;
    logic [(LANES*IN_W)-1:0] in_values = '0;
    logic request_legal;
    logic busy;
    logic [COUNT_W-1:0] accepted_count;
    logic [COUNT_W-1:0] active_population;
    logic protocol_error;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [COUNT_W-1:0] result_count;
    logic [(LANES*SUM_W)-1:0] result_sum;
    logic [(LANES*SUMSQ_W)-1:0] result_sumsq;

    logic signed [IN_W-1:0] stimulus [0:LANES-1];
    logic signed [SUM_W-1:0] reference_sum [0:LANES-1];
    logic [SUMSQ_W-1:0] reference_sumsq [0:LANES-1];

    integer transaction_count = 0;
    integer legal_beat_count = 0;
    integer illegal_count = 0;
    integer output_stall_count = 0;
    integer input_gap_count = 0;
    integer fixed_populations [0:9];

    always #5 clk_core = ~clk_core;

    qfit_dynamic_bn_moment_tile #(
        .LANES(LANES), .IN_W(IN_W),
        .MAX_REDUCTION_POPULATION(MAX_REDUCTION_POPULATION)
    ) dut (
        .clk_core, .rst_core, .in_valid, .in_ready, .in_first, .in_last,
        .reduction_population, .in_values, .request_legal, .busy,
        .accepted_count, .active_population, .protocol_error,
        .result_valid, .result_ready, .result_count, .result_sum,
        .result_sumsq
    );

    function automatic logic [SQUARE_W-1:0] reference_square(
        input logic signed [IN_W-1:0] value
    );
        // Deliberately independent of the DUT's magnitude-square
        // implementation: sign-extend into a 2*IN_W big integer and retain a
        // 4*IN_W multiplication result before checking the mathematical bound.
        logic signed [(2*IN_W)-1:0] wide_value;
        logic signed [(4*IN_W)-1:0] wide_product;
        begin
            wide_value = {{IN_W{value[IN_W-1]}}, value};
            wide_product = wide_value * wide_value;
            if (|wide_product[(4*IN_W)-1:SQUARE_W])
                $fatal(1, "reference square exceeded mathematical SQUARE_W bound");
            reference_square = wide_product[SQUARE_W-1:0];
        end
    endfunction

    task automatic pack_stimulus;
        for (int lane = 0; lane < LANES; lane++)
            in_values[(lane*IN_W) +: IN_W] = stimulus[lane];
    endtask

    task automatic clear_reference;
        for (int lane = 0; lane < LANES; lane++) begin
            reference_sum[lane] = '0;
            reference_sumsq[lane] = '0;
        end
    endtask

    task automatic update_reference(input bit first_beat);
        logic signed [SUM_W-1:0] value_extended;
        logic [SUMSQ_W-1:0] square_extended;
        for (int lane = 0; lane < LANES; lane++) begin
            value_extended = {{(SUM_W-IN_W){stimulus[lane][IN_W-1]}}, stimulus[lane]};
            square_extended = {
                {(SUMSQ_W-SQUARE_W){1'b0}}, reference_square(stimulus[lane])
            };
            if (first_beat) begin
                reference_sum[lane] = value_extended;
                reference_sumsq[lane] = square_extended;
            end else begin
                reference_sum[lane] = reference_sum[lane] + value_extended;
                reference_sumsq[lane] = reference_sumsq[lane] + square_extended;
            end
        end
    endtask

    task automatic fill_stimulus(input int mode, input int beat);
        for (int lane = 0; lane < LANES; lane++) begin
            case (mode)
                1: stimulus[lane] = {1'b1, {(IN_W-1){1'b0}}};
                2: stimulus[lane] = {1'b0, {(IN_W-1){1'b1}}};
                default: begin
                    stimulus[lane] = $urandom;
                    if (((beat + lane) % 19) == 0)
                        stimulus[lane] = {1'b1, {(IN_W-1){1'b0}}};
                    else if (((beat + lane) % 23) == 0)
                        stimulus[lane] = {1'b0, {(IN_W-1){1'b1}}};
                end
            endcase
        end
        pack_stimulus();
    endtask

    task automatic reset_dut;
        @(negedge clk_core);
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_first = 1'b0;
        in_last = 1'b0;
        reduction_population = '0;
        in_values = '0;
        result_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        #1;
        if (!in_ready || busy || accepted_count != 0 || result_valid || protocol_error)
            $fatal(1, "reset did not return M20 tile to its idle contract");
    endtask

    task automatic send_legal_beat(
        input bit first_beat, input bit last_beat,
        input int population, input int gap_cycles
    );
        repeat (gap_cycles) begin
            @(negedge clk_core);
            in_valid = 1'b0;
            input_gap_count = input_gap_count + 1;
            @(posedge clk_core);
        end
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = first_beat;
        in_last = last_beat;
        reduction_population = population[COUNT_W-1:0];
        #1;
        if (!in_ready || !request_legal)
            $fatal(1, "legal M20 beat was not admitted first=%0d last=%0d count=%0d pop=%0d",
                   first_beat, last_beat, accepted_count, population);
        @(posedge clk_core);
        legal_beat_count = legal_beat_count + 1;
        update_reference(first_beat);
        @(negedge clk_core);
        in_valid = 1'b0;
        in_first = 1'b0;
        in_last = 1'b0;
    endtask

    task automatic check_result;
        if (!result_valid)
            $fatal(1, "M20 result disappeared before comparison");
        for (int lane = 0; lane < LANES; lane++) begin
            if ($signed(result_sum[(lane*SUM_W) +: SUM_W]) !== reference_sum[lane])
                $fatal(1, "M20 signed-sum mismatch lane=%0d got=%0h expected=%0h",
                       lane, result_sum[(lane*SUM_W) +: SUM_W], reference_sum[lane]);
            if (result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W] !== reference_sumsq[lane])
                $fatal(1, "M20 unsigned-sumsq mismatch lane=%0d got=%0h expected=%0h",
                       lane, result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W],
                       reference_sumsq[lane]);
        end
    endtask

    task automatic retire_result(input int population, input int stall_cycles);
        // Present a stable next request while result backpressure is active.
        // It must not fire and must not be misclassified as a protocol error.
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = 1'b1;
        in_last = 1'b1;
        reduction_population = 1;
        fill_stimulus(0, population + 1);
        result_ready = 1'b0;
        repeat (2) begin
            @(posedge clk_core);
            #1;
            check_result();
            if (in_ready || protocol_error)
                $fatal(1, "result backpressure did not block input cleanly");
            output_stall_count = output_stall_count + 1;
        end
        @(negedge clk_core);
        in_valid = 1'b0;
        repeat (stall_cycles) begin
            @(posedge clk_core);
            #1;
            check_result();
            output_stall_count = output_stall_count + 1;
        end
        @(negedge clk_core);
        check_result();
        if (result_count != population[COUNT_W-1:0])
            $fatal(1, "M20 result count mismatch got=%0d expected=%0d",
                   result_count, population);
        result_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        result_ready = 1'b0;
        if (result_valid || protocol_error || !in_ready)
            $fatal(1, "M20 result handshake did not return to idle");
    endtask

    task automatic run_transaction(
        input int population, input int mode, input int stall_cycles
    );
        clear_reference();
        for (int beat = 0; beat < population; beat++) begin
            fill_stimulus(mode, beat);
            send_legal_beat(
                beat == 0, beat == population - 1, population,
                $urandom_range(0, 2)
            );
        end
        retire_result(population, stall_cycles);
        transaction_count = transaction_count + 1;
    endtask

    task automatic expect_idle_illegal(
        input bit first_value, input bit last_value, input int population
    );
        reset_dut();
        fill_stimulus(0, illegal_count);
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = first_value;
        in_last = last_value;
        reduction_population = population[COUNT_W-1:0];
        #1;
        if (!in_ready || request_legal)
            $fatal(1, "idle illegal request was not classified as a rejection");
        @(posedge clk_core);
        #1;
        if (!protocol_error || in_ready || busy || result_valid)
            $fatal(1, "idle illegal request did not latch protocol_error");
        illegal_count = illegal_count + 1;
        @(negedge clk_core);
        in_valid = 1'b0;
        @(posedge clk_core);
        #1;
        if (!protocol_error)
            $fatal(1, "M20 protocol_error is not sticky");
    endtask

    task automatic start_population(input int population);
        clear_reference();
        fill_stimulus(0, 0);
        send_legal_beat(1'b1, 1'b0, population, 0);
    endtask

    task automatic expect_active_illegal(
        input bit first_value, input bit last_value, input int population
    );
        fill_stimulus(0, illegal_count + 11);
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = first_value;
        in_last = last_value;
        reduction_population = population[COUNT_W-1:0];
        #1;
        if (!in_ready || request_legal)
            $fatal(1, "active illegal request was not classified as a rejection");
        @(posedge clk_core);
        #1;
        if (!protocol_error || in_ready || busy || result_valid)
            $fatal(1, "active illegal request did not latch protocol_error");
        illegal_count = illegal_count + 1;
        @(negedge clk_core);
        in_valid = 1'b0;
        @(posedge clk_core);
        #1;
        if (!protocol_error)
            $fatal(1, "active protocol_error is not sticky");
    endtask

    initial begin
        $display("SIMULATOR=Synopsys VCS");
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M20 test requires bound SVA runtime");
`endif
        fixed_populations[0] = 1;
        fixed_populations[1] = 2;
        fixed_populations[2] = 3;
        fixed_populations[3] = 7;
        fixed_populations[4] = 16;
        fixed_populations[5] = 31;
        fixed_populations[6] = 64;
        fixed_populations[7] = 79;
        fixed_populations[8] = 127;
        fixed_populations[9] = 257;

        reset_dut();
        for (int index = 0; index < 10; index++)
            run_transaction(fixed_populations[index], 0, (index % 4) + 1);
        run_transaction(MAX_REDUCTION_POPULATION, 1, 5);
        run_transaction(MAX_REDUCTION_POPULATION, 2, 6);

        expect_idle_illegal(1'b0, 1'b0, 3); // missing first
        expect_idle_illegal(1'b1, 1'b0, 0); // zero population
        expect_idle_illegal(1'b1, 1'b1, 3); // early last at first

        reset_dut();
        start_population(3);
        expect_active_illegal(1'b1, 1'b0, 3); // repeated first
        reset_dut();
        start_population(3);
        expect_active_illegal(1'b0, 1'b0, 4); // population changed
        reset_dut();
        start_population(3);
        expect_active_illegal(1'b0, 1'b1, 3); // early last
        reset_dut();
        start_population(2);
        expect_active_illegal(1'b0, 1'b0, 2); // missing required last

        // 1101 admitted miter beats plus four legal setup beats for active
        // protocol-rejection cases.
        if (transaction_count != 12 || legal_beat_count != 1105 || illegal_count != 7)
            $fatal(1, "M20 test population drift transactions=%0d legal=%0d illegal=%0d",
                   transaction_count, legal_beat_count, illegal_count);
        if (output_stall_count <= 0 || input_gap_count <= 0)
            $fatal(1, "M20 randomized backpressure/gap coverage is empty");
        $display("M20_RESULT transactions=%0d legal_beats=%0d illegal=%0d output_stalls=%0d input_gaps=%0d",
                 transaction_count, legal_beat_count, illegal_count,
                 output_stall_count, input_gap_count);
        $display("PASS: Synopsys VCS M20 exact 16-lane dynamic-BN moment tile miter");
        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "M20 simulation timeout");
    end
endmodule

`default_nettype wire
