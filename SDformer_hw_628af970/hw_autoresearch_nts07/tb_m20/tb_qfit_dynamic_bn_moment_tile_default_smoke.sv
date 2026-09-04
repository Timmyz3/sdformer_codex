`timescale 1ns/1ps
`default_nettype none

// Default-parameter elaboration and protocol smoke.  The exhaustive arithmetic
// miter intentionally uses MAX=257 so it can exercise every accumulator bit in
// a short run; this separate top proves that the paper/DC MAX=4,194,304 shape
// elaborates and that its boundary/reset handshakes are live in Synopsys VCS.
module tb_qfit_dynamic_bn_moment_tile_default_smoke;
    localparam int LANES = 16;
    localparam int IN_W = 32;
    localparam int MAX_REDUCTION_POPULATION = 4194304;
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1);
    localparam int POP_GROWTH_W = $clog2(MAX_REDUCTION_POPULATION);
    localparam int SUM_W = IN_W + POP_GROWTH_W;
    localparam int SUMSQ_W = (2 * IN_W) - 1 + POP_GROWTH_W;

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

    always #5 clk_core = ~clk_core;

    date_m20_dynamic_bn_moment_tile_dc_top dut (
        .clk_core, .rst_core, .in_valid, .in_ready, .in_first, .in_last,
        .reduction_population, .in_values, .request_legal, .busy,
        .accepted_count, .active_population, .protocol_error,
        .result_valid, .result_ready, .result_count, .result_sum,
        .result_sumsq
    );

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_first = 1'b0;
        in_last = 1'b0;
        result_ready = 1'b0;
        reduction_population = '0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        #1;
        if (!in_ready || busy || result_valid || protocol_error)
            $fatal(1, "default-parameter reset did not restore idle state");
    endtask

    task automatic fire_beat(
        input bit first_value, input bit last_value, input integer population
    );
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = first_value;
        in_last = last_value;
        reduction_population = population[COUNT_W-1:0];
        #1;
        if (!in_ready || !request_legal)
            $fatal(1, "default-parameter legal beat was rejected");
        @(posedge clk_core);
        @(negedge clk_core);
        in_valid = 1'b0;
        in_first = 1'b0;
        in_last = 1'b0;
    endtask

    initial begin
        if (COUNT_W != 23 || SUM_W != 54 || SUMSQ_W != 85)
            $fatal(1, "default-parameter derived widths drifted");
        for (int lane = 0; lane < LANES; lane++)
            in_values[(lane*IN_W) +: IN_W] = $signed(lane - 8);

        // A representable value above MAX must fail closed.
        apply_reset();
        @(negedge clk_core);
        in_valid = 1'b1;
        in_first = 1'b1;
        in_last = 1'b0;
        reduction_population = MAX_REDUCTION_POPULATION + 1;
        #1;
        if (!in_ready || request_legal)
            $fatal(1, "population above default MAX was not rejected");
        @(posedge clk_core);
        #1;
        if (!protocol_error || in_ready)
            $fatal(1, "above-MAX rejection did not latch fail-closed state");

        // Reset while a population is live must discard all partial moments.
        apply_reset();
        fire_beat(1'b1, 1'b0, 2);
        if (!busy || accepted_count != 1)
            $fatal(1, "default-parameter midflight state was not established");
        apply_reset();

        // Reset is also the explicit cancellation mechanism for a held result.
        fire_beat(1'b1, 1'b1, 1);
        #1;
        if (!result_valid || result_count != 1)
            $fatal(1, "default-parameter result was not published");
        apply_reset();

        // result_ready may be asserted before publication; result_valid still
        // remains observable for exactly one registered cycle.
        @(negedge clk_core);
        result_ready = 1'b1;
        in_valid = 1'b1;
        in_first = 1'b1;
        in_last = 1'b1;
        reduction_population = 1;
        #1;
        if (!request_legal)
            $fatal(1, "early result_ready corrupted request legality");
        @(posedge clk_core);
        #1;
        if (!result_valid || result_count != 1)
            $fatal(1, "early result_ready suppressed the registered result");
        for (int lane = 0; lane < LANES; lane++) begin
            if ($signed(result_sum[(lane*SUM_W) +: SUM_W]) != lane - 8)
                $fatal(1, "default smoke signed sum mismatch lane=%0d", lane);
            if (result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W] != (lane - 8) * (lane - 8))
                $fatal(1, "default smoke sumsq mismatch lane=%0d", lane);
        end
        @(posedge clk_core);
        #1;
        if (result_valid)
            $fatal(1, "early-ready result did not retire after one cycle");

        $display("M20_DEFAULT_SMOKE max_population=%0d count_w=%0d sum_w=%0d sumsq_w=%0d above_max=PASS midflight_reset=PASS result_reset=PASS early_ready=PASS",
                 MAX_REDUCTION_POPULATION, COUNT_W, SUM_W, SUMSQ_W);
        $display("PASS: Synopsys VCS M20 default-parameter elaboration and protocol smoke");
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "M20 default-parameter smoke timeout");
    end
endmodule

`default_nettype wire
