`timescale 1ns/1ps
`default_nettype none

module qfit_dynamic_bn_moment_tile_assertions #(
    parameter int LANES = 16,
    parameter int IN_W = 32,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int POP_GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + POP_GROWTH_W,
    localparam int SUMSQ_W = (2 * IN_W) - 1 + POP_GROWTH_W
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic in_first,
    input logic in_last,
    input logic [COUNT_W-1:0] reduction_population,
    input logic request_legal,
    input logic busy,
    input logic [COUNT_W-1:0] accepted_count,
    input logic [COUNT_W-1:0] active_population,
    input logic protocol_error,
    input logic result_valid,
    input logic result_ready,
    input logic [COUNT_W-1:0] result_count,
    input logic [(LANES*SUM_W)-1:0] result_sum,
    input logic [(LANES*SUMSQ_W)-1:0] result_sumsq
);
    integer legal_seen = 0;
    integer illegal_seen = 0;
    integer final_seen = 0;
    integer input_gap_seen = 0;
    integer output_stall_seen = 0;

    wire input_fire = in_valid && in_ready;

    property p_ready_is_capacity;
        @(posedge clk_core)
            in_ready == (!rst_core && !protocol_error && !result_valid);
    endproperty

    property p_request_legal_is_exact;
        @(posedge clk_core)
            request_legal ==
                (!rst_core && !protocol_error && !result_valid
                 && reduction_population != '0
                 && reduction_population <= MAX_REDUCTION_POPULATION
                 && (!busy
                     ? (in_first &&
                        (in_last == (reduction_population == 1)))
                     : (!in_first
                        && reduction_population == active_population
                        && (in_last ==
                            ((accepted_count + 1'b1) == active_population)))));
    endproperty

    property p_illegal_fire_sets_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            input_fire && !request_legal |=> protocol_error;
    endproperty

    property p_protocol_error_is_sticky;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |=> protocol_error;
    endproperty

    property p_error_blocks_input;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> !in_ready;
    endproperty

    property p_nonfinal_legal_fire_advances_count;
        @(posedge clk_core) disable iff (rst_core)
            input_fire && request_legal && !in_last
            |=> busy && !result_valid
                && accepted_count == ($past(busy) ? $past(accepted_count) + 1'b1 : 1);
    endproperty

    property p_final_legal_fire_publishes_result;
        @(posedge clk_core) disable iff (rst_core)
            input_fire && request_legal && in_last
            |=> result_valid && !busy && accepted_count == '0
                && result_count == $past(reduction_population);
    endproperty

    property p_result_only_follows_legal_last;
        @(posedge clk_core) disable iff (rst_core)
            $rose(result_valid) |->
                $past(input_fire && request_legal && in_last);
    endproperty

    property p_busy_count_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
            busy |-> accepted_count > 0;
    endproperty

    property p_result_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            result_valid && !result_ready |=>
                result_valid && $stable(result_count)
                && $stable(result_sum) && $stable(result_sumsq);
    endproperty

    assert property (p_ready_is_capacity);
    assert property (p_request_legal_is_exact);
    assert property (p_illegal_fire_sets_sticky_error);
    assert property (p_protocol_error_is_sticky);
    assert property (p_error_blocks_input);
    assert property (p_nonfinal_legal_fire_advances_count);
    assert property (p_final_legal_fire_publishes_result);
    assert property (p_result_only_follows_legal_last);
    assert property (p_busy_count_is_bounded);
    assert property (p_result_stable_under_backpressure);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (input_fire && request_legal) begin
                legal_seen <= legal_seen + 1;
                if (in_last)
                    final_seen <= final_seen + 1;
            end
            if (input_fire && !request_legal)
                illegal_seen <= illegal_seen + 1;
            if (!in_valid && in_ready)
                input_gap_seen <= input_gap_seen + 1;
            if (result_valid && !result_ready)
                output_stall_seen <= output_stall_seen + 1;
        end
    end

    final begin
        $display("M20_SVA_COVERAGE legal=%0d illegal=%0d final=%0d input_gap=%0d output_stall=%0d",
                 legal_seen, illegal_seen, final_seen, input_gap_seen,
                 output_stall_seen);
        if (legal_seen <= 0 || illegal_seen <= 0 || final_seen <= 0
            || input_gap_seen <= 0 || output_stall_seen <= 0)
            $error("M20 bound-SVA runtime coverage is incomplete");
    end
endmodule

bind qfit_dynamic_bn_moment_tile
qfit_dynamic_bn_moment_tile_assertions #(
    .LANES(LANES), .IN_W(IN_W),
    .MAX_REDUCTION_POPULATION(MAX_REDUCTION_POPULATION)
) u_qfit_dynamic_bn_moment_tile_assertions (
    .clk_core, .rst_core, .in_valid, .in_ready, .in_first, .in_last,
    .reduction_population, .request_legal, .busy, .accepted_count,
    .active_population,
    .protocol_error, .result_valid, .result_ready, .result_count,
    .result_sum, .result_sumsq
);

`default_nettype wire
