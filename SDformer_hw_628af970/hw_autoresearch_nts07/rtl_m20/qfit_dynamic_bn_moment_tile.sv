`timescale 1ns/1ps

// Exact tile-local first and second raw moments for dynamic BatchNorm.
//
// Scheduling contract: one fixed 16-channel tile remains resident while
// reduction_population vectors are presented.  A wider channel population
// therefore requires channel-tile-major scheduling or an external state
// backing store.  This block does not implement mean division, variance,
// rsqrt, affine normalization, ATLIF, or inter-operator scheduling.
module qfit_dynamic_bn_moment_tile #(
    parameter int LANES = 16,
    parameter int IN_W = 32,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int POP_GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + POP_GROWTH_W,
    localparam int SQUARE_W = (2 * IN_W) - 1,
    localparam int SUMSQ_W = SQUARE_W + POP_GROWTH_W
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              in_valid,
    output logic                              in_ready,
    input  logic                              in_first,
    input  logic                              in_last,
    input  logic [COUNT_W-1:0]                reduction_population,
    input  logic [(LANES*IN_W)-1:0]           in_values,
    output logic                              request_legal,

    output logic                              busy,
    output logic [COUNT_W-1:0]                accepted_count,
    output logic [COUNT_W-1:0]                active_population,
    output logic                              protocol_error,

    output logic                              result_valid,
    input  logic                              result_ready,
    output logic [COUNT_W-1:0]                result_count,
    output logic [(LANES*SUM_W)-1:0]          result_sum,
    output logic [(LANES*SUMSQ_W)-1:0]        result_sumsq
);
    logic busy_q;
    logic [COUNT_W-1:0] accepted_count_q;
    logic [COUNT_W-1:0] population_q;
    logic protocol_error_q;
    logic result_valid_q;
    logic [COUNT_W-1:0] result_count_q;
    logic signed [SUM_W-1:0] sum_q [0:LANES-1];
    logic [SUMSQ_W-1:0] sumsq_q [0:LANES-1];
    logic signed [SUM_W-1:0] final_sum_q [0:LANES-1];
    logic [SUMSQ_W-1:0] final_sumsq_q [0:LANES-1];

    logic capacity_available;
    logic input_fire;
    logic [COUNT_W-1:0] next_count;
    logic population_in_range;

    function automatic logic [SQUARE_W-1:0] exact_square(
        input logic signed [IN_W-1:0] value
    );
        logic [IN_W-1:0] magnitude;
        logic [(2*IN_W)-1:0] product;
        begin
            magnitude = value[IN_W-1] ? (~$unsigned(value) + 1'b1) : $unsigned(value);
            product = magnitude * magnitude;
            // |signed IN_W minimum|^2 is exactly bit 2*IN_W-2, so the
            // mathematical square occupies SQUARE_W bits; product's top bit
            // is provably zero for every IN_W-bit input.
            exact_square = product[SQUARE_W-1:0];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (LANES != 16)
            $fatal(1, "M20 moment tile freezes exactly 16 channels");
        if (IN_W < 2)
            $fatal(1, "M20 IN_W must be at least two signed bits");
        if (MAX_REDUCTION_POPULATION < 1)
            $fatal(1, "M20 maximum reduction population must be positive");
    end
`endif

    assign capacity_available = !rst_core && !protocol_error_q && !result_valid_q;
    assign in_ready = capacity_available;
    assign input_fire = in_valid && in_ready;
    assign next_count = busy_q ? accepted_count_q + {{(COUNT_W-1){1'b0}}, 1'b1}
                               : {{(COUNT_W-1){1'b0}}, 1'b1};
    assign population_in_range =
        reduction_population != {COUNT_W{1'b0}}
        && reduction_population <= MAX_REDUCTION_POPULATION;
    assign request_legal = capacity_available && population_in_range
        && (!busy_q
            ? (in_first && (in_last == (reduction_population == 1)))
            : (!in_first && reduction_population == population_q
               && (in_last == (next_count == population_q))));

    assign busy = busy_q;
    assign accepted_count = accepted_count_q;
    assign active_population = population_q;
    assign protocol_error = protocol_error_q;
    assign result_valid = result_valid_q;
    assign result_count = result_count_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : gen_result_pack
            assign result_sum[(lane*SUM_W) +: SUM_W] = final_sum_q[lane];
            assign result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W] = final_sumsq_q[lane];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : moment_state
        if (rst_core) begin
            busy_q <= 1'b0;
            accepted_count_q <= '0;
            population_q <= '0;
            protocol_error_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_count_q <= '0;
            for (int lane = 0; lane < LANES; lane++) begin
                sum_q[lane] <= '0;
                sumsq_q[lane] <= '0;
                final_sum_q[lane] <= '0;
                final_sumsq_q[lane] <= '0;
            end
        end else begin
            if (result_valid_q && result_ready)
                result_valid_q <= 1'b0;

            if (input_fire && !request_legal) begin
                protocol_error_q <= 1'b1;
                busy_q <= 1'b0;
                accepted_count_q <= '0;
                population_q <= '0;
            end else if (input_fire) begin
                for (int lane = 0; lane < LANES; lane++) begin : accumulate_lanes
                    logic signed [IN_W-1:0] lane_value;
                    logic signed [SUM_W-1:0] lane_value_extended;
                    logic [SQUARE_W-1:0] lane_square;
                    logic [SUMSQ_W-1:0] lane_square_extended;
                    logic signed [SUM_W-1:0] next_sum_value;
                    logic [SUMSQ_W-1:0] next_sumsq_value;

                    lane_value = $signed(in_values[(lane*IN_W) +: IN_W]);
                    lane_value_extended = {{(SUM_W-IN_W){lane_value[IN_W-1]}}, lane_value};
                    lane_square = exact_square(lane_value);
                    lane_square_extended = {
                        {(SUMSQ_W-SQUARE_W){1'b0}}, lane_square
                    };
                    next_sum_value = busy_q
                        ? sum_q[lane] + lane_value_extended : lane_value_extended;
                    next_sumsq_value = busy_q
                        ? sumsq_q[lane] + lane_square_extended : lane_square_extended;

                    sum_q[lane] <= next_sum_value;
                    sumsq_q[lane] <= next_sumsq_value;
                    if (in_last) begin
                        final_sum_q[lane] <= next_sum_value;
                        final_sumsq_q[lane] <= next_sumsq_value;
                    end
                end

                if (!busy_q) begin
                    population_q <= reduction_population;
                    accepted_count_q <= {{(COUNT_W-1){1'b0}}, 1'b1};
                    busy_q <= !in_last;
                end else begin
                    accepted_count_q <= next_count;
                    busy_q <= !in_last;
                end
                if (in_last) begin
                    result_valid_q <= 1'b1;
                    result_count_q <= reduction_population;
                    accepted_count_q <= '0;
                    population_q <= '0;
                end
            end
        end
    end
endmodule
