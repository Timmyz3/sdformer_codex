`timescale 1ns/1ps
`default_nettype none

module qfit_score_leaf_assertions #(
    parameter bit ARCH_QFSA = 1'b1,
    parameter bit PIPE_COMPACTOR = 1'b0,
    parameter bit XBF_BANKED = 1'b0,
    parameter bit USE_BANK_PRESSURE_ROUTE = 1'b0,
    parameter int BANK_PRESSURE_THRESHOLD = 2,
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic [5*SCORE_W-1:0] out_score_q7,
    input logic [5*GATE_W-1:0] out_gate_q17,
    input logic [15:0] perf_service_cycles
);

    localparam int RESIDUAL_BOUND =
        4 * BANK_PRESSURE_THRESHOLD + (PIPE_COMPACTOR ? 1 : 0);
    localparam int SERVICE_BOUND =
        RESIDUAL_BOUND > 4 ? RESIDUAL_BOUND : 4;

    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && !out_ready
            |=> out_valid
                && $stable(out_tag)
                && $stable(out_score_q7)
                && $stable(out_gate_q17)
                && $stable(perf_service_cycles);
    endproperty

    assert property (p_output_stable_under_backpressure);

    generate
        if (
            ARCH_QFSA
            && XBF_BANKED
            && USE_BANK_PRESSURE_ROUTE
        ) begin : g_dbdr_bound
            property p_dbdr_service_bound;
                @(posedge clk_core) disable iff (rst_core)
                    out_valid
                    |-> perf_service_cycles <= 16'(SERVICE_BOUND);
            endproperty

            assert property (p_dbdr_service_bound);
        end
    endgenerate

endmodule

bind qfit_local5_score_leaf qfit_score_leaf_assertions #(
    .ARCH_QFSA(ARCH_QFSA),
    .PIPE_COMPACTOR(PIPE_COMPACTOR),
    .XBF_BANKED(XBF_BANKED),
    .USE_BANK_PRESSURE_ROUTE(USE_BANK_PRESSURE_ROUTE),
    .BANK_PRESSURE_THRESHOLD(BANK_PRESSURE_THRESHOLD),
    .TAG_W(TAG_W),
    .SCORE_W(SCORE_W),
    .GATE_W(GATE_W)
) u_qfit_score_leaf_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_tag(out_tag),
    .out_score_q7(out_score_q7),
    .out_gate_q17(out_gate_q17),
    .perf_service_cycles(perf_service_cycles)
);

`default_nettype wire
