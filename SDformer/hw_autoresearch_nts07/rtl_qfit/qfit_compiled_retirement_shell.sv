`timescale 1ns/1ps
`default_nettype none

// Topology-independent ordered shell for compiler-generated retirement rules.
// The rule generator owns geometry and priority; this module owns only
// ready/valid, pending events, and ordered retirement.
module qfit_compiled_retirement_shell #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int RULES = 3,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int COUNT_W = (RULES <= 1) ? 1 : $clog2(RULES + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [RULES-1:0]           rule_valid,
    input  logic [RULES*Y_W-1:0]       rule_y,
    input  logic [RULES*X_W-1:0]       rule_x,
    input  logic [RULES-1:0]           rule_candidate_valid,
    output logic                       retire_valid,
    input  logic                       retire_ready,
    output logic [SOURCE_ID_W-1:0]     retire_source_id,
    output logic [Y_W-1:0]             retire_y,
    output logic [X_W-1:0]             retire_x,
    output logic                       plane_idle,
    output logic [31:0]                perf_producer_stalls,
    output logic [COUNT_W-1:0]         perf_max_pending
);

    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int EVENT_INDEX_W =
        (RULES <= 1) ? 1 : $clog2(RULES);

    logic [PLANE_W-1:0] plane_q;
    logic retire_valid_q;
    logic [SOURCE_ID_W-1:0] retire_source_q;
    logic [Y_W-1:0] retire_y_q;
    logic [X_W-1:0] retire_x_q;
    logic [31:0] stalls_q;
    logic [COUNT_W-1:0] max_pending_q;
    logic [COUNT_W-1:0] pending_count_q;
    logic [SOURCE_ID_W-1:0] pending_id_q [0:RULES-2];
    logic [Y_W-1:0] pending_y_q [0:RULES-2];
    logic [X_W-1:0] pending_x_q [0:RULES-2];
    logic [SOURCE_ID_W-1:0] event_id [0:RULES-1];
    logic [Y_W-1:0] event_y [0:RULES-1];
    logic [X_W-1:0] event_x [0:RULES-1];
    logic [COUNT_W-1:0] event_count;
    logic output_slot_available;

    function automatic logic [SOURCE_ID_W-1:0] make_source_id(
        input logic [PLANE_W-1:0] p,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        int value;
        value = {{(32-PLANE_W){1'b0}}, p} * TOKENS
              + {{(32-Y_W){1'b0}}, y} * WIDTH
              + {{(32-X_W){1'b0}}, x};
        make_source_id = SOURCE_ID_W'(value);
    endfunction

    always_comb begin
        event_count = '0;
        for (int slot = 0; slot < RULES; slot = slot + 1) begin
            event_id[slot] = '0;
            event_y[slot] = '0;
            event_x[slot] = '0;
        end
        for (int rule = 0; rule < RULES; rule = rule + 1) begin
            if (rule_valid[rule] && rule_candidate_valid[rule]) begin
                event_y[event_count[EVENT_INDEX_W-1:0]] =
                    rule_y[rule*Y_W +: Y_W];
                event_x[event_count[EVENT_INDEX_W-1:0]] =
                    rule_x[rule*X_W +: X_W];
                event_id[event_count[EVENT_INDEX_W-1:0]] = make_source_id(
                    plane_q,
                    rule_y[rule*Y_W +: Y_W],
                    rule_x[rule*X_W +: X_W]
                );
                event_count = event_count + COUNT_W'(1);
            end
        end
    end

    assign output_slot_available = !retire_valid_q || retire_ready;
    assign in_ready = !plane_start
                   && pending_count_q == 0
                   && output_slot_available;
    assign retire_valid = retire_valid_q;
    assign retire_source_id = retire_source_q;
    assign retire_y = retire_y_q;
    assign retire_x = retire_x_q;
    assign plane_idle = pending_count_q == 0 && !retire_valid_q;
    assign perf_producer_stalls = stalls_q;
    assign perf_max_pending = max_pending_q;

    always_ff @(posedge clk_core) begin
        if (rst_core || plane_start) begin
            plane_q <= plane_id;
            retire_valid_q <= 1'b0;
            retire_source_q <= '0;
            retire_y_q <= '0;
            retire_x_q <= '0;
            stalls_q <= '0;
            max_pending_q <= '0;
            pending_count_q <= '0;
            for (int slot = 0; slot < RULES-1; slot = slot + 1) begin
                pending_id_q[slot] <= '0;
                pending_y_q[slot] <= '0;
                pending_x_q[slot] <= '0;
            end
        end else begin
            if (retire_valid_q && retire_ready)
                retire_valid_q <= 1'b0;

            if (pending_count_q != 0 && output_slot_available) begin
                retire_valid_q <= 1'b1;
                retire_source_q <= pending_id_q[0];
                retire_y_q <= pending_y_q[0];
                retire_x_q <= pending_x_q[0];
                for (int slot = 0; slot < RULES-2; slot = slot + 1) begin
                    pending_id_q[slot] <= pending_id_q[slot+1];
                    pending_y_q[slot] <= pending_y_q[slot+1];
                    pending_x_q[slot] <= pending_x_q[slot+1];
                end
                pending_id_q[RULES-2] <= '0;
                pending_y_q[RULES-2] <= '0;
                pending_x_q[RULES-2] <= '0;
                pending_count_q <= pending_count_q - COUNT_W'(1);
            end else if (in_valid && in_ready) begin
                if (event_count != 0) begin
                    retire_valid_q <= 1'b1;
                    retire_source_q <= event_id[0];
                    retire_y_q <= event_y[0];
                    retire_x_q <= event_x[0];
                    for (int slot = 0; slot < RULES-1; slot = slot + 1) begin
                        pending_id_q[slot] <= event_id[slot+1];
                        pending_y_q[slot] <= event_y[slot+1];
                        pending_x_q[slot] <= event_x[slot+1];
                    end
                    pending_count_q <= event_count - COUNT_W'(1);
                end
            end

            if (in_valid && !in_ready)
                stalls_q <= stalls_q + 32'd1;
            if (pending_count_q > max_pending_q)
                max_pending_q <= pending_count_q;
            if (
                in_valid
                && in_ready
                && event_count != 0
                && event_count - COUNT_W'(1) > max_pending_q
            )
                max_pending_q <= event_count - COUNT_W'(1);
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (RULES >= 2) else
            $fatal(1, "qfit_compiled_retirement_shell requires RULES>=2");
    end

    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            assert (pending_count_q < COUNT_W'(RULES)) else
                $fatal(1, "compiled retirement pending overflow");
        end
    end
`endif

endmodule

`default_nettype wire
