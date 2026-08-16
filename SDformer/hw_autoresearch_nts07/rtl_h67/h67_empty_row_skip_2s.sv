`timescale 1ns/1ps
`default_nettype none

// Exact empty-K row skip around RQTB2S. row_k_present is a stored occupancy
// bit produced when K is written; it does not change gated-K semantics.
module h67_empty_row_skip_2s #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int TOKEN_W = $clog2(2 * PAIRS + 1),
    parameter int SLOT_FIFO_DEPTH = 32,
    parameter int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1),
    parameter bit QUOTIENT_ENABLE = 1'b1
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       row_k_present,
    input  logic                       window_seal,
    input  logic                       descriptor_issue_enable,
    input  logic                       cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]     cfg_threshold_q8,
    output logic                       seal_ready,
    output logic                       window_done,
    input  logic                       pair_valid,
    output logic                       pair_ready,
    input  logic [PAIR_ID_W-1:0]       pair_id,
    input  logic [2*HEAD_DIM-1:0]      q_pair,
    input  logic [2*HEAD_DIM-1:0]      k_pair,
    output logic                       out_valid,
    input  logic                       out_ready,
    output logic                       out_last,
    output logic [TOKEN_W-1:0]         out_token_id,
    output logic [HEAD_DIM-1:0]        out_k_bits,
    output logic [GATE_W-1:0]          out_gate_q17,
    output logic                       protocol_error,
    output logic [31:0]                perf_total_cycles,
    output logic [31:0]                perf_skipped_rows,
    output logic                       skipped_row
);

    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_SKIP = 2'd1,
        ST_RUN  = 2'd2
    } state_t;

    state_t state_q;
    logic core_start;
    logic core_seal;
    logic core_seal_ready;
    logic core_done;
    logic core_pair_ready;
    logic core_out_valid;
    logic core_out_last;
    logic [TOKEN_W-1:0] core_out_token;
    logic [HEAD_DIM-1:0] core_out_k;
    logic [GATE_W-1:0] core_out_gate;
    logic core_error;
    logic [31:0] core_cycles;
    logic [31:0] skip_cycles_q;
    logic [31:0] skipped_rows_q;

    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W),
        .THRESHOLD_W(THRESHOLD_W),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
        .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(QUOTIENT_ENABLE)
    ) u_core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(core_start),
        .window_seal(core_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(cfg_preserve_mean),
        .cfg_threshold_q8(cfg_threshold_q8),
        .seal_ready(core_seal_ready),
        .window_done(core_done),
        .pair_valid(pair_valid && (state_q == ST_RUN)),
        .pair_ready(core_pair_ready),
        .pair_id(pair_id),
        .q_pair(q_pair),
        .k_pair(k_pair),
        .out_valid(core_out_valid),
        .out_ready(out_ready),
        .out_last(core_out_last),
        .out_token_id(core_out_token),
        .out_k_bits(core_out_k),
        .out_gate_q17(core_out_gate),
        .out_threshold_q8(),
        .protocol_error(core_error),
        .perf_pairs(),
        .perf_slots(),
        .perf_equal_pairs(),
        .perf_quotient_descriptors(),
        .perf_original_tokens(),
        .perf_active_entries(),
        .perf_class_transactions(),
        .perf_exp_transactions(),
        .perf_emitted_tokens(),
        .perf_k_read_transactions(),
        .perf_k_read_bits(),
        .perf_total_cycles(core_cycles),
        .perf_pair_stall_cycles(),
        .perf_descriptor_stall_cycles(),
        .perf_output_stall_cycles(),
        .perf_fifo_occupancy(),
        .perf_fifo_max_occupancy()
    );

    assign core_start = window_start && row_k_present;
    assign core_seal = window_seal && (state_q == ST_RUN);
    assign skipped_row = (state_q == ST_SKIP);
    assign seal_ready = (state_q == ST_SKIP) || core_seal_ready;
    assign window_done = (state_q == ST_SKIP) || core_done;
    assign pair_ready = (state_q == ST_RUN) && core_pair_ready;
    assign out_valid = (state_q == ST_RUN) && core_out_valid;
    assign out_last = core_out_last;
    assign out_token_id = core_out_token;
    assign out_k_bits = core_out_k;
    assign out_gate_q17 = core_out_gate;
    assign protocol_error = core_error || (pair_valid && state_q == ST_SKIP);
    assign perf_total_cycles = (state_q == ST_SKIP) ? skip_cycles_q : core_cycles;
    assign perf_skipped_rows = skipped_rows_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            skip_cycles_q <= '0;
            skipped_rows_q <= '0;
        end else if (window_start) begin
            skip_cycles_q <= 32'd1;
            if (row_k_present)
                state_q <= ST_RUN;
            else begin
                state_q <= ST_SKIP;
                skipped_rows_q <= skipped_rows_q + 32'd1;
            end
        end else if (state_q == ST_SKIP) begin
            skip_cycles_q <= skip_cycles_q + 32'd1;
            if (window_seal)
                state_q <= ST_IDLE;
        end else if (state_q == ST_RUN && core_done && !window_start)
            state_q <= ST_IDLE;
    end
endmodule

`default_nettype wire
