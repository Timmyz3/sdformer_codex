`timescale 1ns/1ps
`default_nettype none

// Local5-owned multi-bank projection (DCTF-like bank-local Acc).
// Does NOT modify rtl_hitflow. NUM_BANKS parallel issue when dest hashes differ.
// Acc[dest][out] += mult * gate * W[lane][out]
module local5_multibank_projection_top #(
    parameter int HEAD_DIM  = 32,
    parameter int OUT_DIM   = 4,
    parameter int MAX_DEST  = 32,
    parameter int NUM_BANKS = 3,
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter int ACC_W     = 32,
    parameter int W_W       = 8,
    parameter int BANK_W    = (NUM_BANKS <= 1) ? 1 : $clog2(NUM_BANKS)
) (
    input  logic                  clk_core,
    input  logic                  rst_core,

    input  logic                  w_load_valid,
    output logic                  w_load_ready,
    input  logic [LANE_ID_W-1:0]  w_load_lane,
    input  logic [$clog2(OUT_DIM)-1:0] w_load_out,
    input  logic signed [W_W-1:0] w_load_data,
    input  logic                  w_load_last,

    input  logic                  run_start,
    output logic                  run_busy,
    output logic                  run_done,

    input  logic                  cmd_valid,
    output logic                  cmd_ready,
    input  logic [TAG_W-1:0]      cmd_group_tag,
    input  logic [GATE_W-1:0]     cmd_gate_code,
    input  logic [LANE_ID_W-1:0]  cmd_lane_id,
    input  logic [DEST_W-1:0]     cmd_destination_token,
    input  logic [MULT_W-1:0]     cmd_multiplicity,
    input  logic                  cmd_head_last,
    input  logic                  cmd_window_last,

    input  logic                  acc_read_valid,
    output logic                  acc_read_ready,
    input  logic [DEST_W-1:0]     acc_read_dest,
    input  logic [$clog2(OUT_DIM)-1:0] acc_read_out,
    output logic                  acc_data_valid,
    output logic signed [ACC_W-1:0] acc_data,

    output logic                  protocol_error,
    output logic [31:0]           perf_cmd_count,
    output logic [31:0]           perf_product_count,
    output logic [31:0]           perf_bank_conflict_count
);

    typedef enum logic [1:0] {
        ST_LOAD = 2'd0,
        ST_RUN  = 2'd1,
        ST_DONE = 2'd2
    } state_t;

    state_t state_q;
    logic signed [W_W-1:0] weight_q [0:HEAD_DIM-1][0:OUT_DIM-1];
    // Per-bank accumulators: bank holds dests where dest % NUM_BANKS == bank
    logic signed [ACC_W-1:0] acc_q [0:MAX_DEST-1][0:OUT_DIM-1];
    logic protocol_error_q;
    logic [31:0] perf_cmd_q, perf_prod_q, perf_conflict_q;
    logic weights_loaded_q;

    // Single-issue with bank busy mask for conflict accounting (1 IPC, N banks)
    // When same bank consecutive → conflict stall cycle modeled by not accepting
    logic [NUM_BANKS-1:0] bank_busy_q;
    logic [BANK_W-1:0] last_bank_q;

    function automatic logic [BANK_W-1:0] dest_bank(input logic [DEST_W-1:0] d);
        return BANK_W'(32'(d) % NUM_BANKS);
    endfunction

    assign w_load_ready = (state_q == ST_LOAD);
    wire [BANK_W-1:0] cmd_bank = dest_bank(cmd_destination_token);
    wire bank_free = !bank_busy_q[cmd_bank];
    assign cmd_ready = (state_q == ST_RUN) && bank_free;
    assign run_busy = (state_q == ST_RUN);
    assign run_done = (state_q == ST_DONE);
    assign protocol_error = protocol_error_q;
    assign perf_cmd_count = perf_cmd_q;
    assign perf_product_count = perf_prod_q;
    assign perf_bank_conflict_count = perf_conflict_q;

    assign acc_read_ready = 1'b1;
    assign acc_data_valid = acc_read_valid;
    assign acc_data = (32'(acc_read_dest) < MAX_DEST)
                    ? acc_q[int'(acc_read_dest)][int'(acc_read_out)] : '0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            protocol_error_q <= 1'b0;
            perf_cmd_q <= '0;
            perf_prod_q <= '0;
            perf_conflict_q <= '0;
            weights_loaded_q <= 1'b0;
            bank_busy_q <= '0;
            last_bank_q <= '0;
            for (int l = 0; l < HEAD_DIM; l++)
                for (int o = 0; o < OUT_DIM; o++)
                    weight_q[l][o] <= '0;
            for (int d = 0; d < MAX_DEST; d++)
                for (int o = 0; o < OUT_DIM; o++)
                    acc_q[d][o] <= '0;
        end else begin
            unique case (state_q)
                ST_LOAD: begin
                    bank_busy_q <= '0;
                    if (w_load_valid) begin
                        weight_q[w_load_lane][w_load_out] <= w_load_data;
                        if (w_load_last) weights_loaded_q <= 1'b1;
                    end
                    if (run_start && weights_loaded_q) begin
                        for (int d = 0; d < MAX_DEST; d++)
                            for (int o = 0; o < OUT_DIM; o++)
                                acc_q[d][o] <= '0;
                        perf_cmd_q <= '0;
                        perf_prod_q <= '0;
                        perf_conflict_q <= '0;
                        protocol_error_q <= 1'b0;
                        state_q <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    // Default: free all banks next cycle unless we accept below
                    begin
                        logic [NUM_BANKS-1:0] next_busy;
                        next_busy = '0;
                        if (cmd_valid && !bank_free) begin
                            perf_conflict_q <= perf_conflict_q + 1'b1;
                        end
                        if (cmd_valid && cmd_ready) begin
                            if (32'(cmd_destination_token) >= MAX_DEST ||
                                32'(cmd_lane_id) >= HEAD_DIM ||
                                cmd_multiplicity == 0 ||
                                cmd_multiplicity > MULT_W'(5)) begin
                                protocol_error_q <= 1'b1;
                            end else begin
                                perf_cmd_q <= perf_cmd_q + 1'b1;
                                perf_prod_q <= perf_prod_q + 32'(cmd_multiplicity);
                                next_busy[cmd_bank] = 1'b1;
                                last_bank_q <= cmd_bank;
                                for (int o = 0; o < OUT_DIM; o++) begin
                                    acc_q[int'(cmd_destination_token)][o] <=
                                        acc_q[int'(cmd_destination_token)][o]
                                        + (ACC_W'(signed'({1'b0, cmd_gate_code}))
                                           * ACC_W'(weight_q[cmd_lane_id][o])
                                           * ACC_W'(cmd_multiplicity));
                                end
                            end
                            if (cmd_window_last) state_q <= ST_DONE;
                        end else if (cmd_window_last && !cmd_valid) begin
                            state_q <= ST_DONE;
                        end
                        bank_busy_q <= next_busy;
                    end
                end

                ST_DONE: begin
                    bank_busy_q <= '0;
                    // Re-arm straight into RUN when weights already loaded
                    if (run_start) begin
                        if (weights_loaded_q) begin
                            for (int d = 0; d < MAX_DEST; d++)
                                for (int o = 0; o < OUT_DIM; o++)
                                    acc_q[d][o] <= '0;
                            perf_cmd_q <= '0;
                            perf_prod_q <= '0;
                            perf_conflict_q <= '0;
                            protocol_error_q <= 1'b0;
                            state_q <= ST_RUN;
                        end else begin
                            state_q <= ST_LOAD;
                        end
                    end
                end

                default: state_q <= ST_LOAD;
            endcase
        end
    end

endmodule

`default_nettype wire
