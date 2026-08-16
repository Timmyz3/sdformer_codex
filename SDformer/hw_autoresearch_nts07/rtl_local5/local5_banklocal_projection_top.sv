`timescale 1ns/1ps
`default_nettype none

// Lightweight bank-local projection consumer for Local5 MFEP commands.
// Models DCTF bank-local Acc semantics with multiplicity scaling:
//   Acc[dest][out] += multiplicity * gate_q17 * W[lane][out]
// OUT_DIM kept small for sim; not a full DCTF-96 dual-context fabric.
// Weight ROM is loaded before GO. Final readback dumps Acc rows.
module local5_banklocal_projection_top #(
    parameter int HEAD_DIM  = 32,
    parameter int OUT_DIM   = 4,
    parameter int MAX_DEST  = 16,
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter int ACC_W     = 32,
    parameter int W_W       = 8
) (
    input  logic                  clk_core,
    input  logic                  rst_core,

    // Weight load (blocking before run)
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
    // When set with a cmd (or alone while RUN), finish the window.
    // Allows multi-destination streaming without ending on every term last.
    input  logic                  cmd_window_last,

    // Acc readback
    input  logic                  acc_read_valid,
    output logic                  acc_read_ready,
    input  logic [DEST_W-1:0]     acc_read_dest,
    input  logic [$clog2(OUT_DIM)-1:0] acc_read_out,
    output logic                  acc_data_valid,
    output logic signed [ACC_W-1:0] acc_data,

    output logic                  protocol_error,
    output logic [31:0]           perf_cmd_count,
    output logic [31:0]           perf_product_count
);

    localparam int CLEAR_W = (MAX_DEST <= 1) ? 1 : $clog2(MAX_DEST);

    typedef enum logic [1:0] {
        ST_LOAD  = 2'd0,
        ST_CLEAR = 2'd1,
        ST_RUN   = 2'd2,
        ST_DONE  = 2'd3
    } state_t;

    state_t state_q;
    logic signed [W_W-1:0] weight_q [0:HEAD_DIM-1][0:OUT_DIM-1];
    logic signed [ACC_W-1:0] acc_q [0:MAX_DEST-1][0:OUT_DIM-1];
    logic protocol_error_q;
    logic [31:0] perf_cmd_q;
    logic [31:0] perf_prod_q;
    logic weights_loaded_q;
    logic [CLEAR_W-1:0] clear_dest_q;

    logic dest_in_range;
    logic lane_in_range;
    logic read_dest_in_range;
    logic read_out_in_range;

    assign w_load_ready = (state_q == ST_LOAD);
    assign cmd_ready = (state_q == ST_RUN);
    assign run_busy = (state_q == ST_CLEAR || state_q == ST_RUN);
    assign run_done = (state_q == ST_DONE);
    assign protocol_error = protocol_error_q;
    assign perf_cmd_count = perf_cmd_q;
    assign perf_product_count = perf_prod_q;

    assign dest_in_range = (32'(cmd_destination_token) < MAX_DEST);
    assign lane_in_range = (32'(cmd_lane_id) < HEAD_DIM);
    assign read_dest_in_range = (32'(acc_read_dest) < MAX_DEST);
    assign read_out_in_range = (32'(acc_read_out) < OUT_DIM);

    assign acc_read_ready = (state_q == ST_DONE)
                         && read_dest_in_range && read_out_in_range;
    assign acc_data_valid = acc_read_valid && acc_read_ready;
    assign acc_data = acc_data_valid
                    ? acc_q[acc_read_dest][acc_read_out]
                    : '0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            protocol_error_q <= 1'b0;
            perf_cmd_q <= '0;
            perf_prod_q <= '0;
            weights_loaded_q <= 1'b0;
            clear_dest_q <= '0;
            for (int l = 0; l < HEAD_DIM; l = l + 1) begin
                for (int o = 0; o < OUT_DIM; o = o + 1) begin
                    weight_q[l][o] <= '0;
                end
            end
        end else begin
            if (state_q == ST_DONE && acc_read_valid
                && (!read_dest_in_range || !read_out_in_range))
                protocol_error_q <= 1'b1;
            case (state_q)
                ST_LOAD: begin
                    if (w_load_valid) begin
                        weight_q[w_load_lane][w_load_out] <= w_load_data;
                        if (w_load_last) begin
                            weights_loaded_q <= 1'b1;
                        end
                    end
                    if (run_start && weights_loaded_q) begin
                        clear_dest_q <= '0;
                        perf_cmd_q <= '0;
                        perf_prod_q <= '0;
                        protocol_error_q <= 1'b0;
                        state_q <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    for (int o = 0; o < OUT_DIM; o = o + 1)
                        acc_q[clear_dest_q][o] <= '0;
                    if (32'(clear_dest_q) == MAX_DEST-1) begin
                        state_q <= ST_RUN;
                    end else begin
                        clear_dest_q <= clear_dest_q + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (cmd_valid) begin
                        if (!dest_in_range || !lane_in_range ||
                            cmd_multiplicity == 0 || cmd_multiplicity > MULT_W'(5)) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            perf_cmd_q <= perf_cmd_q + 1'b1;
                            perf_prod_q <= perf_prod_q + 32'(cmd_multiplicity);
                            for (int o = 0; o < OUT_DIM; o = o + 1) begin
                                // mult * gate * W  (gate unsigned Q1.7, W signed)
                                acc_q[cmd_destination_token][o] <=
                                    acc_q[cmd_destination_token][o]
                                    + (ACC_W'(signed'({1'b0, cmd_gate_code}))
                                       * ACC_W'(weight_q[cmd_lane_id][o])
                                       * ACC_W'(cmd_multiplicity));
                            end
                        end
                    end
                    // A zero-term window may close without a command. For a
                    // nonempty window, close only when the final command is
                    // accepted (cmd_ready is high throughout ST_RUN).
                    if (cmd_window_last && (!cmd_valid || cmd_ready))
                        state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (run_start) begin
                        if (weights_loaded_q) begin
                            clear_dest_q <= '0;
                            perf_cmd_q <= '0;
                            perf_prod_q <= '0;
                            protocol_error_q <= 1'b0;
                            state_q <= ST_CLEAR;
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
