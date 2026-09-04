`timescale 1ns/1ps
`default_nettype none

// Strong destination-major baseline: precompute one 6-bit popcount per K when
// it enters a three-row stripe, then gather five narrow statistics per query.
// This module deliberately excludes Shiftmax and the residual Q!=0 path.
module qfit_local5_qsilent_popcount_sidecar #(
    parameter int ROW_TOKENS = 15,
    parameter int X_W = (ROW_TOKENS <= 1) ? 1 : $clog2(ROW_TOKENS)
) (
    input  logic                 clk_core,
    input  logic                 rst_core,
    input  logic                 write_valid,
    output logic                 write_ready,
    input  logic [1:0]           write_row_sel,
    input  logic [X_W-1:0]       write_x,
    input  logic [31:0]          write_k,
    input  logic                 write_token_valid,
    input  logic                 read_valid,
    output logic                 read_ready,
    input  logic [X_W-1:0]       read_x,
    output logic                 rsp_valid,
    input  logic                 rsp_ready,
    output logic [5*6-1:0]       rsp_popcount,
    output logic [4:0]           rsp_valid_mask,
    output logic                 protocol_error
);

    logic [5:0] pop_prev [0:ROW_TOKENS-1];
    logic [5:0] pop_curr [0:ROW_TOKENS-1];
    logic [5:0] pop_next [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] valid_prev_q;
    logic [ROW_TOKENS-1:0] valid_curr_q;
    logic [ROW_TOKENS-1:0] valid_next_q;
    logic hold_q;
    logic [29:0] rsp_pop_q;
    logic [4:0] rsp_mask_q;
    logic error_q;
    logic [5:0] write_pop_w;

    function automatic logic [5:0] popcount32(input logic [31:0] bits);
        logic [5:0] count;
        count = '0;
        for (int lane = 0; lane < 32; lane = lane + 1)
            count = count + 6'(bits[lane]);
        popcount32 = count;
    endfunction

    assign write_pop_w = popcount32(write_k);
    assign write_ready = 1'b1;
    assign read_ready = !hold_q || rsp_ready;
    assign rsp_valid = hold_q;
    assign rsp_popcount = rsp_pop_q;
    assign rsp_valid_mask = rsp_mask_q;
    assign protocol_error = error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_prev_q <= '0;
            valid_curr_q <= '0;
            valid_next_q <= '0;
            hold_q <= 1'b0;
            rsp_pop_q <= '0;
            rsp_mask_q <= '0;
            error_q <= 1'b0;
            for (int idx = 0; idx < ROW_TOKENS; idx = idx + 1) begin
                pop_prev[idx] <= '0;
                pop_curr[idx] <= '0;
                pop_next[idx] <= '0;
            end
        end else begin
            if (write_valid) begin
                if ((write_row_sel > 2) || (32'(write_x) >= ROW_TOKENS)) begin
                    error_q <= 1'b1;
                end else begin
                    unique case (write_row_sel)
                        0: begin
                            pop_prev[write_x] <= write_pop_w;
                            valid_prev_q[write_x] <= write_token_valid;
                        end
                        1: begin
                            pop_curr[write_x] <= write_pop_w;
                            valid_curr_q[write_x] <= write_token_valid;
                        end
                        default: begin
                            pop_next[write_x] <= write_pop_w;
                            valid_next_q[write_x] <= write_token_valid;
                        end
                    endcase
                end
            end

            if (hold_q && rsp_ready)
                hold_q <= 1'b0;
            if (read_valid && read_ready) begin
                if (32'(read_x) >= ROW_TOKENS) begin
                    error_q <= 1'b1;
                    hold_q <= 1'b1;
                    rsp_pop_q <= '0;
                    rsp_mask_q <= '0;
                end else begin
                    hold_q <= 1'b1;
                    rsp_pop_q[0*6 +: 6] <= pop_curr[read_x];
                    rsp_pop_q[1*6 +: 6] <= pop_prev[read_x];
                    rsp_pop_q[2*6 +: 6] <= pop_next[read_x];
                    rsp_pop_q[3*6 +: 6] <=
                        (32'(read_x) + 1 < ROW_TOKENS)
                        ? pop_curr[read_x + X_W'(1)] : 6'd0;
                    rsp_pop_q[4*6 +: 6] <=
                        (read_x != '0)
                        ? pop_curr[read_x - X_W'(1)] : 6'd0;
                    rsp_mask_q[0] <= valid_curr_q[read_x];
                    rsp_mask_q[1] <= valid_prev_q[read_x];
                    rsp_mask_q[2] <= valid_next_q[read_x];
                    rsp_mask_q[3] <= (32'(read_x) + 1 < ROW_TOKENS)
                                          && valid_curr_q[read_x + X_W'(1)];
                    rsp_mask_q[4] <= (read_x != '0)
                                          && valid_curr_q[read_x - X_W'(1)];
                end
            end
        end
    end
endmodule

`default_nettype wire
