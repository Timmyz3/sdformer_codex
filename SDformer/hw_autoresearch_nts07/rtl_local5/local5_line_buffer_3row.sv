`timescale 1ns/1ps
`default_nettype none

// Three-row K/Q line buffer for Local5 source-stationary stencil.
// Rows are addressed relatively as prev/curr/next (N/self/S geometric).
// Does not modify Motion storage; Local5-only front-end SRAM model.
module local5_line_buffer_3row #(
    parameter int HEAD_DIM   = 32,
    parameter int ROW_TOKENS = 18,  // typical window width fragment
    parameter int TOKEN_W    = (ROW_TOKENS <= 1) ? 1 : $clog2(ROW_TOKENS),
    parameter int TAG_W      = 16
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    // Shift in a new image row (push: prev<-curr, curr<-next, next<-data)
    input  logic                     row_push_valid,
    output logic                     row_push_ready,
    input  logic [TAG_W-1:0]         row_push_tag,
    input  logic [HEAD_DIM-1:0]      row_push_q [0:ROW_TOKENS-1],
    input  logic [HEAD_DIM-1:0]      row_push_k [0:ROW_TOKENS-1],
    input  logic [ROW_TOKENS-1:0]    row_push_valid_mask,

    // Random-access read for stencil assembly
    input  logic                     rd_valid,
    output logic                     rd_ready,
    input  logic [1:0]               rd_row_sel,   // 0=prev,1=curr,2=next
    input  logic [TOKEN_W-1:0]       rd_x,
    output logic                     rd_rsp_valid,
    input  logic                     rd_rsp_ready,
    output logic [HEAD_DIM-1:0]      rd_q_bits,
    output logic [HEAD_DIM-1:0]      rd_k_bits,
    output logic                     rd_token_valid,

    output logic [TAG_W-1:0]         curr_row_tag,
    output logic [1:0]               rows_filled,  // 0..3
    output logic                     protocol_error
);

    logic [HEAD_DIM-1:0] q_prev [0:ROW_TOKENS-1];
    logic [HEAD_DIM-1:0] k_prev [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] v_prev;
    logic [HEAD_DIM-1:0] q_curr [0:ROW_TOKENS-1];
    logic [HEAD_DIM-1:0] k_curr [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] v_curr;
    logic [HEAD_DIM-1:0] q_next [0:ROW_TOKENS-1];
    logic [HEAD_DIM-1:0] k_next [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] v_next;

    logic [TAG_W-1:0] tag_prev_q, tag_curr_q, tag_next_q;
    logic [1:0] filled_q;
    logic protocol_error_q;

    logic rsp_hold_q;
    logic [HEAD_DIM-1:0] rsp_q_q, rsp_k_q;
    logic rsp_v_q;

    assign row_push_ready = 1'b1;
    assign rd_ready = !rsp_hold_q || rd_rsp_ready;
    assign rd_rsp_valid = rsp_hold_q;
    assign rd_q_bits = rsp_q_q;
    assign rd_k_bits = rsp_k_q;
    assign rd_token_valid = rsp_v_q;
    assign curr_row_tag = tag_curr_q;
    assign rows_filled = filled_q;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            filled_q <= 2'd0;
            protocol_error_q <= 1'b0;
            v_prev <= '0;
            v_curr <= '0;
            v_next <= '0;
            tag_prev_q <= '0;
            tag_curr_q <= '0;
            tag_next_q <= '0;
            rsp_hold_q <= 1'b0;
            rsp_q_q <= '0;
            rsp_k_q <= '0;
            rsp_v_q <= 1'b0;
            for (int i = 0; i < ROW_TOKENS; i = i + 1) begin
                q_prev[i] <= '0; k_prev[i] <= '0;
                q_curr[i] <= '0; k_curr[i] <= '0;
                q_next[i] <= '0; k_next[i] <= '0;
            end
        end else begin
            if (row_push_valid && row_push_ready) begin
                // rotate
                for (int i = 0; i < ROW_TOKENS; i = i + 1) begin
                    q_prev[i] <= q_curr[i];
                    k_prev[i] <= k_curr[i];
                    q_curr[i] <= q_next[i];
                    k_curr[i] <= k_next[i];
                    q_next[i] <= row_push_q[i];
                    k_next[i] <= row_push_k[i];
                end
                v_prev <= v_curr;
                v_curr <= v_next;
                v_next <= row_push_valid_mask;
                tag_prev_q <= tag_curr_q;
                tag_curr_q <= tag_next_q;
                tag_next_q <= row_push_tag;
                if (filled_q < 2'd3) begin
                    filled_q <= filled_q + 2'd1;
                end
            end

            if (rsp_hold_q && rd_rsp_ready) begin
                rsp_hold_q <= 1'b0;
            end
            if (rd_valid && rd_ready) begin
                if (rd_row_sel > 2'd2 || 32'(rd_x) >= ROW_TOKENS) begin
                    protocol_error_q <= 1'b1;
                    rsp_hold_q <= 1'b1;
                    rsp_q_q <= '0;
                    rsp_k_q <= '0;
                    rsp_v_q <= 1'b0;
                end else begin
                    rsp_hold_q <= 1'b1;
                    unique case (rd_row_sel)
                        2'd0: begin
                            rsp_q_q <= q_prev[rd_x];
                            rsp_k_q <= k_prev[rd_x];
                            rsp_v_q <= v_prev[rd_x];
                        end
                        2'd1: begin
                            rsp_q_q <= q_curr[rd_x];
                            rsp_k_q <= k_curr[rd_x];
                            rsp_v_q <= v_curr[rd_x];
                        end
                        default: begin
                            rsp_q_q <= q_next[rd_x];
                            rsp_k_q <= k_next[rd_x];
                            rsp_v_q <= v_next[rd_x];
                        end
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire
