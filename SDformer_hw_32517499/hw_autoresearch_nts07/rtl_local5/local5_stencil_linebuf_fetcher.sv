`timescale 1ns/1ps
`default_nettype none

// Assemble one Local5 stencil (self+N+S+E+W) by reading the 3-row line buffer.
// Geometry: curr row x = center; N=prev[x], S=next[x], E=curr[x+1], W=curr[x-1].
// Boundary: invalid directions reported in valid_mask.
module local5_stencil_linebuf_fetcher #(
    parameter int HEAD_DIM   = 32,
    parameter int ROW_TOKENS = 16,
    parameter int TOKEN_W    = (ROW_TOKENS <= 1) ? 1 : $clog2(ROW_TOKENS),
    parameter int TAG_W      = 16,
    parameter int DEST_W     = 8
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    // request one center token on current row
    input  logic                     req_valid,
    output logic                     req_ready,
    input  logic [TAG_W-1:0]         req_tag,
    input  logic [DEST_W-1:0]        req_dest_id,
    input  logic [TOKEN_W-1:0]       req_x,
    input  logic                     req_last,

    // line buffer read port
    output logic                     rd_valid,
    input  logic                     rd_ready,
    output logic [1:0]               rd_row_sel,
    output logic [TOKEN_W-1:0]       rd_x,
    input  logic                     rd_rsp_valid,
    output logic                     rd_rsp_ready,
    input  logic [HEAD_DIM-1:0]      rd_q_bits,
    input  logic [HEAD_DIM-1:0]      rd_k_bits,
    input  logic                     rd_token_valid,

    // assembled stencil
    output logic                     stencil_valid,
    input  logic                     stencil_ready,
    output logic [TAG_W-1:0]         stencil_tag,
    output logic [DEST_W-1:0]        stencil_dest_id,
    output logic [HEAD_DIM-1:0]      stencil_q,
    output logic [HEAD_DIM-1:0]      stencil_k_self,
    output logic [HEAD_DIM-1:0]      stencil_k_n,
    output logic [HEAD_DIM-1:0]      stencil_k_s,
    output logic [HEAD_DIM-1:0]      stencil_k_e,
    output logic [HEAD_DIM-1:0]      stencil_k_w,
    output logic [4:0]               stencil_valid_mask, // self N S E W
    output logic                     stencil_last,

    output logic                     protocol_error
);

    typedef enum logic [3:0] {
        ST_IDLE  = 4'd0,
        ST_SELF  = 4'd1,
        ST_N     = 4'd2,
        ST_S     = 4'd3,
        ST_W     = 4'd4,
        ST_E     = 4'd5,
        ST_OUT   = 4'd6
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [TOKEN_W-1:0] x_q;
    logic last_q;
    logic [HEAD_DIM-1:0] q_q, ks_q, kn_q, ksou_q, ke_q, kw_q;
    logic v_self_q, v_n_q, v_s_q, v_e_q, v_w_q;
    logic protocol_error_q;
    logic issued_q;

    assign req_ready = (state_q == ST_IDLE);
    assign stencil_valid = (state_q == ST_OUT);
    assign stencil_tag = tag_q;
    assign stencil_dest_id = dest_q;
    assign stencil_q = q_q;
    assign stencil_k_self = ks_q;
    assign stencil_k_n = kn_q;
    assign stencil_k_s = ksou_q;
    assign stencil_k_e = ke_q;
    assign stencil_k_w = kw_q;
    assign stencil_valid_mask = {v_w_q, v_e_q, v_s_q, v_n_q, v_self_q};
    // pack order W E S N self in bits 4:0 → self is bit0
    // Fix: {v_w, v_e, v_s, v_n, v_self} is correct for [4:0]
    assign stencil_last = last_q;
    assign protocol_error = protocol_error_q;
    assign rd_rsp_ready = 1'b1;

    always_comb begin
        rd_valid = 1'b0;
        rd_row_sel = 2'd1;
        rd_x = x_q;
        unique case (state_q)
            ST_SELF: begin
                rd_valid = !issued_q;
                rd_row_sel = 2'd1; // curr
                rd_x = x_q;
            end
            ST_N: begin
                rd_valid = !issued_q;
                rd_row_sel = 2'd0; // prev
                rd_x = x_q;
            end
            ST_S: begin
                rd_valid = !issued_q;
                rd_row_sel = 2'd2; // next
                rd_x = x_q;
            end
            ST_W: begin
                rd_valid = !issued_q && (x_q != '0);
                rd_row_sel = 2'd1;
                rd_x = x_q - TOKEN_W'(1);
            end
            ST_E: begin
                rd_valid = !issued_q && (32'(x_q) + 1 < ROW_TOKENS);
                rd_row_sel = 2'd1;
                rd_x = x_q + TOKEN_W'(1);
            end
            default: begin
                rd_valid = 1'b0;
            end
        endcase
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0; dest_q <= '0; x_q <= '0; last_q <= 1'b0;
            q_q <= '0; ks_q <= '0; kn_q <= '0; ksou_q <= '0; ke_q <= '0; kw_q <= '0;
            v_self_q <= 1'b0; v_n_q <= 1'b0; v_s_q <= 1'b0; v_e_q <= 1'b0; v_w_q <= 1'b0;
            protocol_error_q <= 1'b0;
            issued_q <= 1'b0;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    issued_q <= 1'b0;
                    if (req_valid) begin
                        tag_q <= req_tag;
                        dest_q <= req_dest_id;
                        x_q <= req_x;
                        last_q <= req_last;
                        kn_q <= '0; ksou_q <= '0; ke_q <= '0; kw_q <= '0;
                        v_n_q <= 1'b0; v_s_q <= 1'b0; v_e_q <= 1'b0; v_w_q <= 1'b0;
                        state_q <= ST_SELF;
                    end
                end
                ST_SELF: begin
                    if (rd_valid && rd_ready) issued_q <= 1'b1;
                    if (rd_rsp_valid && issued_q) begin
                        q_q <= rd_q_bits;
                        ks_q <= rd_k_bits;
                        v_self_q <= rd_token_valid;
                        issued_q <= 1'b0;
                        state_q <= ST_N;
                    end
                end
                ST_N: begin
                    if (rd_valid && rd_ready) issued_q <= 1'b1;
                    if (rd_rsp_valid && issued_q) begin
                        kn_q <= rd_k_bits;
                        v_n_q <= rd_token_valid;
                        issued_q <= 1'b0;
                        state_q <= ST_S;
                    end
                end
                ST_S: begin
                    if (rd_valid && rd_ready) issued_q <= 1'b1;
                    if (rd_rsp_valid && issued_q) begin
                        ksou_q <= rd_k_bits;
                        v_s_q <= rd_token_valid;
                        issued_q <= 1'b0;
                        state_q <= ST_W;
                    end
                end
                ST_W: begin
                    if (x_q == '0) begin
                        v_w_q <= 1'b0;
                        kw_q <= '0;
                        issued_q <= 1'b0;
                        state_q <= ST_E;
                    end else begin
                        if (rd_valid && rd_ready) issued_q <= 1'b1;
                        if (rd_rsp_valid && issued_q) begin
                            kw_q <= rd_k_bits;
                            v_w_q <= rd_token_valid;
                            issued_q <= 1'b0;
                            state_q <= ST_E;
                        end
                    end
                end
                ST_E: begin
                    if (32'(x_q) + 1 >= ROW_TOKENS) begin
                        v_e_q <= 1'b0;
                        ke_q <= '0;
                        issued_q <= 1'b0;
                        state_q <= ST_OUT;
                    end else begin
                        if (rd_valid && rd_ready) issued_q <= 1'b1;
                        if (rd_rsp_valid && issued_q) begin
                            ke_q <= rd_k_bits;
                            v_e_q <= rd_token_valid;
                            issued_q <= 1'b0;
                            state_q <= ST_OUT;
                        end
                    end
                end
                ST_OUT: begin
                    if (stencil_ready) state_q <= ST_IDLE;
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
