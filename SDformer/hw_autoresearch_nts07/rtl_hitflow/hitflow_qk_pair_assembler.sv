`timescale 1ns/1ps
`default_nettype none

module hitflow_qk_pair_assembler #(
    parameter int DATA_W = 32,
    parameter int TAG_W  = 48
) (
    input  logic                  clk_core,
    input  logic                  rst_core,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [1:0]            in_slot,
    input  logic [DATA_W-1:0]     in_data,
    input  logic [TAG_W-1:0]      in_tag,
    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [(4*DATA_W)-1:0] out_pair,
    output logic [TAG_W-1:0]      out_tag,
    output logic                  tag_mismatch,
    output logic                  duplicate_slot
);

    logic [3:0]        present_q;
    logic [TAG_W-1:0]  tag_q;
    logic [DATA_W-1:0] q0_q;
    logic [DATA_W-1:0] q1_q;
    logic [DATA_W-1:0] k0_q;
    logic [DATA_W-1:0] k1_q;
    logic              has_partial;
    logic              tag_matches;
    logic              slot_empty;
    logic              retire_pair;
    logic              accept_slot;

    assign has_partial = |present_q;
    assign out_valid = &present_q;
    assign retire_pair = out_valid & out_ready;
    assign tag_matches = ~has_partial | retire_pair | (in_tag == tag_q);
    assign slot_empty = retire_pair | ~present_q[in_slot];
    assign in_ready = (~out_valid | out_ready) & tag_matches & slot_empty;
    assign accept_slot = in_valid & in_ready;
    assign out_pair = {k1_q, k0_q, q1_q, q0_q};
    assign out_tag = tag_q;
    assign tag_mismatch = in_valid & has_partial & ~retire_pair &
                          (in_tag != tag_q);
    assign duplicate_slot = in_valid & has_partial & ~retire_pair &
                            (in_tag == tag_q) & ~slot_empty;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            present_q <= 4'b0000;
            tag_q     <= '0;
            q0_q      <= '0;
            q1_q      <= '0;
            k0_q      <= '0;
            k1_q      <= '0;
        end else if (accept_slot) begin
                if (~has_partial | retire_pair) begin
                    tag_q <= in_tag;
                    present_q <= 4'b0001 << in_slot;
                end else begin
                    present_q[in_slot] <= 1'b1;
                end
                unique case (in_slot)
                    2'd0: q0_q <= in_data;
                    2'd1: q1_q <= in_data;
                    2'd2: k0_q <= in_data;
                    2'd3: k1_q <= in_data;
                    default: begin end
                endcase
        end else if (retire_pair) begin
            present_q <= 4'b0000;
        end
    end

endmodule

`default_nettype wire
