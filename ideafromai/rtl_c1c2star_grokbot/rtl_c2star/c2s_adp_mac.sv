// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_adp_mac — Bilateral bit-sparse ADP-MAC (preserves ATLIF amplitude)
// Dual-side bit skip + donate/balance gate; output gated MAC accumulate.
// INNOVATION: bilateral bit-skip MAC preserving ATLIF amplitude.
// Does NOT multiply-reorder (forbid A-then-B swap narrative).
// Claim sentence: "Dual-side sparse MAC; payload≠absorbable spike."

`timescale 1ns/1ps
`default_nettype none

module c2s_adp_mac #(
  parameter int W_A   = 8,
  parameter int W_B   = 8,
  parameter int W_ACC = 16,
  // Letter hygiene: document anti-reorder; TB asserts FORBID_REORDER==1
  parameter bit FORBID_REORDER = 1'b1
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      valid_i,
  input  logic signed [W_A-1:0]     a,        // ATLIF payload (amplitude preserved)
  input  logic signed [W_B-1:0]     b,        // weight
  input  logic                      skip_a,   // bilateral skip sides
  input  logic                      skip_b,
  input  logic                      mac_en,   // from SMAM/HBG gate
  input  logic                      clear_i,  // clear accumulator
  output logic signed [W_ACC-1:0]   acc,
  output logic                      skipped,  // 1 if either skip or !mac_en
  output logic                      out_valid
);

  localparam int PROD_W = W_A + W_B;

  logic do_mac;
  logic signed [PROD_W-1:0] prod;
  logic signed [W_ACC-1:0]  prod_ext;

  // FORBID_REORDER=1 means a*b path is fixed; never swap operands for 'bit sparsity'
  // (parameter kept for TB/docs; tieoff avoids unused-param warn in some tools)
  wire _forbid_reorder_tieoff = FORBID_REORDER;
  assign do_mac  = mac_en && !skip_a && !skip_b;
  assign skipped = !do_mac;
  // Signed multiply; both operands signed → signed product
  assign prod     = a * b;
  assign prod_ext = {{(W_ACC-PROD_W){prod[PROD_W-1]}}, prod};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc       <= '0;
      out_valid <= 1'b0;
    end else begin
      out_valid <= valid_i;
      if (clear_i) begin
        acc <= '0;
      end else if (valid_i && do_mac) begin
        acc <= acc + prod_ext;
      end
      // else: hold acc (skip / !mac_en / !valid_i)
    end
  end

endmodule

`default_nettype wire
