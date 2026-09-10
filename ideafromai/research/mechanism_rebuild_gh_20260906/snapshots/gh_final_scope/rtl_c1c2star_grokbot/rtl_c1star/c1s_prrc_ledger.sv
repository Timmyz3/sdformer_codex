// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_prrc_ledger — Pyramid Residual Budget / ledger (PRRC)
// Coarse levels write residual budget; fine levels only capture when budget > 0.
// Tracks remaining exact-capture budget per pyramid level.
// Port refinements (see README):
//   - budget PACKED [N_LEVEL*BUDGET_W-1:0]; level i at [i*BUDGET_W +: BUDGET_W]
//   - level_idx select; level_sel optional one-hot override when |sel != 0
//   - Default N_LEVEL=3 with explicit b0/b1/b2 banks (iverilog-friendly)

`timescale 1ns/1ps
`default_nettype none

module c1s_prrc_ledger #(
  parameter int N_LEVEL = 3,
  parameter int BUDGET_W = 8,
  parameter logic [BUDGET_W-1:0] INIT_BUDGET = 8'd16
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [N_LEVEL-1:0]           level_sel,
  input  logic [$clog2(N_LEVEL)-1:0]   level_idx,
  input  logic                         spend_i,
  input  logic                         refill_i,
  output logic [N_LEVEL*BUDGET_W-1:0]  budget,
  output logic                         allow_exact,
  output logic                         ledger_valid
);

  // Explicit banks for N_LEVEL=3 (TB/synth default)
  logic [BUDGET_W-1:0] b0, b1, b2;
  logic [1:0]          sel_idx;
  logic [BUDGET_W-1:0] cur;

  // one-hot override (unique case priority low→high via cascading if)
  always @(*) begin
    if (level_sel[2])
      sel_idx = 2'd2;
    else if (level_sel[1])
      sel_idx = 2'd1;
    else if (level_sel[0])
      sel_idx = 2'd0;
    else
      sel_idx = level_idx[1:0];
  end

  always @(*) begin
    case (sel_idx)
      2'd0:    cur = b0;
      2'd1:    cur = b1;
      default: cur = b2;
    endcase
  end

  assign allow_exact = (cur != {BUDGET_W{1'b0}});
  // level0 in LSBs: {b2,b1,b0}
  assign budget = {b2, b1, b0};  // N_LEVEL=3: level0 LSBs

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      b0 <= INIT_BUDGET;
      b1 <= INIT_BUDGET;
      b2 <= INIT_BUDGET;
      ledger_valid <= 1'b0;
    end else begin
      ledger_valid <= valid_i;
      if (valid_i) begin
        if (refill_i) begin
          if (sel_idx == 2'd0) b0 <= INIT_BUDGET;
          else if (sel_idx == 2'd1) b1 <= INIT_BUDGET;
          else b2 <= INIT_BUDGET;
        end else if (spend_i && (cur != {BUDGET_W{1'b0}})) begin
          if (sel_idx == 2'd0) b0 <= b0 - {{(BUDGET_W-1){1'b0}}, 1'b1};
          else if (sel_idx == 2'd1) b1 <= b1 - {{(BUDGET_W-1){1'b0}}, 1'b1};
          else b2 <= b2 - {{(BUDGET_W-1){1'b0}}, 1'b1};
        end
      end
    end
  end

endmodule

`default_nettype wire
