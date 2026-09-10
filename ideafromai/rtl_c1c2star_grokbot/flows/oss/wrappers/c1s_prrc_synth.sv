// GROKBOT NEW FILE -- iscas_ssh
// Synth wrapper: PRRC packed budget (instantiates RTL @ N_LEVEL=3).

`timescale 1ns/1ps
`default_nettype none

module c1s_prrc_ledger_synth #(
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

  c1s_prrc_ledger #(
    .N_LEVEL(N_LEVEL), .BUDGET_W(BUDGET_W), .INIT_BUDGET(INIT_BUDGET)
  ) u_prrc (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .level_sel(level_sel), .level_idx(level_idx),
    .spend_i(spend_i), .refill_i(refill_i),
    .budget(budget), .allow_exact(allow_exact), .ledger_valid(ledger_valid)
  );

endmodule

`default_nettype wire
