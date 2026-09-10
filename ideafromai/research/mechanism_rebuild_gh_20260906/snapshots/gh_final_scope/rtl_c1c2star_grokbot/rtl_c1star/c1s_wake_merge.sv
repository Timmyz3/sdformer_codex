// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_wake_merge — thin OR-merge of OP-STW | TDE3-Prior | MW-Δ | NL-STMFA
//         then AND-NOT BL-VetoPrior mask (Card I optional lanes).
// Additive glue: does NOT rewrite OP-STW; optional tde/nl/veto lanes.
// Policy: wake = (op_stw | tde | delta_nz | nl_wake | pd_wake) & ~veto_mask
// Tie unused lanes to 0.

`timescale 1ns/1ps
`default_nettype none

module c1s_wake_merge #(
  parameter int N_TILE = 8
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  input  logic [N_TILE-1:0]    op_stw_wake,
  input  logic [N_TILE-1:0]    tde_wake,
  input  logic [N_TILE-1:0]    delta_nz,
  // Card I optional: NL-STMFA residual wake (OR)
  input  logic [N_TILE-1:0]    nl_wake,
  // Card I optional: BL preferred-dir wake (OR) + veto mask (AND-NOT)
  input  logic [N_TILE-1:0]    pd_wake,
  input  logic [N_TILE-1:0]    veto_mask,
  output logic [N_TILE-1:0]    wake_merged,
  output logic                 merge_valid
);

  logic [N_TILE-1:0] merge_comb;
  assign merge_comb = (op_stw_wake | tde_wake | delta_nz | nl_wake | pd_wake) & ~veto_mask;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wake_merged <= '0;
      merge_valid <= 1'b0;
    end else begin
      merge_valid <= valid_i;
      if (valid_i)
        wake_merged <= merge_comb;
    end
  end

endmodule

`default_nettype wire
