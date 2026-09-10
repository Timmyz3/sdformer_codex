// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_sci_cleanexit — SCI-CleanExit (Card H3)
// INNOVATION: Per-iter SCI warp-consistency quality → scrub / exit_ok / exact_hold
//             sharpens PredExit-like halt and protects exact budget on dirty iters.
// Policy:
//   g_sci = invert(|f1 - f2_warp|)  (high = consistent)
//   scrub_en when g_sci < TH_SCRUB (disabled if ABLATE_EXIT_ONLY)
//   exit_ok when mean g_sci > TH_EXIT
//   exact_hold = scrub_any | !exit_ok | bisat_occ_exit_i
// Ports PACKED for multi-bit tile vectors (iverilog/yosys-friendly).
// NOT claimed: SciFlow frame AEE; Snapdragon on-device; RFL training.

`timescale 1ns/1ps
`default_nettype none

module c1s_sci_cleanexit #(
  parameter int N_TILE  = 8,
  parameter int FEAT_W  = 8,
  parameter int FLOW_W  = 8,
  parameter int SCI_W   = 8,
  parameter logic [SCI_W-1:0] TH_EXIT  = 8'd200,
  parameter logic [SCI_W-1:0] TH_SCRUB = 8'd80,
  parameter bit ABLATE_EXIT_ONLY = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         it_valid_i,
  input  logic [N_TILE*FEAT_W-1:0]     f1_bus,
  input  logic [N_TILE*FEAT_W-1:0]     f2_warp_bus,
  input  logic [N_TILE*FLOW_W-1:0]     f_hat_bus,
  input  logic                         bisat_occ_exit_i,
  output logic [N_TILE*SCI_W-1:0]      g_sci_bus,
  output logic [N_TILE-1:0]            scrub_en,
  output logic                         exit_ok,
  output logic                         exact_hold,
  output logic                         sci_valid
);

  logic [N_TILE*SCI_W-1:0] g_c;
  logic [N_TILE-1:0]       scrub_c;
  logic                    exit_c;
  logic                    hold_c;
  logic                    scrub_any;
  integer                  ti;
  logic signed [FEAT_W-1:0] f1_i, f2_i;
  logic signed [FEAT_W:0]   diff_x;
  logic [FEAT_W:0]          abs_x;
  logic [SCI_W-1:0]         abs_sat;
  logic [SCI_W-1:0]         g_i;
  logic [SCI_W+7:0]         sum_g;
  logic [SCI_W-1:0]         mean_g;
  logic [FLOW_W-1:0]        fhat_unused;

  always @(*) begin
    g_c       = {N_TILE*SCI_W{1'b0}};
    scrub_c   = {N_TILE{1'b0}};
    scrub_any = 1'b0;
    sum_g     = {(SCI_W+8){1'b0}};
    fhat_unused = {FLOW_W{1'b0}};
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      f1_i = f1_bus[ti*FEAT_W +: FEAT_W];
      f2_i = f2_warp_bus[ti*FEAT_W +: FEAT_W];
      diff_x = $signed({f1_i[FEAT_W-1], f1_i}) - $signed({f2_i[FEAT_W-1], f2_i});
      if (diff_x[FEAT_W])
        abs_x = (~diff_x) + {{FEAT_W{1'b0}}, 1'b1};
      else
        abs_x = {1'b0, diff_x[FEAT_W-1:0]};

      // Saturate to SCI_W then invert
      if (abs_x > {{(FEAT_W+1-SCI_W){1'b0}}, {SCI_W{1'b1}}})
        abs_sat = {SCI_W{1'b1}};
      else
        abs_sat = abs_x[SCI_W-1:0];
      g_i = {SCI_W{1'b1}} - abs_sat;

      g_c[ti*SCI_W +: SCI_W] = g_i;
      sum_g = sum_g + {{8{1'b0}}, g_i};

      if (!ABLATE_EXIT_ONLY && (g_i < TH_SCRUB)) begin
        scrub_c[ti] = 1'b1;
        scrub_any   = 1'b1;
      end

      fhat_unused = f_hat_bus[ti*FLOW_W +: FLOW_W];
    end

    mean_g = sum_g / N_TILE;
    exit_c = (mean_g > TH_EXIT);
    hold_c = scrub_any || (!exit_c) || bisat_occ_exit_i;
    if (fhat_unused == {FLOW_W{1'b0}})
      hold_c = hold_c;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sci_valid  <= 1'b0;
      g_sci_bus  <= '0;
      scrub_en   <= '0;
      exit_ok    <= 1'b0;
      exact_hold <= 1'b0;
    end else begin
      sci_valid <= it_valid_i;
      if (it_valid_i) begin
        g_sci_bus  <= g_c;
        scrub_en   <= scrub_c;
        exit_ok    <= exit_c;
        exact_hold <= hold_c;
      end
    end
  end

endmodule

`default_nettype wire
