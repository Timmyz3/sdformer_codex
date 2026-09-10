// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_cfp_sci_exact_glue — pipe glue: CFP-ConfGate × SCI-CleanExit → exact_capture
// INNOVATION: Additive combiners wire H2/H3 into OGEC×PRRC→exact_capture without rewriting
//             green OGEC / PRRC / exact_capture / CFP / SCI modules.
// Recommended policy (19_CARD_H_IF_SKETCH_RED4):
//   exact_en_to_capture = ogec.exact_en & cfp.exact_req
//   allow_exact_final   = prrc.allow_exact && (popcount(exact_req) <= prrc_grant)
//   capture_en_gated    = capture_en_raw & ~sci.exact_hold
//   pe_clk_en           = ~(|sci.scrub_en)   // optional PE path gate
// Ablation:
//   ABLATE_CFP_BYPASS=1 → treat exact_req as all-1 (ALWAYS-like)
//   ABLATE_SCI_BYPASS=1 → ignore exact_hold / scrub (no hold gate)
// NOT claimed: AEE; silicon PPA; replacement of OGEC/PRRC.

`timescale 1ns/1ps
`default_nettype none

module c1s_cfp_sci_exact_glue #(
  parameter int N_TILE   = 8,
  parameter int BUDGET_W = 8,
  parameter bit ABLATE_CFP_BYPASS = 1'b0,
  parameter bit ABLATE_SCI_BYPASS = 1'b0
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  // From OGEC
  input  logic [N_TILE-1:0]    ogec_exact_en,
  // From CFP-ConfGate
  input  logic [N_TILE-1:0]    cfp_exact_req,
  input  logic [BUDGET_W-1:0]  cfp_prrc_grant,
  // From PRRC ledger
  input  logic                 prrc_allow_exact,
  // From SCI-CleanExit
  input  logic                 sci_exact_hold,
  input  logic [N_TILE-1:0]    sci_scrub_en,
  // Raw capture enable from schedule FSM
  input  logic                 capture_en_raw,
  // Gated outputs → exact_capture_wrap / optional PE
  output logic [N_TILE-1:0]    exact_en_to_capture,
  output logic                 allow_exact_final,
  output logic                 capture_en_gated,
  output logic                 pe_clk_en,
  output logic                 glue_valid,
  // Prove-by taps
  output logic [BUDGET_W-1:0]  pop_exact_req,
  output logic                 grant_ok,
  output logic                 hold_active
);

  logic [N_TILE-1:0] req_eff;
  logic [BUDGET_W-1:0] pop_c;
  logic              grant_ok_c;
  logic              allow_c;
  logic              hold_c;
  logic              scrub_any;
  logic [N_TILE-1:0] exact_c;
  logic              cap_c;
  logic              pe_c;
  integer            ti;

  always @(*) begin
    req_eff = ABLATE_CFP_BYPASS ? {N_TILE{1'b1}} : cfp_exact_req;
    pop_c = {BUDGET_W{1'b0}};
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      if (req_eff[ti])
        pop_c = pop_c + {{(BUDGET_W-1){1'b0}}, 1'b1};
    end
    grant_ok_c = (pop_c <= cfp_prrc_grant);
    allow_c = prrc_allow_exact && grant_ok_c;
    exact_c = ogec_exact_en & req_eff;

    hold_c = ABLATE_SCI_BYPASS ? 1'b0 : sci_exact_hold;
    scrub_any = ABLATE_SCI_BYPASS ? 1'b0 : (|sci_scrub_en);
    cap_c = capture_en_raw & ~hold_c;
    pe_c  = ~scrub_any;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      exact_en_to_capture <= '0;
      allow_exact_final   <= 1'b0;
      capture_en_gated    <= 1'b0;
      pe_clk_en           <= 1'b1;
      glue_valid          <= 1'b0;
      pop_exact_req       <= '0;
      grant_ok            <= 1'b0;
      hold_active         <= 1'b0;
    end else begin
      glue_valid <= valid_i;
      if (valid_i) begin
        exact_en_to_capture <= exact_c;
        allow_exact_final   <= allow_c;
        capture_en_gated    <= cap_c;
        pe_clk_en           <= pe_c;
        pop_exact_req       <= pop_c;
        grant_ok            <= grant_ok_c;
        hold_active         <= hold_c;
      end else begin
        // Hold capture_en_gated tracking raw even when !valid (FSM may drop)
        capture_en_gated <= capture_en_raw & ~(ABLATE_SCI_BYPASS ? 1'b0 : sci_exact_hold);
        pe_clk_en        <= ABLATE_SCI_BYPASS ? 1'b1 : ~(|sci_scrub_en);
      end
    end
  end

endmodule

`default_nettype wire
