// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_cfp_confgate — CFP-ConfGate (Card H2)
// INNOVATION: corr-peak confidence gates exact vs propagate and soft-scales
//             PRRC grant — inserts between OGEC match binary and PRRC budget.
// Policy (recommended default):
//   high conf (>= TH_HI) + match_ok → prop_pref (release grant)
//   low/mid conf (< TH_HI) + match_ok → exact_req
//   ABLATE_ALWAYS_EXACT=1 → all match_ok → exact_req, prop_pref=0
// Combine (TB/doc): exact_en_to_capture = ogec.exact_en & exact_req
//                   allow_exact_final = prrc.allow_exact && (popcount(exact_req) <= prrc_grant)
// Ports PACKED: tile i at [i*W +: W] (iverilog/yosys-friendly).
// NOT claimed: first occlusion OF; AEE without algo cal; PredExit frame exit.

`timescale 1ns/1ps
`default_nettype none

module c1s_cfp_confgate #(
  parameter int N_TILE   = 8,
  parameter int PEAK_W   = 16,
  parameter int CONF_W   = 8,
  parameter int BUDGET_W = 8,
  parameter logic [CONF_W-1:0] TH_HI = 8'd160,
  parameter logic [CONF_W-1:0] TH_LO = 8'd64,  // documented band; exact uses TH_HI cut
  parameter bit ABLATE_ALWAYS_EXACT = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [N_TILE*PEAK_W-1:0]     corr_peak_bus,
  input  logic [N_TILE-1:0]            match_ok,
  input  logic [BUDGET_W-1:0]          prrc_budget_i,
  output logic [N_TILE*CONF_W-1:0]     conf_bus,
  output logic [N_TILE-1:0]            exact_req,
  output logic [N_TILE-1:0]            prop_pref,
  output logic [BUDGET_W-1:0]          prrc_grant,
  output logic                         conf_valid
);

  logic [N_TILE*CONF_W-1:0] conf_c;
  logic [N_TILE-1:0]        exact_c;
  logic [N_TILE-1:0]        prop_c;
  logic [BUDGET_W-1:0]      grant_c;
  logic [BUDGET_W-1:0]      n_prop;
  logic [BUDGET_W-1:0]      n_need;
  integer                   ti;
  logic [PEAK_W-1:0]        peak_i;
  logic [CONF_W-1:0]        conf_i;
  logic [PEAK_W-1:0]        conf_max_ext;

  assign conf_max_ext = {{(PEAK_W-CONF_W){1'b0}}, {CONF_W{1'b1}}};

  always @(*) begin
    conf_c  = {N_TILE*CONF_W{1'b0}};
    exact_c = {N_TILE{1'b0}};
    prop_c  = {N_TILE{1'b0}};
    n_prop  = {BUDGET_W{1'b0}};
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      peak_i = corr_peak_bus[ti*PEAK_W +: PEAK_W];
      // Map peak → conf: saturate into CONF_W
      if (peak_i > conf_max_ext)
        conf_i = {CONF_W{1'b1}};
      else
        conf_i = peak_i[CONF_W-1:0];
      conf_c[ti*CONF_W +: CONF_W] = conf_i;

      if (ABLATE_ALWAYS_EXACT) begin
        exact_c[ti] = match_ok[ti];
        prop_c[ti]  = 1'b0;
      end else if (match_ok[ti]) begin
        if (conf_i >= TH_HI) begin
          prop_c[ti]  = 1'b1;   // high conf → prefer propagate, release grant
          exact_c[ti] = 1'b0;
        end else begin
          exact_c[ti] = 1'b1;   // low/mid conf → exact_req
          prop_c[ti]  = 1'b0;
        end
      end

      if (prop_c[ti] && (n_prop != {BUDGET_W{1'b1}}))
        n_prop = n_prop + {{(BUDGET_W-1){1'b0}}, 1'b1};
    end

    // Soft grant: budget scaled by non-prop (need) tiles — high-conf releases
    n_need = N_TILE[BUDGET_W-1:0] - n_prop;
    if (N_TILE == 0)
      grant_c = {BUDGET_W{1'b0}};
    else
      grant_c = (prrc_budget_i * n_need) / N_TILE[BUDGET_W-1:0];
    // TH_LO kept as documented mid-band parameter (exact cut uses TH_HI)
    if (TH_LO == {CONF_W{1'b0}})
      grant_c = grant_c;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      conf_valid  <= 1'b0;
      conf_bus    <= '0;
      exact_req   <= '0;
      prop_pref   <= '0;
      prrc_grant  <= '0;
    end else begin
      conf_valid <= valid_i;
      if (valid_i) begin
        conf_bus   <= conf_c;
        exact_req  <= exact_c;
        prop_pref  <= prop_c;
        prrc_grant <= grant_c;
      end
    end
  end

endmodule

`default_nettype wire
