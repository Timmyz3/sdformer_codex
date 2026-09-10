// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_bl_veto_prior — Barlow–Levick inhibition-veto digital direction bank (Card I)
// INNOVATION: Pure inhibition veto / null-direction ban bank — ORTHOGONAL to TDE3 facilitator.
//             Preferred facilitator→trigger within Δt enhances pd_wake; null-spike in window
//             asserts veto_mask that ANDs out of wake_merge (inhibition veto vs TDE3 facilitation).
// Cite: Haessig et al. TrueNorth BL OF (arXiv:1710.09820); TDE-3 Frontiers 2025 ≠ BL veto.
// NOT claimed: first bio OF HW; Loihi/TrueNorth silicon; AEE.
// Ports PACKED where multi-bit per tile (iverilog/yosys-friendly).

`timescale 1ns/1ps
`default_nettype none

module c1s_bl_veto_prior #(
  parameter int N_TILE  = 8,
  parameter int AGE_W   = 8,
  parameter int CONF_W  = 4,
  parameter logic [AGE_W-1:0] DT_WIN = 8'd6,
  parameter logic [CONF_W-1:0] TH_VETO = 4'd2,
  parameter bit ABLATE_NO_VETO = 1'b0  // 1 = never assert veto_mask
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  // Per-tile event lanes (BL triad)
  input  logic [N_TILE-1:0]            fac_pulse,   // preferred facilitator
  input  logic [N_TILE-1:0]            trg_pulse,   // preferred trigger
  input  logic [N_TILE-1:0]            null_pulse,  // null / anti-preferred
  output logic [N_TILE-1:0]            pd_wake,     // preferred-direction wake (OR into merge)
  output logic [N_TILE-1:0]            veto_mask,   // 1 = inhibit (AND-NOT from wake_merge)
  output logic [N_TILE*CONF_W-1:0]     veto_conf_bus,
  output logic                         bl_valid
);

  logic [AGE_W-1:0] age_fac [N_TILE];
  logic [AGE_W-1:0] age_null [N_TILE];
  logic [N_TILE-1:0] pd_c;
  logic [N_TILE-1:0] veto_c;
  logic [N_TILE*CONF_W-1:0] conf_c;
  integer ti;

  always @(*) begin
    pd_c   = {N_TILE{1'b0}};
    veto_c = {N_TILE{1'b0}};
    conf_c = {N_TILE*CONF_W{1'b0}};
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      // Preferred: trigger while facilitator still young
      if (trg_pulse[ti] && (age_fac[ti] <= DT_WIN)) begin
        pd_c[ti] = 1'b1;
        conf_c[ti*CONF_W +: CONF_W] = (age_fac[ti] < 4) ? 4'd4 :
                                      (age_fac[ti] < DT_WIN) ? 4'd3 : 4'd2;
      end
      // Null veto: null arrives while fac or recent pd window young
      if (!ABLATE_NO_VETO && null_pulse[ti] &&
          ((age_fac[ti] <= DT_WIN) || (age_null[ti] <= DT_WIN))) begin
        veto_c[ti] = 1'b1;
        // Stronger veto conf when null lands close after fac
        if (age_fac[ti] <= 4)
          conf_c[ti*CONF_W +: CONF_W] = 4'd7;
        else if (conf_c[ti*CONF_W +: CONF_W] < TH_VETO)
          conf_c[ti*CONF_W +: CONF_W] = TH_VETO;
      end
      // Sustained veto while null age young (inhibition lingering)
      if (!ABLATE_NO_VETO && (age_null[ti] <= DT_WIN) && (age_null[ti] != {AGE_W{1'b1}}))
        veto_c[ti] = 1'b1;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bl_valid     <= 1'b0;
      pd_wake      <= '0;
      veto_mask    <= '0;
      veto_conf_bus <= '0;
      for (ti = 0; ti < N_TILE; ti = ti + 1) begin
        age_fac[ti]  <= {AGE_W{1'b1}};
        age_null[ti] <= {AGE_W{1'b1}};
      end
    end else begin
      bl_valid <= valid_i;
      if (valid_i) begin
        pd_wake       <= pd_c;
        veto_mask     <= veto_c;
        veto_conf_bus <= conf_c;
        for (ti = 0; ti < N_TILE; ti = ti + 1) begin
          if (fac_pulse[ti])
            age_fac[ti] <= {AGE_W{1'b0}};
          else if (age_fac[ti] != {AGE_W{1'b1}})
            age_fac[ti] <= age_fac[ti] + {{(AGE_W-1){1'b0}}, 1'b1};

          if (null_pulse[ti])
            age_null[ti] <= {AGE_W{1'b0}};
          else if (age_null[ti] != {AGE_W{1'b1}})
            age_null[ti] <= age_null[ti] + {{(AGE_W-1){1'b0}}, 1'b1};
        end
      end
    end
  end

endmodule

`default_nettype wire
