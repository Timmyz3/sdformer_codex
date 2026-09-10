// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c2s_bui_guard_sdsa — BUI-GuardSDSA bit-bound token guard (Card H4)
// INNOVATION: Cheap digital bit-guard / magnitude+bound width check on SDSA
//             scores → token_keep / payload_en into HBG path (enhances HBG,
//             does NOT replace packetizer; payload stays int8 proposal).
// Policy:
//   Drop if ub < TH_DROP (cannot reach useful score)
//   Keep if width=(ub-lb) <= TH_DROP OR score_msb has enough magnitude
//   ABLATE_KEEP_ALL=1 → always keep
//   payload_en = token_keep & {N{hbg_g_i}}
//   ooo_ready when all tokens decided by bounds this round
// Ports PACKED for multi-bit score/bound buses.
// NOT claimed: first sparse-attn ASIC; PADE/SOFA LLM accuracy; absorb ATLIF→binary.

`timescale 1ns/1ps
`default_nettype none

module c2s_bui_guard_sdsa #(
  parameter int N_TOKEN = 8,
  parameter int SCORE_W = 8,
  parameter int BOUND_W = 8,
  parameter logic [BOUND_W-1:0] TH_DROP = 8'd16,
  parameter bit ABLATE_KEEP_ALL = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         bit_round_valid_i,
  input  logic [N_TOKEN*SCORE_W-1:0]   score_msb_bus,
  input  logic [N_TOKEN*BOUND_W-1:0]   ub_bus,
  input  logic [N_TOKEN*BOUND_W-1:0]   lb_bus,
  input  logic                         hbg_g_i,
  output logic [N_TOKEN-1:0]           token_keep,
  output logic [N_TOKEN-1:0]           payload_en,
  output logic                         ooo_ready,
  output logic                         guard_valid
);

  logic [N_TOKEN-1:0] keep_c;
  logic [N_TOKEN-1:0] pay_c;
  logic               ooo_c;
  integer             ti;
  logic [SCORE_W-1:0] score_i;
  logic [BOUND_W-1:0] ub_i, lb_i, width_i;
  logic               mag_ok, decided_i;
  logic               all_decided;

  always @(*) begin
    keep_c      = {N_TOKEN{1'b0}};
    pay_c       = {N_TOKEN{1'b0}};
    all_decided = 1'b1;
    for (ti = 0; ti < N_TOKEN; ti = ti + 1) begin
      score_i = score_msb_bus[ti*SCORE_W +: SCORE_W];
      ub_i    = ub_bus[ti*BOUND_W +: BOUND_W];
      lb_i    = lb_bus[ti*BOUND_W +: BOUND_W];
      if (ub_i >= lb_i)
        width_i = ub_i - lb_i;
      else
        width_i = {BOUND_W{1'b1}};

      // Magnitude / leading-nonzero guard on partial MSB score
      mag_ok = (score_i >= (TH_DROP >> 1));

      if (ABLATE_KEEP_ALL) begin
        keep_c[ti]  = 1'b1;
        decided_i   = 1'b1;
      end else if (ub_i < TH_DROP) begin
        // Upper bound too low → definite drop
        keep_c[ti]  = 1'b0;
        decided_i   = 1'b1;
      end else if (width_i <= TH_DROP) begin
        // Bounds tight enough → keep if mag or lb already useful
        keep_c[ti]  = mag_ok || (lb_i >= (TH_DROP >> 1));
        decided_i   = 1'b1;
      end else begin
        // Wide uncertainty: keep only strong MSB magnitude (bit-guard)
        keep_c[ti]  = mag_ok;
        decided_i   = mag_ok;  // undecided weak tokens → not ooo
      end

      if (!decided_i)
        all_decided = 1'b0;

      pay_c[ti] = keep_c[ti] & hbg_g_i;
    end
    ooo_c = bit_round_valid_i && all_decided;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      guard_valid <= 1'b0;
      token_keep  <= '0;
      payload_en  <= '0;
      ooo_ready   <= 1'b0;
    end else begin
      guard_valid <= bit_round_valid_i;
      if (bit_round_valid_i) begin
        token_keep <= keep_c;
        payload_en <= pay_c;
        ooo_ready  <= ooo_c;
      end else begin
        ooo_ready  <= 1'b0;
      end
    end
  end

endmodule

`default_nettype wire
