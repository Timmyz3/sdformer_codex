// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_motion_ttb_packer — Motion-TTB packer (C2*/midend)
// Idea packs: Bishop TTB / Motion-TTB+OF-ECP — pack wake tiles into
// motion time-bundles (tile_id, dt_bin, hyp_id) for MFBD-style delivery.
// Note: multi-bit bundle outputs are packed buses — iverilog cannot reliably
// drive multi-bit unpacked array output ports; TB / top unpack as needed.
// Yosys: use flows/oss/wrappers/c2s_motion_ttb_synth.sv (flat inputs too).

`timescale 1ns/1ps
`default_nettype none

module c2s_motion_ttb_packer #(
  parameter int N_TILE      = 16,
  parameter int DT_W        = 3,
  parameter int HYP_W       = 2,
  parameter int MAX_BUNDLES = 8,
  parameter int TILE_W      = (N_TILE <= 1) ? 1 : $clog2(N_TILE),
  parameter int CNT_W       = $clog2(MAX_BUNDLES + 1)
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic [N_TILE-1:0]                 wake_bitmap,
  input  logic [DT_W-1:0]                   dt_bin  [N_TILE],
  input  logic [HYP_W-1:0]                  hyp_id  [N_TILE],
  // Packed bundles: entry i in bits [i*W +: W]; scan order ascending tile
  output logic [MAX_BUNDLES*TILE_W-1:0]     bundle_tile,
  output logic [MAX_BUNDLES*DT_W-1:0]       bundle_dt,
  output logic [MAX_BUNDLES*HYP_W-1:0]      bundle_hyp,
  output logic [CNT_W-1:0]                  bundle_count,
  output logic                              bundle_valid
);

  logic [MAX_BUNDLES*TILE_W-1:0] pack_tile;
  logic [MAX_BUNDLES*DT_W-1:0]   pack_dt;
  logic [MAX_BUNDLES*HYP_W-1:0]  pack_hyp;
  logic [CNT_W-1:0]              pack_count;

  integer ti, cnt;

  always_comb begin
    pack_tile  = '0;
    pack_dt    = '0;
    pack_hyp   = '0;
    pack_count = '0;
    cnt        = 0;
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      if (wake_bitmap[ti] && (cnt < MAX_BUNDLES)) begin
        pack_tile[cnt*TILE_W +: TILE_W] = ti[TILE_W-1:0];
        pack_dt  [cnt*DT_W   +: DT_W]   = dt_bin[ti];
        pack_hyp [cnt*HYP_W  +: HYP_W]  = hyp_id[ti];
        cnt = cnt + 1;
      end
    end
    pack_count = cnt[CNT_W-1:0];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bundle_tile  <= '0;
      bundle_dt    <= '0;
      bundle_hyp   <= '0;
      bundle_count <= '0;
      bundle_valid <= 1'b0;
    end else begin
      bundle_valid <= valid_i;
      if (valid_i) begin
        bundle_tile  <= pack_tile;
        bundle_dt    <= pack_dt;
        bundle_hyp   <= pack_hyp;
        bundle_count <= pack_count;
      end
    end
  end

endmodule

`default_nettype wire
