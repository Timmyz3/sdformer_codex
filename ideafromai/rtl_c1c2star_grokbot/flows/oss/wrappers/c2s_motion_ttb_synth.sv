// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened Motion-TTB packer (Yosys cannot parse unpacked array ports).

`timescale 1ns/1ps
`default_nettype none

module c2s_motion_ttb_packer_synth #(
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
  input  logic [N_TILE*DT_W-1:0]            dt_bin_flat,
  input  logic [N_TILE*HYP_W-1:0]           hyp_id_flat,
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
        pack_dt  [cnt*DT_W   +: DT_W]   = dt_bin_flat[ti*DT_W +: DT_W];
        pack_hyp [cnt*HYP_W  +: HYP_W]  = hyp_id_flat[ti*HYP_W +: HYP_W];
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
