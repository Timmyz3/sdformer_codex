// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_tid_deblur_loop — TID-DeblurLoop (Card I1)
// INNOVATION: Corr-free iterative motion-compensate / deblur on event bins →
//             Δflow warm-start + deblur_wake / exact_pref. No 4D corr SRAM.
// Cite: IDNet ICRA'24 / arXiv:2211.13726 (ID+TID); enhance K1 wake + K3 exact.
// MODE_TID=1: use flow_hat_prev online; MODE_TID=0: ID same-batch from seed.
// ABLATE_NO_DEBLUR: passthrough raw bins, dflow=0 path weakened.
// NOT claimed: invent IDNet; Jetson latency as ASIC; MVSEC AEE.

`timescale 1ns/1ps
`default_nettype none

module c1s_tid_deblur_loop #(
  parameter int N_TILE   = 8,
  parameter int FLOW_W   = 8,
  parameter int BIN_W    = 8,
  parameter int N_BIN    = 4,
  parameter int N_ITER   = 2,
  parameter logic [FLOW_W-1:0] TH_DRES = 8'd6,
  parameter bit MODE_TID = 1'b1,
  parameter bit ABLATE_NO_DEBLUR = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic                         iter_fire_i,
  // Packed: tile t, bin b at [(t*N_BIN + b)*BIN_W +: BIN_W]
  input  logic [N_TILE*N_BIN*BIN_W-1:0] evt_bin_bus,
  input  logic [N_TILE*FLOW_W-1:0]     flow_seed_bus,
  input  logic [N_TILE*FLOW_W-1:0]     flow_hat_prev_bus,
  output logic [N_TILE*N_BIN*BIN_W-1:0] evt_deblur_bus,
  output logic [N_TILE*FLOW_W-1:0]     dflow_bus,
  output logic [N_TILE*FLOW_W-1:0]     flow_out_bus,
  output logic [N_TILE*FLOW_W-1:0]     flow_hat_next_bus,
  output logic [N_TILE-1:0]            deblur_wake,
  output logic                         exact_pref,
  output logic                         loop_valid
);

  logic [N_TILE*N_BIN*BIN_W-1:0] deb_c;
  logic [N_TILE*FLOW_W-1:0] dflow_c, fout_c, fhat_c;
  logic [N_TILE-1:0] wake_c;
  logic exact_c;
  integer t, b, src;
  logic [FLOW_W-1:0] seed_i, hat_i, fuse_i, d_i, fout_i;
  logic [BIN_W-1:0] bin_v, acc_l, acc_r, e_raw, e_deb;
  logic [FLOW_W:0]  abs_asym;
  logic [1:0]       shift_q;
  logic [FLOW_W-1:0] n_wake;

  always @(*) begin
    deb_c   = {N_TILE*N_BIN*BIN_W{1'b0}};
    dflow_c = {N_TILE*FLOW_W{1'b0}};
    fout_c  = {N_TILE*FLOW_W{1'b0}};
    fhat_c  = {N_TILE*FLOW_W{1'b0}};
    wake_c  = {N_TILE{1'b0}};
    n_wake  = {FLOW_W{1'b0}};
    for (t = 0; t < N_TILE; t = t + 1) begin
      seed_i = flow_seed_bus[t*FLOW_W +: FLOW_W];
      hat_i  = flow_hat_prev_bus[t*FLOW_W +: FLOW_W];
      fuse_i = MODE_TID ? hat_i : seed_i;
      // Quantize shift from fused flow (corr-free MC)
      shift_q = fuse_i[FLOW_W-1:FLOW_W-2]; // top 2 bits → 0..3

      acc_l = {BIN_W{1'b0}};
      acc_r = {BIN_W{1'b0}};
      e_raw = {BIN_W{1'b0}};
      e_deb = {BIN_W{1'b0}};
      for (b = 0; b < N_BIN; b = b + 1) begin
        bin_v = evt_bin_bus[(t*N_BIN + b)*BIN_W +: BIN_W];
        e_raw = e_raw + bin_v;
        if (ABLATE_NO_DEBLUR) begin
          deb_c[(t*N_BIN + b)*BIN_W +: BIN_W] = bin_v;
          src = b;
        end else begin
          // Circular shift bins opposite to motion (digital deblur stub)
          src = b - shift_q;
          if (src < 0) src = src + N_BIN;
          deb_c[(t*N_BIN + b)*BIN_W +: BIN_W] =
              evt_bin_bus[(t*N_BIN + src)*BIN_W +: BIN_W];
        end
        e_deb = e_deb + deb_c[(t*N_BIN + b)*BIN_W +: BIN_W];
        if (b < (N_BIN/2))
          acc_l = acc_l + deb_c[(t*N_BIN + b)*BIN_W +: BIN_W];
        else
          acc_r = acc_r + deb_c[(t*N_BIN + b)*BIN_W +: BIN_W];
      end

      if (acc_l >= acc_r)
        abs_asym = {1'b0, acc_l} - {1'b0, acc_r};
      else
        abs_asym = {1'b0, acc_r} - {1'b0, acc_l};

      // Residual Δflow: asymmetry + |e_deb - e_raw|/2 (+ seed-hat gap in TID)
      d_i = abs_asym[FLOW_W-1:0];
      if (!ABLATE_NO_DEBLUR) begin
        if (e_deb >= e_raw)
          d_i = d_i + ((e_deb - e_raw) >> 1);
        else
          d_i = d_i + ((e_raw - e_deb) >> 1);
        if (MODE_TID) begin
          if (seed_i >= hat_i)
            d_i = d_i + ((seed_i - hat_i) >> 2);
          else
            d_i = d_i + ((hat_i - seed_i) >> 2);
        end
      end else begin
        d_i = {FLOW_W{1'b0}};
      end

      // Warm-start flow_out = seed + sat(d/2) (unsigned sketch toward larger residual)
      fout_i = seed_i + (d_i >> 1);
      if (fout_i < seed_i) // overflow wrap guard
        fout_i = {FLOW_W{1'b1}};

      dflow_c[t*FLOW_W +: FLOW_W] = d_i;
      fout_c[t*FLOW_W +: FLOW_W]  = fout_i;
      // TID next hat = flow_out; ID can stay on seed+d for multi-iter
      fhat_c[t*FLOW_W +: FLOW_W]  = MODE_TID ? fout_i : fout_i;

      if (d_i >= TH_DRES) begin
        wake_c[t] = 1'b1;
        if (n_wake != {FLOW_W{1'b1}})
          n_wake = n_wake + {{(FLOW_W-1){1'b0}}, 1'b1};
      end
    end
    exact_c = (n_wake > (N_TILE[FLOW_W-1:0] >> 2)); // > N_TILE/4
    // Keep N_ITER visible to synth (multi-iter fire handled by TB/FSM)
    if (N_ITER == 0)
      exact_c = exact_c;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      loop_valid        <= 1'b0;
      evt_deblur_bus    <= '0;
      dflow_bus         <= '0;
      flow_out_bus      <= '0;
      flow_hat_next_bus <= '0;
      deblur_wake       <= '0;
      exact_pref        <= 1'b0;
    end else begin
      loop_valid <= valid_i && iter_fire_i;
      if (valid_i && iter_fire_i) begin
        evt_deblur_bus    <= deb_c;
        dflow_bus         <= dflow_c;
        flow_out_bus      <= fout_c;
        flow_hat_next_bus <= fhat_c;
        deblur_wake       <= wake_c;
        exact_pref        <= exact_c;
      end
    end
  end

endmodule

`default_nettype wire
