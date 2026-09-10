// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened back_pipe @ N_HEAD=8 (Yosys + unpacked STH scores).
// Instantiates HBG + SMAM RTL + STH flat synth wrapper; g/p hold for SMAM.

`timescale 1ns/1ps
`default_nettype none

module c2s_back_pipe_synth #(
  parameter int                 AMP_W   = 8,
  parameter int                 ACC_W   = 16,
  parameter int                 N_HEAD  = 8,
  parameter int                 SCORE_W = 8,
  parameter logic [SCORE_W-1:0] TH_SP   = 8'd2,
  parameter logic [SCORE_W-1:0] TH_TP   = 8'd2
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         amp_valid,
  input  logic signed [AMP_W-1:0]      amp,
  output logic                         g,
  output logic signed [AMP_W-1:0]      p,
  output logic                         gp_valid,
  output logic                         pe_clk_en,
  output logic                         mac_en,
  output logic                         mask_add_en,
  output logic signed [AMP_W-1:0]      mac_payload,
  output logic signed [ACC_W-1:0]      mask_add_result,
  output logic                         smam_out_valid,
  input  logic                         sth_valid_i,
  input  logic [N_HEAD*SCORE_W-1:0]    sth_spat_score_flat,
  input  logic [N_HEAD*SCORE_W-1:0]    sth_temp_score_flat,
  output logic [N_HEAD-1:0]            sth_spat_en,
  output logic [N_HEAD-1:0]            sth_temp_en,
  output logic [N_HEAD-1:0]            sth_dual_en,
  output logic [N_HEAD-1:0]            sth_skip_en,
  output logic                         sth_gate_valid
);

  logic                      g_q;
  logic signed [AMP_W-1:0]   p_q;

  c2s_hbg_rp_packetizer #(
    .AMP_W(AMP_W)
  ) u_hbg_rp (
    .clk      (clk),
    .rst_n    (rst_n),
    .amp_valid(amp_valid),
    .amp      (amp),
    .g        (g),
    .p        (p),
    .gp_valid (gp_valid),
    .pe_clk_en(pe_clk_en)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      g_q <= 1'b0;
      p_q <= '0;
    end else if (amp_valid) begin
      g_q <= g;
      p_q <= p;
    end
  end

  c2s_smam_rp #(
    .AMP_W(AMP_W),
    .ACC_W(ACC_W)
  ) u_smam_rp (
    .clk            (clk),
    .rst_n          (rst_n),
    .valid          (gp_valid),
    .spike_gate     (g_q),
    .payload        (p_q),
    .mac_en         (mac_en),
    .mask_add_en    (mask_add_en),
    .mac_payload    (mac_payload),
    .mask_add_result(mask_add_result),
    .out_valid      (smam_out_valid)
  );

  c2s_sth_gate_synth #(
    .N_HEAD (N_HEAD),
    .SCORE_W(SCORE_W),
    .TH_SP  (TH_SP),
    .TH_TP  (TH_TP)
  ) u_sth_gate (
    .clk            (clk),
    .rst_n          (rst_n),
    .valid_i        (sth_valid_i),
    .spat_score_flat(sth_spat_score_flat),
    .temp_score_flat(sth_temp_score_flat),
    .spat_en        (sth_spat_en),
    .temp_en        (sth_temp_en),
    .dual_en        (sth_dual_en),
    .skip_en        (sth_skip_en),
    .gate_valid     (sth_gate_valid)
  );

endmodule

`default_nettype wire
