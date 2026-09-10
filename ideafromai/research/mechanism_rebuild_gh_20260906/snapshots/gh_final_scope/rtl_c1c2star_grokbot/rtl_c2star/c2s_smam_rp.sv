// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_smam_rp — SMAM-RP dual-rail Mask-Add × real ATLIF payload (C2*)
// Gate rail: accumulate +1 when spike_gate=1; payload rail: MAC enable + pass-through.

`timescale 1ns/1ps
`default_nettype none

module c2s_smam_rp #(
  parameter int AMP_W = 8,
  parameter int ACC_W = 16
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    valid,       // beat / handshake valid
  input  logic                    spike_gate,  // binary spike gate
  input  logic signed [AMP_W-1:0] payload,     // signed ATLIF payload
  // Dual-rail enables
  output logic                    mac_en,      // valid && spike_gate
  output logic                    mask_add_en, // same condition (mask-add path)
  // Payload rail: enabled only when gate=1
  output logic signed [AMP_W-1:0] mac_payload,
  // Mask-Add accumulate: +1 per gated beat (simple count-style)
  output logic signed [ACC_W-1:0] mask_add_result,
  output logic                    out_valid
);

  wire gate_fire = valid && spike_gate;

  assign mac_en      = gate_fire;
  assign mask_add_en = gate_fire;
  assign mac_payload = gate_fire ? payload : '0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mask_add_result <= '0;
      out_valid       <= 1'b0;
    end else begin
      out_valid <= valid;
      if (gate_fire)
        mask_add_result <= mask_add_result + {{(ACC_W-1){1'b0}}, 1'b1};
    end
  end

endmodule

`default_nettype wire
