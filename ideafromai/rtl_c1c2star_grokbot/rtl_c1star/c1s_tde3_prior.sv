// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_tde3_prior — TDE3-Prior digital time-difference OF wake prior (C1*)
// INNOVATION: Bio-inspired TDE-3 time-difference prior seeds OF wake —
//             enhances OP-STW, NOT zero-skip / flat quota.
// NOT claimed: Loihi silicon, analog TDE, DualRail-CIM arrays.
// Ports: dir/conf PACKED (iverilog-friendly); tile i at [i*W +: W].

`timescale 1ns/1ps
`default_nettype none

module c1s_tde3_prior #(
  parameter int N_TILE  = 8,
  parameter int AGE_W   = 8,
  parameter int CONF_W  = 4,
  parameter int DIR_W   = 2,
  parameter logic [AGE_W-1:0]  TH_AGE  = 8'd8,
  parameter logic [CONF_W-1:0] TH_WAKE = 4'd2,
  parameter logic [CONF_W-1:0] INHIB_PENALTY = 4'd2
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [N_TILE-1:0]            event_pulse,
  input  logic [N_TILE-1:0]            polarity,
  input  logic [N_TILE*2-1:0]          flow_hint_bus, // tile i: [i*2 +: 2]
  output logic [N_TILE*DIR_W-1:0]      dir_code_bus,
  output logic [N_TILE*CONF_W-1:0]     tde_conf_bus,
  output logic [N_TILE-1:0]            tde_wake,
  output logic                         tde_valid
);

  logic [AGE_W-1:0] age [N_TILE];
  logic [N_TILE*DIR_W-1:0]  dir_c;
  logic [N_TILE*CONF_W-1:0] conf_c;
  logic [N_TILE-1:0]        wake_c;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tde
      wire [1:0] flow_hint_i = flow_hint_bus[gi*2 +: 2];
      wire [AGE_W-1:0] age_l;
      wire [AGE_W-1:0] age_r;
      if (gi == 0) begin : g_l0
        assign age_l = {AGE_W{1'b1}};
      end else begin : g_ln
        assign age_l = age[gi-1];
      end
      if (gi == (N_TILE-1)) begin : g_r0
        assign age_r = {AGE_W{1'b1}};
      end else begin : g_rn
        assign age_r = age[gi+1];
      end

      wire [AGE_W:0] a_l_x = {1'b0, age_l};
      wire [AGE_W:0] a_r_x = {1'b0, age_r};
      wire left_younger  = (age_l < age_r);
      wire right_younger = (age_r < age_l);
      wire [AGE_W:0] gap_raw = left_younger  ? (a_r_x - a_l_x) :
                               right_younger ? (a_l_x - a_r_x) :
                                               {(AGE_W+1){1'b0}};
      wire [CONF_W-1:0] gap_c = (gap_raw > {{(AGE_W+1-4){1'b0}}, 4'd7})
                                  ? 4'd7 : gap_raw[CONF_W-1:0];
      wire [DIR_W-1:0] dir_raw =
          (!event_pulse[gi]) ? 2'd0 :
          left_younger       ? 2'd2 :
          right_younger      ? 2'd1 : 2'd3;

      wire boost_hint = event_pulse[gi] && (flow_hint_i != 2'b00) &&
                        (flow_hint_i == dir_raw) && (dir_raw != 2'd3);
      wire boost_pol  = event_pulse[gi] && polarity[gi];
      wire [CONF_W-1:0] base0 = event_pulse[gi] ? gap_c : {CONF_W{1'b0}};
      wire [CONF_W-1:0] base1 = (boost_hint && (base0 < 4'd7)) ? (base0 + 4'd1) : base0;
      wire [CONF_W-1:0] base2 = (boost_pol  && (base1 < 4'd7)) ? (base1 + 4'd1) : base1;

      wire both_fresh = (age_l < TH_AGE) && (age_r < TH_AGE);
      wire conflict = event_pulse[gi] && both_fresh &&
                      ((dir_raw == 2'd3) || (gap_raw <= {{(AGE_W+1-2){1'b0}}, 2'd1}));
      wire [CONF_W-1:0] conf_raw =
          (!event_pulse[gi]) ? {CONF_W{1'b0}} :
          (conflict && (base2 > INHIB_PENALTY)) ? (base2 - INHIB_PENALTY) :
          conflict ? {CONF_W{1'b0}} : base2;

      assign dir_c[gi*DIR_W +: DIR_W]   = dir_raw;
      assign conf_c[gi*CONF_W +: CONF_W] = conf_raw;
      assign wake_c[gi] = event_pulse[gi] && (conf_raw >= TH_WAKE);
    end
  endgenerate

  integer ti;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tde_valid    <= 1'b0;
      tde_wake     <= '0;
      dir_code_bus <= '0;
      tde_conf_bus <= '0;
      for (ti = 0; ti < N_TILE; ti = ti + 1)
        age[ti] <= {AGE_W{1'b1}};
    end else begin
      tde_valid <= valid_i;
      if (valid_i) begin
        tde_wake     <= wake_c;
        dir_code_bus <= dir_c;
        tde_conf_bus <= conf_c;
        for (ti = 0; ti < N_TILE; ti = ti + 1) begin
          if (event_pulse[ti])
            age[ti] <= {AGE_W{1'b0}};
          else if (age[ti] != {AGE_W{1'b1}})
            age[ti] <= age[ti] + {{(AGE_W-1){1'b0}}, 1'b1};
        end
      end
    end
  end

endmodule

`default_nettype wire
