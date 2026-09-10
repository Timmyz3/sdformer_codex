// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_stats — C1* window counters (wake pops / proj skips / delta_nz)
// INNOVATION: quantifies OP-STW wake vs ECP skip vs MW residual sparsity —
//             ablation metrics, not multiply-reorder bookkeeping.
// Over a window (window_en): accumulate popcounts; clear_i / !window_en resets.

`timescale 1ns/1ps
`default_nettype none

module c1s_stats #(
  parameter int N_TILE = 8,
  parameter int CNT_W  = 16
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  input  logic                 window_en,
  input  logic                 clear_i,
  input  logic [N_TILE-1:0]    wake_bitmap,
  input  logic [N_TILE-1:0]    skip_bitmap,
  input  logic [N_TILE-1:0]    delta_nz,
  output logic [CNT_W-1:0]     wake_pop_cnt,
  output logic [CNT_W-1:0]     proj_skip_cnt,
  output logic [CNT_W-1:0]     delta_nz_cnt,
  output logic                 stats_valid
);

  logic [CNT_W-1:0] pop_wake, pop_skip, pop_nz;
  integer           i;

  always @(*) begin
    pop_wake = {CNT_W{1'b0}};
    pop_skip = {CNT_W{1'b0}};
    pop_nz   = {CNT_W{1'b0}};
    for (i = 0; i < N_TILE; i = i + 1) begin
      if (wake_bitmap[i]) pop_wake = pop_wake + {{(CNT_W-1){1'b0}}, 1'b1};
      if (skip_bitmap[i]) pop_skip = pop_skip + {{(CNT_W-1){1'b0}}, 1'b1};
      if (delta_nz[i])    pop_nz   = pop_nz   + {{(CNT_W-1){1'b0}}, 1'b1};
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wake_pop_cnt  <= {CNT_W{1'b0}};
      proj_skip_cnt <= {CNT_W{1'b0}};
      delta_nz_cnt  <= {CNT_W{1'b0}};
      stats_valid   <= 1'b0;
    end else if (clear_i || !window_en) begin
      wake_pop_cnt  <= {CNT_W{1'b0}};
      proj_skip_cnt <= {CNT_W{1'b0}};
      delta_nz_cnt  <= {CNT_W{1'b0}};
      stats_valid   <= 1'b0;
    end else begin
      stats_valid <= valid_i;
      if (valid_i) begin
        wake_pop_cnt  <= wake_pop_cnt  + pop_wake;
        proj_skip_cnt <= proj_skip_cnt + pop_skip;
        delta_nz_cnt  <= delta_nz_cnt  + pop_nz;
      end
    end
  end

endmodule

`default_nettype wire
