// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_bisat_agg — N_TILE=8; FWD_ONLY vs FULL ablation + prove-by counters

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_bisat_agg;
  localparam int N_TILE  = 8;
  localparam int N_SLICE = 4;
  localparam int FEAT_W  = 4;
  localparam int DIR_W   = 2;
  localparam int HIT_W   = 4;
  localparam int SL_W    = $clog2(N_SLICE);

  logic clk, rst_n, valid_i, bwd_en_i;
  logic [SL_W-1:0]          slice_idx;
  logic [N_TILE*FEAT_W-1:0] feat_fwd_bus, feat_bwd_bus, tma_agg_flow_bus;
  logic [N_TILE*DIR_W-1:0]  dir_code_bus;

  // DUT FULL
  logic [N_TILE*FEAT_W-1:0] fuse_flow_bus;
  logic [N_TILE-1:0]        fuse_hit;
  logic                     occ_exit_hint;
  logic [1:0]               hyp_hint_o;
  logic                     fuse_valid;

  // Ablation twin: FWD_ONLY
  logic [N_TILE*FEAT_W-1:0] fuse_flow_a;
  logic [N_TILE-1:0]        fuse_hit_a;
  logic                     occ_exit_a;
  logic [1:0]               hyp_a;
  logic                     fuse_valid_a;

  integer i, k;
  int n_pass;
  longint cnt_fwd_only, cnt_bwd_used, cnt_fuse_hit, cnt_occ_exit;
  longint cnt_fuse_hit_a, cnt_occ_a;
  int pop_hit;
  logic differ;

  c2s_bisat_agg #(
    .N_TILE(N_TILE), .N_SLICE(N_SLICE), .FEAT_W(FEAT_W), .DIR_W(DIR_W),
    .HIT_W(HIT_W), .TH_FUSE(4'd3), .ABLATE_FWD_ONLY(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .bwd_en_i(bwd_en_i),
    .slice_idx(slice_idx),
    .feat_fwd_bus(feat_fwd_bus), .feat_bwd_bus(feat_bwd_bus),
    .dir_code_bus(dir_code_bus), .tma_agg_flow_bus(tma_agg_flow_bus),
    .fuse_flow_bus(fuse_flow_bus), .fuse_hit(fuse_hit),
    .occ_exit_hint(occ_exit_hint), .hyp_hint_o(hyp_hint_o), .fuse_valid(fuse_valid)
  );

  c2s_bisat_agg #(
    .N_TILE(N_TILE), .N_SLICE(N_SLICE), .FEAT_W(FEAT_W), .DIR_W(DIR_W),
    .HIT_W(HIT_W), .TH_FUSE(4'd3), .ABLATE_FWD_ONLY(1'b1)
  ) dut_ablate (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .bwd_en_i(bwd_en_i),
    .slice_idx(slice_idx),
    .feat_fwd_bus(feat_fwd_bus), .feat_bwd_bus(feat_bwd_bus),
    .dir_code_bus(dir_code_bus), .tma_agg_flow_bus(tma_agg_flow_bus),
    .fuse_flow_bus(fuse_flow_a), .fuse_hit(fuse_hit_a),
    .occ_exit_hint(occ_exit_a), .hyp_hint_o(hyp_a), .fuse_valid(fuse_valid_a)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_bisat_agg.vcd");
      $dumpvars(0, tb_c2s_bisat_agg);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic int popc(input logic [N_TILE-1:0] b);
    int kk, c;
    begin
      c = 0;
      for (kk = 0; kk < N_TILE; kk = kk + 1)
        if (b[kk]) c = c + 1;
      popc = c;
    end
  endfunction

  task automatic clear_in;
    begin
      valid_i = 1'b0;
      bwd_en_i = 1'b0;
      slice_idx = '0;
      feat_fwd_bus = '0;
      feat_bwd_bus = '0;
      dir_code_bus = '0;
      tma_agg_flow_bus = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk);
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      valid_i = 1'b0;
    end
  endtask

  task automatic accum;
    begin
      if (!bwd_en_i)
        cnt_fwd_only = cnt_fwd_only + 1;
      else begin
        differ = 1'b0;
        for (k = 0; k < N_TILE; k = k + 1)
          if (fuse_flow_bus[k*FEAT_W +: FEAT_W] !== tma_agg_flow_bus[k*FEAT_W +: FEAT_W])
            differ = 1'b1;
        if (differ)
          cnt_bwd_used = cnt_bwd_used + 1;
      end
      pop_hit = popc(fuse_hit);
      cnt_fuse_hit = cnt_fuse_hit + pop_hit;
      cnt_fuse_hit_a = cnt_fuse_hit_a + popc(fuse_hit_a);
      if (occ_exit_hint) cnt_occ_exit = cnt_occ_exit + 1;
      if (occ_exit_a)    cnt_occ_a    = cnt_occ_a + 1;
    end
  endtask

  initial begin
    n_pass = 0;
    cnt_fwd_only = 0; cnt_bwd_used = 0; cnt_fuse_hit = 0; cnt_occ_exit = 0;
    cnt_fuse_hit_a = 0; cnt_occ_a = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: bwd disabled → fuse ≈ tma, fuse_hit=0
    clear_in();
    bwd_en_i = 1'b0;
    slice_idx = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_fwd_bus[i*FEAT_W +: FEAT_W] = 4'd5;
      feat_bwd_bus[i*FEAT_W +: FEAT_W] = 4'd5;
      tma_agg_flow_bus[i*FEAT_W +: FEAT_W] = 4'd7;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat();
    accum();
    if (!fuse_valid) begin $display("FAIL Case1: fuse_valid"); $fatal(1); end
    if (fuse_flow_bus !== tma_agg_flow_bus) begin
      $display("FAIL Case1: fuse!=tma %h vs %h", fuse_flow_bus, tma_agg_flow_bus);
      $fatal(1);
    end
    if (fuse_hit !== {N_TILE{1'b0}}) begin
      $display("FAIL Case1: fuse_hit=%b", fuse_hit); $fatal(1);
    end
    if (occ_exit_hint !== 1'b0) begin
      $display("FAIL Case1: unexpected occ"); $fatal(1);
    end
    $display("PASS Case1: bwd_off → fuse=tma fuse_hit=0");
    n_pass = n_pass + 1;

    // Case2: agree path builds fuse_hit over TH_FUSE=3
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    bwd_en_i = 1'b1;
    for (int s = 0; s < 4; s = s + 1) begin
      slice_idx = s[SL_W-1:0];
      for (i = 0; i < N_TILE; i = i + 1) begin
        feat_fwd_bus[i*FEAT_W +: FEAT_W] = 4'd4;
        feat_bwd_bus[i*FEAT_W +: FEAT_W] = 4'd4;
        tma_agg_flow_bus[i*FEAT_W +: FEAT_W] = 4'd2;
        dir_code_bus[i*DIR_W +: DIR_W] = 2'd1;
      end
      beat();
      accum();
    end
    if (fuse_hit !== {N_TILE{1'b1}}) begin
      $display("FAIL Case2: expected all fuse_hit got %b", fuse_hit); $fatal(1);
    end
    if (hyp_hint_o !== 2'd1) begin
      $display("FAIL Case2: hyp_hint_o=%0d expect 1", hyp_hint_o); $fatal(1);
    end
    // Ablation FWD_ONLY must stay hit-free
    if (fuse_hit_a !== {N_TILE{1'b0}}) begin
      $display("FAIL Case2: ablate hit=%b", fuse_hit_a); $fatal(1);
    end
    $display("PASS Case2: agree → fuse_hit=all hyp=1 ablate_hit=0");
    n_pass = n_pass + 1;

    // Case3: disagree on both edges → occ_exit_hint
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    bwd_en_i = 1'b1;
    slice_idx = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_fwd_bus[i*FEAT_W +: FEAT_W] = 4'd6;
      feat_bwd_bus[i*FEAT_W +: FEAT_W] = 4'd1;  // disagree
      tma_agg_flow_bus[i*FEAT_W +: FEAT_W] = 4'd4;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
    end
    beat();
    accum();
    if (occ_exit_hint !== 1'b1) begin
      $display("FAIL Case3: expected occ_exit_hint"); $fatal(1);
    end
    if (occ_exit_a !== 1'b0) begin
      $display("FAIL Case3: ablate should not occ"); $fatal(1);
    end
    // Suppressed fuse should be tma>>1 = 2
    if (fuse_flow_bus[FEAT_W-1:0] !== 4'd2) begin
      $display("FAIL Case3: fuse0=%0d expect 2", fuse_flow_bus[FEAT_W-1:0]);
      $fatal(1);
    end
    $display("PASS Case3: edge disagree → occ_exit fuse_suppressed");
    n_pass = n_pass + 1;

    // Case4: mixed + counters dump
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    bwd_en_i = 1'b1;
    for (int s = 0; s < N_SLICE; s = s + 1) begin
      slice_idx = s[SL_W-1:0];
      for (i = 0; i < N_TILE; i = i + 1) begin
        if (i < 4) begin
          feat_fwd_bus[i*FEAT_W +: FEAT_W] = 4'd3;
          feat_bwd_bus[i*FEAT_W +: FEAT_W] = 4'd3;
          dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
        end else begin
          feat_fwd_bus[i*FEAT_W +: FEAT_W] = 4'd5;
          feat_bwd_bus[i*FEAT_W +: FEAT_W] = 4'd2;
          dir_code_bus[i*DIR_W +: DIR_W] = 2'd1;
        end
        tma_agg_flow_bus[i*FEAT_W +: FEAT_W] = 4'd3;
      end
      beat();
      accum();
    end
    $display("COUNTERS cnt_fwd_only=%0d cnt_bwd_used=%0d cnt_fuse_hit=%0d cnt_occ_exit=%0d",
             cnt_fwd_only, cnt_bwd_used, cnt_fuse_hit, cnt_occ_exit);
    $display("ABLATION FULL_hit_sum=%0d FWD_ONLY_hit_sum=%0d occ_FULL=%0d occ_FWD=%0d",
             cnt_fuse_hit, cnt_fuse_hit_a, cnt_occ_exit, cnt_occ_a);
    if (cnt_fuse_hit <= cnt_fuse_hit_a) begin
      $display("FAIL Case4: FULL should accumulate more fuse_hit than FWD_ONLY");
      $fatal(1);
    end
    if (cnt_bwd_used == 0) begin
      $display("FAIL Case4: expected cnt_bwd_used>0"); $fatal(1);
    end
    $display("PASS Case4: ablation FULL>hit FWD_ONLY; counters OK");
    n_pass = n_pass + 1;

    $display("PASS: all BiSAT-Agg cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
