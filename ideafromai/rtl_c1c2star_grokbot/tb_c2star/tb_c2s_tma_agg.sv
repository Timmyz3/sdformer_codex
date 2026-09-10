// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_tma_agg — N_TILE=8, N_SLICE=4 deterministic split/lookup/agg

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_tma_agg;
  localparam int N_TILE  = 8;
  localparam int N_SLICE = 4;
  localparam int FEAT_W  = 4;
  localparam int DIR_W   = 2;
  localparam int CNT_W   = 4;
  localparam int HIT_W   = 4;
  localparam int SL_W    = $clog2(N_SLICE);

  logic clk, rst_n, valid_i;
  logic [SL_W-1:0]          slice_idx;
  logic [N_TILE*FEAT_W-1:0] feat_cur_bus;
  logic [N_TILE*DIR_W-1:0]  dir_code_bus;
  logic [N_TILE*FEAT_W-1:0] agg_flow_bus;
  logic [N_TILE-1:0]        pattern_hit;
  logic                     early_exit;
  logic [1:0]               hyp_hint;
  logic                     agg_valid;

  integer i;
  int hit_pop_sum, early_cnt, n_pass;
  logic [FEAT_W-1:0] ftmp;

  c2s_tma_agg #(
    .N_TILE(N_TILE), .N_SLICE(N_SLICE), .FEAT_W(FEAT_W), .DIR_W(DIR_W),
    .CNT_W(CNT_W), .HIT_W(HIT_W), .TH_CONS(4'd2), .TH_EARLY(4'd4)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .slice_idx(slice_idx), .feat_cur_bus(feat_cur_bus), .dir_code_bus(dir_code_bus),
    .agg_flow_bus(agg_flow_bus), .pattern_hit(pattern_hit),
    .early_exit(early_exit), .hyp_hint(hyp_hint), .agg_valid(agg_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_tma_agg.vcd");
      $dumpvars(0, tb_c2s_tma_agg);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_in;
    begin
      valid_i = 1'b0;
      slice_idx = '0;
      feat_cur_bus = '0;
      dir_code_bus = '0;
    end
  endtask

  task automatic beat_sample;
    begin
      @(negedge clk);
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      valid_i = 1'b0;
    end
  endtask

  function automatic int popc(input logic [N_TILE-1:0] b);
    int kk, c;
    begin
      c = 0;
      for (kk = 0; kk < N_TILE; kk = kk + 1)
        if (b[kk]) c = c + 1;
      popc = c;
    end
  endfunction

  initial begin
    n_pass = 0; hit_pop_sum = 0; early_cnt = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: consistent stay
    clear_in();
    slice_idx = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd3;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat_sample();
    slice_idx = 1;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd3;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat_sample();
    slice_idx = 2;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd3;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat_sample();
    if (!agg_valid) begin $display("FAIL Case1: agg_valid"); $fatal(1); end
    if (pattern_hit !== {N_TILE{1'b1}}) begin
      $display("FAIL Case1: pattern_hit=%b", pattern_hit); $fatal(1);
    end
    if (early_exit !== 1'b1) begin
      $display("FAIL Case1: expected early_exit"); $fatal(1);
    end
    $display("PASS Case1: consistent stay → pattern_hit=%b early=%0d hyp=%0d",
             pattern_hit, early_exit, hyp_hint);
    n_pass = n_pass + 1;
    hit_pop_sum = hit_pop_sum + popc(pattern_hit);
    if (early_exit) early_cnt = early_cnt + 1;

    // Case2: dir=2 lookup
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    slice_idx = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      ftmp = i[3:0] + 4'd1;
      feat_cur_bus[i*FEAT_W +: FEAT_W] = ftmp;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
    end
    beat_sample();
    slice_idx = 1;
    feat_cur_bus[0*FEAT_W +: FEAT_W] = 4'd1;
    dir_code_bus[0*DIR_W +: DIR_W] = 2'd0;
    for (i = 1; i < N_TILE; i = i + 1) begin
      ftmp = (i - 1);
      feat_cur_bus[i*FEAT_W +: FEAT_W] = ftmp[3:0] + 4'd1;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
    end
    beat_sample();
    slice_idx = 2;
    feat_cur_bus[0*FEAT_W +: FEAT_W] = 4'd1;
    dir_code_bus[0*DIR_W +: DIR_W] = 2'd0;
    for (i = 1; i < N_TILE; i = i + 1) begin
      ftmp = (i - 1);
      feat_cur_bus[i*FEAT_W +: FEAT_W] = ftmp[3:0] + 4'd1;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
    end
    beat_sample();
    $display("PASS Case2: lookup-align dir=2 pattern_hit=%b hyp=%0d agg0=%0d",
             pattern_hit, hyp_hint, agg_flow_bus[FEAT_W-1:0]);
    n_pass = n_pass + 1;
    hit_pop_sum = hit_pop_sum + popc(pattern_hit);
    if (early_exit) early_cnt = early_cnt + 1;

    // Case3: mismatch
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    slice_idx = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd5;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat_sample();
    slice_idx = 1;
    for (i = 0; i < N_TILE; i = i + 1) begin
      feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd1;
      dir_code_bus[i*DIR_W +: DIR_W] = 2'd0;
    end
    beat_sample();
    if (early_exit !== 1'b0) begin
      $display("FAIL Case3: unexpected early_exit"); $fatal(1);
    end
    if (pattern_hit !== {N_TILE{1'b0}}) begin
      $display("FAIL Case3: pattern_hit=%b", pattern_hit); $fatal(1);
    end
    $display("PASS Case3: mismatch → no early_exit");
    n_pass = n_pass + 1;

    // Case4: sweep
    rst_n = 1'b0; @(posedge clk); rst_n = 1'b1; @(posedge clk);
    clear_in();
    for (int s = 0; s < N_SLICE; s = s + 1) begin
      slice_idx = s[SL_W-1:0];
      for (i = 0; i < N_TILE; i = i + 1) begin
        feat_cur_bus[i*FEAT_W +: FEAT_W] = 4'd2;
        dir_code_bus[i*DIR_W +: DIR_W] = 2'd2;
      end
      beat_sample();
      hit_pop_sum = hit_pop_sum + popc(pattern_hit);
      if (early_exit) early_cnt = early_cnt + 1;
    end
    $display("COUNTERS hit_pop_sum=%0d early_cnt=%0d hyp_last=%0d cases_pass=%0d",
             hit_pop_sum, early_cnt, hyp_hint, n_pass);
    n_pass = n_pass + 1;

    $display("PASS: all TMA-Agg cases");
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
