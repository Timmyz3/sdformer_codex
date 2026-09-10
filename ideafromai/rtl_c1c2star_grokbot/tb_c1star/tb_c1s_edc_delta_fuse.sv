// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_edc_delta_fuse — N_TILE=8; FULL vs CORR_ONLY vs DIFF_ONLY

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_edc_delta_fuse;
  localparam int N_TILE = 8;
  localparam int FEAT_W = 8;
  localparam int CORR_W = 16;
  localparam int MOT_W  = 8;

  logic clk, rst_n, valid_i;
  logic [N_TILE*FEAT_W-1:0] feat_t0_bus, feat_tn_bus;
  logic [N_TILE*CORR_W-1:0] corr_lo_bus;
  logic [N_TILE-1:0] match_ok;

  logic [N_TILE*MOT_W-1:0] mot_diff, mot_fuse, mot_diff_c, mot_fuse_c;
  logic [N_TILE*MOT_W-1:0] mot_diff_d, mot_fuse_d;
  logic [N_TILE-1:0] detail_hit, exact_boost;
  logic [N_TILE-1:0] detail_c, boost_c, detail_d, boost_d;
  logic fuse_valid, fuse_valid_c, fuse_valid_d;

  integer t;
  int n_pass, sum_diff, sum_fuse, cnt_detail, cnt_boost;
  int sum_fuse_c, sum_fuse_d, cnt_detail_c, cnt_detail_d;

  c1s_edc_delta_fuse #(
    .N_TILE(N_TILE), .TH_DETAIL(8'd32),
    .ABLATE_CORR_ONLY(1'b0), .ABLATE_DIFF_ONLY(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .feat_t0_bus(feat_t0_bus), .feat_tn_bus(feat_tn_bus),
    .corr_lo_bus(corr_lo_bus), .match_ok(match_ok),
    .mot_diff_bus(mot_diff), .mot_fuse_bus(mot_fuse),
    .detail_hit(detail_hit), .exact_boost(exact_boost), .fuse_valid(fuse_valid)
  );

  c1s_edc_delta_fuse #(
    .N_TILE(N_TILE), .TH_DETAIL(8'd32),
    .ABLATE_CORR_ONLY(1'b1), .ABLATE_DIFF_ONLY(1'b0)
  ) dut_corr (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .feat_t0_bus(feat_t0_bus), .feat_tn_bus(feat_tn_bus),
    .corr_lo_bus(corr_lo_bus), .match_ok(match_ok),
    .mot_diff_bus(mot_diff_c), .mot_fuse_bus(mot_fuse_c),
    .detail_hit(detail_c), .exact_boost(boost_c), .fuse_valid(fuse_valid_c)
  );

  c1s_edc_delta_fuse #(
    .N_TILE(N_TILE), .TH_DETAIL(8'd32),
    .ABLATE_CORR_ONLY(1'b0), .ABLATE_DIFF_ONLY(1'b1)
  ) dut_diff (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .feat_t0_bus(feat_t0_bus), .feat_tn_bus(feat_tn_bus),
    .corr_lo_bus(corr_lo_bus), .match_ok(match_ok),
    .mot_diff_bus(mot_diff_d), .mot_fuse_bus(mot_fuse_d),
    .detail_hit(detail_d), .exact_boost(boost_d), .fuse_valid(fuse_valid_d)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_edc_delta_fuse.vcd");
      $dumpvars(0, tb_c1s_edc_delta_fuse);
    end
  end
  initial clk=0; always #5 clk=~clk;

  function automatic int popc(input logic [N_TILE-1:0] m);
    int k,c; begin c=0; for(k=0;k<N_TILE;k=k+1) if(m[k]) c=c+1; popc=c; end
  endfunction

  task automatic clear_in;
    begin valid_i=0; feat_t0_bus='0; feat_tn_bus='0; corr_lo_bus='0; match_ok='0; end
  endtask

  task automatic beat;
    begin @(negedge clk); valid_i=1; @(posedge clk); #1; valid_i=0; end
  endtask

  task automatic accum;
    begin
      for (t=0;t<N_TILE;t=t+1) begin
        sum_diff = sum_diff + mot_diff[t*MOT_W +: MOT_W];
        sum_fuse = sum_fuse + mot_fuse[t*MOT_W +: MOT_W];
        sum_fuse_c = sum_fuse_c + mot_fuse_c[t*MOT_W +: MOT_W];
        sum_fuse_d = sum_fuse_d + mot_fuse_d[t*MOT_W +: MOT_W];
      end
      cnt_detail = cnt_detail + popc(detail_hit);
      cnt_boost  = cnt_boost  + popc(exact_boost);
      cnt_detail_c = cnt_detail_c + popc(detail_c);
      cnt_detail_d = cnt_detail_d + popc(detail_d);
    end
  endtask

  initial begin
    n_pass=0; sum_diff=0; sum_fuse=0; cnt_detail=0; cnt_boost=0;
    sum_fuse_c=0; sum_fuse_d=0; cnt_detail_c=0; cnt_detail_d=0;
    rst_n=0; clear_in();
    repeat(3) @(posedge clk); rst_n=1; @(posedge clk);

    // Case1: large Δ + mid corr → FULL detail
    clear_in();
    match_ok = 8'hFF;
    for (t=0;t<N_TILE;t=t+1) begin
      feat_t0_bus[t*FEAT_W +: FEAT_W] = 8'sd80;
      feat_tn_bus[t*FEAT_W +: FEAT_W] = 8'sd10; // |Δ|=70
      corr_lo_bus[t*CORR_W +: CORR_W] = 16'd40;
    end
    beat(); accum();
    if (!fuse_valid) begin $display("FAIL C1: fuse_valid"); $fatal(1); end
    if (detail_hit !== 8'hFF) begin
      $display("FAIL C1: detail=%b fuse0=%0d", detail_hit, mot_fuse[0*MOT_W +: MOT_W]);
      $fatal(1);
    end
    if (exact_boost !== detail_hit) begin
      $display("FAIL C1: boost=%b", exact_boost); $fatal(1);
    end
    $display("PASS Case1: detail=%b fuse0=%0d diff0=%0d",
             detail_hit, mot_fuse[0*MOT_W +: MOT_W], mot_diff[0*MOT_W +: MOT_W]);
    n_pass = n_pass + 1;

    // Case2: tiny Δ + tiny corr → no detail
    clear_in();
    match_ok = 8'hFF;
    for (t=0;t<N_TILE;t=t+1) begin
      feat_t0_bus[t*FEAT_W +: FEAT_W] = 8'sd5;
      feat_tn_bus[t*FEAT_W +: FEAT_W] = 8'sd4;
      corr_lo_bus[t*CORR_W +: CORR_W] = 16'd2;
    end
    beat(); accum();
    if (detail_hit !== '0) begin $display("FAIL C2: detail=%b", detail_hit); $fatal(1); end
    $display("PASS Case2: no detail");
    n_pass = n_pass + 1;

    // Case3: CORR_ONLY with high corr, low Δ — CORR hits, DIFF may miss
    clear_in();
    match_ok = 8'h0F;
    for (t=0;t<N_TILE;t=t+1) begin
      feat_t0_bus[t*FEAT_W +: FEAT_W] = 8'sd20;
      feat_tn_bus[t*FEAT_W +: FEAT_W] = 8'sd18; // small Δ
      corr_lo_bus[t*CORR_W +: CORR_W] = 16'd100; // high corr
    end
    beat(); accum();
    if (detail_c === '0) begin
      $display("FAIL C3: CORR_ONLY should hit fuse=%0d", mot_fuse_c[0*MOT_W +: MOT_W]);
      $fatal(1);
    end
    $display("PASS Case3: CORR_ONLY detail=%b DIFF detail=%b FULL=%b",
             detail_c, detail_d, detail_hit);
    n_pass = n_pass + 1;

    // Ablation energy: FULL fuse between pure extremes typically
    if (sum_fuse == 0) begin $display("FAIL: sum_fuse=0"); $fatal(1); end
    $display("ABLATION sum_mot_diff=%0d sum_mot_fuse=%0d sum_CORR_ONLY=%0d sum_DIFF_ONLY=%0d",
             sum_diff, sum_fuse, sum_fuse_c, sum_fuse_d);
    $display("COUNTERS cnt_detail_hit=%0d cnt_exact_boost=%0d cnt_detail_CORR=%0d cnt_detail_DIFF=%0d",
             cnt_detail, cnt_boost, cnt_detail_c, cnt_detail_d);
    $display("PASS: all EDC-ΔFuse cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin #200000; $display("FAIL: timeout"); $fatal(1); end
endmodule
`default_nettype wire
