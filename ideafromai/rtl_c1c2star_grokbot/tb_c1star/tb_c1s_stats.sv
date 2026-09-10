// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_stats — wake / skip / delta_nz window counts

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_stats;
  localparam int N_TILE = 8;
  localparam int CNT_W  = 16;

  logic clk, rst_n, valid_i, window_en, clear_i;
  logic [N_TILE-1:0] wake_bitmap, skip_bitmap, delta_nz;
  logic [CNT_W-1:0]  wake_pop_cnt, proj_skip_cnt, delta_nz_cnt;
  logic              stats_valid;

  c1s_stats #(.N_TILE(N_TILE), .CNT_W(CNT_W)) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .window_en(window_en), .clear_i(clear_i),
    .wake_bitmap(wake_bitmap), .skip_bitmap(skip_bitmap), .delta_nz(delta_nz),
    .wake_pop_cnt(wake_pop_cnt), .proj_skip_cnt(proj_skip_cnt), .delta_nz_cnt(delta_nz_cnt),
    .stats_valid(stats_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_stats.vcd");
      $dumpvars(0, tb_c1s_stats);
    end
  end

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0; valid_i = 0; window_en = 0; clear_i = 0;
    wake_bitmap = 0; skip_bitmap = 0; delta_nz = 0;
    repeat (3) @(posedge clk);
    rst_n = 1; @(posedge clk); #1;

    // Case1: window accumulate two beats
    window_en = 1;
    wake_bitmap = 8'h0F; skip_bitmap = 8'hF0; delta_nz = 8'h03; // 4,4,2
    valid_i = 1; @(posedge clk); #1;
    if (!stats_valid || wake_pop_cnt !== 4 || proj_skip_cnt !== 4 || delta_nz_cnt !== 2) begin
      $display("FAIL Case1a %0d %0d %0d", wake_pop_cnt, proj_skip_cnt, delta_nz_cnt); $fatal(1);
    end
    wake_bitmap = 8'h01; skip_bitmap = 8'h01; delta_nz = 8'h01; // +1 each
    @(posedge clk); #1;
    if (wake_pop_cnt !== 5 || proj_skip_cnt !== 5 || delta_nz_cnt !== 3) begin
      $display("FAIL Case1b %0d %0d %0d", wake_pop_cnt, proj_skip_cnt, delta_nz_cnt); $fatal(1);
    end
    valid_i = 0; @(posedge clk); #1;
    $display("PASS Case1 accumulate");

    // Case2: clear
    clear_i = 1; @(posedge clk); #1;
    if (wake_pop_cnt !== 0 || proj_skip_cnt !== 0 || delta_nz_cnt !== 0 || stats_valid !== 0) begin
      $display("FAIL Case2 clear"); $fatal(1);
    end
    clear_i = 0;
    $display("PASS Case2 clear");

    // Case3: !window_en clears
    wake_bitmap = 8'hFF; valid_i = 1; @(posedge clk); #1;
    if (wake_pop_cnt !== 8) begin $display("FAIL Case3 setup"); $fatal(1); end
    window_en = 0; valid_i = 0; @(posedge clk); #1;
    if (wake_pop_cnt !== 0) begin $display("FAIL Case3 window off"); $fatal(1); end
    $display("PASS Case3 window_en off clears");

    $display("PASS: all c1s_stats cases");
    $finish;
  end

  initial begin #100000; $display("FAIL: timeout"); $fatal(1); end
endmodule

`default_nettype wire
