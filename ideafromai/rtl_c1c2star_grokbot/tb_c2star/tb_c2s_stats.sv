// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_stats — mac_en / skip / gate_fire window counts

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_stats;
  localparam int CNT_W = 16;

  logic clk, rst_n, valid_i, window_en, clear_i;
  logic mac_en, skipped, gate_fire;
  logic [CNT_W-1:0] mac_en_cnt, skip_cnt, gate_fire_cnt;
  logic stats_valid;

  c2s_stats #(.CNT_W(CNT_W)) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .window_en(window_en), .clear_i(clear_i),
    .mac_en(mac_en), .skipped(skipped), .gate_fire(gate_fire),
    .mac_en_cnt(mac_en_cnt), .skip_cnt(skip_cnt), .gate_fire_cnt(gate_fire_cnt),
    .stats_valid(stats_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_stats.vcd");
      $dumpvars(0, tb_c2s_stats);
    end
  end

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0; valid_i = 0; window_en = 0; clear_i = 0;
    mac_en = 0; skipped = 0; gate_fire = 0;
    repeat (3) @(posedge clk);
    rst_n = 1; @(posedge clk); #1;

    window_en = 1;
    // Case1: three beats — mac, skip, gate variously
    mac_en = 1; skipped = 0; gate_fire = 1; valid_i = 1;
    @(posedge clk); #1;
    mac_en = 1; skipped = 1; gate_fire = 0;
    @(posedge clk); #1;
    mac_en = 0; skipped = 1; gate_fire = 1;
    @(posedge clk); #1;
    valid_i = 0;
    if (mac_en_cnt !== 2 || skip_cnt !== 2 || gate_fire_cnt !== 2) begin
      $display("FAIL Case1 mac=%0d skip=%0d gate=%0d", mac_en_cnt, skip_cnt, gate_fire_cnt);
      $fatal(1);
    end
    $display("PASS Case1 counts");

    clear_i = 1; @(posedge clk); #1;
    if (mac_en_cnt !== 0 || skip_cnt !== 0 || gate_fire_cnt !== 0) begin
      $display("FAIL Case2 clear"); $fatal(1);
    end
    clear_i = 0;
    $display("PASS Case2 clear");

    mac_en = 1; valid_i = 1; @(posedge clk); #1;
    if (mac_en_cnt !== 1) begin $display("FAIL Case3 setup"); $fatal(1); end
    window_en = 0; valid_i = 0; @(posedge clk); #1;
    if (mac_en_cnt !== 0) begin $display("FAIL Case3 window"); $fatal(1); end
    $display("PASS Case3 window_en off");

    $display("PASS: all c2s_stats cases");
    $finish;
  end

  initial begin #100000; $display("FAIL: timeout"); $fatal(1); end
endmodule

`default_nettype wire
