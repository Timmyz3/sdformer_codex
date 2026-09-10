// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_exact_capture_wrap — OGEC×PRRC gate, count, capacity done

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_exact_capture_wrap;
  localparam int N_TILE   = 8;
  localparam int CNT_W    = 16;
  localparam int CAPACITY = 8;

  logic                 clk, rst_n, valid_i, capture_en, allow_exact;
  logic [N_TILE-1:0]    exact_en, hit_bitmap;
  logic [CNT_W-1:0]     capture_cnt;
  logic                 busy, done;

  c1s_exact_capture_wrap #(
    .N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .capture_en(capture_en),
    .exact_en(exact_en), .allow_exact(allow_exact),
    .hit_bitmap(hit_bitmap), .capture_cnt(capture_cnt), .busy(busy), .done(done)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_exact_capture.vcd");
      $dumpvars(0, tb_c1s_exact_capture_wrap);
    end
  end

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0; valid_i = 0; capture_en = 0; allow_exact = 0; exact_en = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk); #1;

    // Case1: allow_exact=0 → no hits counted even with exact_en
    capture_en = 1; allow_exact = 0; exact_en = 8'hFF;
    @(posedge clk); #1;
    if (!busy) begin $display("FAIL Case1 expect busy"); $fatal(1); end
    valid_i = 1; @(posedge clk); #1;
    if (capture_cnt !== 0 || hit_bitmap !== 0) begin
      $display("FAIL Case1 gated: cnt=%0d hit=%b", capture_cnt, hit_bitmap); $fatal(1);
    end
    valid_i = 0; capture_en = 0;
    @(posedge clk); #1;
    if (!done) begin $display("FAIL Case1 expect done"); $fatal(1); end
    @(posedge clk); #1; // stay in DONE while capture_en still 0 → IDLE next
    // wait for IDLE
    repeat (2) @(posedge clk); #1;
    $display("PASS Case1 PRRC deny gates all hits");

    // Case2: allow + exact_en=0x0F → 4 hits; stay under capacity
    capture_en = 1; allow_exact = 1; exact_en = 8'h0F;
    @(posedge clk); #1;
    if (!busy || done) begin $display("FAIL Case2 busy"); $fatal(1); end
    valid_i = 1; @(posedge clk); #1;
    if (capture_cnt !== 4 || hit_bitmap !== 8'h0F) begin
      $display("FAIL Case2 cnt=%0d hit=%b", capture_cnt, hit_bitmap); $fatal(1);
    end
    valid_i = 0; capture_en = 0;
    @(posedge clk); #1;
    if (!done || capture_cnt !== 4) begin $display("FAIL Case2 done"); $fatal(1); end
    capture_en = 0;
    @(posedge clk); #1; // → IDLE
    $display("PASS Case2 count gated hits");

    // Case3: capacity trip → done while capture_en still high
    capture_en = 1; allow_exact = 1; exact_en = 8'hFF; // 8 pops = CAPACITY
    @(posedge clk); #1;
    valid_i = 1; @(posedge clk); #1;
    if (capture_cnt !== 8 || !done || busy) begin
      $display("FAIL Case3 cap cnt=%0d done=%0d busy=%0d", capture_cnt, done, busy);
      $fatal(1);
    end
    valid_i = 0; capture_en = 0;
    @(posedge clk); #1;
    $display("PASS Case3 CAPACITY → done");

    // Case4: second beat OR-latch + accumulate before capacity
    capture_en = 1; allow_exact = 1; exact_en = 8'h03; // 2
    @(posedge clk); #1;
    valid_i = 1; @(posedge clk); #1;
    exact_en = 8'h0C; // +2 → 4 total, OR bitmap 0x0F
    @(posedge clk); #1;
    if (capture_cnt !== 4 || hit_bitmap !== 8'h0F) begin
      $display("FAIL Case4 cnt=%0d hit=%b", capture_cnt, hit_bitmap); $fatal(1);
    end
    valid_i = 0; capture_en = 0;
    @(posedge clk); #1;
    $display("PASS Case4 accumulate + OR latch");

    $display("PASS: all exact_capture cases");
    $finish;
  end

  initial begin
    #100000; $display("FAIL: timeout"); $fatal(1);
  end
endmodule

`default_nettype wire
