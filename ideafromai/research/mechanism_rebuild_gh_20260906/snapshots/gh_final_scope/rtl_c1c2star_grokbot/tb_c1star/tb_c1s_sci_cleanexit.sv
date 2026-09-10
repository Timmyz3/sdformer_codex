// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_sci_cleanexit — N_TILE=8; EXIT_ONLY vs FULL scrub ablation

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_sci_cleanexit;
  localparam int N_TILE = 8;
  localparam int FEAT_W = 8;
  localparam int FLOW_W = 8;
  localparam int SCI_W  = 8;

  logic clk, rst_n, it_valid_i, bisat_occ_exit_i;
  logic [N_TILE*FEAT_W-1:0] f1_bus, f2_warp_bus;
  logic [N_TILE*FLOW_W-1:0] f_hat_bus;

  logic [N_TILE*SCI_W-1:0] g_sci_bus, g_sci_e;
  logic [N_TILE-1:0]       scrub_en, scrub_e;
  logic                    exit_ok, exit_e;
  logic                    exact_hold, hold_e;
  logic                    sci_valid, sci_e;

  integer i, k;
  int n_pass;
  longint cnt_iter, cnt_scrub, cnt_exit, cnt_exact_hold, sum_g;
  longint cnt_iter_e, cnt_scrub_e, cnt_exit_e, cnt_hold_e, sum_g_e;
  int capture_full, capture_exit_only;
  logic [SCI_W-1:0] g0;

  // FULL scrub
  c1s_sci_cleanexit #(
    .N_TILE(N_TILE), .FEAT_W(FEAT_W), .FLOW_W(FLOW_W), .SCI_W(SCI_W),
    .TH_EXIT(8'd200), .TH_SCRUB(8'd80), .ABLATE_EXIT_ONLY(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .it_valid_i(it_valid_i),
    .f1_bus(f1_bus), .f2_warp_bus(f2_warp_bus), .f_hat_bus(f_hat_bus),
    .bisat_occ_exit_i(bisat_occ_exit_i),
    .g_sci_bus(g_sci_bus), .scrub_en(scrub_en),
    .exit_ok(exit_ok), .exact_hold(exact_hold), .sci_valid(sci_valid)
  );

  // EXIT_ONLY ablation
  c1s_sci_cleanexit #(
    .N_TILE(N_TILE), .FEAT_W(FEAT_W), .FLOW_W(FLOW_W), .SCI_W(SCI_W),
    .TH_EXIT(8'd200), .TH_SCRUB(8'd80), .ABLATE_EXIT_ONLY(1'b1)
  ) dut_exit (
    .clk(clk), .rst_n(rst_n), .it_valid_i(it_valid_i),
    .f1_bus(f1_bus), .f2_warp_bus(f2_warp_bus), .f_hat_bus(f_hat_bus),
    .bisat_occ_exit_i(bisat_occ_exit_i),
    .g_sci_bus(g_sci_e), .scrub_en(scrub_e),
    .exit_ok(exit_e), .exact_hold(hold_e), .sci_valid(sci_e)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_sci_cleanexit.vcd");
      $dumpvars(0, tb_c1s_sci_cleanexit);
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
      it_valid_i = 1'b0;
      bisat_occ_exit_i = 1'b0;
      f1_bus = '0;
      f2_warp_bus = '0;
      f_hat_bus = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk);
      it_valid_i = 1'b1;
      @(posedge clk);
      #1;
      it_valid_i = 1'b0;
    end
  endtask

  task automatic accum;
    begin
      cnt_iter = cnt_iter + 1;
      cnt_iter_e = cnt_iter_e + 1;
      cnt_scrub = cnt_scrub + popc(scrub_en);
      cnt_scrub_e = cnt_scrub_e + popc(scrub_e);
      if (exit_ok) cnt_exit = cnt_exit + 1;
      if (exit_e)  cnt_exit_e = cnt_exit_e + 1;
      if (exact_hold) cnt_exact_hold = cnt_exact_hold + 1;
      if (hold_e)     cnt_hold_e = cnt_hold_e + 1;
      for (k = 0; k < N_TILE; k = k + 1) begin
        sum_g   = sum_g   + g_sci_bus[k*SCI_W +: SCI_W];
        sum_g_e = sum_g_e + g_sci_e[k*SCI_W +: SCI_W];
      end
      // Stub capture: allowed only when !exact_hold
      if (!exact_hold) capture_full = capture_full + 1;
      if (!hold_e)     capture_exit_only = capture_exit_only + 1;
    end
  endtask

  initial begin
    n_pass = 0;
    cnt_iter = 0; cnt_scrub = 0; cnt_exit = 0; cnt_exact_hold = 0; sum_g = 0;
    cnt_iter_e = 0; cnt_scrub_e = 0; cnt_exit_e = 0; cnt_hold_e = 0; sum_g_e = 0;
    capture_full = 0; capture_exit_only = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: perfect match → high g_sci, exit_ok, no scrub, no hold
    clear_in();
    for (i = 0; i < N_TILE; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd20;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = 8'sd20;
      f_hat_bus[i*FLOW_W +: FLOW_W] = 8'd1;
    end
    beat();
    accum();
    g0 = g_sci_bus[0*SCI_W +: SCI_W];
    if (!sci_valid) begin $display("FAIL Case1: sci_valid"); $fatal(1); end
    if (g0 !== 8'd255) begin $display("FAIL Case1: g=%0d expected 255", g0); $fatal(1); end
    if (!exit_ok) begin $display("FAIL Case1: exit_ok"); $fatal(1); end
    if (scrub_en !== 8'h00) begin $display("FAIL Case1: scrub=%b", scrub_en); $fatal(1); end
    if (exact_hold !== 1'b0) begin $display("FAIL Case1: hold"); $fatal(1); end
    $display("PASS Case1: clean exit g=%0d exit=%0b hold=%0b", g0, exit_ok, exact_hold);
    n_pass = n_pass + 1;

    // Case2: large mismatch → low g, scrub, hold, !exit
    clear_in();
    for (i = 0; i < N_TILE; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd100;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = 8'sd0;  // abs=100 → g=155? wait 255-100=155
      // 155 > TH_SCRUB=80, so need larger diff for scrub
    end
    // Use abs=200 → g=55 < 80 scrub; mean 55 < TH_EXIT=200 → !exit
    for (i = 0; i < N_TILE; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd100;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = -8'sd100; // abs=200 → g=55
    end
    beat();
    accum();
    g0 = g_sci_bus[0*SCI_W +: SCI_W];
    if (g0 !== 8'd55) begin $display("FAIL Case2: g=%0d expected 55", g0); $fatal(1); end
    if (exit_ok) begin $display("FAIL Case2: exit should be 0"); $fatal(1); end
    if (scrub_en !== 8'hFF) begin $display("FAIL Case2: scrub=%b", scrub_en); $fatal(1); end
    if (!exact_hold) begin $display("FAIL Case2: hold expected"); $fatal(1); end
    // EXIT_ONLY: no scrub but still hold via !exit
    if (scrub_e !== 8'h00) begin $display("FAIL Case2: EXIT scrub=%b", scrub_e); $fatal(1); end
    if (!hold_e) begin $display("FAIL Case2: EXIT hold"); $fatal(1); end
    $display("PASS Case2: dirty g=%0d scrub=%b hold=%0b EXIT_scrub=%b",
             g0, scrub_en, exact_hold, scrub_e);
    n_pass = n_pass + 1;

    // Case3: mid quality — abs=20 → g=235 > TH_EXIT mean, no scrub
    clear_in();
    for (i = 0; i < N_TILE; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd30;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = 8'sd10; // abs=20 → g=235
    end
    beat();
    accum();
    g0 = g_sci_bus[0*SCI_W +: SCI_W];
    if (g0 !== 8'd235) begin $display("FAIL Case3: g=%0d", g0); $fatal(1); end
    if (!exit_ok) begin $display("FAIL Case3: exit"); $fatal(1); end
    if (scrub_en !== 0) begin $display("FAIL Case3: scrub"); $fatal(1); end
    $display("PASS Case3: mid-clean g=%0d exit=%0b", g0, exit_ok);
    n_pass = n_pass + 1;

    // Case4: mixed tiles — some scrub some not; mean may !exit
    // 4 tiles abs=0 (g=255), 4 tiles abs=200 (g=55); mean=(4*255+4*55)/8=155 < 200
    clear_in();
    for (i = 0; i < 4; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd5;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = 8'sd5;
    end
    for (i = 4; i < 8; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd100;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = -8'sd100;
    end
    beat();
    accum();
    if (exit_ok) begin $display("FAIL Case4: mean should not exit"); $fatal(1); end
    if (scrub_en !== 8'b1111_0000) begin
      $display("FAIL Case4: scrub=%b", scrub_en); $fatal(1);
    end
    if (!exact_hold) begin $display("FAIL Case4: hold"); $fatal(1); end
    if (scrub_e !== 0) begin $display("FAIL Case4: EXIT scrub nonzero"); $fatal(1); end
    $display("PASS Case4: mixed scrub=%b exit=%0b hold=%0b", scrub_en, exit_ok, exact_hold);
    n_pass = n_pass + 1;

    // Case5: bisat_occ_exit forces hold even when clean
    clear_in();
    bisat_occ_exit_i = 1'b1;
    for (i = 0; i < N_TILE; i = i + 1) begin
      f1_bus[i*FEAT_W +: FEAT_W] = 8'sd7;
      f2_warp_bus[i*FEAT_W +: FEAT_W] = 8'sd7;
    end
    beat();
    accum();
    if (!exit_ok) begin $display("FAIL Case5: exit should still assert"); $fatal(1); end
    if (!exact_hold) begin $display("FAIL Case5: bisat hold"); $fatal(1); end
    $display("PASS Case5: bisat_occ forces hold exit=%0b hold=%0b", exit_ok, exact_hold);
    n_pass = n_pass + 1;

    if (cnt_scrub_e != 0) begin
      $display("FAIL ablation: EXIT_ONLY scrub cnt=%0d", cnt_scrub_e); $fatal(1);
    end
    if (cnt_scrub <= cnt_scrub_e) begin
      $display("FAIL ablation: FULL scrub=%0d should exceed EXIT=%0d", cnt_scrub, cnt_scrub_e);
      $fatal(1);
    end
    // FULL holds more often when scrub fires on partial-dirty that still exits? 
    // In our cases capture_full should be <= capture_exit_only (more holds with scrub)
    if (capture_full > capture_exit_only) begin
      $display("FAIL ablation: capture_full=%0d > EXIT_ONLY=%0d",
               capture_full, capture_exit_only);
      $fatal(1);
    end

    $display("ABLATION EXIT_ONLY scrub=%0d hold=%0d capture=%0d | FULL scrub=%0d hold=%0d capture=%0d",
             cnt_scrub_e, cnt_hold_e, capture_exit_only, cnt_scrub, cnt_exact_hold, capture_full);
    $display("COUNTERS cnt_iter=%0d cnt_scrub=%0d cnt_exit=%0d cnt_exact_hold=%0d avg_g_sci_xN=%0d",
             cnt_iter, cnt_scrub, cnt_exit, cnt_exact_hold, sum_g);
    $display("PASS: all SCI-CleanExit cases");
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
