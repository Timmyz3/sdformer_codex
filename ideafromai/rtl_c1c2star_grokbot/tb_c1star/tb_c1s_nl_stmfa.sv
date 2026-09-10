// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_nl_stmfa + wake_merge — N_TILE=8; nonlinear vs LINEAR_ONLY ablation

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_nl_stmfa;
  localparam int N_TILE  = 8;
  localparam int N_SCALE = 4;
  localparam int FEAT_W  = 8;
  localparam int RES_W   = 8;

  logic clk, rst_n, valid_i;
  logic [N_SCALE*N_TILE*FEAT_W-1:0] f_prev_bus;
  logic [N_TILE*FEAT_W-1:0] f_cur_bus, f_lin_bus;
  logic [N_TILE-1:0] nl_wake, nl_wake_lin;
  logic [N_TILE*RES_W-1:0] r_nl_bus, r_lin_bus;
  logic [N_SCALE-1:0] amfe_w, amfe_lin;
  logic nl_valid, nl_valid_lin;

  logic [N_TILE-1:0] op_stw_wake, tde_wake, delta_nz, pd_wake, veto_mask;
  logic [N_TILE-1:0] wake_merged;
  logic merge_valid;

  integer t, s;
  int n_pass, wake_nl, wake_lin, res_sum_nl, res_sum_lin;

  c1s_nl_stmfa #(
    .N_TILE(N_TILE), .N_SCALE(N_SCALE), .FEAT_W(FEAT_W), .RES_W(RES_W),
    .TH_WAKE(8'd12), .ABLATE_LINEAR_ONLY(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .f_prev_bus(f_prev_bus), .f_cur_bus(f_cur_bus), .f_lin_bus(f_lin_bus),
    .nl_wake(nl_wake), .r_nl_bus(r_nl_bus), .amfe_w(amfe_w), .nl_valid(nl_valid)
  );

  c1s_nl_stmfa #(
    .N_TILE(N_TILE), .N_SCALE(N_SCALE), .FEAT_W(FEAT_W), .RES_W(RES_W),
    .TH_WAKE(8'd12), .ABLATE_LINEAR_ONLY(1'b1)
  ) dut_lin (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .f_prev_bus(f_prev_bus), .f_cur_bus(f_cur_bus), .f_lin_bus(f_lin_bus),
    .nl_wake(nl_wake_lin), .r_nl_bus(r_lin_bus), .amfe_w(amfe_lin), .nl_valid(nl_valid_lin)
  );

  c1s_wake_merge #(.N_TILE(N_TILE)) u_merge (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .op_stw_wake(op_stw_wake), .tde_wake(tde_wake), .delta_nz(delta_nz),
    .nl_wake(nl_wake), .pd_wake(pd_wake), .veto_mask(veto_mask),
    .wake_merged(wake_merged), .merge_valid(merge_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_nl_stmfa.vcd");
      $dumpvars(0, tb_c1s_nl_stmfa);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic int popc(input logic [N_TILE-1:0] b);
    int kk, c; begin
      c = 0;
      for (kk = 0; kk < N_TILE; kk = kk + 1) if (b[kk]) c = c + 1;
      popc = c;
    end
  endfunction

  task automatic clear_in;
    begin
      valid_i = 1'b0;
      f_prev_bus = '0; f_cur_bus = '0; f_lin_bus = '0;
      op_stw_wake = '0; tde_wake = '0; delta_nz = '0;
      pd_wake = '0; veto_mask = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk); valid_i = 1'b1; @(posedge clk); #1; valid_i = 1'b0;
    end
  endtask

  initial begin
    n_pass = 0; wake_nl = 0; wake_lin = 0; res_sum_nl = 0; res_sum_lin = 0;
    rst_n = 1'b0; clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1; @(posedge clk);

    // Case1: identical prev/cur → no wake
    clear_in();
    for (t = 0; t < N_TILE; t = t + 1) begin
      f_cur_bus[t*FEAT_W +: FEAT_W] = 8'sd20;
      f_lin_bus[t*FEAT_W +: FEAT_W] = 8'sd20;
      for (s = 0; s < N_SCALE; s = s + 1)
        f_prev_bus[(s*N_TILE + t)*FEAT_W +: FEAT_W] = 8'sd20;
    end
    beat();
    if (nl_wake !== '0) begin $display("FAIL C1: wake=%b", nl_wake); $fatal(1); end
    if (amfe_w !== '0) begin $display("FAIL C1: amfe=%b", amfe_w); $fatal(1); end
    $display("PASS Case1: flat residual no wake");
    n_pass = n_pass + 1;

    // Case2: mid residual — NL may wake where LINEAR_ONLY may not (boost)
    clear_in();
    for (t = 0; t < N_TILE; t = t + 1) begin
      f_cur_bus[t*FEAT_W +: FEAT_W] = 8'sd30;
      f_lin_bus[t*FEAT_W +: FEAT_W] = 8'sd20; // |cur-lin|=10 → +2.5
      for (s = 0; s < N_SCALE; s = s + 1) begin
        // |30-20|=10; NL ≈15 +2 =17 >=12; linear-only=10 <12
        f_prev_bus[(s*N_TILE + t)*FEAT_W +: FEAT_W] = 8'sd20;
      end
    end
    beat();
    if (nl_wake !== 8'hFF) begin
      $display("FAIL C2: NL wake=%b r0=%0d", nl_wake, r_nl_bus[0*RES_W +: RES_W]); $fatal(1);
    end
    if (nl_wake_lin !== '0) begin
      $display("FAIL C2: LINEAR_ONLY should not wake got %b r=%0d",
               nl_wake_lin, r_lin_bus[0*RES_W +: RES_W]); $fatal(1);
    end
    if (amfe_w === '0) begin $display("FAIL C2: amfe_w"); $fatal(1); end
    $display("PASS Case2: NL wake=%b LINEAR=%b r_nl=%0d r_lin=%0d amfe=%b",
             nl_wake, nl_wake_lin, r_nl_bus[0*RES_W +: RES_W],
             r_lin_bus[0*RES_W +: RES_W], amfe_w);
    n_pass = n_pass + 1;

    // Case3: merge OR nl_wake
    clear_in();
    op_stw_wake = 8'b0000_0001;
    for (t = 0; t < N_TILE; t = t + 1) begin
      f_cur_bus[t*FEAT_W +: FEAT_W] = 8'sd40;
      f_lin_bus[t*FEAT_W +: FEAT_W] = 8'sd0;
      for (s = 0; s < N_SCALE; s = s + 1)
        f_prev_bus[(s*N_TILE + t)*FEAT_W +: FEAT_W] = 8'sd0;
    end
    beat();
    if ((wake_merged & 8'b0000_0001) !== 8'b0000_0001) begin
      $display("FAIL C3: merge missing op bit %b", wake_merged); $fatal(1);
    end
    if (popc(wake_merged) < 2) begin
      $display("FAIL C3: expected nl OR into merge pop=%0d", popc(wake_merged)); $fatal(1);
    end
    $display("PASS Case3: wake_merge=%b", wake_merged);
    n_pass = n_pass + 1;

    // Ablation counters
    clear_in();
    for (t = 0; t < N_TILE; t = t + 1) begin
      f_cur_bus[t*FEAT_W +: FEAT_W] = 8'sd25 + t;
      f_lin_bus[t*FEAT_W +: FEAT_W] = 8'sd10;
      for (s = 0; s < N_SCALE; s = s + 1)
        f_prev_bus[(s*N_TILE + t)*FEAT_W +: FEAT_W] = 8'sd10 + s;
    end
    beat();
    wake_nl  = popc(nl_wake);
    wake_lin = popc(nl_wake_lin);
    for (t = 0; t < N_TILE; t = t + 1) begin
      res_sum_nl  = res_sum_nl  + r_nl_bus[t*RES_W +: RES_W];
      res_sum_lin = res_sum_lin + r_lin_bus[t*RES_W +: RES_W];
    end
    if (res_sum_nl <= res_sum_lin) begin
      $display("FAIL ablation: res_nl=%0d should be > lin=%0d", res_sum_nl, res_sum_lin);
      $fatal(1);
    end
    $display("ABLATION wake_nl=%0d wake_lin=%0d res_sum_nl=%0d res_sum_lin=%0d",
             wake_nl, wake_lin, res_sum_nl, res_sum_lin);
    $display("PASS: all NL-STMFA cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin
    #200000; $display("FAIL: timeout"); $fatal(1);
  end
endmodule

`default_nettype wire
