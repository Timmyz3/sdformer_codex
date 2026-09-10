// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_bl_veto_prior + wake_merge — N_TILE=8; veto vs no-veto ablation

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_bl_veto_prior;
  localparam int N_TILE = 8;
  localparam int CONF_W = 4;

  logic clk, rst_n, valid_i;
  logic [N_TILE-1:0] fac_pulse, trg_pulse, null_pulse;
  logic [N_TILE-1:0] pd_wake, veto_mask, pd_wake_nv, veto_mask_nv;
  logic [N_TILE*CONF_W-1:0] veto_conf_bus, veto_conf_nv;
  logic bl_valid, bl_valid_nv;

  logic [N_TILE-1:0] op_stw_wake, tde_wake, delta_nz, nl_wake;
  logic [N_TILE-1:0] wake_merged, wake_merged_nv;
  logic merge_valid, merge_valid_nv;

  integer i;
  int n_pass, cnt_veto, cnt_pd, merge_pop, merge_pop_nv;

  c1s_bl_veto_prior #(
    .N_TILE(N_TILE), .DT_WIN(8'd6), .TH_VETO(4'd2), .ABLATE_NO_VETO(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .fac_pulse(fac_pulse), .trg_pulse(trg_pulse), .null_pulse(null_pulse),
    .pd_wake(pd_wake), .veto_mask(veto_mask), .veto_conf_bus(veto_conf_bus),
    .bl_valid(bl_valid)
  );

  c1s_bl_veto_prior #(
    .N_TILE(N_TILE), .DT_WIN(8'd6), .TH_VETO(4'd2), .ABLATE_NO_VETO(1'b1)
  ) dut_noveto (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .fac_pulse(fac_pulse), .trg_pulse(trg_pulse), .null_pulse(null_pulse),
    .pd_wake(pd_wake_nv), .veto_mask(veto_mask_nv), .veto_conf_bus(veto_conf_nv),
    .bl_valid(bl_valid_nv)
  );

  c1s_wake_merge #(.N_TILE(N_TILE)) u_merge (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .op_stw_wake(op_stw_wake), .tde_wake(tde_wake), .delta_nz(delta_nz),
    .nl_wake(nl_wake), .pd_wake(pd_wake), .veto_mask(veto_mask),
    .wake_merged(wake_merged), .merge_valid(merge_valid)
  );

  c1s_wake_merge #(.N_TILE(N_TILE)) u_merge_nv (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .op_stw_wake(op_stw_wake), .tde_wake(tde_wake), .delta_nz(delta_nz),
    .nl_wake(nl_wake), .pd_wake(pd_wake_nv), .veto_mask(veto_mask_nv),
    .wake_merged(wake_merged_nv), .merge_valid(merge_valid_nv)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_bl_veto_prior.vcd");
      $dumpvars(0, tb_c1s_bl_veto_prior);
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
      fac_pulse = '0; trg_pulse = '0; null_pulse = '0;
      op_stw_wake = '0; tde_wake = '0; delta_nz = '0; nl_wake = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk); valid_i = 1'b1; @(posedge clk); #1; valid_i = 1'b0;
    end
  endtask

  task automatic idle;
    begin
      @(negedge clk);
      fac_pulse = '0; trg_pulse = '0; null_pulse = '0;
      valid_i = 1'b1; @(posedge clk); #1; valid_i = 1'b0;
    end
  endtask

  initial begin
    n_pass = 0; cnt_veto = 0; cnt_pd = 0;
    rst_n = 1'b0; clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1; @(posedge clk);

    // Case1: fac then trg → pd_wake, no veto
    clear_in();
    fac_pulse = 8'b0000_1000; // tile3
    beat();
    idle(); idle();
    clear_in();
    trg_pulse = 8'b0000_1000;
    beat();
    if (pd_wake[3] !== 1'b1) begin $display("FAIL C1: pd_wake=%b", pd_wake); $fatal(1); end
    if (veto_mask[3] !== 1'b0) begin $display("FAIL C1: unexpected veto"); $fatal(1); end
    $display("PASS Case1: pd_wake tile3 conf=%0d", veto_conf_bus[3*CONF_W +: CONF_W]);
    n_pass = n_pass + 1;

    // Case2: fac then null → veto_mask; next beat AND-NOT into wake_merge
    // (veto_mask is registered — merge sees it one beat after null)
    rst_n = 1'b0; @(posedge clk); #1; rst_n = 1'b1; @(posedge clk);
    clear_in();
    fac_pulse = 8'b0001_0000; // tile4
    beat();
    idle();
    clear_in();
    null_pulse = 8'b0001_0000;
    beat();
    if (veto_mask[4] !== 1'b1) begin $display("FAIL C2: veto=%b", veto_mask); $fatal(1); end
    if (veto_mask_nv[4] !== 1'b0) begin $display("FAIL C2: ABLATE should no veto"); $fatal(1); end
    // Sustained veto (age_null young) + drive wake lanes
    clear_in();
    op_stw_wake = 8'b0001_0000;
    tde_wake = 8'b0001_0000;
    beat();
    if (veto_mask[4] !== 1'b1) begin $display("FAIL C2b: sustained veto=%b", veto_mask); $fatal(1); end
    if (wake_merged[4] !== 1'b0) begin
      $display("FAIL C2: merge should AND-NOT veto got %b", wake_merged); $fatal(1);
    end
    if (wake_merged_nv[4] !== 1'b1) begin
      $display("FAIL C2: no-veto merge should keep wake"); $fatal(1);
    end
    $display("PASS Case2: veto_mask=%b merge=%b merge_noveto=%b",
             veto_mask, wake_merged, wake_merged_nv);
    n_pass = n_pass + 1;

    // Case3: idle
    clear_in();
    beat();
    if (pd_wake !== '0) begin $display("FAIL C3: pd=%b", pd_wake); $fatal(1); end
    $display("PASS Case3: idle");
    n_pass = n_pass + 1;

    // Counters over a sweep (null path: register veto, then probe wake)
    rst_n = 1'b0; @(posedge clk); #1; rst_n = 1'b1; @(posedge clk);
    clear_in();
    merge_pop = 0; merge_pop_nv = 0; cnt_veto = 0; cnt_pd = 0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      fac_pulse = (8'b1 << i);
      beat();
      idle();
      clear_in();
      if (i[0]) begin
        null_pulse = (8'b1 << i);
        beat(); // latch veto_mask
        cnt_veto = cnt_veto + popc(veto_mask);
        clear_in();
        op_stw_wake = (8'b1 << i);
        beat(); // merge sees sustained veto
      end else begin
        trg_pulse = (8'b1 << i);
        op_stw_wake = (8'b1 << i);
        beat();
        cnt_pd = cnt_pd + popc(pd_wake);
      end
      merge_pop = merge_pop + popc(wake_merged);
      merge_pop_nv = merge_pop_nv + popc(wake_merged_nv);
      clear_in();
    end
    if (cnt_veto == 0) begin
      $display("FAIL ablation: expected veto counts"); $fatal(1);
    end
    if (merge_pop >= merge_pop_nv) begin
      $display("FAIL ablation: merge_pop=%0d should be < noveto=%0d", merge_pop, merge_pop_nv);
      $fatal(1);
    end
    $display("ABLATION merge_pop_veto=%0d merge_pop_noveto=%0d cnt_veto=%0d cnt_pd=%0d",
             merge_pop, merge_pop_nv, cnt_veto, cnt_pd);
    $display("PASS: all BL-VetoPrior cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin
    #200000; $display("FAIL: timeout"); $fatal(1);
  end
endmodule

`default_nettype wire
