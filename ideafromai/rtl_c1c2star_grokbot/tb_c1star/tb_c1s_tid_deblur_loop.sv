// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_tid_deblur_loop — N_TILE=8; NO_DEBLUR vs TID ablation + prove-by

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_tid_deblur_loop;
  localparam int N_TILE = 8;
  localparam int FLOW_W = 8;
  localparam int BIN_W  = 8;
  localparam int N_BIN  = 4;

  logic clk, rst_n, valid_i, iter_fire_i;
  logic [N_TILE*N_BIN*BIN_W-1:0] evt_bin_bus, evt_deblur_bus, evt_deblur_nd;
  logic [N_TILE*FLOW_W-1:0] flow_seed_bus, flow_hat_prev_bus;
  logic [N_TILE*FLOW_W-1:0] dflow_bus, flow_out_bus, flow_hat_next_bus;
  logic [N_TILE*FLOW_W-1:0] dflow_nd, flow_out_nd, flow_hat_nd;
  logic [N_TILE-1:0] deblur_wake, deblur_wake_nd;
  logic exact_pref, exact_pref_nd, loop_valid, loop_valid_nd;

  integer t, b;
  int n_pass, cnt_deblur, sum_abs_dflow, cnt_exact_pref, cnt_wake;
  int sum_abs_nd, cnt_wake_nd;

  c1s_tid_deblur_loop #(
    .N_TILE(N_TILE), .FLOW_W(FLOW_W), .BIN_W(BIN_W), .N_BIN(N_BIN),
    .TH_DRES(8'd6), .MODE_TID(1'b1), .ABLATE_NO_DEBLUR(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .iter_fire_i(iter_fire_i),
    .evt_bin_bus(evt_bin_bus), .flow_seed_bus(flow_seed_bus),
    .flow_hat_prev_bus(flow_hat_prev_bus),
    .evt_deblur_bus(evt_deblur_bus), .dflow_bus(dflow_bus),
    .flow_out_bus(flow_out_bus), .flow_hat_next_bus(flow_hat_next_bus),
    .deblur_wake(deblur_wake), .exact_pref(exact_pref), .loop_valid(loop_valid)
  );

  c1s_tid_deblur_loop #(
    .N_TILE(N_TILE), .FLOW_W(FLOW_W), .BIN_W(BIN_W), .N_BIN(N_BIN),
    .TH_DRES(8'd6), .MODE_TID(1'b1), .ABLATE_NO_DEBLUR(1'b1)
  ) dut_nd (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i), .iter_fire_i(iter_fire_i),
    .evt_bin_bus(evt_bin_bus), .flow_seed_bus(flow_seed_bus),
    .flow_hat_prev_bus(flow_hat_prev_bus),
    .evt_deblur_bus(evt_deblur_nd), .dflow_bus(dflow_nd),
    .flow_out_bus(flow_out_nd), .flow_hat_next_bus(flow_hat_nd),
    .deblur_wake(deblur_wake_nd), .exact_pref(exact_pref_nd), .loop_valid(loop_valid_nd)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_tid_deblur_loop.vcd");
      $dumpvars(0, tb_c1s_tid_deblur_loop);
    end
  end
  initial clk = 0;
  always #5 clk = ~clk;

  function automatic int popc(input logic [N_TILE-1:0] m);
    int k,c; begin c=0; for(k=0;k<N_TILE;k=k+1) if(m[k]) c=c+1; popc=c; end
  endfunction

  task automatic clear_in;
    begin
      valid_i=0; iter_fire_i=0;
      evt_bin_bus='0; flow_seed_bus='0; flow_hat_prev_bus='0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk); valid_i=1; iter_fire_i=1; @(posedge clk); #1;
      valid_i=0; iter_fire_i=0;
    end
  endtask

  task automatic accum;
    begin
      for (t=0;t<N_TILE;t=t+1) begin
        sum_abs_dflow = sum_abs_dflow + dflow_bus[t*FLOW_W +: FLOW_W];
        sum_abs_nd    = sum_abs_nd    + dflow_nd[t*FLOW_W +: FLOW_W];
      end
      cnt_wake    = cnt_wake + popc(deblur_wake);
      cnt_wake_nd = cnt_wake_nd + popc(deblur_wake_nd);
      if (exact_pref) cnt_exact_pref = cnt_exact_pref + 1;
      cnt_deblur = cnt_deblur + 1;
    end
  endtask

  initial begin
    n_pass=0; cnt_deblur=0; sum_abs_dflow=0; cnt_exact_pref=0; cnt_wake=0;
    sum_abs_nd=0; cnt_wake_nd=0;
    rst_n=0; clear_in();
    repeat(3) @(posedge clk); rst_n=1; @(posedge clk);

    // Case1: asymmetric bins + nonzero hat → deblur wake
    clear_in();
    for (t=0;t<N_TILE;t=t+1) begin
      flow_seed_bus[t*FLOW_W +: FLOW_W] = 8'd10;
      flow_hat_prev_bus[t*FLOW_W +: FLOW_W] = 8'd40; // TID uses hat
      evt_bin_bus[(t*N_BIN+0)*BIN_W +: BIN_W] = 8'd40;
      evt_bin_bus[(t*N_BIN+1)*BIN_W +: BIN_W] = 8'd5;
      evt_bin_bus[(t*N_BIN+2)*BIN_W +: BIN_W] = 8'd5;
      evt_bin_bus[(t*N_BIN+3)*BIN_W +: BIN_W] = 8'd5;
    end
    beat(); accum();
    if (!loop_valid) begin $display("FAIL C1: loop_valid"); $fatal(1); end
    if (deblur_wake === '0) begin
      $display("FAIL C1: expected deblur_wake d0=%0d", dflow_bus[0*FLOW_W +: FLOW_W]);
      $fatal(1);
    end
    if (deblur_wake_nd !== '0) begin
      $display("FAIL C1: NO_DEBLUR should not wake got %b", deblur_wake_nd); $fatal(1);
    end
    if (evt_deblur_nd !== evt_bin_bus) begin
      $display("FAIL C1: NO_DEBLUR passthrough"); $fatal(1);
    end
    $display("PASS Case1: wake=%b exact_pref=%0d dflow0=%0d",
             deblur_wake, exact_pref, dflow_bus[0*FLOW_W +: FLOW_W]);
    n_pass = n_pass + 1;

    // Case2: flat bins → little/no wake
    clear_in();
    for (t=0;t<N_TILE;t=t+1) begin
      flow_seed_bus[t*FLOW_W +: FLOW_W] = 8'd4;
      flow_hat_prev_bus[t*FLOW_W +: FLOW_W] = 8'd4;
      for (b=0;b<N_BIN;b=b+1)
        evt_bin_bus[(t*N_BIN+b)*BIN_W +: BIN_W] = 8'd8;
    end
    beat(); accum();
    if (popc(deblur_wake) > 2) begin
      $display("FAIL C2: unexpected wake=%b", deblur_wake); $fatal(1);
    end
    $display("PASS Case2: flat wake=%b", deblur_wake);
    n_pass = n_pass + 1;

    // Case3: warm-start loop — hat_next feeds next beat
    clear_in();
    for (t=0;t<N_TILE;t=t+1) begin
      flow_seed_bus[t*FLOW_W +: FLOW_W] = 8'd8;
      flow_hat_prev_bus[t*FLOW_W +: FLOW_W] = 8'd8;
      evt_bin_bus[(t*N_BIN+0)*BIN_W +: BIN_W] = 8'd30;
      evt_bin_bus[(t*N_BIN+1)*BIN_W +: BIN_W] = 8'd2;
      evt_bin_bus[(t*N_BIN+2)*BIN_W +: BIN_W] = 8'd2;
      evt_bin_bus[(t*N_BIN+3)*BIN_W +: BIN_W] = 8'd2;
    end
    beat(); accum();
    flow_hat_prev_bus = flow_hat_next_bus;
    beat(); accum();
    if (flow_hat_next_bus === '0) begin $display("FAIL C3: hat_next"); $fatal(1); end
    $display("PASS Case3: loop hat_next0=%0d flow_out0=%0d",
             flow_hat_next_bus[0*FLOW_W +: FLOW_W], flow_out_bus[0*FLOW_W +: FLOW_W]);
    n_pass = n_pass + 1;

    if (sum_abs_dflow <= sum_abs_nd) begin
      $display("FAIL ablation: sum_dflow=%0d should be > NO_DEBLUR=%0d",
               sum_abs_dflow, sum_abs_nd);
      $fatal(1);
    end
    if (cnt_wake <= cnt_wake_nd) begin
      $display("FAIL ablation: cnt_wake=%0d vs nd=%0d", cnt_wake, cnt_wake_nd);
      $fatal(1);
    end
    $display("ABLATION sum_abs_dflow=%0d sum_NO_DEBLUR=%0d cnt_wake=%0d cnt_wake_nd=%0d cnt_exact_pref=%0d cnt_deblur=%0d",
             sum_abs_dflow, sum_abs_nd, cnt_wake, cnt_wake_nd, cnt_exact_pref, cnt_deblur);
    $display("PASS: all TID-DeblurLoop cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin #200000; $display("FAIL: timeout"); $fatal(1); end
endmodule
`default_nettype wire
