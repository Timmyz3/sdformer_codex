// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_tde3_prior + c1s_wake_merge — N_TILE=8 deterministic cases

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_tde3_prior;
  localparam int N_TILE = 8;
  localparam int AGE_W  = 8;
  localparam int CONF_W = 4;
  localparam int DIR_W  = 2;

  logic clk, rst_n, valid_i;
  logic [N_TILE-1:0] event_pulse, polarity;
  logic [N_TILE*2-1:0]      flow_hint_bus;
  logic [N_TILE*DIR_W-1:0]  dir_code_bus;
  logic [N_TILE*CONF_W-1:0] tde_conf_bus;
  logic [N_TILE-1:0] tde_wake;
  logic              tde_valid;

  logic [N_TILE-1:0] op_stw_wake, delta_nz, wake_merged;
  logic              merge_valid;

  integer i, k;
  int wake_pop, merge_pop, conf_sum;
  int n_pass;
  logic [DIR_W-1:0]  d3, d4;
  logic [CONF_W-1:0] c3, c4;
  logic [N_TILE-1:0] held_tde;

  c1s_tde3_prior #(
    .N_TILE(N_TILE), .AGE_W(AGE_W), .CONF_W(CONF_W), .DIR_W(DIR_W),
    .TH_AGE(8'd8), .TH_WAKE(4'd2), .INHIB_PENALTY(4'd2)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .event_pulse(event_pulse), .polarity(polarity), .flow_hint_bus(flow_hint_bus),
    .dir_code_bus(dir_code_bus), .tde_conf_bus(tde_conf_bus),
    .tde_wake(tde_wake), .tde_valid(tde_valid)
  );

  c1s_wake_merge #(.N_TILE(N_TILE)) u_merge (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .op_stw_wake(op_stw_wake), .tde_wake(tde_wake), .delta_nz(delta_nz),
    .nl_wake({N_TILE{1'b0}}), .pd_wake({N_TILE{1'b0}}), .veto_mask({N_TILE{1'b0}}),
    .wake_merged(wake_merged), .merge_valid(merge_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_tde3_prior.vcd");
      $dumpvars(0, tb_c1s_tde3_prior);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_in;
    begin
      valid_i = 1'b0;
      event_pulse = '0;
      polarity = '0;
      flow_hint_bus = '0;
      op_stw_wake = '0;
      delta_nz = '0;
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

  task automatic idle_beat;
    begin
      @(negedge clk);
      event_pulse = '0;
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
    n_pass = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: facilitator tile2, event tile3 → rightward (dir=2)
    clear_in();
    event_pulse = 8'b0000_0100;
    beat_sample();
    idle_beat(); idle_beat(); idle_beat();
    clear_in();
    event_pulse = 8'b0000_1000;
    polarity[3] = 1'b1;
    flow_hint_bus[3*2 +: 2] = 2'd2;
    beat_sample();
    d3 = dir_code_bus[3*DIR_W +: DIR_W];
    c3 = tde_conf_bus[3*CONF_W +: CONF_W];
    if (!tde_valid) begin $display("FAIL Case1: tde_valid"); $fatal(1); end
    if (tde_wake[3] !== 1'b1) begin
      $display("FAIL Case1: wake[3] conf=%0d dir=%0d wake=%b", c3, d3, tde_wake);
      $fatal(1);
    end
    if (d3 !== 2'd2) begin
      $display("FAIL Case1: dir=%0d expected 2 conf=%0d", d3, c3);
      $fatal(1);
    end
    $display("PASS Case1: tile3 wake dir=%0d conf=%0d tde_wake=%b", d3, c3, tde_wake);
    n_pass = n_pass + 1;

    // Case2: reset ages, facilitator tile5, event tile4 → leftward (dir=1)
    rst_n = 1'b0; @(posedge clk); #1; rst_n = 1'b1; @(posedge clk);
    clear_in();
    event_pulse = 8'b0010_0000;
    beat_sample();
    idle_beat(); idle_beat(); idle_beat();
    clear_in();
    event_pulse = 8'b0001_0000;
    polarity[4] = 1'b1;
    beat_sample();
    d4 = dir_code_bus[4*DIR_W +: DIR_W];
    c4 = tde_conf_bus[4*CONF_W +: CONF_W];
    if (d4 !== 2'd1) begin
      $display("FAIL Case2: dir=%0d expected 1 wake=%b conf=%0d", d4, tde_wake, c4);
      $fatal(1);
    end
    if (tde_wake[4] !== 1'b1) begin
      $display("FAIL Case2: wake[4] conf=%0d", c4); $fatal(1);
    end
    $display("PASS Case2: tile4 wake dir=%0d conf=%0d", d4, c4);
    n_pass = n_pass + 1;

    // Case3: idle
    clear_in();
    beat_sample();
    if (tde_wake !== {N_TILE{1'b0}}) begin
      $display("FAIL Case3: wake=%b", tde_wake); $fatal(1);
    end
    $display("PASS Case3: idle no wake");
    n_pass = n_pass + 1;

    // Case4: wake_merge OR
    clear_in();
    event_pulse = 8'b0000_0100;
    beat_sample();
    idle_beat(); idle_beat();
    clear_in();
    event_pulse = 8'b0000_1000;
    polarity[3] = 1'b1;
    flow_hint_bus[3*2 +: 2] = 2'd2;
    op_stw_wake = 8'b0000_0001;
    delta_nz    = 8'b1000_0000;
    beat_sample();
    held_tde = tde_wake;
    clear_in();
    op_stw_wake = held_tde | 8'b0000_0001;
    delta_nz    = 8'b1000_0000;
    beat_sample();
    if ((wake_merged & (held_tde | 8'b1000_0001)) !== (held_tde | 8'b1000_0001)) begin
      $display("FAIL Case4: merge=%b held=%b", wake_merged, held_tde); $fatal(1);
    end
    merge_pop = popc(wake_merged);
    $display("PASS Case4: wake_merge=%b pop=%0d held_tde=%b", wake_merged, merge_pop, held_tde);
    n_pass = n_pass + 1;

    wake_pop = 0; conf_sum = 0;
    clear_in();
    for (i = 0; i < N_TILE; i = i + 1) begin
      event_pulse = (8'b1 << i);
      beat_sample();
      wake_pop = wake_pop + popc(tde_wake);
      for (k = 0; k < N_TILE; k = k + 1)
        conf_sum = conf_sum + tde_conf_bus[k*CONF_W +: CONF_W];
    end
    $display("COUNTERS tde_wake_pop_sum=%0d conf_sum=%0d cases_pass=%0d", wake_pop, conf_sum, n_pass);
    $display("PASS: all TDE3-Prior / wake_merge cases");
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
