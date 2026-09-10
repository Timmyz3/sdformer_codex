// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_cfp_confgate — N_TILE=8; ALWAYS_EXACT vs CFP ablation + prove-by counters

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_cfp_confgate;
  localparam int N_TILE   = 8;
  localparam int PEAK_W   = 16;
  localparam int CONF_W   = 8;
  localparam int BUDGET_W = 8;

  logic clk, rst_n, valid_i;
  logic [N_TILE*PEAK_W-1:0] corr_peak_bus;
  logic [N_TILE-1:0]        match_ok;
  logic [BUDGET_W-1:0]      prrc_budget_i;

  // DUT: CFP mode
  logic [N_TILE*CONF_W-1:0] conf_bus;
  logic [N_TILE-1:0]        exact_req, prop_pref;
  logic [BUDGET_W-1:0]      prrc_grant;
  logic                     conf_valid;

  // Ablation twin: ALWAYS_EXACT
  logic [N_TILE*CONF_W-1:0] conf_bus_a;
  logic [N_TILE-1:0]        exact_req_a, prop_pref_a;
  logic [BUDGET_W-1:0]      prrc_grant_a;
  logic                     conf_valid_a;

  // Stub OGEC exact_en (= match_ok) combine
  logic [N_TILE-1:0] ogec_exact_en;
  logic [N_TILE-1:0] exact_en_cfp, exact_en_always;
  logic              allow_cfp, allow_always;
  int                capture_cfp, capture_always;

  integer i, k;
  int n_pass;
  longint sum_conf, cnt_exact_req, cnt_prop_pref, cnt_grant_starve;
  longint sum_conf_a, cnt_exact_a, cnt_prop_a;
  int pop_ex, pop_pr;
  logic [CONF_W-1:0] c0, c1;

  c1s_cfp_confgate #(
    .N_TILE(N_TILE), .PEAK_W(PEAK_W), .CONF_W(CONF_W), .BUDGET_W(BUDGET_W),
    .TH_HI(8'd160), .TH_LO(8'd64), .ABLATE_ALWAYS_EXACT(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .corr_peak_bus(corr_peak_bus), .match_ok(match_ok), .prrc_budget_i(prrc_budget_i),
    .conf_bus(conf_bus), .exact_req(exact_req), .prop_pref(prop_pref),
    .prrc_grant(prrc_grant), .conf_valid(conf_valid)
  );

  c1s_cfp_confgate #(
    .N_TILE(N_TILE), .PEAK_W(PEAK_W), .CONF_W(CONF_W), .BUDGET_W(BUDGET_W),
    .TH_HI(8'd160), .TH_LO(8'd64), .ABLATE_ALWAYS_EXACT(1'b1)
  ) dut_ablate (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .corr_peak_bus(corr_peak_bus), .match_ok(match_ok), .prrc_budget_i(prrc_budget_i),
    .conf_bus(conf_bus_a), .exact_req(exact_req_a), .prop_pref(prop_pref_a),
    .prrc_grant(prrc_grant_a), .conf_valid(conf_valid_a)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_cfp_confgate.vcd");
      $dumpvars(0, tb_c1s_cfp_confgate);
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
      valid_i = 1'b0;
      corr_peak_bus = '0;
      match_ok = '0;
      prrc_budget_i = 8'd16;
      ogec_exact_en = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk);
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      valid_i = 1'b0;
    end
  endtask

  task automatic accum_counters;
    begin
      for (k = 0; k < N_TILE; k = k + 1)
        sum_conf = sum_conf + conf_bus[k*CONF_W +: CONF_W];
      pop_ex = popc(exact_req);
      pop_pr = popc(prop_pref);
      cnt_exact_req = cnt_exact_req + pop_ex;
      cnt_prop_pref = cnt_prop_pref + pop_pr;
      if ((pop_ex > 0) && (prrc_grant == 0))
        cnt_grant_starve = cnt_grant_starve + 1;

      for (k = 0; k < N_TILE; k = k + 1)
        sum_conf_a = sum_conf_a + conf_bus_a[k*CONF_W +: CONF_W];
      cnt_exact_a = cnt_exact_a + popc(exact_req_a);
      cnt_prop_a  = cnt_prop_a  + popc(prop_pref_a);

      // Stub combine with OGEC
      ogec_exact_en = match_ok;
      exact_en_cfp    = ogec_exact_en & exact_req;
      exact_en_always = ogec_exact_en & exact_req_a;
      allow_cfp    = (prrc_budget_i != 0) && (popc(exact_req)   <= prrc_grant);
      allow_always = (prrc_budget_i != 0) && (popc(exact_req_a) <= prrc_grant_a);
      if (allow_cfp)    capture_cfp    = capture_cfp    + popc(exact_en_cfp);
      if (allow_always) capture_always = capture_always + popc(exact_en_always);
    end
  endtask

  initial begin
    n_pass = 0;
    sum_conf = 0; cnt_exact_req = 0; cnt_prop_pref = 0; cnt_grant_starve = 0;
    sum_conf_a = 0; cnt_exact_a = 0; cnt_prop_a = 0;
    capture_cfp = 0; capture_always = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: mixed peaks — high vs low conf split
    clear_in();
    match_ok = 8'hFF;
    prrc_budget_i = 8'd16;
    // tiles 0..3 low peak (conf=40), 4..7 high peak (conf=200)
    for (i = 0; i < 4; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd40;
    for (i = 4; i < 8; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd200;
    beat();
    accum_counters();
    c0 = conf_bus[0*CONF_W +: CONF_W];
    c1 = conf_bus[4*CONF_W +: CONF_W];
    if (!conf_valid) begin $display("FAIL Case1: conf_valid"); $fatal(1); end
    if (c0 !== 8'd40) begin $display("FAIL Case1: conf0=%0d", c0); $fatal(1); end
    if (c1 !== 8'd200) begin $display("FAIL Case1: conf4=%0d", c1); $fatal(1); end
    if (exact_req !== 8'b0000_1111) begin
      $display("FAIL Case1: exact_req=%b", exact_req); $fatal(1);
    end
    if (prop_pref !== 8'b1111_0000) begin
      $display("FAIL Case1: prop_pref=%b", prop_pref); $fatal(1);
    end
    // grant = 16 * 4 / 8 = 8
    if (prrc_grant !== 8'd8) begin
      $display("FAIL Case1: grant=%0d expected 8", prrc_grant); $fatal(1);
    end
    if (exact_req_a !== 8'hFF) begin
      $display("FAIL Case1: ALWAYS exact_req=%b", exact_req_a); $fatal(1);
    end
    if (prop_pref_a !== 8'h00) begin
      $display("FAIL Case1: ALWAYS prop=%b", prop_pref_a); $fatal(1);
    end
    $display("PASS Case1: split exact=%b prop=%b grant=%0d conf0=%0d conf4=%0d",
             exact_req, prop_pref, prrc_grant, c0, c1);
    n_pass = n_pass + 1;

    // Case2: all high conf → prop_pref all, grant=0, starve if any exact
    clear_in();
    match_ok = 8'hFF;
    prrc_budget_i = 8'd16;
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd255;
    beat();
    accum_counters();
    if (exact_req !== 8'h00) begin $display("FAIL Case2: exact=%b", exact_req); $fatal(1); end
    if (prop_pref !== 8'hFF) begin $display("FAIL Case2: prop=%b", prop_pref); $fatal(1); end
    if (prrc_grant !== 8'd0) begin $display("FAIL Case2: grant=%0d", prrc_grant); $fatal(1); end
    $display("PASS Case2: all-high prop=%b grant=%0d", prop_pref, prrc_grant);
    n_pass = n_pass + 1;

    // Case3: all low conf → all exact_req, grant=budget
    clear_in();
    match_ok = 8'hFF;
    prrc_budget_i = 8'd16;
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd50;
    beat();
    accum_counters();
    if (exact_req !== 8'hFF) begin $display("FAIL Case3: exact=%b", exact_req); $fatal(1); end
    if (prop_pref !== 8'h00) begin $display("FAIL Case3: prop=%b", prop_pref); $fatal(1); end
    if (prrc_grant !== 8'd16) begin $display("FAIL Case3: grant=%0d", prrc_grant); $fatal(1); end
    $display("PASS Case3: all-low exact=%b grant=%0d", exact_req, prrc_grant);
    n_pass = n_pass + 1;

    // Case4: saturate peak > 255 → conf=255; partial match_ok
    clear_in();
    match_ok = 8'b0000_1111;
    prrc_budget_i = 8'd8;
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd1000;
    beat();
    accum_counters();
    c0 = conf_bus[0*CONF_W +: CONF_W];
    if (c0 !== 8'd255) begin $display("FAIL Case4: sat conf=%0d", c0); $fatal(1); end
    // matched high → prop on low nibble only
    if (prop_pref !== 8'b0000_1111) begin
      $display("FAIL Case4: prop=%b", prop_pref); $fatal(1);
    end
    if (exact_req !== 8'h00) begin $display("FAIL Case4: exact=%b", exact_req); $fatal(1); end
    $display("PASS Case4: saturate conf=%0d prop=%b exact=%b grant=%0d",
             c0, prop_pref, exact_req, prrc_grant);
    n_pass = n_pass + 1;

    // Case5: grant starve — low conf but budget scaled to 0 via all-prop sibling beat
    // Force: match all, half high half — already counted; explicit starve:
    clear_in();
    match_ok = 8'hFF;
    prrc_budget_i = 8'd0;  // zero budget → grant 0; with low conf exact_req>0 → starve
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd10;
    beat();
    accum_counters();
    if (prrc_grant !== 8'd0) begin $display("FAIL Case5: grant=%0d", prrc_grant); $fatal(1); end
    if (exact_req === 8'h00) begin $display("FAIL Case5: expected exact_req"); $fatal(1); end
    $display("PASS Case5: starve grant=0 exact_pop=%0d", popc(exact_req));
    n_pass = n_pass + 1;

    if (capture_cfp >= capture_always) begin
      $display("FAIL ablation: capture_cfp=%0d should be < ALWAYS=%0d",
               capture_cfp, capture_always);
      $fatal(1);
    end
    $display("ABLATION capture_cfp=%0d capture_ALWAYS_EXACT=%0d (CFP lowers capture)",
             capture_cfp, capture_always);
    $display("COUNTERS sum_conf=%0d cnt_exact_req=%0d cnt_prop_pref=%0d cnt_grant_starve=%0d",
             sum_conf, cnt_exact_req, cnt_prop_pref, cnt_grant_starve);
    $display("COUNTERS_ABL sum_conf=%0d cnt_exact_req=%0d cnt_prop_pref=%0d",
             sum_conf_a, cnt_exact_a, cnt_prop_a);
    $display("PASS: all CFP-ConfGate cases");
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
