// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_cfp_sci_exact_glue — prove exact_hit deltas: CFP vs ALWAYS + SCI hold
// Instantiates real CFP / SCI / OGEC / PRRC / exact_capture + glue (additive path).

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_cfp_sci_exact_glue;
  localparam int N_TILE   = 8;
  localparam int PEAK_W   = 16;
  localparam int CONF_W   = 8;
  localparam int BUDGET_W = 8;
  localparam int CNT_W    = 16;
  localparam int CAPACITY = 64;
  localparam int FEAT_W   = 8;
  localparam int FLOW_W   = 8;
  localparam int SCI_W    = 8;

  logic clk, rst_n, valid_i;
  logic [N_TILE*PEAK_W-1:0] corr_peak_bus;
  logic [N_TILE-1:0]        match_ok;
  logic [BUDGET_W-1:0]      prrc_budget_slice;

  // CFP (normal) + ALWAYS ablate twin
  logic [N_TILE*CONF_W-1:0] conf_bus, conf_bus_a;
  logic [N_TILE-1:0]        exact_req, prop_pref, exact_req_a, prop_pref_a;
  logic [BUDGET_W-1:0]      prrc_grant, prrc_grant_a;
  logic                     conf_valid, conf_valid_a;

  // SCI
  logic                     it_valid_i;
  logic [N_TILE*FEAT_W-1:0] f1_bus, f2_warp_bus;
  logic [N_TILE*FLOW_W-1:0] f_hat_bus;
  logic                     bisat_occ_exit_i;
  logic [N_TILE*SCI_W-1:0]  g_sci_bus;
  logic [N_TILE-1:0]        scrub_en;
  logic                     exit_ok, exact_hold, sci_valid;

  // OGEC
  logic [N_TILE-1:0] exact_en_ogec, prop_en_ogec;
  logic              ogec_valid;

  // PRRC (shared budget bank for both paths — we drive allow via spend policy)
  logic              prrc_valid_i, prrc_spend, prrc_refill;
  logic [2:0]        prrc_level_sel;
  logic [1:0]        prrc_level_idx;
  logic [3*BUDGET_W-1:0] prrc_budget;
  logic              allow_exact_prrc, prrc_ledger_valid;

  // Glue CFP path + ALWAYS bypass path + SCI bypass
  logic [N_TILE-1:0] exact_en_cfp, exact_en_alw, exact_en_sci_bypass;
  logic              allow_cfp, allow_alw, allow_sci_bypass;
  logic              cap_en_cfp, cap_en_alw, cap_en_hold, cap_en_sci_bypass;
  logic              pe_cfp, pe_alw, pe_hold, pe_sci_bypass;
  logic              glue_v_cfp, glue_v_alw, glue_v_hold, glue_v_sci_bypass;
  logic [BUDGET_W-1:0] pop_req_cfp, pop_req_alw;
  logic              grant_ok_cfp, grant_ok_alw;
  logic              hold_act_cfp, hold_act_hold;

  logic capture_en_raw;

  // Captures
  logic [N_TILE-1:0] hit_cfp, hit_alw, hit_hold, hit_sci_bypass;
  logic [CNT_W-1:0]  cnt_cfp, cnt_alw, cnt_hold, cnt_sci_bypass;
  logic              busy_cfp, done_cfp, busy_alw, done_alw;
  logic              busy_hold, done_hold, busy_sb, done_sb;

  integer i;
  int n_pass;
  int saved_cfp, saved_alw, saved_hold, saved_sb;

  c1s_cfp_confgate #(
    .N_TILE(N_TILE), .PEAK_W(PEAK_W), .CONF_W(CONF_W), .BUDGET_W(BUDGET_W),
    .TH_HI(8'd160), .TH_LO(8'd64), .ABLATE_ALWAYS_EXACT(1'b0)
  ) u_cfp (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .corr_peak_bus(corr_peak_bus), .match_ok(match_ok),
    .prrc_budget_i(prrc_budget_slice),
    .conf_bus(conf_bus), .exact_req(exact_req), .prop_pref(prop_pref),
    .prrc_grant(prrc_grant), .conf_valid(conf_valid)
  );

  c1s_cfp_confgate #(
    .N_TILE(N_TILE), .PEAK_W(PEAK_W), .CONF_W(CONF_W), .BUDGET_W(BUDGET_W),
    .TH_HI(8'd160), .TH_LO(8'd64), .ABLATE_ALWAYS_EXACT(1'b1)
  ) u_cfp_always (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .corr_peak_bus(corr_peak_bus), .match_ok(match_ok),
    .prrc_budget_i(prrc_budget_slice),
    .conf_bus(conf_bus_a), .exact_req(exact_req_a), .prop_pref(prop_pref_a),
    .prrc_grant(prrc_grant_a), .conf_valid(conf_valid_a)
  );

  c1s_sci_cleanexit #(
    .N_TILE(N_TILE), .FEAT_W(FEAT_W), .FLOW_W(FLOW_W), .SCI_W(SCI_W),
    .TH_EXIT(8'd200), .TH_SCRUB(8'd80), .ABLATE_EXIT_ONLY(1'b0)
  ) u_sci (
    .clk(clk), .rst_n(rst_n), .it_valid_i(it_valid_i),
    .f1_bus(f1_bus), .f2_warp_bus(f2_warp_bus), .f_hat_bus(f_hat_bus),
    .bisat_occ_exit_i(bisat_occ_exit_i),
    .g_sci_bus(g_sci_bus), .scrub_en(scrub_en),
    .exit_ok(exit_ok), .exact_hold(exact_hold), .sci_valid(sci_valid)
  );

  c1s_ogec_gate #(.N_TILE(N_TILE)) u_ogec (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .match_ok(match_ok), .exact_en(exact_en_ogec), .prop_en(prop_en_ogec),
    .ogec_valid(ogec_valid)
  );

  assign prrc_level_sel = 3'b001;
  assign prrc_level_idx = 2'd0;
  assign prrc_refill    = 1'b0;
  // Feed CFP grant soft-budget as INIT; spend lightly so allow stays high for ablation window
  c1s_prrc_ledger #(
    .N_LEVEL(3), .BUDGET_W(BUDGET_W), .INIT_BUDGET(8'd32)
  ) u_prrc (
    .clk(clk), .rst_n(rst_n), .valid_i(prrc_valid_i),
    .level_sel(prrc_level_sel), .level_idx(prrc_level_idx),
    .spend_i(prrc_spend), .refill_i(prrc_refill),
    .budget(prrc_budget), .allow_exact(allow_exact_prrc),
    .ledger_valid(prrc_ledger_valid)
  );

  // Path A: full CFP + SCI glue
  c1s_cfp_sci_exact_glue #(
    .N_TILE(N_TILE), .BUDGET_W(BUDGET_W),
    .ABLATE_CFP_BYPASS(1'b0), .ABLATE_SCI_BYPASS(1'b0)
  ) u_glue_cfp (
    .clk(clk), .rst_n(rst_n), .valid_i(conf_valid && ogec_valid),
    .ogec_exact_en(exact_en_ogec),
    .cfp_exact_req(exact_req), .cfp_prrc_grant(prrc_grant),
    .prrc_allow_exact(allow_exact_prrc),
    .sci_exact_hold(exact_hold), .sci_scrub_en(scrub_en),
    .capture_en_raw(capture_en_raw),
    .exact_en_to_capture(exact_en_cfp), .allow_exact_final(allow_cfp),
    .capture_en_gated(cap_en_cfp), .pe_clk_en(pe_cfp), .glue_valid(glue_v_cfp),
    .pop_exact_req(pop_req_cfp), .grant_ok(grant_ok_cfp), .hold_active(hold_act_cfp)
  );

  // Path B: CFP ALWAYS (bypass) + SCI still on — for CFP vs ALWAYS delta
  c1s_cfp_sci_exact_glue #(
    .N_TILE(N_TILE), .BUDGET_W(BUDGET_W),
    .ABLATE_CFP_BYPASS(1'b1), .ABLATE_SCI_BYPASS(1'b0)
  ) u_glue_alw (
    .clk(clk), .rst_n(rst_n), .valid_i(conf_valid_a && ogec_valid),
    .ogec_exact_en(exact_en_ogec),
    .cfp_exact_req(exact_req_a), .cfp_prrc_grant(prrc_grant_a),
    .prrc_allow_exact(allow_exact_prrc),
    .sci_exact_hold(exact_hold), .sci_scrub_en(scrub_en),
    .capture_en_raw(capture_en_raw),
    .exact_en_to_capture(exact_en_alw), .allow_exact_final(allow_alw),
    .capture_en_gated(cap_en_alw), .pe_clk_en(pe_alw), .glue_valid(glue_v_alw),
    .pop_exact_req(pop_req_alw), .grant_ok(grant_ok_alw), .hold_active()
  );

  // Path C: CFP on + SCI hold forced-dirty window (same as path A; separate capture)
  // Path D: SCI bypass — hold ignored (exact_hit higher under dirty iters)
  c1s_cfp_sci_exact_glue #(
    .N_TILE(N_TILE), .BUDGET_W(BUDGET_W),
    .ABLATE_CFP_BYPASS(1'b0), .ABLATE_SCI_BYPASS(1'b1)
  ) u_glue_sci_bypass (
    .clk(clk), .rst_n(rst_n), .valid_i(conf_valid && ogec_valid),
    .ogec_exact_en(exact_en_ogec),
    .cfp_exact_req(exact_req), .cfp_prrc_grant(prrc_grant),
    .prrc_allow_exact(allow_exact_prrc),
    .sci_exact_hold(exact_hold), .sci_scrub_en(scrub_en),
    .capture_en_raw(capture_en_raw),
    .exact_en_to_capture(exact_en_sci_bypass), .allow_exact_final(allow_sci_bypass),
    .capture_en_gated(cap_en_sci_bypass), .pe_clk_en(pe_sci_bypass),
    .glue_valid(glue_v_sci_bypass),
    .pop_exact_req(), .grant_ok(), .hold_active()
  );

  c1s_exact_capture_wrap #(.N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)) u_cap_cfp (
    .clk(clk), .rst_n(rst_n), .valid_i(glue_v_cfp),
    .capture_en(cap_en_cfp), .exact_en(exact_en_cfp), .allow_exact(allow_cfp),
    .hit_bitmap(hit_cfp), .capture_cnt(cnt_cfp), .busy(busy_cfp), .done(done_cfp)
  );

  c1s_exact_capture_wrap #(.N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)) u_cap_alw (
    .clk(clk), .rst_n(rst_n), .valid_i(glue_v_alw),
    .capture_en(cap_en_alw), .exact_en(exact_en_alw), .allow_exact(allow_alw),
    .hit_bitmap(hit_alw), .capture_cnt(cnt_alw), .busy(busy_alw), .done(done_alw)
  );

  // Hold path reuses CFP glue capture but we also log hold-gated vs bypass under dirty SCI
  c1s_exact_capture_wrap #(.N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)) u_cap_hold (
    .clk(clk), .rst_n(rst_n), .valid_i(glue_v_cfp),
    .capture_en(cap_en_cfp), .exact_en(exact_en_cfp), .allow_exact(allow_cfp),
    .hit_bitmap(hit_hold), .capture_cnt(cnt_hold), .busy(busy_hold), .done(done_hold)
  );

  c1s_exact_capture_wrap #(.N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)) u_cap_sb (
    .clk(clk), .rst_n(rst_n), .valid_i(glue_v_sci_bypass),
    .capture_en(cap_en_sci_bypass), .exact_en(exact_en_sci_bypass),
    .allow_exact(allow_sci_bypass),
    .hit_bitmap(hit_sci_bypass), .capture_cnt(cnt_sci_bypass),
    .busy(busy_sb), .done(done_sb)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_cfp_sci_exact_glue.vcd");
      $dumpvars(0, tb_c1s_cfp_sci_exact_glue);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_in;
    begin
      valid_i = 1'b0;
      it_valid_i = 1'b0;
      corr_peak_bus = '0;
      match_ok = '0;
      prrc_budget_slice = 8'd16;
      f1_bus = '0;
      f2_warp_bus = '0;
      f_hat_bus = '0;
      bisat_occ_exit_i = 1'b0;
      prrc_valid_i = 1'b0;
      prrc_spend = 1'b0;
      capture_en_raw = 1'b0;
    end
  endtask

  task automatic beat_cfp_sci;
    begin
      @(negedge clk);
      valid_i = 1'b1;
      it_valid_i = 1'b1;
      prrc_valid_i = 1'b1;
      prrc_spend = 1'b0; // keep allow high
      @(posedge clk);
      #1;
      valid_i = 1'b0;
      it_valid_i = 1'b0;
      prrc_valid_i = 1'b0;
    end
  endtask

  task automatic set_clean_sci;
    int t;
    begin
      for (t = 0; t < N_TILE; t = t + 1) begin
        f1_bus[t*FEAT_W +: FEAT_W] = 8'sd10;
        f2_warp_bus[t*FEAT_W +: FEAT_W] = 8'sd10; // |diff|=0 → g=255
        f_hat_bus[t*FLOW_W +: FLOW_W] = 8'd1;
      end
    end
  endtask

  task automatic set_dirty_sci;
    int t;
    begin
      for (t = 0; t < N_TILE; t = t + 1) begin
        // signed FEAT: 127 vs -127 → |diff|≈254 → g_sci≈1 < TH_SCRUB
        f1_bus[t*FEAT_W +: FEAT_W] = 8'sd127;
        f2_warp_bus[t*FEAT_W +: FEAT_W] = -8'sd127;
        f_hat_bus[t*FLOW_W +: FLOW_W] = 8'd1;
      end
    end
  endtask

  task automatic set_mixed_peaks;
    int t;
    begin
      match_ok = 8'hFF;
      prrc_budget_slice = 8'd16;
      for (t = 0; t < 4; t = t + 1)
        corr_peak_bus[t*PEAK_W +: PEAK_W] = 16'd40;   // low → exact_req
      for (t = 4; t < 8; t = t + 1)
        corr_peak_bus[t*PEAK_W +: PEAK_W] = 16'd200;  // high → prop
    end
  endtask

  initial begin
    n_pass = 0;
    saved_cfp = 0; saved_alw = 0; saved_hold = 0; saved_sb = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ---- Phase 1: clean SCI + CFP vs ALWAYS (exact_hit delta) ----
    clear_in();
    capture_en_raw = 1'b1;
    set_clean_sci();
    set_mixed_peaks();
    // Prime SCI once so hold=0
    beat_cfp_sci();
    if (exact_hold !== 1'b0) begin
      $display("FAIL P1 prime: expected hold=0 exit=%0d", exit_ok); $fatal(1);
    end
    // Capture window: several beats
    repeat (6) begin
      set_mixed_peaks();
      set_clean_sci();
      beat_cfp_sci();
    end
    // Drop capture to latch done
    @(negedge clk); capture_en_raw = 1'b0; @(posedge clk); #1;
    repeat (2) @(posedge clk);

    if (cnt_cfp >= cnt_alw) begin
      $display("FAIL P1: CFP exact_hit=%0d should be < ALWAYS=%0d", cnt_cfp, cnt_alw);
      $fatal(1);
    end
    if (cnt_cfp == 0) begin
      $display("FAIL P1: CFP capture should be >0"); $fatal(1);
    end
    $display("PASS P1: CFP vs ALWAYS exact_hit cfp=%0d always=%0d (CFP lowers)",
             cnt_cfp, cnt_alw);
    saved_cfp = cnt_cfp;
    saved_alw = cnt_alw;
    n_pass = n_pass + 1;

    // ---- Phase 2: dirty SCI hold freezes capture vs SCI bypass ----
    // Reset captures via rst; prime SCI hold BEFORE capture_en so no leaked hits
    rst_n = 1'b0; @(posedge clk); #1; rst_n = 1'b1; @(posedge clk);
    clear_in();
    capture_en_raw = 1'b0;
    set_dirty_sci();
    set_mixed_peaks();
    beat_cfp_sci();
    if (exact_hold !== 1'b1) begin
      $display("FAIL P2: expected exact_hold=1 scrub=%b", scrub_en); $fatal(1);
    end
    // Glue samples registered SCI hold — settle then raise capture_en under hold
    @(negedge clk);
    capture_en_raw = 1'b1;
    @(posedge clk); #1;
    if (cap_en_cfp !== 1'b0) begin
      $display("FAIL P2: capture_en_gated should be 0 under hold got %0d hold=%0d",
               cap_en_cfp, exact_hold); $fatal(1);
    end
    if (cap_en_sci_bypass !== 1'b1) begin
      $display("FAIL P2: SCI bypass should keep capture_en=1"); $fatal(1);
    end
    if (pe_cfp !== 1'b0) begin
      $display("FAIL P2: pe_clk_en should be 0 under scrub pe=%0d scrub=%b",
               pe_cfp, scrub_en); $fatal(1);
    end
    repeat (5) begin
      set_mixed_peaks();
      set_dirty_sci();
      beat_cfp_sci();
    end
    @(negedge clk); capture_en_raw = 1'b0; @(posedge clk); #1;
    repeat (2) @(posedge clk);

    if (cnt_hold != 0) begin
      $display("FAIL P2: hold path exact_hit=%0d expected 0", cnt_hold); $fatal(1);
    end
    if (cnt_sci_bypass == 0) begin
      $display("FAIL P2: SCI-bypass exact_hit should be >0 got %0d", cnt_sci_bypass);
      $fatal(1);
    end
    $display("PASS P2: SCI hold exact_hit=%0d vs SCI-bypass=%0d (hold freezes)",
             cnt_hold, cnt_sci_bypass);
    saved_hold = cnt_hold;
    saved_sb = cnt_sci_bypass;
    n_pass = n_pass + 1;

    // ---- Phase 3: grant AND policy — all high conf → grant=0 → allow_final=0 ----
    rst_n = 1'b0; @(posedge clk); #1; rst_n = 1'b1; @(posedge clk);
    clear_in();
    capture_en_raw = 1'b1;
    set_clean_sci();
    match_ok = 8'hFF;
    prrc_budget_slice = 8'd16;
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd255;
    beat_cfp_sci();
    if (prrc_grant !== 8'd0) begin
      $display("FAIL P3: grant=%0d", prrc_grant); $fatal(1);
    end
    // With CFP bypass off and exact_req=0, pop=0 <= grant → allow may still be 1
    // Force low conf with zero budget grant path via CFP: low peaks + budget_slice=0
    clear_in();
    capture_en_raw = 1'b1;
    set_clean_sci();
    match_ok = 8'hFF;
    prrc_budget_slice = 8'd0;
    for (i = 0; i < N_TILE; i = i + 1)
      corr_peak_bus[i*PEAK_W +: PEAK_W] = 16'd20;
    beat_cfp_sci();
    if (exact_req === 8'h00) begin $display("FAIL P3: need exact_req"); $fatal(1); end
    // Glue valid = conf_valid&ogec_valid → latch one cycle after CFP/OGEC
    @(posedge clk); #1;
    if (grant_ok_cfp !== 1'b0) begin
      $display("FAIL P3: grant_ok should be 0 pop=%0d grant=%0d exact_req=%b",
               pop_req_cfp, prrc_grant, exact_req);
      $fatal(1);
    end
    if (allow_cfp !== 1'b0) begin
      $display("FAIL P3: allow_exact_final should be 0 allow=%0d prrc_allow=%0d",
               allow_cfp, allow_exact_prrc); $fatal(1);
    end
    $display("PASS P3: grant AND policy allow_final=0 pop=%0d grant=%0d",
             pop_req_cfp, prrc_grant);
    n_pass = n_pass + 1;

    $display("ABLATION_GLUE exact_hit_cfp=%0d exact_hit_ALWAYS=%0d exact_hit_SCI_hold=%0d exact_hit_SCI_bypass=%0d",
             saved_cfp, saved_alw, saved_hold, saved_sb);
    $display("PASS: all CFP/SCI exact glue cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin
    #500000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
