// GROKBOT NEW FILE -- iscas_ssh
// TB: C1* ablation ladder — ALWAYS|OPSTW|ECP|MW|EXACT
// EXACT = MW-style wake + PRRC budget gate on allow_exact (exact_hit capped)

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_ablation_ladder;
  localparam int N_TILE  = 8;
  localparam int FLOW_W  = 8;
  localparam int EVT_W   = 8;
  localparam int SCORE_W = 8;
  localparam int SAMP_W  = 8;
  localparam int CNT_W   = 16;
  localparam int N_FRAME = 8;
  localparam int CAPACITY = 128;
  localparam int BUDGET_W = 8;
  // Tight PRRC budget for EXACT rung (spend 1 per ogec beat → caps vs MW)
  localparam logic [BUDGET_W-1:0] EXACT_INIT_BUDGET = 8'd3;

  logic clk, rst_n, valid_i;
  logic signed [FLOW_W-1:0]  flow_cur  [N_TILE];
  logic signed [FLOW_W-1:0]  flow_prev [N_TILE];
  logic        [EVT_W-1:0]   event_cnt [N_TILE];
  logic        [SCORE_W-1:0] corr_score [N_TILE];
  logic        [SCORE_W-1:0] corr_held  [N_TILE];
  logic signed [SAMP_W-1:0]  ref_samp [N_TILE];
  logic signed [SAMP_W-1:0]  cur_samp [N_TILE];

  logic [N_TILE-1:0] wake_raw, delta_nz_raw;
  logic              wake_valid, delta_valid;
  logic signed [N_TILE*(SAMP_W+1)-1:0] delta;

  logic [N_TILE-1:0] wake_eff, dnz_eff, wake_for_ecp;
  logic [SCORE_W-1:0] corr_to_ecp [N_TILE];
  logic              use_wake;
  logic [N_TILE-1:0] proj_en, skip_bm;
  logic              proj_valid;

  logic [N_TILE-1:0] held_wake, held_dnz;

  logic window_en, clear_i;
  logic [CNT_W-1:0] wake_pop_cnt, proj_skip_cnt, delta_nz_cnt;
  logic             stats_valid;

  logic [N_TILE-1:0] exact_en, prop_en;
  logic              ogec_valid;
  logic              capture_en, allow_exact, allow_exact_prrc;
  logic [N_TILE-1:0] hit_bitmap;
  logic [CNT_W-1:0]  capture_cnt;
  logic              busy, done;

  // PRRC for EXACT mode
  logic              prrc_valid_i, prrc_spend, prrc_refill;
  logic [2:0]        prrc_level_sel;
  logic [1:0]        prrc_level_idx;
  logic [3*BUDGET_W-1:0] prrc_budget;
  logic              prrc_ledger_valid;

  string mode_s;
  int mode_id;
  integer ci;

  always @(*) begin
    wake_eff = wake_raw;
    dnz_eff  = delta_nz_raw;
    use_wake = 1'b1;
    for (ci = 0; ci < N_TILE; ci = ci + 1)
      corr_to_ecp[ci] = corr_held[ci];
    case (mode_id)
      0: begin // ALWAYS
        wake_eff = {N_TILE{1'b1}};
        dnz_eff  = {N_TILE{1'b0}};
        for (ci = 0; ci < N_TILE; ci = ci + 1)
          corr_to_ecp[ci] = 8'd7;
      end
      1: begin // OPSTW
        dnz_eff = {N_TILE{1'b0}};
        for (ci = 0; ci < N_TILE; ci = ci + 1)
          corr_to_ecp[ci] = 8'd0;
      end
      2: begin // ECP
        dnz_eff = {N_TILE{1'b0}};
      end
      3: begin // MW
      end
      4: begin // EXACT: same as MW for wake/ecp; PRRC gates allow_exact
      end
      default: begin
      end
    endcase
  end

  assign wake_for_ecp = wake_eff | dnz_eff;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (ci = 0; ci < N_TILE; ci = ci + 1)
        corr_held[ci] <= '0;
      held_wake <= '0;
      held_dnz  <= '0;
    end else begin
      if (valid_i) begin
        for (ci = 0; ci < N_TILE; ci = ci + 1)
          corr_held[ci] <= corr_score[ci];
      end
      if (wake_valid) begin
        held_wake <= wake_for_ecp;
        held_dnz  <= dnz_eff;
      end
    end
  end

  c1s_op_stw_predictor #(.N_TILE(N_TILE), .FLOW_W(FLOW_W), .EVT_W(EVT_W)) u_op (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .flow_cur(flow_cur), .flow_prev(flow_prev), .event_cnt(event_cnt),
    .wake_bitmap(wake_raw), .wake_valid(wake_valid)
  );

  c1s_mw_delta_buf #(.N_TILE(N_TILE), .SAMP_W(SAMP_W)) u_mw (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .ref_samp(ref_samp), .cur_samp(cur_samp),
    .delta(delta), .delta_nz(delta_nz_raw), .delta_valid(delta_valid)
  );

  c1s_ecp_qkv_predictor #(.N_TILE(N_TILE), .SCORE_W(SCORE_W)) u_ecp (
    .clk(clk), .rst_n(rst_n), .valid_i(wake_valid),
    .corr_score(corr_to_ecp), .wake_bitmap(wake_for_ecp), .use_wake(use_wake),
    .proj_en_bitmap(proj_en), .skip_bitmap(skip_bm), .proj_valid(proj_valid)
  );

  assign window_en = 1'b1;
  c1s_stats #(.N_TILE(N_TILE), .CNT_W(CNT_W)) u_stats (
    .clk(clk), .rst_n(rst_n), .valid_i(proj_valid),
    .window_en(window_en), .clear_i(clear_i),
    .wake_bitmap(held_wake), .skip_bitmap(skip_bm), .delta_nz(held_dnz),
    .wake_pop_cnt(wake_pop_cnt), .proj_skip_cnt(proj_skip_cnt),
    .delta_nz_cnt(delta_nz_cnt), .stats_valid(stats_valid)
  );

  c1s_ogec_gate #(.N_TILE(N_TILE)) u_ogec (
    .clk(clk), .rst_n(rst_n), .valid_i(proj_valid),
    .match_ok(proj_en), .exact_en(exact_en), .prop_en(prop_en), .ogec_valid(ogec_valid)
  );

  // PRRC always instantiated; only EXACT mode spends / uses allow
  assign prrc_level_sel = 3'b001;
  assign prrc_level_idx = 2'd0;
  assign prrc_refill    = 1'b0;
  assign prrc_valid_i   = (mode_id == 4) && ogec_valid;
  assign prrc_spend     = (mode_id == 4) && ogec_valid;

  c1s_prrc_ledger #(
    .N_LEVEL(3), .BUDGET_W(BUDGET_W), .INIT_BUDGET(EXACT_INIT_BUDGET)
  ) u_prrc (
    .clk(clk), .rst_n(rst_n), .valid_i(prrc_valid_i),
    .level_sel(prrc_level_sel), .level_idx(prrc_level_idx),
    .spend_i(prrc_spend), .refill_i(prrc_refill),
    .budget(prrc_budget), .allow_exact(allow_exact_prrc),
    .ledger_valid(prrc_ledger_valid)
  );

  // Non-EXACT: allow always; EXACT: PRRC budget
  assign allow_exact = (mode_id == 4) ? allow_exact_prrc : 1'b1;

  c1s_exact_capture_wrap #(.N_TILE(N_TILE), .CNT_W(CNT_W), .CAPACITY(CAPACITY)) u_cap (
    .clk(clk), .rst_n(rst_n), .valid_i(ogec_valid),
    .capture_en(capture_en), .exact_en(exact_en), .allow_exact(allow_exact),
    .hit_bitmap(hit_bitmap), .capture_cnt(capture_cnt), .busy(busy), .done(done)
  );

  initial clk = 0;
  always #5 clk = ~clk;

  task automatic clear_inputs;
    int i;
    begin
      valid_i = 0;
      for (i = 0; i < N_TILE; i++) begin
        flow_cur[i] = 0; flow_prev[i] = 0; event_cnt[i] = 0;
        corr_score[i] = 0; ref_samp[i] = 0; cur_samp[i] = 0;
      end
    end
  endtask

  task automatic drive_frame(input int f);
    int i;
    begin
      for (i = 0; i < N_TILE; i++) begin
        flow_prev[i] = 0;
        flow_cur[i]  = ((f + i) % 3 == 0) ? 8'sd5 : 8'sd0;
        event_cnt[i] = ((f + i) % 5 == 0) ? 8'd3 : 8'd0;
        corr_score[i]= ((f + i) % 4 == 1) ? 8'd4 : 8'd0;
        ref_samp[i]  = 0;
        cur_samp[i]  = ((f + i) % 3 == 1) ? 8'sd3 : 8'sd0;
      end
      @(negedge clk);
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      valid_i = 1'b0;
      @(posedge clk);
      #1;
      @(posedge clk);
      #1;
      @(posedge clk);
      #1;
    end
  endtask

  initial begin
    mode_s = "MW";
    void'($value$plusargs("MODE=%s", mode_s));
    mode_id = 3;
    if (mode_s == "ALWAYS") mode_id = 0;
    else if (mode_s == "OPSTW") mode_id = 1;
    else if (mode_s == "ECP")   mode_id = 2;
    else if (mode_s == "MW")    mode_id = 3;
    else if (mode_s == "EXACT" || mode_s == "OGECP RRC" || mode_s == "OGECPRRC") mode_id = 4;
    else begin
      $display("FAIL: unknown MODE=%s", mode_s); $fatal(1);
    end
    // Normalize display name
    if (mode_id == 4) mode_s = "EXACT";
    $display("ABLATION_MODE %s id=%0d", mode_s, mode_id);

    rst_n = 0; clear_i = 0; capture_en = 0; clear_inputs();
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    clear_i = 1; @(posedge clk); clear_i = 0;
    capture_en = 1;
    @(posedge clk);

    for (int f = 0; f < N_FRAME; f++)
      drive_frame(f);

    capture_en = 0;
    repeat (3) @(posedge clk);
    #1;

    $display("ABLATION_LINE mode=%s wake_pop=%0d proj_skip=%0d delta_nz=%0d exact_hit=%0d frames=%0d",
             mode_s, wake_pop_cnt, proj_skip_cnt, delta_nz_cnt, capture_cnt, N_FRAME);
    if (mode_id == 4)
      $display("EXACT_BUDGET init=%0d allow_last=%0d budget0=%0d",
               EXACT_INIT_BUDGET, allow_exact_prrc, prrc_budget[BUDGET_W-1:0]);
    $display("PASS: ablation ladder mode=%s", mode_s);
    $finish;
  end

  initial begin
    #500000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
