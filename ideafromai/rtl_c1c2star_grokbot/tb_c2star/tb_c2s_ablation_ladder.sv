// GROKBOT NEW FILE -- iscas_ssh
// TB: C2* ablation ladder — ALWAYS vs HBG vs HBG+SMAM vs HBG+SMAM+ADP
// Plusarg +MODE=ALWAYS|HBG|HBGSMAM|FULL
// Counters via c2s_stats: mac_en_cnt / skip_cnt / gate_fire_cnt

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_ablation_ladder;
  localparam int AMP_W  = 8;
  localparam int ACC_W  = 16;
  localparam int W_A    = 8;
  localparam int W_B    = 8;
  localparam int W_ACC  = 16;
  localparam int CNT_W  = 16;
  localparam int N_BEAT = 16;
  localparam logic signed [AMP_W-1:0] EPS = 8'sd1;

  logic clk, rst_n;
  logic amp_valid;
  logic signed [AMP_W-1:0] amp;

  // HBG
  logic hbg_g, hbg_gp_valid, hbg_pe_clk_en;
  logic signed [AMP_W-1:0] hbg_p;
  logic hbg_g_q;
  logic signed [AMP_W-1:0] hbg_p_q;

  // SMAM
  logic smam_mac_en, smam_mask_add_en, smam_out_valid;
  logic signed [AMP_W-1:0] smam_mac_payload;
  logic signed [ACC_W-1:0] smam_mask_add_result;

  // ADP
  logic adp_valid_i, adp_skip_a, adp_skip_b, adp_mac_en_i, adp_clear_i;
  logic signed [W_A-1:0] adp_a;
  logic signed [W_B-1:0] adp_b;
  logic signed [W_ACC-1:0] adp_acc;
  logic adp_skipped, adp_out_valid;

  // Stats inputs (mode-muxed)
  logic stats_valid_i, window_en, clear_i;
  logic st_mac_en, st_skipped, st_gate_fire;
  logic [CNT_W-1:0] mac_en_cnt, skip_cnt, gate_fire_cnt;
  logic stats_out_valid;

  string mode_s;
  int mode_id;
  int bi;

  // Deterministic amp / skip side patterns (same all modes)
  // amp schedule: mix of |amp|<=EPS (no gate) and |amp|>EPS
  function automatic logic signed [AMP_W-1:0] amp_of(input int k);
    begin
      case (k % 8)
        0: amp_of = 8'sd0;    // |amp|=0  ≤EPS → no gate
        1: amp_of = 8'sd1;    // |amp|=1  ≤EPS (strict >) → no gate
        2: amp_of = 8'sd2;    // gate
        3: amp_of = -8'sd3;   // gate
        4: amp_of = 8'sd0;    // no gate
        5: amp_of = 8'sd5;    // gate
        6: amp_of = -8'sd1;   // no gate (|amp|==1)
        7: amp_of = 8'sd7;    // gate
        default: amp_of = 8'sd0;
      endcase
    end
  endfunction

  // ADP skip sides on beats where (k%5==0) skip_a, (k%7==0) skip_b
  function automatic logic skip_a_of(input int k);
    skip_a_of = (k % 5 == 0);
  endfunction
  function automatic logic skip_b_of(input int k);
    skip_b_of = (k % 7 == 0);
  endfunction

  c2s_hbg_rp_packetizer #(.AMP_W(AMP_W), .EPS(EPS)) u_hbg (
    .clk(clk), .rst_n(rst_n), .amp_valid(amp_valid), .amp(amp),
    .g(hbg_g), .p(hbg_p), .gp_valid(hbg_gp_valid), .pe_clk_en(hbg_pe_clk_en)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      hbg_g_q <= 1'b0;
      hbg_p_q <= '0;
    end else if (amp_valid) begin
      hbg_g_q <= hbg_g;
      hbg_p_q <= hbg_p;
    end
  end

  c2s_smam_rp #(.AMP_W(AMP_W), .ACC_W(ACC_W)) u_smam (
    .clk(clk), .rst_n(rst_n),
    .valid(hbg_gp_valid), .spike_gate(hbg_g_q), .payload(hbg_p_q),
    .mac_en(smam_mac_en), .mask_add_en(smam_mask_add_en),
    .mac_payload(smam_mac_payload), .mask_add_result(smam_mask_add_result),
    .out_valid(smam_out_valid)
  );

  c2s_adp_mac #(.W_A(W_A), .W_B(W_B), .W_ACC(W_ACC), .FORBID_REORDER(1'b1)) u_adp (
    .clk(clk), .rst_n(rst_n),
    .valid_i(adp_valid_i), .a(adp_a), .b(adp_b),
    .skip_a(adp_skip_a), .skip_b(adp_skip_b), .mac_en(adp_mac_en_i),
    .clear_i(adp_clear_i), .acc(adp_acc), .skipped(adp_skipped), .out_valid(adp_out_valid)
  );

  c2s_stats #(.CNT_W(CNT_W)) u_stats (
    .clk(clk), .rst_n(rst_n),
    .valid_i(stats_valid_i), .window_en(window_en), .clear_i(clear_i),
    .mac_en(st_mac_en), .skipped(st_skipped), .gate_fire(st_gate_fire),
    .mac_en_cnt(mac_en_cnt), .skip_cnt(skip_cnt), .gate_fire_cnt(gate_fire_cnt),
    .stats_valid(stats_out_valid)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  // Drive one amp beat; sample stats when HBG gp_valid & g_q align (SMAM window)
  task automatic drive_beat(input int k);
    begin
      amp = amp_of(k);
      adp_a = amp_of(k);
      adp_b = 8'sd3;
      adp_skip_a = skip_a_of(k);
      adp_skip_b = skip_b_of(k);
      adp_clear_i = 1'b0;

      // Pulse amp_valid for one cycle; HBG registers gp_valid/g_q on that edge
      @(negedge clk);
      amp_valid = 1'b1;
      @(posedge clk);
      #1;
      // Now gp_valid=1 and g_q=g — SMAM mac_en is live combinationally
      amp_valid = 1'b0;

      case (mode_id)
        0: begin // ALWAYS
          st_mac_en     = 1'b1;
          st_skipped    = 1'b0;
          st_gate_fire  = 1'b1;
          stats_valid_i = 1'b1;
          adp_valid_i   = 1'b0;
          adp_mac_en_i  = 1'b0;
        end
        1: begin // HBG-only
          st_mac_en     = hbg_g_q;
          st_skipped    = ~hbg_g_q;
          st_gate_fire  = hbg_g_q;
          stats_valid_i = 1'b1;
          adp_valid_i   = 1'b0;
          adp_mac_en_i  = 1'b0;
        end
        2: begin // HBG+SMAM
          st_mac_en     = smam_mac_en;
          st_skipped    = ~smam_mac_en;
          st_gate_fire  = smam_mac_en;
          stats_valid_i = 1'b1;
          adp_valid_i   = 1'b0;
          adp_mac_en_i  = 1'b0;
        end
        default: begin // FULL: HBG+SMAM+ADP
          adp_mac_en_i  = smam_mac_en;
          adp_valid_i   = 1'b1;
          #0; // ADP comb (skipped/do_mac) settles
          st_mac_en     = (smam_mac_en && !adp_skipped);
          st_skipped    = adp_skipped;
          st_gate_fire  = hbg_g_q;
          stats_valid_i = 1'b1;
        end
      endcase

      @(posedge clk);
      #1;
      stats_valid_i = 1'b0;
      adp_valid_i   = 1'b0;

      // Drain
      @(posedge clk);
      #1;
    end
  endtask

  initial begin
    mode_s = "FULL";
    void'($value$plusargs("MODE=%s", mode_s));
    mode_id = 3;
    if (mode_s == "ALWAYS")  mode_id = 0;
    else if (mode_s == "HBG")     mode_id = 1;
    else if (mode_s == "HBGSMAM") mode_id = 2;
    else if (mode_s == "FULL")    mode_id = 3;
    else begin
      $display("FAIL: unknown MODE=%s (want ALWAYS|HBG|HBGSMAM|FULL)", mode_s);
      $fatal(1);
    end
    $display("ABLATION_MODE %s id=%0d", mode_s, mode_id);

    rst_n = 0;
    amp_valid = 0; amp = '0;
    adp_valid_i = 0; adp_a = 0; adp_b = 0;
    adp_skip_a = 0; adp_skip_b = 0; adp_mac_en_i = 0; adp_clear_i = 0;
    stats_valid_i = 0; // keep window_en so counters hold; sample then finish clear_i = 0;
    st_mac_en = 0; st_skipped = 0; st_gate_fire = 0;

    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    clear_i = 1; @(posedge clk); clear_i = 0;
    window_en = 1;
    @(posedge clk);

    for (bi = 0; bi < N_BEAT; bi++)
      drive_beat(bi);

    // keep window_en so counters hold; sample then finish
    repeat (2) @(posedge clk);
    #1;

    $display("ABLATION_LINE mode=%s mac_en=%0d skipped=%0d gate_fire=%0d beats=%0d",
             mode_s, mac_en_cnt, skip_cnt, gate_fire_cnt, N_BEAT);
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
