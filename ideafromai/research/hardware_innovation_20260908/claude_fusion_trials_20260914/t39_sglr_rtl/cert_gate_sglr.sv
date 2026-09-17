// T25 生成文件（勿手改）：掩码加法树门核，变体 = k4。
// dot10(t,w) = Σ_{c∈supp(t)} aq[t][c]·w[c]（4 项选位常数和，烘焙字面量）。
// 端口/时序/判决逻辑与 t15_synth/cert_gate_bitl_synth.sv 逐位一致；
// 差异仅在：无 a/pn/lut ROM、无 $readmemh，P_t/N_t/dot 常数全部为 RTL 字面量。
// 生成：t25_k4_gate/t25_gen.py（A 来源：results/t21b_k4_gate_params.npz: L8_A（t21f k=4 微调））
module cert_gate_bitl_synth (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_sop,
    input  wire [11:0] in_h,
    input  wire [9:0]  in_sign,
    input  wire [4:0]  in_e,
    input  wire        in_plane_valid,
    input  wire [9:0]  in_plane,
    input  wire [639:0] in_thr_row,
    output wire [9:0]  out_dec,
    output wire [9:0]  dbg_locked,
    output wire [9:0]  out_need
);

    reg [11:0] h_reg;
    reg [4:0]  m_chk;
    reg signed [47:0] vtop  [0:9];
    reg signed [47:0] pmsk  [0:9];
    reg signed [47:0] nmsk  [0:9];
    reg [63:0] tauw   [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t;

    // sop 拍与数据平面拍共用一套选位加法树（in_sop 选 in_sign，否则选 in_plane）
    wire [9:0] wsel = in_sop ? in_sign : in_plane;

    wire signed [47:0] lutw [0:9];
    assign lutw[0] = (wsel[0] ? 48'sd796 : 48'sd0) + (wsel[3] ? 48'sd1077 : 48'sd0) + (wsel[6] ? -48'sd831 : 48'sd0) + (wsel[7] ? -48'sd840 : 48'sd0);
    assign lutw[1] = (wsel[1] ? -48'sd1312 : 48'sd0) + (wsel[2] ? -48'sd1414 : 48'sd0) + (wsel[4] ? -48'sd1431 : 48'sd0) + (wsel[6] ? -48'sd1079 : 48'sd0);
    assign lutw[2] = (wsel[3] ? 48'sd995 : 48'sd0) + (wsel[4] ? 48'sd715 : 48'sd0) + (wsel[5] ? 48'sd768 : 48'sd0) + (wsel[9] ? 48'sd908 : 48'sd0);
    assign lutw[3] = (wsel[1] ? -48'sd1141 : 48'sd0) + (wsel[5] ? -48'sd1018 : 48'sd0) + (wsel[7] ? 48'sd745 : 48'sd0) + (wsel[8] ? -48'sd1133 : 48'sd0);
    assign lutw[4] = (wsel[3] ? -48'sd596 : 48'sd0) + (wsel[4] ? -48'sd1574 : 48'sd0) + (wsel[5] ? -48'sd1008 : 48'sd0) + (wsel[8] ? -48'sd1048 : 48'sd0);
    assign lutw[5] = (wsel[0] ? -48'sd1359 : 48'sd0) + (wsel[1] ? -48'sd1146 : 48'sd0) + (wsel[4] ? -48'sd1121 : 48'sd0) + (wsel[7] ? -48'sd1116 : 48'sd0);
    assign lutw[6] = (wsel[1] ? 48'sd1260 : 48'sd0) + (wsel[2] ? -48'sd1669 : 48'sd0) + (wsel[4] ? 48'sd1656 : 48'sd0) + (wsel[5] ? 48'sd1358 : 48'sd0);
    assign lutw[7] = (wsel[3] ? 48'sd1361 : 48'sd0) + (wsel[6] ? 48'sd1545 : 48'sd0) + (wsel[8] ? -48'sd1345 : 48'sd0) + (wsel[9] ? -48'sd1366 : 48'sd0);
    assign lutw[8] = (wsel[1] ? 48'sd1354 : 48'sd0) + (wsel[3] ? -48'sd1350 : 48'sd0) + (wsel[6] ? 48'sd1093 : 48'sd0) + (wsel[7] ? -48'sd1712 : 48'sd0);
    assign lutw[9] = (wsel[0] ? 48'sd1320 : 48'sd0) + (wsel[1] ? -48'sd1330 : 48'sd0) + (wsel[6] ? -48'sd1262 : 48'sd0) + (wsel[8] ? -48'sd1138 : 48'sd0);

    wire signed [47:0] P_t [0:9];
    wire signed [47:0] N_t [0:9];
    assign P_t[0] = 48'sd1873;  assign N_t[0] = -48'sd1671;
    assign P_t[1] = 48'sd0;  assign N_t[1] = -48'sd5236;
    assign P_t[2] = 48'sd3386;  assign N_t[2] = 48'sd0;
    assign P_t[3] = 48'sd745;  assign N_t[3] = -48'sd3292;
    assign P_t[4] = 48'sd0;  assign N_t[4] = -48'sd4226;
    assign P_t[5] = 48'sd0;  assign N_t[5] = -48'sd4742;
    assign P_t[6] = 48'sd4274;  assign N_t[6] = -48'sd1669;
    assign P_t[7] = 48'sd2906;  assign N_t[7] = -48'sd2711;
    assign P_t[8] = 48'sd2447;  assign N_t[8] = -48'sd3062;
    assign P_t[9] = 48'sd1320;  assign N_t[9] = -48'sd3730;

    wire [63:0]        tau_sel [0:9];
    wire signed [47:0] thrw   [0:9];
    wire signed [47:0] nv     [0:9];
    wire signed [47:0] vmin_n [0:9];
    wire signed [47:0] vmax_n [0:9];
    wire               lock_n [0:9];
    wire signed [47:0] thr0   [0:9];
    wire signed [47:0] vtop0  [0:9];
    wire signed [47:0] vmin0  [0:9];
    wire signed [47:0] vmax0  [0:9];
    wire               lock0  [0:9];
    wire               dec0   [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tau_sel[gt]    = tauw[gt];
            assign thrw[gt]       = $signed(tau_sel[gt][47:0]);
            assign nv[gt]         = (vtop[gt] << 1) + lutw[gt];
            assign vmin_n[gt]     = (nv[gt] << m_chk) + nmsk[gt];
            assign vmax_n[gt]     = (nv[gt] << m_chk) + pmsk[gt];
            assign lock_n[gt]     = ~locked[gt]
                                  & ((vmin_n[gt] >= thrw[gt]) | (vmax_n[gt] < thrw[gt]));
            // T39b：sop 拍粗区间（e 个低位全未知，V ∈ vtop·2^e + [N(2^e−1), P(2^e−1)]）
            // 与首平面细区间同形，只是 d₁ 取极值 2N / 2P——故是 k=0 那一档检查。
            assign thr0[gt]  = $signed(in_thr_row[gt*64 +: 48]);
            assign vtop0[gt] = -lutw[gt];
            assign vmin0[gt] = (vtop0[gt] <<< in_e) + ((N_t[gt] <<< in_e) - N_t[gt]);
            assign vmax0[gt] = (vtop0[gt] <<< in_e) + ((P_t[gt] <<< in_e) - P_t[gt]);
            assign lock0[gt] = (vmin0[gt] >= thr0[gt]) | (vmax0[gt] < thr0[gt]);
            assign dec0[gt]  = (vmin0[gt] >= thr0[gt]);
        end
    endgenerate

    assign dbg_locked = locked;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_out
            assign out_dec[gt] = tauw[gt][63] ? dec_raw[gt] : ~dec_raw[gt];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            h_reg <= 12'd0; m_chk <= 5'd0;
            locked <= 10'd0; dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= 48'sd0; pmsk[t] <= 48'sd0; nmsk[t] <= 48'sd0;
                tauw[t] <= 64'd0;
            end
        end else if (in_sop) begin
            h_reg <= in_h;
            m_chk <= (in_e == 5'd0) ? 5'd0 : (in_e - 5'd1);
            for (t = 0; t < 10; t = t + 1) begin
                locked[t]  <= lock0[t];
                dec_raw[t] <= dec0[t];
                tauw[t] <= in_thr_row[t*64 +: 64];
                vtop[t] <= -lutw[t];
                pmsk[t] <= (P_t[t] <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1))) - P_t[t];
                nmsk[t] <= (N_t[t] <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1))) - N_t[t];
            end
        end else if (in_plane_valid) begin
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= nv[t];
                if (m_chk != 5'd0) begin
                    pmsk[t] <= (pmsk[t] - P_t[t])   >>> 1;
                    nmsk[t] <= (nmsk[t] - N_t[t])   >>> 1;
                end
                if (lock_n[t]) begin
                    locked[t] <= 1'b1;
                    dec_raw[t] <= (vmin_n[t] >= thrw[t]);
                end
            end
            if (m_chk != 5'd0)
                m_chk <= m_chk - 5'd1;
        end
    end

    // ---- SGLR（T39）：支撑感知 lane 退役 ----
    // need[k]=1 ⟺ 存在未锁定判决 t 使 A_q[t][k]≠0。sup_t 为部署常量，
    // 故 need 是 10 个 10bit 二选一 OR——纯组合逻辑，无 LUT/无元数据流。
    wire [9:0] sglr_need;
    assign sglr_need = 10'd0
        | (locked[0] ? 10'd0 : 10'd201)
        | (locked[1] ? 10'd0 : 10'd86)
        | (locked[2] ? 10'd0 : 10'd568)
        | (locked[3] ? 10'd0 : 10'd418)
        | (locked[4] ? 10'd0 : 10'd312)
        | (locked[5] ? 10'd0 : 10'd147)
        | (locked[6] ? 10'd0 : 10'd54)
        | (locked[7] ? 10'd0 : 10'd840)
        | (locked[8] ? 10'd0 : 10'd202)
        | (locked[9] ? 10'd0 : 10'd323);
    assign out_need = sglr_need;

endmodule
