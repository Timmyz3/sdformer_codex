// T25 生成文件（勿手改）：掩码加法树门核，变体 = full。
// dot10(t,w) = Σ_{c∈supp(t)} aq[t][c]·w[c]（10 项选位常数和，烘焙字面量）。
// 端口/时序/判决逻辑与 t15_synth/cert_gate_bitl_synth.sv 逐位一致；
// 差异仅在：无 a/pn/lut ROM、无 $readmemh，P_t/N_t/dot 常数全部为 RTL 字面量。
// 生成：t25_k4_gate/t25_gen.py（A 来源：bn_state/trace_s0_stage0.npz: A（稠密基线））
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
    output wire [9:0]  dbg_locked
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
    assign lutw[0] = (wsel[0] ? 48'sd787 : 48'sd0) + (wsel[1] ? 48'sd324 : 48'sd0) + (wsel[2] ? 48'sd312 : 48'sd0) + (wsel[3] ? 48'sd1069 : 48'sd0) + (wsel[4] ? 48'sd230 : 48'sd0) + (wsel[5] ? 48'sd208 : 48'sd0) + (wsel[6] ? -48'sd822 : 48'sd0) + (wsel[7] ? -48'sd851 : 48'sd0) + (wsel[8] ? 48'sd719 : 48'sd0) + (wsel[9] ? 48'sd686 : 48'sd0);
    assign lutw[1] = (wsel[0] ? -48'sd139 : 48'sd0) + (wsel[1] ? -48'sd1329 : 48'sd0) + (wsel[2] ? -48'sd1393 : 48'sd0) + (wsel[3] ? -48'sd607 : 48'sd0) + (wsel[4] ? -48'sd1446 : 48'sd0) + (wsel[5] ? -48'sd826 : 48'sd0) + (wsel[6] ? -48'sd1097 : 48'sd0) + (wsel[7] ? 48'sd141 : 48'sd0) + (wsel[8] ? -48'sd367 : 48'sd0) + (wsel[9] ? -48'sd716 : 48'sd0);
    assign lutw[2] = (wsel[0] ? -48'sd57 : 48'sd0) + (wsel[1] ? -48'sd170 : 48'sd0) + (wsel[2] ? -48'sd674 : 48'sd0) + (wsel[3] ? 48'sd990 : 48'sd0) + (wsel[4] ? 48'sd705 : 48'sd0) + (wsel[5] ? 48'sd759 : 48'sd0) + (wsel[6] ? -48'sd294 : 48'sd0) + (wsel[7] ? 48'sd208 : 48'sd0) + (wsel[8] ? 48'sd421 : 48'sd0) + (wsel[9] ? 48'sd900 : 48'sd0);
    assign lutw[3] = (wsel[0] ? -48'sd454 : 48'sd0) + (wsel[1] ? -48'sd1125 : 48'sd0) + (wsel[2] ? 48'sd619 : 48'sd0) + (wsel[3] ? -48'sd10 : 48'sd0) + (wsel[4] ? -48'sd396 : 48'sd0) + (wsel[5] ? -48'sd1004 : 48'sd0) + (wsel[6] ? -48'sd157 : 48'sd0) + (wsel[7] ? 48'sd728 : 48'sd0) + (wsel[8] ? -48'sd1120 : 48'sd0) + (wsel[9] ? 48'sd578 : 48'sd0);
    assign lutw[4] = (wsel[0] ? 48'sd318 : 48'sd0) + (wsel[1] ? -48'sd348 : 48'sd0) + (wsel[2] ? 48'sd396 : 48'sd0) + (wsel[3] ? -48'sd602 : 48'sd0) + (wsel[4] ? -48'sd1530 : 48'sd0) + (wsel[5] ? -48'sd970 : 48'sd0) + (wsel[6] ? -48'sd552 : 48'sd0) + (wsel[7] ? 48'sd110 : 48'sd0) + (wsel[8] ? -48'sd1012 : 48'sd0) + (wsel[9] ? 48'sd463 : 48'sd0);
    assign lutw[5] = (wsel[0] ? -48'sd1349 : 48'sd0) + (wsel[1] ? -48'sd1165 : 48'sd0) + (wsel[2] ? -48'sd845 : 48'sd0) + (wsel[3] ? 48'sd308 : 48'sd0) + (wsel[4] ? -48'sd1134 : 48'sd0) + (wsel[5] ? 48'sd50 : 48'sd0) + (wsel[6] ? 48'sd102 : 48'sd0) + (wsel[7] ? -48'sd1104 : 48'sd0) + (wsel[8] ? 48'sd818 : 48'sd0) + (wsel[9] ? -48'sd694 : 48'sd0);
    assign lutw[6] = (wsel[0] ? -48'sd1067 : 48'sd0) + (wsel[1] ? 48'sd1239 : 48'sd0) + (wsel[2] ? -48'sd1645 : 48'sd0) + (wsel[3] ? -48'sd261 : 48'sd0) + (wsel[4] ? 48'sd1640 : 48'sd0) + (wsel[5] ? 48'sd1346 : 48'sd0) + (wsel[6] ? 48'sd747 : 48'sd0) + (wsel[7] ? -48'sd100 : 48'sd0) + (wsel[8] ? 48'sd643 : 48'sd0) + (wsel[9] ? -48'sd1063 : 48'sd0);
    assign lutw[7] = (wsel[0] ? 48'sd188 : 48'sd0) + (wsel[1] ? 48'sd691 : 48'sd0) + (wsel[2] ? -48'sd746 : 48'sd0) + (wsel[3] ? 48'sd1351 : 48'sd0) + (wsel[4] ? 48'sd290 : 48'sd0) + (wsel[5] ? -48'sd274 : 48'sd0) + (wsel[6] ? 48'sd1539 : 48'sd0) + (wsel[7] ? 48'sd1157 : 48'sd0) + (wsel[8] ? -48'sd1335 : 48'sd0) + (wsel[9] ? -48'sd1358 : 48'sd0);
    assign lutw[8] = (wsel[0] ? -48'sd544 : 48'sd0) + (wsel[1] ? 48'sd1356 : 48'sd0) + (wsel[2] ? -48'sd975 : 48'sd0) + (wsel[3] ? -48'sd1346 : 48'sd0) + (wsel[4] ? -48'sd521 : 48'sd0) + (wsel[5] ? 48'sd48 : 48'sd0) + (wsel[6] ? 48'sd1092 : 48'sd0) + (wsel[7] ? -48'sd1708 : 48'sd0) + (wsel[8] ? -48'sd462 : 48'sd0) + (wsel[9] ? -48'sd92 : 48'sd0);
    assign lutw[9] = (wsel[0] ? 48'sd1300 : 48'sd0) + (wsel[1] ? -48'sd1295 : 48'sd0) + (wsel[2] ? 48'sd1085 : 48'sd0) + (wsel[3] ? -48'sd402 : 48'sd0) + (wsel[4] ? -48'sd1037 : 48'sd0) + (wsel[5] ? -48'sd884 : 48'sd0) + (wsel[6] ? -48'sd1236 : 48'sd0) + (wsel[7] ? 48'sd696 : 48'sd0) + (wsel[8] ? -48'sd1117 : 48'sd0) + (wsel[9] ? -48'sd248 : 48'sd0);

    wire signed [47:0] P_t [0:9];
    wire signed [47:0] N_t [0:9];
    assign P_t[0] = 48'sd4335;  assign N_t[0] = -48'sd1673;
    assign P_t[1] = 48'sd141;  assign N_t[1] = -48'sd7920;
    assign P_t[2] = 48'sd3983;  assign N_t[2] = -48'sd1195;
    assign P_t[3] = 48'sd1925;  assign N_t[3] = -48'sd4266;
    assign P_t[4] = 48'sd1287;  assign N_t[4] = -48'sd5014;
    assign P_t[5] = 48'sd1278;  assign N_t[5] = -48'sd6291;
    assign P_t[6] = 48'sd5615;  assign N_t[6] = -48'sd4136;
    assign P_t[7] = 48'sd5216;  assign N_t[7] = -48'sd3713;
    assign P_t[8] = 48'sd2496;  assign N_t[8] = -48'sd5648;
    assign P_t[9] = 48'sd3081;  assign N_t[9] = -48'sd6219;

    wire [63:0]        tau_sel [0:9];
    wire signed [47:0] thrw   [0:9];
    wire signed [47:0] nv     [0:9];
    wire signed [47:0] vmin_n [0:9];
    wire signed [47:0] vmax_n [0:9];
    wire               lock_n [0:9];
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
            locked <= 10'd0;
            dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
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
endmodule
