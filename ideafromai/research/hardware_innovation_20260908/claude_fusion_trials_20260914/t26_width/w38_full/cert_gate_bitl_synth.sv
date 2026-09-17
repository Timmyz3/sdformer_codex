// T26 生成文件（勿手改）：位宽参数化门核，W=38 bit，变体 = full。
// 生成：t26_width/t26_gen.py。常数（P_t/N_t/dot/|A_q|）同 t25_k4_gate/full。
// 相对 T25（48b）仅改区间通路位宽：38b，thr 仍取 640b 端口低 38 位。
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
    reg signed [37:0] vtop  [0:9];
    reg signed [37:0] pmsk  [0:9];
    reg signed [37:0] nmsk  [0:9];
    reg [63:0] tauw   [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t;

    wire [9:0] wsel = in_sop ? in_sign : in_plane;

    wire signed [37:0] lutw [0:9];
    assign lutw[0] = (wsel[0] ? 38'sd787 : 38'sd0) + (wsel[1] ? 38'sd324 : 38'sd0) + (wsel[2] ? 38'sd312 : 38'sd0) + (wsel[3] ? 38'sd1069 : 38'sd0) + (wsel[4] ? 38'sd230 : 38'sd0) + (wsel[5] ? 38'sd208 : 38'sd0) + (wsel[6] ? -38'sd822 : 38'sd0) + (wsel[7] ? -38'sd851 : 38'sd0) + (wsel[8] ? 38'sd719 : 38'sd0) + (wsel[9] ? 38'sd686 : 38'sd0);
    assign lutw[1] = (wsel[0] ? -38'sd139 : 38'sd0) + (wsel[1] ? -38'sd1329 : 38'sd0) + (wsel[2] ? -38'sd1393 : 38'sd0) + (wsel[3] ? -38'sd607 : 38'sd0) + (wsel[4] ? -38'sd1446 : 38'sd0) + (wsel[5] ? -38'sd826 : 38'sd0) + (wsel[6] ? -38'sd1097 : 38'sd0) + (wsel[7] ? 38'sd141 : 38'sd0) + (wsel[8] ? -38'sd367 : 38'sd0) + (wsel[9] ? -38'sd716 : 38'sd0);
    assign lutw[2] = (wsel[0] ? -38'sd57 : 38'sd0) + (wsel[1] ? -38'sd170 : 38'sd0) + (wsel[2] ? -38'sd674 : 38'sd0) + (wsel[3] ? 38'sd990 : 38'sd0) + (wsel[4] ? 38'sd705 : 38'sd0) + (wsel[5] ? 38'sd759 : 38'sd0) + (wsel[6] ? -38'sd294 : 38'sd0) + (wsel[7] ? 38'sd208 : 38'sd0) + (wsel[8] ? 38'sd421 : 38'sd0) + (wsel[9] ? 38'sd900 : 38'sd0);
    assign lutw[3] = (wsel[0] ? -38'sd454 : 38'sd0) + (wsel[1] ? -38'sd1125 : 38'sd0) + (wsel[2] ? 38'sd619 : 38'sd0) + (wsel[3] ? -38'sd10 : 38'sd0) + (wsel[4] ? -38'sd396 : 38'sd0) + (wsel[5] ? -38'sd1004 : 38'sd0) + (wsel[6] ? -38'sd157 : 38'sd0) + (wsel[7] ? 38'sd728 : 38'sd0) + (wsel[8] ? -38'sd1120 : 38'sd0) + (wsel[9] ? 38'sd578 : 38'sd0);
    assign lutw[4] = (wsel[0] ? 38'sd318 : 38'sd0) + (wsel[1] ? -38'sd348 : 38'sd0) + (wsel[2] ? 38'sd396 : 38'sd0) + (wsel[3] ? -38'sd602 : 38'sd0) + (wsel[4] ? -38'sd1530 : 38'sd0) + (wsel[5] ? -38'sd970 : 38'sd0) + (wsel[6] ? -38'sd552 : 38'sd0) + (wsel[7] ? 38'sd110 : 38'sd0) + (wsel[8] ? -38'sd1012 : 38'sd0) + (wsel[9] ? 38'sd463 : 38'sd0);
    assign lutw[5] = (wsel[0] ? -38'sd1349 : 38'sd0) + (wsel[1] ? -38'sd1165 : 38'sd0) + (wsel[2] ? -38'sd845 : 38'sd0) + (wsel[3] ? 38'sd308 : 38'sd0) + (wsel[4] ? -38'sd1134 : 38'sd0) + (wsel[5] ? 38'sd50 : 38'sd0) + (wsel[6] ? 38'sd102 : 38'sd0) + (wsel[7] ? -38'sd1104 : 38'sd0) + (wsel[8] ? 38'sd818 : 38'sd0) + (wsel[9] ? -38'sd694 : 38'sd0);
    assign lutw[6] = (wsel[0] ? -38'sd1067 : 38'sd0) + (wsel[1] ? 38'sd1239 : 38'sd0) + (wsel[2] ? -38'sd1645 : 38'sd0) + (wsel[3] ? -38'sd261 : 38'sd0) + (wsel[4] ? 38'sd1640 : 38'sd0) + (wsel[5] ? 38'sd1346 : 38'sd0) + (wsel[6] ? 38'sd747 : 38'sd0) + (wsel[7] ? -38'sd100 : 38'sd0) + (wsel[8] ? 38'sd643 : 38'sd0) + (wsel[9] ? -38'sd1063 : 38'sd0);
    assign lutw[7] = (wsel[0] ? 38'sd188 : 38'sd0) + (wsel[1] ? 38'sd691 : 38'sd0) + (wsel[2] ? -38'sd746 : 38'sd0) + (wsel[3] ? 38'sd1351 : 38'sd0) + (wsel[4] ? 38'sd290 : 38'sd0) + (wsel[5] ? -38'sd274 : 38'sd0) + (wsel[6] ? 38'sd1539 : 38'sd0) + (wsel[7] ? 38'sd1157 : 38'sd0) + (wsel[8] ? -38'sd1335 : 38'sd0) + (wsel[9] ? -38'sd1358 : 38'sd0);
    assign lutw[8] = (wsel[0] ? -38'sd544 : 38'sd0) + (wsel[1] ? 38'sd1356 : 38'sd0) + (wsel[2] ? -38'sd975 : 38'sd0) + (wsel[3] ? -38'sd1346 : 38'sd0) + (wsel[4] ? -38'sd521 : 38'sd0) + (wsel[5] ? 38'sd48 : 38'sd0) + (wsel[6] ? 38'sd1092 : 38'sd0) + (wsel[7] ? -38'sd1708 : 38'sd0) + (wsel[8] ? -38'sd462 : 38'sd0) + (wsel[9] ? -38'sd92 : 38'sd0);
    assign lutw[9] = (wsel[0] ? 38'sd1300 : 38'sd0) + (wsel[1] ? -38'sd1295 : 38'sd0) + (wsel[2] ? 38'sd1085 : 38'sd0) + (wsel[3] ? -38'sd402 : 38'sd0) + (wsel[4] ? -38'sd1037 : 38'sd0) + (wsel[5] ? -38'sd884 : 38'sd0) + (wsel[6] ? -38'sd1236 : 38'sd0) + (wsel[7] ? 38'sd696 : 38'sd0) + (wsel[8] ? -38'sd1117 : 38'sd0) + (wsel[9] ? -38'sd248 : 38'sd0);

    wire signed [37:0] P_t [0:9];
    wire signed [37:0] N_t [0:9];
    assign P_t[0] = 38'sd4335;  assign N_t[0] = -38'sd1673;
    assign P_t[1] = 38'sd141;  assign N_t[1] = -38'sd7920;
    assign P_t[2] = 38'sd3983;  assign N_t[2] = -38'sd1195;
    assign P_t[3] = 38'sd1925;  assign N_t[3] = -38'sd4266;
    assign P_t[4] = 38'sd1287;  assign N_t[4] = -38'sd5014;
    assign P_t[5] = 38'sd1278;  assign N_t[5] = -38'sd6291;
    assign P_t[6] = 38'sd5615;  assign N_t[6] = -38'sd4136;
    assign P_t[7] = 38'sd5216;  assign N_t[7] = -38'sd3713;
    assign P_t[8] = 38'sd2496;  assign N_t[8] = -38'sd5648;
    assign P_t[9] = 38'sd3081;  assign N_t[9] = -38'sd6219;

    wire [63:0]        tau_sel [0:9];
    wire signed [37:0] thrw   [0:9];
    wire signed [37:0] nv     [0:9];
    wire signed [37:0] vmin_n [0:9];
    wire signed [37:0] vmax_n [0:9];
    wire               lock_n [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tau_sel[gt]    = tauw[gt];
            assign thrw[gt]       = $signed(tau_sel[gt][37:0]);
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
                vtop[t] <= 38'sd0; pmsk[t] <= 38'sd0; nmsk[t] <= 38'sd0;
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
