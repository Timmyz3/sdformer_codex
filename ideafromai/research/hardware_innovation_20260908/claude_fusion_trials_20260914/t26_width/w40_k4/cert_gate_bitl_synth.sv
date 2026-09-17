// T26 生成文件（勿手改）：位宽参数化门核，W=40 bit，变体 = k4。
// 生成：t26_width/t26_gen.py。常数（P_t/N_t/dot/|A_q|）同 t25_k4_gate/k4。
// 相对 T25（48b）仅改区间通路位宽：40b，thr 仍取 640b 端口低 40 位。
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
    reg signed [39:0] vtop  [0:9];
    reg signed [39:0] pmsk  [0:9];
    reg signed [39:0] nmsk  [0:9];
    reg [63:0] tauw   [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t;

    wire [9:0] wsel = in_sop ? in_sign : in_plane;

    wire signed [39:0] lutw [0:9];
    assign lutw[0] = (wsel[0] ? 40'sd796 : 40'sd0) + (wsel[3] ? 40'sd1077 : 40'sd0) + (wsel[6] ? -40'sd831 : 40'sd0) + (wsel[7] ? -40'sd840 : 40'sd0);
    assign lutw[1] = (wsel[1] ? -40'sd1312 : 40'sd0) + (wsel[2] ? -40'sd1414 : 40'sd0) + (wsel[4] ? -40'sd1431 : 40'sd0) + (wsel[6] ? -40'sd1079 : 40'sd0);
    assign lutw[2] = (wsel[3] ? 40'sd995 : 40'sd0) + (wsel[4] ? 40'sd715 : 40'sd0) + (wsel[5] ? 40'sd768 : 40'sd0) + (wsel[9] ? 40'sd908 : 40'sd0);
    assign lutw[3] = (wsel[1] ? -40'sd1141 : 40'sd0) + (wsel[5] ? -40'sd1018 : 40'sd0) + (wsel[7] ? 40'sd745 : 40'sd0) + (wsel[8] ? -40'sd1133 : 40'sd0);
    assign lutw[4] = (wsel[3] ? -40'sd596 : 40'sd0) + (wsel[4] ? -40'sd1574 : 40'sd0) + (wsel[5] ? -40'sd1008 : 40'sd0) + (wsel[8] ? -40'sd1048 : 40'sd0);
    assign lutw[5] = (wsel[0] ? -40'sd1359 : 40'sd0) + (wsel[1] ? -40'sd1146 : 40'sd0) + (wsel[4] ? -40'sd1121 : 40'sd0) + (wsel[7] ? -40'sd1116 : 40'sd0);
    assign lutw[6] = (wsel[1] ? 40'sd1260 : 40'sd0) + (wsel[2] ? -40'sd1669 : 40'sd0) + (wsel[4] ? 40'sd1656 : 40'sd0) + (wsel[5] ? 40'sd1358 : 40'sd0);
    assign lutw[7] = (wsel[3] ? 40'sd1361 : 40'sd0) + (wsel[6] ? 40'sd1545 : 40'sd0) + (wsel[8] ? -40'sd1345 : 40'sd0) + (wsel[9] ? -40'sd1366 : 40'sd0);
    assign lutw[8] = (wsel[1] ? 40'sd1354 : 40'sd0) + (wsel[3] ? -40'sd1350 : 40'sd0) + (wsel[6] ? 40'sd1093 : 40'sd0) + (wsel[7] ? -40'sd1712 : 40'sd0);
    assign lutw[9] = (wsel[0] ? 40'sd1320 : 40'sd0) + (wsel[1] ? -40'sd1330 : 40'sd0) + (wsel[6] ? -40'sd1262 : 40'sd0) + (wsel[8] ? -40'sd1138 : 40'sd0);

    wire signed [39:0] P_t [0:9];
    wire signed [39:0] N_t [0:9];
    assign P_t[0] = 40'sd1873;  assign N_t[0] = -40'sd1671;
    assign P_t[1] = 40'sd0;  assign N_t[1] = -40'sd5236;
    assign P_t[2] = 40'sd3386;  assign N_t[2] = 40'sd0;
    assign P_t[3] = 40'sd745;  assign N_t[3] = -40'sd3292;
    assign P_t[4] = 40'sd0;  assign N_t[4] = -40'sd4226;
    assign P_t[5] = 40'sd0;  assign N_t[5] = -40'sd4742;
    assign P_t[6] = 40'sd4274;  assign N_t[6] = -40'sd1669;
    assign P_t[7] = 40'sd2906;  assign N_t[7] = -40'sd2711;
    assign P_t[8] = 40'sd2447;  assign N_t[8] = -40'sd3062;
    assign P_t[9] = 40'sd1320;  assign N_t[9] = -40'sd3730;

    wire [63:0]        tau_sel [0:9];
    wire signed [39:0] thrw   [0:9];
    wire signed [39:0] nv     [0:9];
    wire signed [39:0] vmin_n [0:9];
    wire signed [39:0] vmax_n [0:9];
    wire               lock_n [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tau_sel[gt]    = tauw[gt];
            assign thrw[gt]       = $signed(tau_sel[gt][39:0]);
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
                vtop[t] <= 40'sd0; pmsk[t] <= 40'sd0; nmsk[t] <= 40'sd0;
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
