// T10（L1，照抄对象 BitL MICRO25）：证书终止门核——BitL 式 LUT 数据通路优化版。自有代码。
//
// 与 T5 cert_gate_core.sv 的接口/协议/数学完全一致（零差目标），数据通路两处优化：
// 1. **dot10 查表化**（BitL 横/纵 bit 级查表思路）：dot10(t,w) = Σ_{s:w[s]=1} A[t][s]
//    是 10-bit 模式的子集和 → 分解为两个 5-bit 半模式查表：
//    dot10 = lut_lo[t][w[4:0]] + lut_hi[t][w[9:5]]，
//    LUT 2×10×32×20bit ≈ 12.8kb，initial 块从 a_mem 离线算出。
//    原 10 输入选择-加法树（深度~4、16bit 符号扩展）→ 2 次 LUTRAM 读 + 1 个 20bit 加法。
// 2. **pn·msk 乘法消除**：P_t·(2^m−1) 随 m 逐平面递减：msk' = msk>>1，
//    pmsk' = (pmsk − P)>>>1（pmsk−P 恒为偶，算术移位精确），nmsk 对称。
//    每平面 10 单元 ×2 个 48bit 乘法 → 减法+移位；sop 拍一次性变移位初始化。
module cert_gate_bitl (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_sop,
    input  wire [11:0] in_h,
    input  wire [9:0]  in_sign,
    input  wire [4:0]  in_e,
    input  wire        in_plane_valid,
    input  wire [9:0]  in_plane,
    output wire [9:0]  out_dec,
    output wire [9:0]  dbg_locked
);

    reg signed [15:0] a_mem  [0:99];    // A_q[t][s] f12，t*10+s
    reg signed [47:0] pn_mem [0:19];    // {P_t, N_t} 交替，t*2 / t*2+1
    reg        [63:0] tau0   [0:3071];
    reg        [63:0] tau1   [0:3071];
    reg        [63:0] tau2   [0:3071];
    reg        [63:0] tau3   [0:3071];
    reg        [63:0] tau4   [0:3071];
    reg        [63:0] tau5   [0:3071];
    reg        [63:0] tau6   [0:3071];
    reg        [63:0] tau7   [0:3071];
    reg        [63:0] tau8   [0:3071];
    reg        [63:0] tau9   [0:3071];
    reg signed [19:0] lut_lo [0:319];   // t*32+code：lanes 0..4 子集和（离线 init）
    reg signed [19:0] lut_hi [0:319];   // t*32+code：lanes 5..9 子集和

    reg [11:0] h_reg;
    reg [4:0]  m_chk;
    reg signed [47:0] vtop  [0:9];
    reg signed [47:0] pmsk  [0:9];      // P_t·(2^m_chk −1)，逐平面递推
    reg signed [47:0] nmsk  [0:9];      // N_t·(2^m_chk −1)
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t, tt, cc, ss;

    function [63:0] tau_sel;
        input integer tt;
        case (tt)
            0: tau_sel = tau0[h_reg];
            1: tau_sel = tau1[h_reg];
            2: tau_sel = tau2[h_reg];
            3: tau_sel = tau3[h_reg];
            4: tau_sel = tau4[h_reg];
            5: tau_sel = tau5[h_reg];
            6: tau_sel = tau6[h_reg];
            7: tau_sel = tau7[h_reg];
            8: tau_sel = tau8[h_reg];
            9: tau_sel = tau9[h_reg];
        endcase
    endfunction

    function signed [47:0] lut10;
        input integer tt;
        input [9:0] w;
        begin
            lut10 = $signed({{28{lut_lo[tt*32+w[4:0]][19]}}, lut_lo[tt*32+w[4:0]]})
                  + $signed({{28{lut_hi[tt*32+w[9:5]][19]}}, lut_hi[tt*32+w[9:5]]});
        end
    endfunction

    // 平面沿组合逻辑：更新后 Vtop（nv）与 m=m_chk 的区间判定
    wire [63:0]        tauw   [0:9];
    wire signed [47:0] thrw   [0:9];
    wire signed [47:0] nv     [0:9];
    wire signed [47:0] vmin_n [0:9];
    wire signed [47:0] vmax_n [0:9];
    wire               lock_n [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tauw[gt]    = tau_sel(gt);
            assign thrw[gt]    = $signed(tauw[gt][47:0]);
            assign nv[gt]     = (vtop[gt] << 1) + lut10(gt, in_plane);
            assign vmin_n[gt] = (nv[gt] << m_chk) + nmsk[gt];
            assign vmax_n[gt] = (nv[gt] << m_chk) + pmsk[gt];
            assign lock_n[gt] = ~locked[gt]
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
            end
        end else if (in_sop) begin
            h_reg <= in_h;
            m_chk <= (in_e == 5'd0) ? 5'd0 : (in_e - 5'd1);
            locked <= 10'd0;
            dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= -lut10(t, in_sign);
                // pmsk 初始化 = P·(2^m−1) = (P<<<m) − P（m=0 时为 0）
                pmsk[t] <= (pn_mem[t*2]   <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1)))
                          - pn_mem[t*2];
                nmsk[t] <= (pn_mem[t*2+1] <<< ((in_e == 5'd0) ? 5'd0 : (in_e - 5'd1)))
                          - pn_mem[t*2+1];
            end
        end else if (in_plane_valid) begin
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= nv[t];
                if (m_chk != 5'd0) begin
                    // m→m−1：msk' = msk>>1，P·msk' = (P·msk − P)>>>1（差恒为偶）
                    pmsk[t] <= (pmsk[t] - pn_mem[t*2])   >>> 1;
                    nmsk[t] <= (nmsk[t] - pn_mem[t*2+1]) >>> 1;
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

    reg [8*512-1:0] fa, fp, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9;
    initial begin
        if (!$value$plusargs("a=%s", fa))  begin $display("missing +a=");  $finish; end
        if (!$value$plusargs("pn=%s", fp)) begin $display("missing +pn="); $finish; end
        $readmemh(fa, a_mem);
        $readmemh(fp, pn_mem);
        if ($value$plusargs("tau_t0=%s", f0)) $readmemh(f0, tau0);
        if ($value$plusargs("tau_t1=%s", f1)) $readmemh(f1, tau1);
        if ($value$plusargs("tau_t2=%s", f2)) $readmemh(f2, tau2);
        if ($value$plusargs("tau_t3=%s", f3)) $readmemh(f3, tau3);
        if ($value$plusargs("tau_t4=%s", f4)) $readmemh(f4, tau4);
        if ($value$plusargs("tau_t5=%s", f5)) $readmemh(f5, tau5);
        if ($value$plusargs("tau_t6=%s", f6)) $readmemh(f6, tau6);
        if ($value$plusargs("tau_t7=%s", f7)) $readmemh(f7, tau7);
        if ($value$plusargs("tau_t8=%s", f8)) $readmemh(f8, tau8);
        if ($value$plusargs("tau_t9=%s", f9)) $readmemh(f9, tau9);
        // LUT 离线构建：每 t、每 5-bit code 的子集和（Σ_5|A| < 2^18，20bit 足够）
        for (tt = 0; tt < 10; tt = tt + 1)
            for (cc = 0; cc < 32; cc = cc + 1) begin
                lut_lo[tt*32+cc] = 20'sd0;
                lut_hi[tt*32+cc] = 20'sd0;
                for (ss = 0; ss < 5; ss = ss + 1) begin
                    if (cc[ss])
                        lut_lo[tt*32+cc] = lut_lo[tt*32+cc] + a_mem[tt*10+ss];
                    if (cc[ss])
                        lut_hi[tt*32+cc] = lut_hi[tt*32+cc] + a_mem[tt*10+5+ss];
                end
            end
    end
endmodule
