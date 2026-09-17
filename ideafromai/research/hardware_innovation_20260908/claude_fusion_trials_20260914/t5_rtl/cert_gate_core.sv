// T5（C1 卡）：证书终止门核——位平面 MSB-first + 逐判决区间锁定。自有代码。
//
// 每组 (p,h)：10 个判决单元 t=0..9 共享贡献向量 Y[0..9]（signed24/f14）。
// 协议：sop 拍送 in_h/in_sign/in_e（in_sign = bit s 为 Y[s]<0；FX 模式 in_e=24，
// 此时 in_sign 与首平面（bit23）相同，signed24 的 bit23 即符号位）；
// 之后每拍 1 个数据平面（in_plane bit s = Y[s] 当前位，从 bit e-1 往下）。
//
// 状态递推（两补码恒等式 Y>>(p-1) = 2*(Y>>p) + b_{p-1}，且 Y>>e = -sign(Y)）：
//   sop：Vtop[t] = -dot10(t, in_sign) = Σ A[t][s]·(Y[s]>>e)
//   平面沿：Vtop'[t] = 2·Vtop[t] + dot10(t, in_plane)
// 平面沿后剩余 m=m_chk 位，真值 V ∈ [Vtop'<<m + N_t·(2^m-1), Vtop'<<m + P_t·(2^m-1)]
// （区间随 m 减小单调嵌套，锁定后判决冻结且与全深度一致）。
// 锁定 = (Vmin >= thr) | (Vmax < thr)；判决 = (Vmin >= thr)。
// thr/方向折入 tau 存储：bit63=Dflag（1=同向原判），[47:0]=thr。
module cert_gate_core (
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

    reg [11:0] h_reg;
    reg [4:0]  m_chk;
    reg signed [47:0] vtop [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t;

    function signed [47:0] dot10;
        input integer tt;
        input [9:0] w;
        integer ss;
        reg signed [47:0] acc;
        begin
            acc = 48'sd0;
            for (ss = 0; ss < 10; ss = ss + 1)
                if (w[ss])
                    acc = acc + $signed({{32{a_mem[tt*10+ss][15]}}, a_mem[tt*10+ss]});
            dot10 = acc;
        end
    endfunction

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

    // 平面沿组合逻辑：更新后 Vtop（nv）与 m=m_chk 的区间判定
    wire [63:0]        tauw   [0:9];
    wire signed [47:0] thrw   [0:9];
    wire signed [47:0] nv     [0:9];
    wire signed [47:0] vmin_n [0:9];
    wire signed [47:0] vmax_n [0:9];
    wire               lock_n [0:9];
    wire signed [47:0] msk = (48'sd1 << m_chk) - 48'sd1;
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tauw[gt]    = tau_sel(gt);
            assign thrw[gt]    = $signed(tauw[gt][47:0]);
            assign nv[gt]     = (vtop[gt] << 1) + dot10(gt, in_plane);
            assign vmin_n[gt] = (nv[gt] << m_chk) + pn_mem[gt*2+1] * msk;  // N_t
            assign vmax_n[gt] = (nv[gt] << m_chk) + pn_mem[gt*2]   * msk;  // P_t
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
            for (t = 0; t < 10; t = t + 1) vtop[t] <= 48'sd0;
        end else if (in_sop) begin
            h_reg <= in_h;
            m_chk <= (in_e == 5'd0) ? 5'd0 : (in_e - 5'd1);
            locked <= 10'd0;
            dec_raw <= 10'd0;
            for (t = 0; t < 10; t = t + 1)
                vtop[t] <= -dot10(t, in_sign);
        end else if (in_plane_valid) begin
            for (t = 0; t < 10; t = t + 1) begin
                vtop[t] <= nv[t];
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
    end
endmodule
