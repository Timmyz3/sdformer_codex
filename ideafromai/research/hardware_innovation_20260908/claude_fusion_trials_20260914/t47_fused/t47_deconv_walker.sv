// T47：stride-2 转置卷积「行走器」两版对照（K=3, S=2, P=1, OP=1，输出平稳、单 MAC 时分复用）。
//
// 为什么做这个：T46-2 已在真实权重+真实脉冲上证明零插值口径与相位分解**逐元素 0 误差**，
// 且非零乘法次数完全相同（都是 active×9×Cout）。所以两者的一切硬件差异**全在控制路径**：
//   - t47_deconv_z：稠密零插值行走器。对每个输出位置枚举 3×3=9 个 (kh,kw) 槽位，
//     用「(2*oh-1+kh) 必须为非负偶数」判有效 ⇒ 每 4 个槽位只用 1 个。
//   - t47_deconv_p：相位分解行走器。输出按 (a,b,j,l) 相位优先遍历，抽头表静态
//     (a=0→1 抽头, a=1→2 抽头)，直接算 ih=j+dih ⇒ 无奇偶判定、无空槽。
// 两者的 MAC 完全相同，且因为操作数是二值脉冲，乘法退化成 select ⇒ 期望 0 DSP。
//
// 用法：verilator --cc --exe --Mdir obj --top-module t47_walk_top t47_deconv_walker.sv t47_tb.cpp
module t47_deconv_z #(
    parameter integer H = 4, W = 5, CIN = 4,
    parameter integer ACCW = 24
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output reg                           done,
    output wire [$clog2(CIN*9)-1:0]      waddr,
    input  wire signed [7:0]             wdata,
    output wire [$clog2(CIN*H*W)-1:0]    xaddr,
    input  wire                          xbit,
    output reg  [$clog2(4*H*W)-1:0]      yaddr,
    output reg  signed [ACCW-1:0]        ydata,
    output reg                           ywe,
    output reg  [31:0]                   nslot
);
    localparam integer HO = 2 * H, WO = 2 * W;

    reg  [15:0] oh, ow, kh, kw, ci;
    reg  [15:0] run;
    reg  signed [ACCW-1:0] acc;

    // 零插值口径的逆映射：输出位置 oh 的第 kh 个抽头来自 ih = (oh + P - kh) / S，
    // 仅当 (oh + P - kh) 非负、能被 S 整除、且 ih < H 时该槽位有效；S=2 ⇒ 整除 ⟺ 最低位为 0。
    // 枚举的 9 个槽位里有 3/4 过不了整除判定 —— 这就是要量的浪费。
    wire signed [16:0] nhs = $signed({1'b0, oh}) + 17'sd1 - $signed({1'b0, kh});
    wire signed [16:0] nws = $signed({1'b0, ow}) + 17'sd1 - $signed({1'b0, kw});
    wire        th_ok = (nhs >= 0) && (nhs[0] == 1'b0) && ((nhs >> 1) < H);
    wire        tw_ok = (nws >= 0) && (nws[0] == 1'b0) && ((nws >> 1) < W);
    wire        tap_ok = th_ok && tw_ok;
    wire [15:0] ih = nhs[15:0] >> 1;
    wire [15:0] iw = nws[15:0] >> 1;

    assign waddr = ci * 9 + kh * 3 + kw;
    assign xaddr = ci * (H * W) + (tap_ok ? ih : 16'd0) * W + (tap_ok ? iw : 16'd0);

    wire signed [ACCW-1:0] prod =
        (xbit && tap_ok) ? {{(ACCW - 8) {wdata[7]}}, wdata} : {ACCW{1'b0}};

    wire last_slot = (kh == 16'd2) && (kw == 16'd2) && (ci == CIN - 1);

    always @(posedge clk) begin
        if (!rst_n) begin
            oh <= 0; ow <= 0; kh <= 0; kw <= 0; ci <= 0;
            run <= 0; done <= 0; ywe <= 0; nslot <= 0; acc <= 0; ydata <= 0;
        end else begin
            ywe <= 1'b0;
            if (start) begin
                oh <= 0; ow <= 0; kh <= 0; kw <= 0; ci <= 0;
                run <= 1; done <= 0; nslot <= 0; acc <= 0;
            end else if (run) begin
                nslot <= nslot + 1;
                if (last_slot) begin
                    ydata <= acc + prod;
                    yaddr <= oh * WO + ow;
                    ywe   <= 1'b1;
                    acc   <= 0;
                    kh <= 0; kw <= 0; ci <= 0;
                    if (ow == WO - 1) begin
                        ow <= 0;
                        if (oh == HO - 1) begin
                            oh <= 0; run <= 0; done <= 1;
                        end else begin
                            oh <= oh + 1;
                        end
                    end else begin
                        ow <= ow + 1;
                    end
                end else begin
                    acc <= acc + prod;
                    if (ci + 1 < CIN) begin
                        ci <= ci + 1;
                    end else begin
                        ci <= 0;
                        if (kw + 1 < 3) begin
                            kw <= kw + 1;
                        end else begin
                            kw <= 0;
                            kh <= kh + 1;
                        end
                    end
                end
            end
        end
    end
endmodule


module t47_deconv_p #(
    parameter integer H = 4, W = 5, CIN = 4,
    parameter integer ACCW = 24
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output reg                           done,
    output wire [$clog2(CIN*9)-1:0]      waddr,
    input  wire signed [7:0]             wdata,
    output wire [$clog2(CIN*H*W)-1:0]    xaddr,
    input  wire                          xbit,
    output reg  [$clog2(4*H*W)-1:0]      yaddr,
    output reg  signed [ACCW-1:0]        ydata,
    output reg                           ywe,
    output reg  [31:0]                   nslot
);
    reg  [15:0] pa, pb, jj, ll, ti, tj, ci;
    reg  [15:0] run;
    reg  signed [ACCW-1:0] acc;

    // 静态抽头表：a=0 → 1 抽头 (kh=1, dih=0)；a=1 → 2 抽头 (kh=0,dih=+1) / (kh=2,dih=0)
    wire [15:0] dih = pa ? (ti == 0 ? 16'd1 : 16'd0) : 16'd0;
    wire [15:0] khi = pa ? (ti == 0 ? 16'd0 : 16'd2) : 16'd1;
    wire [15:0] diw = pb ? (tj == 0 ? 16'd1 : 16'd0) : 16'd0;
    wire [15:0] kwi = pb ? (tj == 0 ? 16'd0 : 16'd2) : 16'd1;
    wire [15:0] nth = pa ? 16'd2 : 16'd1;
    wire [15:0] ntw = pb ? 16'd2 : 16'd1;

    // 直接偏移，无奇偶判定；只需越界判定
    wire [16:0] ih = {1'b0, jj} + {1'b0, dih};
    wire [16:0] iw = {1'b0, ll} + {1'b0, diw};
    wire        tap_ok = (ih < H) && (iw < W);

    assign waddr = ci * 9 + khi * 3 + kwi;
    assign xaddr = ci * (H * W) + (tap_ok ? ih[15:0] : 16'd0) * W + (tap_ok ? iw[15:0] : 16'd0);

    wire signed [ACCW-1:0] prod =
        (xbit && tap_ok) ? {{(ACCW - 8) {wdata[7]}}, wdata} : {ACCW{1'b0}};

    wire last_slot = (ti == (nth - 1)) && (tj == (ntw - 1)) && (ci == CIN - 1);

    always @(posedge clk) begin
        if (!rst_n) begin
            pa <= 0; pb <= 0; jj <= 0; ll <= 0; ti <= 0; tj <= 0; ci <= 0;
            run <= 0; done <= 0; ywe <= 0; nslot <= 0; acc <= 0; ydata <= 0;
        end else begin
            ywe <= 1'b0;
            if (start) begin
                pa <= 0; pb <= 0; jj <= 0; ll <= 0; ti <= 0; tj <= 0; ci <= 0;
                run <= 1; done <= 0; nslot <= 0; acc <= 0;
            end else if (run) begin
                nslot <= nslot + 1;
                if (last_slot) begin
                    ydata <= acc + prod;
                    yaddr <= (jj * 2 + pa) * (W * 2) + (ll * 2 + pb);
                    ywe   <= 1'b1;
                    acc   <= 0;
                    ti <= 0; tj <= 0; ci <= 0;
                    if (ll + 1 < W) begin
                        ll <= ll + 1;
                    end else begin
                        ll <= 0;
                        if (jj + 1 < H) begin
                            jj <= jj + 1;
                        end else begin
                            jj <= 0;
                            if (pb == 0) begin
                                pb <= 1;
                            end else begin
                                pb <= 0;
                                if (pa == 0) begin
                                    pa <= 1;
                                end else begin
                                    pa <= 0; run <= 0; done <= 1;
                                end
                            end
                        end
                    end
                end else begin
                    acc <= acc + prod;
                    if (ci + 1 < CIN) begin
                        ci <= ci + 1;
                    end else begin
                        ci <= 0;
                        if (tj + 1 < ntw) begin
                            tj <= tj + 1;
                        end else begin
                            tj <= 0;
                            ti <= ti + 1;
                        end
                    end
                end
            end
        end
    end
endmodule


// 仿真用的一体化顶层：两个 DUT 并列，便于同一份激励逐周期对照。
module t47_walk_top #(
    parameter integer H = 4, W = 5, CIN = 4,
    parameter integer ACCW = 24
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start_z,
    input  wire                       start_p,
    output wire [$clog2(CIN*9)-1:0]   zw_addr,
    input  wire signed [7:0]          zw_data,
    output wire [$clog2(CIN*H*W)-1:0] zx_addr,
    input  wire                       zx_bit,
    output wire [$clog2(4*H*W)-1:0]   zy_addr,
    output wire signed [ACCW-1:0]     zy_data,
    output wire                       zy_we,
    output wire [31:0]                z_nslot,
    output wire                       z_done,
    output wire [$clog2(CIN*9)-1:0]   pw_addr,
    input  wire signed [7:0]          pw_data,
    output wire [$clog2(CIN*H*W)-1:0] px_addr,
    input  wire                       px_bit,
    output wire [$clog2(4*H*W)-1:0]   py_addr,
    output wire signed [ACCW-1:0]     py_data,
    output wire                       py_we,
    output wire [31:0]                p_nslot,
    output wire                       p_done
);
    t47_deconv_z #(.H(H), .W(W), .CIN(CIN), .ACCW(ACCW)) u_z (
        .clk(clk), .rst_n(rst_n), .start(start_z), .done(z_done),
        .waddr(zw_addr), .wdata(zw_data), .xaddr(zx_addr), .xbit(zx_bit),
        .yaddr(zy_addr), .ydata(zy_data), .ywe(zy_we), .nslot(z_nslot));

    t47_deconv_p #(.H(H), .W(W), .CIN(CIN), .ACCW(ACCW)) u_p (
        .clk(clk), .rst_n(rst_n), .start(start_p), .done(p_done),
        .waddr(pw_addr), .wdata(pw_data), .xaddr(px_addr), .xbit(px_bit),
        .yaddr(py_addr), .ydata(py_data), .ywe(py_we), .nslot(p_nslot));
endmodule
