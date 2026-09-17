#!/usr/bin/env python3
"""T41：生成「融合后送判决」基线 RTL —— 生产者做完 A·Y + 门，线上只留 10 个判决位。

## 这条基线为什么是主对照（用户 2026-09-16 定稿口径）

C1 的传输合法性依赖显式的模块化约束 B（层间合同固定为多比特 Y）。若允许把 A（10×10）
折进生产者侧，则生产者已精确知道全部判决，线上只需 T=10 个判决位 = **10.0000 bits/组**。
本模块就是那一侧的 RTL：**它同时替代 plane_ser 生产者与门核**（C1 侧两者都要）。

- 输入：`in_yv` = 10 lane × 24b signed（生产者 lane 契约 signed24/f14，e≤23）
- 权重：A_q 的 k=4 掩码（与 T25 k4 同源 `results/t21b_k4_gate_params.npz: L%d_A`），
        烘焙为字面量，每行 4 项
- 判决：`dec_t = (V_t ≥ thr_t)`，与证书递归的终态判决**恒等**
- 输出：10 位，寄存器一级

A 是**逐层**的（10×10 门矩阵每层不同），故本脚本对 T40 用的四层各生成一份 RTL，
模块名统一为 `t41_fused_gate`（Verilator 各自独立编译，互不冲突）。综合只做 L8
（与 T25 k4 门核同层同源）。

用法：python t41_fused/t41_gen.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 't41_fused'
NPZ = ROOT / 'results' / 't21b_k4_gate_params.npz'
LIDS = [8, 14, 20, 28]          # T19/T40 口径的四层（stage0..3 各一层）


def lit(v):
    return ('48\'sd%d' % v) if v >= 0 else ('-48\'sd%d' % (-v))


def gen_layer(A, lid):
    Aq = np.rint(A * 4096).astype(np.int64)
    Aq = np.where(Aq >= (1 << 15), Aq - (1 << 16), Aq)
    T, C = Aq.shape

    rows = []
    for t in range(T):
        terms = ' + '.join('(yv[%d] * %s)' % (c, lit(int(Aq[t][c])))
                           for c in np.nonzero(Aq[t])[0])
        rows.append('    assign V[%d] = %s;' % (t, terms))

    sv = f"""// T41 生成文件（勿手改）：融合基线门 —— 生产者算完 A·Y 后只发 {T} 个判决位。
// A 来源：results/t21b_k4_gate_params.npz: L{lid}_A（t21f k=4 微调，与 T25 k4 门核同源）。
// 与 C1 的差别：C1 把 Y 位平面串行送出、消费者逐平面维护区间证书；本模块把
// A_q·Y 一次算完并直接比较，线上只有 {T} 位。判决与证书递归终态恒等。
module t41_fused_gate (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    input  wire [239:0] in_yv,        // {T} lane x 24b signed，切片 [24*c +: 24]
    input  wire [639:0] in_thr_row,   // {T} lane x 64b，与 T25/T15 同端口口径
    output reg  [9:0]   out_dec,
    output reg          out_valid
);

    wire signed [23:0] yv [0:9];
    wire signed [47:0] thrw [0:9];
    wire signed [47:0] V [0:9];

    genvar g;
    generate
        for (g = 0; g < {T}; g = g + 1) begin : g_lane
            assign yv[g]   = $signed(in_yv[24*g +: 24]);
            assign thrw[g] = $signed(in_thr_row[64*g +: 48]);
        end
    endgenerate

{chr(10).join(rows)}

    integer t;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_dec   <= {T}'d0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid;
            for (t = 0; t < {T}; t = t + 1)
                out_dec[t] <= (V[t] >= thrw[t]);
        end
    end
endmodule
"""
    meta = {
        'layer': lid,
        'A_source': 'results/t21b_k4_gate_params.npz: L%d_A' % lid,
        'nonzero_per_row': [int((Aq[t] != 0).sum()) for t in range(T)],
        'A_q_rows': [[int(x) for x in Aq[t]] for t in range(T)],
        'y_port_bits': 24 * C,
        'thr_port_bits': 64 * T,
        'note': '融合基线：同时替代 plane_ser 生产者与门核。线上 10 bits/组，1 拍/组。',
    }
    return sv, meta


def main():
    z = np.load(NPZ)
    (OUT / 'gates').mkdir(exist_ok=True)
    for lid in LIDS:
        sv, meta = gen_layer(z['L%d_A' % lid], lid)
        (OUT / 'gates' / ('L%d_gate.sv' % lid)).write_text(sv)
        if lid == LIDS[0]:
            # 综合脚本读这一份（L8，与 T25 k4 门核同层同源）
            (OUT / 't41_fused_gate.sv').write_text(sv)
            (OUT / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    ids = {'layers': LIDS, 'note': '四层各一份 RTL，模块名同为 t41_fused_gate；'
                                   't41_fused_gate.sv = L%d（综合用）。' % LIDS[0]}
    (OUT / 'gates' / 'meta.json').write_text(json.dumps(ids, indent=1) + '\n')
    print('wrote t41_fused/t41_fused_gate.sv (L%d) + gates/L{%s}_gate.sv'
          % (LIDS[0], ','.join(str(x) for x in LIDS)))


if __name__ == '__main__':
    main()
