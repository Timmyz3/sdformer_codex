#!/usr/bin/env python3
"""T26b：位宽参数化的门核生成器（k4 掩码 A，加法树形态）。

T26a 结论：48b 是为 e≤31（5 bit 端口的全量程）留的保守尺寸，但生产者的 lane 是
signed24/f14，|Yq| 的 MSB 下标**物理上 ≤ 23**——所以 48 位中高 10 位不可达。

本脚本给出**可证明**的位宽下界（与观测数据无关，只依赖网络常量）：
    W_min = bit_length( max_t L1_t · 2^23 ) + 1        # 1 = 符号位
    L1_t = Σ_c |A_q[t][c]|（部署网络常量，逐行）
并把 W 作为参数生成 SV：所有区间通路寄存器/常量/thr 切片都用 W 位。
thr 仍从 640b 外置端口取（接口不变），内部取低 W 位——两补码截断对"能装下的值"
是精确的，生成时 assert 保证。

用法：python t26_width/t26_gen.py <W> [full|k4]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 't25_k4_gate'))
from t25_gen import K4_PARAMS, TRACE  # noqa: E402

T = 10


def load_A(variant):
    if variant == 'full':
        return np.load(TRACE)['A'].astype(np.float64)
    return np.load(K4_PARAMS)['L8_A'].astype(np.float64)


def load_tau48(d, H):
    tau = np.zeros((T, H), np.int64)
    dflag = np.zeros((T, H), np.int64)
    for t in range(T):
        v = np.array([int(x, 16) for x in (d / f'tau_t{t}.hex').read_text().split()],
                     np.uint64)
        dflag[t] = (v >> np.uint64(63)).astype(np.int64)
        thr = (v & np.uint64((1 << 48) - 1)).astype(np.int64)
        tau[t] = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    return tau, dflag


SV = '''// T26 生成文件（勿手改）：位宽参数化门核，W={W} bit，变体 = {variant}。
// 生成：t26_width/t26_gen.py。常数（P_t/N_t/dot/|A_q|）同 t25_k4_gate/{variant}。
// 相对 T25（48b）仅改区间通路位宽：{W}b，thr 仍取 640b 端口低 {W} 位。
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
    reg signed [{W1}:0] vtop  [0:9];
    reg signed [{W1}:0] pmsk  [0:9];
    reg signed [{W1}:0] nmsk  [0:9];
    reg [63:0] tauw   [0:9];
    reg [9:0]  locked;
    reg [9:0]  dec_raw;

    integer t;

    wire [9:0] wsel = in_sop ? in_sign : in_plane;

    wire signed [{W1}:0] lutw [0:9];
{lut_assigns}

    wire signed [{W1}:0] P_t [0:9];
    wire signed [{W1}:0] N_t [0:9];
{pn_assigns}

    wire [63:0]        tau_sel [0:9];
    wire signed [{W1}:0] thrw   [0:9];
    wire signed [{W1}:0] nv     [0:9];
    wire signed [{W1}:0] vmin_n [0:9];
    wire signed [{W1}:0] vmax_n [0:9];
    wire               lock_n [0:9];
    genvar gt;
    generate
        for (gt = 0; gt < 10; gt = gt + 1) begin : g_chk
            assign tau_sel[gt]    = tauw[gt];
            assign thrw[gt]       = $signed(tau_sel[gt][{W1}:0]);
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
                vtop[t] <= {W}'sd0; pmsk[t] <= {W}'sd0; nmsk[t] <= {W}'sd0;
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
'''


def lit(v, W):
    v = int(v)
    return f"-{W}'sd{-v}" if v < 0 else f"{W}'sd{v}"


def main():
    W = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else 'k4'
    W1 = W - 1
    src = ROOT / 't25_k4_gate' / variant
    outdir = ROOT / 't26_width' / f'w{W}_{variant}'
    outdir.mkdir(parents=True, exist_ok=True)

    A_q = np.array([int(x, 16) for x in (src / 'a.hex').read_text().split()], np.int64)
    A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q).reshape(T, T)
    pnv = np.array([int(x, 16) for x in (src / 'pn.hex').read_text().split()], np.int64)
    pnv = np.where(pnv >= (1 << 47), pnv - (1 << 48), pnv)
    P_t, N_t = pnv[0::2], pnv[1::2]

    # 可证明的位宽下界：L1·2^23（lane 是 signed24，MSB 下标物理 ≤ 23）
    L1 = np.abs(A_q).sum(1)
    provable = int(L1.max() * (1 << 23)).bit_length() + 1
    # 生成期 assert：所有实际常数必须装得下 W 位
    lim = 1 << (W - 1)
    consts = list(P_t) + list(N_t) + list(A_q.ravel())
    assert max(abs(v) for v in consts) < lim, f'constant overflow at W={W}'

    lut_assigns, pn_assigns = [], []
    for t in range(T):
        supp = [int(c) for c in np.nonzero(A_q[t])[0]]
        if not supp:
            lut_assigns.append(f"    assign lutw[{t}] = {W}'sd0;")
        else:
            terms = ' + '.join(f"(wsel[{c}] ? {lit(A_q[t][c], W)} : {W}'sd0)"
                               for c in supp)
            lut_assigns.append(f"    assign lutw[{t}] = {terms};")
        pn_assigns.append(f"    assign P_t[{t}] = {lit(A_q[t].clip(min=0).sum(), W)};"
                          f"  assign N_t[{t}] = {lit(A_q[t].clip(max=0).sum(), W)};")

    (outdir / 'cert_gate_bitl_synth.sv').write_text(
        SV.format(W=W, W1=W1, variant=variant, lut_assigns='\n'.join(lut_assigns),
                  pn_assigns='\n'.join(pn_assigns)))

    # thr 检查（部署 tau 常量必须装得下）+ 零差校验所需的 hex 原样复制
    thr_chk = 0
    for t in range(T):
        (outdir / f'tau_t{t}.hex').write_text((src / f'tau_t{t}.hex').read_text())
    thr, _ = load_tau48(src, 384)
    thr_chk = int(np.abs(thr).max())
    assert thr_chk < lim, f'thr overflow: |thr|max={thr_chk} >= 2^{W-1}'
    (outdir / 'a.hex').write_text((src / 'a.hex').read_text())
    (outdir / 'pn.hex').write_text((src / 'pn.hex').read_text())

    meta = {'variant': variant, 'W': W, 'provable_W_min': provable,
            'L1_row_max': int(L1.max()), 'thr_abs_max': thr_chk,
            'nz_per_row': (A_q != 0).sum(1).tolist()}
    (outdir / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'W={W} {variant}: provable_min={provable} L1_max={int(L1.max())} '
          f'|thr|max={thr_chk} -> {outdir}')


if __name__ == '__main__':
    main()
