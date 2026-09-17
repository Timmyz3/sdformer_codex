#!/usr/bin/env python3
"""T25 生成器：掩码加法树门核（无 ROM 版 cert_gate_bitl）。

动机（T24 §3）：ROM 版门核的 dot10（12.8kb 子集和 ROM）在 FPGA 上落 distributed
RAM，吃 5,905 LUT。T21 的证书感知掩码把 A 行压到 k 个非零后，dot10 退化为
k 项选位常数和——**只要常数是 RTL 字面量**（而非 $readmemh），Vivado 即可把它
综合成几十个 LUT 的加法树，不再推断 ROM。

本脚本：
1. 从 trace + A 重算 A_q / P_t / N_t / tau_q / thr（与 t5_c1_cert_core_model.py 同式）；
2. 写 tau_t*.hex（thr 仍走 640b 外置端口，接口不变）；
3. 生成 cert_gate_bitl_synth.sv —— 端口/功能与 t15_synth 版逐位一致，仅把
   lut_lo/lut_hi ROM + $readmemh 常数换成烘焙字面量的选位加法树。

变体：
  full —— A = trace 原始稠密 A（10 项/行）：与 ROM 版同常数，隔离"ROM→树"这一步；
  k4   —— A = t21f k=4 微调网络的 L8_A（4 项/行）：T25 目标设计点。

用法：python t25_k4_gate/t25_gen.py [full|k4]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT.parent
TRACE = HW / 'bn_state' / 'trace_s0_stage0.npz'
K4_PARAMS = ROOT / 'results' / 't21b_k4_gate_params.npz'
T = 10


def hexlines(arr, bits):
    return '\n'.join(format(int(x) & ((1 << bits) - 1), '0%dx' % (bits // 4))
                     for x in np.asarray(arr).ravel()) + '\n'


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1), \
        'overflow %d bits' % bits
    return out


def build_params(A):
    """A → A_q/P_t/N_t/tau_q/thr（照抄 t5_c1_cert_core_model.py 的部署态静态常数口径）。"""
    z = np.load(TRACE)
    W = z['W'].astype(np.float64)
    gamma, beta = z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    H = W.shape[0]
    A = np.asarray(A, dtype=np.float64)
    assert A.shape == (T, T)

    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    S = None
    # tau 只用 Y 的矩（mu/var），按 h 分块算，避免整块 (T,P,H) 常驻
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[
        :, :W.shape[1]].astype(np.float64)
    P = S.shape[0] // T
    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (theta + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]
        del Y

    A_q = to_signed(np.rint(A * 4096), 16)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
    thr = to_signed(thr, 48)
    Dflag = (direction > 0).astype(np.int64)
    return A_q, P_t, N_t, thr, Dflag


def s48(v):
    """48 位有符号字面量（负号必须在位宽之前：-48'sd5）。"""
    v = int(v)
    return f"-48'sd{-v}" if v < 0 else f"48'sd{v}"


SV_TMPL = '''// T25 生成文件（勿手改）：掩码加法树门核，变体 = {variant}。
// dot10(t,w) = Σ_{{c∈supp(t)}} aq[t][c]·w[c]（{nterms} 项选位常数和，烘焙字面量）。
// 端口/时序/判决逻辑与 t15_synth/cert_gate_bitl_synth.sv 逐位一致；
// 差异仅在：无 a/pn/lut ROM、无 $readmemh，P_t/N_t/dot 常数全部为 RTL 字面量。
// 生成：t25_k4_gate/t25_gen.py（A 来源：{asrc}）
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
{lut_assigns}

    wire signed [47:0] P_t [0:9];
    wire signed [47:0] N_t [0:9];
{pn_assigns}

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
'''


def gen_sv(A_q, outdir, variant, asrc):
    lut_assigns, pn_assigns = [], []
    nterm = set()
    for t in range(T):
        supp = [int(c) for c in np.nonzero(A_q[t])[0]]
        nterm.add(len(supp))
        if not supp:                       # 全零行：dot ≡ 0
            lut_assigns.append(f"    assign lutw[{t}] = 48'sd0;")
        else:
            terms = ' + '.join(f"(wsel[{c}] ? {s48(A_q[t][c])} : 48'sd0)" for c in supp)
            lut_assigns.append(f"    assign lutw[{t}] = {terms};")
        pn_assigns.append(f"    assign P_t[{t}] = {s48(A_q[t].clip(min=0).sum())};"
                          f"  assign N_t[{t}] = {s48(A_q[t].clip(max=0).sum())};")
    nt = ','.join(str(x) for x in sorted(nterm))
    sv = SV_TMPL.format(variant=variant, asrc=asrc, nterms=nt,
                        lut_assigns='\n'.join(lut_assigns),
                        pn_assigns='\n'.join(pn_assigns))
    (outdir / 'cert_gate_bitl_synth.sv').write_text(sv)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'k4'
    if variant == 'full':
        z = np.load(TRACE)
        A = z['A'].astype(np.float64)
        asrc = 'bn_state/trace_s0_stage0.npz: A（稠密基线）'
    elif variant == 'k4':
        A = np.load(K4_PARAMS)['L8_A'].astype(np.float64)
        asrc = 'results/t21b_k4_gate_params.npz: L8_A（t21f k=4 微调）'
    else:
        sys.exit('variant must be full|k4')

    outdir = ROOT / 't25_k4_gate' / variant
    outdir.mkdir(parents=True, exist_ok=True)

    A_q, P_t, N_t, thr, Dflag = build_params(A)
    gen_sv(A_q, outdir, variant, asrc)

    # thr 走 640b 端口：与 t5 生成器同格式（bit63 = Dflag，低 48 位 = thr）
    for t in range(T):
        word = (Dflag << 63) | (thr[t] & ((1 << 48) - 1))
        (outdir / f'tau_t{t}.hex').write_text(hexlines(word, 64))
    (outdir / 'a.hex').write_text(hexlines(A_q, 16))
    (outdir / 'pn.hex').write_text(hexlines(np.stack([P_t, N_t], 1).ravel(), 48))

    nz = (A_q != 0).sum(1)
    n_terms = int(np.unique(nz).sum() if len(np.unique(nz)) == 1 else 0)
    meta = {
        'variant': variant, 'A_source': asrc,
        'nonzero_per_row': nz.tolist(),
        'dot_terms_per_row': n_terms or int(nz.max()),
        'A_q_max_abs': int(np.abs(A_q).max()),
        'lut_ops_per_row': int(nz.max()), 'total_mux_terms': int(nz.sum()),
        'P_t': P_t.tolist(), 'N_t': N_t.tolist(),
        'tau_hex_dir': str(outdir),
    }
    (outdir / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'{variant}: nz/row={nz.tolist()} total_mux_terms={int(nz.sum())} '
          f'|A_q|max={int(np.abs(A_q).max())} -> {outdir}')


if __name__ == '__main__':
    main()
