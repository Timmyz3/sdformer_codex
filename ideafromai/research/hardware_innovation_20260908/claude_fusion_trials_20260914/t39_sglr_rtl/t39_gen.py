#!/usr/bin/env python3
"""T39 生成器：在已验证的 T25 k=4 门核上注入两处供数削减。

不重写门核：直接读 `t25_k4_gate/k4/cert_gate_bitl_synth.sv`（T25 已零差验证的 k=4
加法树门核），只做三处注入——
  1. 端口加 `out_need[9:0]`；
  2. 末尾加 need 组合逻辑（SGLR）：
       need[k] = 1  ⟺  存在未锁定判决 t（~locked[t]）使 A_q[t][k] != 0
     即 `need = OR_{t: ~locked[t]} sup_t`，sup_t 是部署常量（A_q 第 t 行非零 lane 位图）。
  3. sop 拍粗区间判定（T39b）：RTL 原先只在"送完第 1 个平面后"（m=e−1）才判，
     而 numpy 口径 (t32_lever.cert_planes 的 j 从 23 起扫) 允许在**发送前**用 m=e
     的粗区间直接锁定判决。补上这一拍即把 24.29% → 32.78%（T5 组样本，见
     t39_model.py --preplane）。判定安全：粗区间 ⊇ 细区间，锁定只提前不改变符号。

消费者侧用 need 反压生产者：只传 need 里的 lane，其余 lane 的比特不发。
数学模型（T38 §1.1）：被省略的 lane 对所有未解析判决权重恒 0 ⇒ 判决零差。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_SV = ROOT / 't25_k4_gate' / 'k4' / 'cert_gate_bitl_synth.sv'
SRC_A = ROOT / 't25_k4_gate' / 'k4' / 'a.hex'
OUT = ROOT / 't39_sglr_rtl'
T = 10


def load_A_q(path):
    vals = [int(x, 16) for x in path.read_text().split()]
    vals = np.array([v - (1 << 16) if v >= (1 << 15) else v for v in vals], np.int64)
    return vals.reshape(T, T)


def build(sv_src, sup_mask, preplane, name):
    """在 T25 k=4 门核源码上注入 T39 机制；preplane 控制是否含 sop 拍粗区间判定。"""
    sv = sv_src
    # [1] 端口注入
    port_old = "    output wire [9:0]  dbg_locked\n);"
    assert sv.count(port_old) == 1, 'port anchor not found'
    sv = sv.replace(port_old,
                    "    output wire [9:0]  dbg_locked,\n"
                    "    output wire [9:0]  out_need\n);")

    if preplane:
        # [2] sop 拍粗区间判定：在 generate 块尾加组合逻辑
        gen_anchor = "            assign lock_n[gt]     = ~locked[gt]\n" \
                     "                                  & ((vmin_n[gt] >= thrw[gt]) | (vmax_n[gt] < thrw[gt]));\n"
        assert sv.count(gen_anchor) == 1, 'generate anchor not found'
        gen_add = (
            "            // T39b：sop 拍粗区间（e 个低位全未知，V ∈ vtop·2^e + [N(2^e−1), P(2^e−1)]）\n"
            "            // 与首平面细区间同形，只是 d₁ 取极值 2N / 2P——故是 k=0 那一档检查。\n"
            "            assign thr0[gt]  = $signed(in_thr_row[gt*64 +: 48]);\n"
            "            assign vtop0[gt] = -lutw[gt];\n"
            "            assign vmin0[gt] = (vtop0[gt] <<< in_e) + ((N_t[gt] <<< in_e) - N_t[gt]);\n"
            "            assign vmax0[gt] = (vtop0[gt] <<< in_e) + ((P_t[gt] <<< in_e) - P_t[gt]);\n"
            "            assign lock0[gt] = (vmin0[gt] >= thr0[gt]) | (vmax0[gt] < thr0[gt]);\n"
            "            assign dec0[gt]  = (vmin0[gt] >= thr0[gt]);\n")
        sv = sv.replace(gen_anchor, gen_anchor + gen_add)

        decl_anchor = "    wire               lock_n [0:9];\n"
        assert sv.count(decl_anchor) == 1, 'decl anchor not found'
        sv = sv.replace(decl_anchor, decl_anchor +
                        "    wire signed [47:0] thr0   [0:9];\n"
                        "    wire signed [47:0] vtop0  [0:9];\n"
                        "    wire signed [47:0] vmin0  [0:9];\n"
                        "    wire signed [47:0] vmax0  [0:9];\n"
                        "    wire               lock0  [0:9];\n"
                        "    wire               dec0   [0:9];\n")

        sop_old = ("            locked <= 10'd0;\n"
                   "            dec_raw <= 10'd0;\n"
                   "            for (t = 0; t < 10; t = t + 1) begin\n")
        assert sv.count(sop_old) == 1, 'sop anchor not found'
        sv = sv.replace(sop_old,
                        "            for (t = 0; t < 10; t = t + 1) begin\n"
                        "                locked[t]  <= lock0[t];\n"
                        "                dec_raw[t] <= dec0[t];\n")

    # [3] SGLR 逻辑注入
    terms = '\n'.join(
        "        | (locked[%d] ? 10'd0 : 10'd%d)" % (t, sup_mask[t]) for t in range(T))
    body = ("\n    // ---- SGLR（T39）：支撑感知 lane 退役 ----\n"
            "    // need[k]=1 ⟺ 存在未锁定判决 t 使 A_q[t][k]≠0。sup_t 为部署常量，\n"
            "    // 故 need 是 10 个 10bit 二选一 OR——纯组合逻辑，无 LUT/无元数据流。\n"
            "    wire [9:0] sglr_need;\n"
            "    assign sglr_need = 10'd0\n" + terms + ";\n"
            "    assign out_need = sglr_need;\n\nendmodule\n")
    idx = sv.rindex('endmodule')
    sv = sv[:idx] + body
    (OUT / name).write_text(sv)


def main():
    OUT.mkdir(exist_ok=True)
    sv_src = SRC_SV.read_text()
    A_q = load_A_q(SRC_A)
    sup_mask = [sum(int(1 << k) for k in range(T) if A_q[t, k] != 0) for t in range(T)]

    build(sv_src, sup_mask, True, 'cert_gate_sglr.sv')
    build(sv_src, sup_mask, False, 'cert_gate_sglr_np.sv')

    meta = {
        'source_sv': str(SRC_SV.relative_to(ROOT)), 'source_a': str(SRC_A.relative_to(ROOT)),
        'A_q': A_q.tolist(),
        'support_per_row': sup_mask,
        'support_union': int(np.bitwise_or.reduce(np.array(sup_mask, np.int64))),
        'nonzero_per_row': (A_q != 0).sum(1).tolist(),
        'variants': {
            'cert_gate_sglr.sv': 'SGLR + sop 拍粗区间判定（T39 设计点）',
            'cert_gate_sglr_np.sv': '仅 SGLR（消融：去掉粗区间判定）',
        },
        'note': 'need = OR over unlocked t of sup_t；被省略 lane 对未解析判决权重恒 0 ⇒ 判决零差。'
                'sop 粗区间 ⊇ 首平面细区间，故只提前锁定、不改符号 ⇒ 判决零差。',
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print('wrote t39_sglr_rtl/cert_gate_sglr.sv (+ sop 粗区间) 与 cert_gate_sglr_np.sv (仅 SGLR)')
    print('  support/row =', sup_mask, ' union =', bin(meta['support_union']))
    print('  nonzero/row =', meta['nonzero_per_row'])


if __name__ == '__main__':
    sys.exit(main())
