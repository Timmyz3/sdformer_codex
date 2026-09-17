# Grok 照抄-先、再改（2026-09-16）

独立目录，不复用 Claude/Codex 脚本。只读 `bn_state/trace_*.npz`。

用户纠正：C1 现在就是「位串行 + 区间早停」，一条线太窄、创新不够。
路线改回：**先把顶会机制完整抄到本网接口上（A），测过再针对性改（X）**；
体系结构文章不限 SNN，ANN/DNN 按需迁移。

## 两篇必须先抄全的（Claude 没抄全）

| 论文 | 原作真正机制 | Claude 实际做了什么 | 本目录 A |
|---|---|---|---|
| **BitFair** JETCAS'26 / arXiv:2607.05445 | 权值 bit-serial；**每个 PE 独立**早停；部分和 vs **可学习层阈值** 预测 ReLU=0；**贪心自适应位序**；软门+温度退火训练 | T1b 只测了「逐判决相对位深」数字，定点组粒度杀了；**没有**逐 PE 停、没有学习阈值、没有 ABO、没有训练目标 | `bitfair/bitfair_copy.py` |
| **BitL** MICRO'25 DOI 10.1145/3725843.3756044 | bit 矩阵上 **横/纵查找切换（dynamic pivot）** 缩短零比特跳过的关键路径 | T10 只做了 5+5 bit 子集和 LUT，拍数不变。Codex 已写明「不是完整 BitL」 | `bitl/bitl_copy.py` |

## 运行

```bash
cd grok_copyfirst_20260916
/opt/anaconda3/bin/python bitfair/bitfair_copy.py
/opt/anaconda3/bin/python bitl/bitl_copy.py
```

## 文献降维

见 `ARCH_IDEAS.md`：HPCA/ISCA/MICRO/DAC/DATE/ICCAD 的 ANN/DNN 机制 → 本网挂点。
