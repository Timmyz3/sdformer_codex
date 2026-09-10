# Card I Soft Knives — BL-VetoPrior + NL-STMFA (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot` only · `// GROKBOT NEW FILE -- iscas_ssh`  
**Role:** Knife-1 novelty enhancers (soft → real RTL). Orthogonal to Card I main knives TID/EDC.

---

## 中文摘要

1. **BL-VetoPrior（`c1s_bl_veto_prior`）**：Barlow–Levick **抑制 veto** 数字方向银行；与 TDE3 **促进型**先验正交。`pd_wake` OR 进 `wake_merge`；`veto_mask` AND-NOT 抑制 null 方向。
2. **NL-STMFA（`c1s_nl_stmfa`）**：多尺度 prev feature stub + 非线性残差（≈1.5|Δ|+线性种子项）→ `nl_wake` / `r_nl`。**诚实：** 算法启发数字草图，非 ICRA AEE 声称。

`c1s_wake_merge` 扩展：`(op|tde|delta|nl|pd) & ~veto`。

---

## EN

| | BL-VetoPrior | NL-STMFA |
|---|---|---|
| **What** | Inhibition veto bank (fac/trg/null Δt) | Multi-scale nonlinear residual wake |
| **vs stack** | Orthogonal to TDE3 facilitation | Beyond linear MW/TMA |
| **Cite** | TrueNorth BL OF arXiv:1710.09820 | E-NMSTFlow / STMFA ICRA'25 |
| **Ablation** | NO_VETO → merge_pop ↑ | LINEAR_ONLY → res_sum ↓ / may miss wake |
| **Do-not-claim** | first bio OF HW; TrueNorth≠this RTL | ICRA accuracy; unsupervised loss as HW |

### Files
| Path | Role |
|---|---|
| `rtl_c1star/c1s_bl_veto_prior.sv` | BL RTL |
| `tb_c1star/tb_c1s_bl_veto_prior.sv` | BL TB |
| `flows/oss/sim_bl_veto_prior.sh` | BL sim |
| `rtl_c1star/c1s_nl_stmfa.sv` | NL RTL |
| `tb_c1star/tb_c1s_nl_stmfa.sv` | NL TB |
| `flows/oss/sim_nl_stmfa.sh` | NL sim |
| `rtl_c1star/c1s_wake_merge.sv` | extended merge |

### CN novelty one-liner
刀1 新颖增强器：**BL 抑制否决库 + NL-STMFA 非线性残差唤醒**（相对 TDE3 促进 / 线性 MW·TMA）。
