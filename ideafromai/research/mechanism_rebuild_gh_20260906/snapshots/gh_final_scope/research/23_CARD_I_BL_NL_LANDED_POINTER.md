# Card I + BL/NL landed (pointer)

**Date:** 2026-09-06 (Asia/Shanghai)  
**RTL tree:** `/workspace/sdformer_c1c2star_grokbot/` (iscas_ssh)  
**Commits:** `f19016a` / `4aca061` · regress **30/30**  
**Sketch:** `22_CARD_I_IF_TID_EDC.md` · idea Top5: `21_NOVELTY_BOOST_TOP5_FORKA.md`

## Landed this wave
| Idea name | RTL (expected) | Role |
|---|---|---|
| TID-DeblurLoop | `c1s_tid_deblur_loop` | Card I1 — corr-free deblur loop |
| EDC-ΔFuse | `c1s_edc_delta_fuse` | Card I2 — Δ-feat × lo-corr fuse |
| BL-VetoPrior | (BL module per iscas_ssh) | 辅先验 — orthogonal to TDE3 |
| NL-STMFA | (NL module per iscas_ssh) | 辅先验 — nonlinear residual |
| CFP/SCI glue | front/back pipe wiring | exact/wake 真接入 |

## Still idea-only (not this wave)
ResHTR-Refiner · WinTok-CoSparse · AdjEvt-Compress（见 `21_*`；ResHTR = Card I+1 候补）

## Hygiene
三刀脊柱不变。OpenROAD ≠ 贡献。AEE → sd5。禁 MX3P / CIM。
