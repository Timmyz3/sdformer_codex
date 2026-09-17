# Fusion Screen 记分牌（ROUND1–ROUND3）

日期：2026-09-13（Asia/Shanghai）  
根目录：`/workspace/rtl_fusion_screen/`  
工具：iverilog 12.0 + yosys 0.52 techmap

## 范围诚实
周期=同资源 TB；cells=techmap 代理。**不是** Stage B service% / ASIC PPA / 生产树。无 `ADVANCE_X`。

## ROUND1–2 摘要
KEEP_PROBE：无。BASE_ONLY：cand1c、cand8。KILL：cand1b、cand2–7、cand9。

## ROUND3

| 候选 | 主题 | sim | synth | Cells | 周期 A vs X | verdict | 一行 |
|------|------|-----|-------|------:|-------------|---------|------|
| **cand10** 弹性直供 | R3-A | PASS | PASS | 3692 | 16/23/**903** vs 8/15/**903** | **KILL_LAYOUT** | 长背压打平 |
| **cand11** 基底+例外 | R3-B | PASS | PASS | 1198 | 8 vs **10** | **KILL_LAYOUT** | 例外税 |
| **cand12** gate-summary×NRV | R3-C | PASS | PASS | 1066 | 32/47 vs **34/49** | **KILL_LAYOUT** | 并集满读 |
| **cand13** 驻留 R24 | R3-D | PASS | PASS | 1534 | 64 vs **48** | **BASE_ONLY** | 通用驻留 |
| **cand14** 例外打包 | 可选 | PASS | PASS | 1319 | 16 vs 18/20/24 | **KILL_LAYOUT** | 三密度更慢 |

## 幸存
- **KEEP_PROBE**：无
- **BASE_ONLY**：cand1c、cand8、**cand13**
- 无标题 X
