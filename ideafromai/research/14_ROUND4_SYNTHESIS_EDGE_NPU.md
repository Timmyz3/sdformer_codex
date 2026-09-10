# ROUND 4 合成 — Edge NPU 控制环 → C1*/C2*（Grok Bot）

**日期:** 2026-09-05  
**输入:** `13_isscc_vlsi_hotchips_edge_npus.md`  
**定位:** 横向对标，不是再造手机 NPU；用 edge-NPU **控制环** 增强 R1–R3 包。

## 最强转移（叠在 09/12 上）
| 代号 | 作用 | 挂哪 |
|---|---|---|
| **Entropy-EE-OF / TileVFS-Motion** | 残差熵/运动驱动早停与 VFS | C1* exit / 功耗 |
| **MP-Pred-PRRC** | 预测谓词 → 金字塔预算 | PRRC |
| **SalCascade-EVWake / AoV-Cascade** | 显著度/Always-on 级联唤醒 | EV-Wake / Card H |
| **Asymp-OoO-ECP / FMSkip-OPSTW** | 异步乱序 / feature-map skip → wake | ECP-QKV / OP-STW |
| **HetSchedule-OF / DynPort-MFBD** | 异构调度 / 动态端口 | MFBD / Motion-TTB |
| **BigLittle-DualRail / BLT-CIM-SDSA** | big.LITTLE × DualRail-CIM | C2* fabric |

## Card H（系统）
Always-on OF orchestrator：SalCascade + EE/VFS + HetSchedule — **不是**又一个 MAC 阵列。

## DO NOT CLAIM
峰值 TOPS、首个 edge/sparse NPU、QNAP-ECP≠ECP-QKV、营销内部数字。

## RTL 顺序（用户锁）
**先 Card A OP-STW + Card B HBG-RP**（本轮立刻开干）；Card F/G/H 后置。
