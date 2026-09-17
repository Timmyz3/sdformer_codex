# 融合 RTL 微探针第三轮（MP5）— 2026-09-11

**工具**：iverilog 12.0 + yosys 0.52 techmap（box）  
**根目录**：`/workspace/rtl_microprobes/`  
**性质**：表达力 / 同槽周期差教学计数。**不是** Stage B `service%`，**不是** TCAS-II PPA，**不是**标题级 X。

---

## 记分牌（含既有）

| 探针 | sim | synth | Cells | 关键周期/行为 |
|------|-----|-------|------:|----------------|
| MP1 same_port_credit | PASS | PASS | 179 | C0/C1/C2 服务与 stall（既有） |
| MP2 group_accept | PASS | PASS | 921 / 985 | 接受 1 拍；重算 ~6–7 拍（既有） |
| MP3 shared_source_prune（F1） | PASS | PASS | 887 / 1143 | 剪枝 emit / 双消费者（既有） |
| MP4 last_n_merge_cert（F4） | PASS | PASS | 2407 | 全合 8 / 末N接受 6 / 回退 14（既有） |
| **MP5** fuse_legality | **PASS** | **PASS** | **2091**（MODE1）/ **1922**（MODE0） | 同 1-slot：通用 8 拍 vs lifting-iface 11 拍（基线 16） |
| half-step RNE（旁路既有） | PASS | PASS | ~76 | 半步边界粗验 |

---

## MP5 通过了什么 / 不声称什么

**通过**：在**相同**每拍 1-slot 契约下，可综合（无 latch）实现两臂 fuse legality——
- Arm A「通用」：任意相邻对可融 → **8** 拍（相对基线 16 削减 8，玩具 −50%）
- Arm B「lifting 结构化接口」：仅 `iface` 标签相等可融 → **11** 拍（削减 5，玩具 −31.25%）
- 两臂退役同一 16-op 流、checksum 一致；`fuse+single == cycles`。

**教学用途**：口头说明为何软件侧 **通用指令融合 −13.56%**（lifting 6194→5354 风格）**不得**写成 lifting 标题级 X——通用合法与 lifting 接口合法不是同一谓词，同槽下削减可以更大，但那部分不是「结构化 lifting 创新」。

**不声称**：真实 −13.56% 被本玩具复现；lifting 融合净 service% 优势；任何标题 X；cell 数作 PPA。

---

## 与 ROUND2 的关系

ROUND2 完成 MP3/MP4。本 ROUND3 只追加 **MP5**（ROUND2「下一探针想法」第 1 条）。  
未做：MP3b 索引重映射税、MP2×RNE 粘合、生产 RTL / Stage B 替代。

---

## 绝对路径速查

```
/workspace/rtl_microprobes/mp5_fuse_legality/
/workspace/rtl_microprobes/mp5_fuse_legality/RESULTS.md
/workspace/rtl_microprobes/OVERNIGHT_RTL_SUMMARY.md
/workspace/overnight_20260911/rtl_probes/FUSION_RTL_ROUND3.md
/workspace/overnight_20260911/FUSION_RTL_ROUND3.md
/workspace/overnight_20260911/rtl_probes/FUSION_RTL_ROUND2.md
```
