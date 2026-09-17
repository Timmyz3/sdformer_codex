# 融合 RTL 微探针第二轮（MP3 / MP4）— 2026-09-11

**工具**：iverilog 12.0 + yosys 0.52 techmap（box）  
**根目录**：`/workspace/rtl_microprobes/`  
**性质**：表达力 / 周期差 / cell 代理粗验。**不是** Stage B `service%`，**不是** TCAS-II PPA，**不是**标题级 X。

---

## 记分牌

| 探针 | sim | synth | Cells | 关键周期/行为 |
|------|-----|-------|------:|----------------|
| MP1 same_port_credit | PASS | PASS | 179 | C0/C1/C2 服务与 stall（既有） |
| MP2 group_accept | PASS | PASS | 921 / 985 | 接受 1 拍；重算 ~6–7 拍（既有） |
| **MP3** shared_source_prune（F1） | **PASS** | **PASS** | **887 / 1143**（W=8/16） | 稠密 emit=W；剪枝 emit=popcount；双消费者重构正确；重叠 start→stall |
| **MP4** last_n_merge_cert（F4） | **PASS** | **PASS** | **2407**（N=5,GATES=8） | 全合并 8 拍；末N接受 6 拍；回退 14 拍 |
| half-step RNE（旁路既有） | PASS | PASS | ~76 | 半步边界粗验 |

---

## MP3 通过了什么 / 不声称什么

**通过**：共享源字库上，lifting 形 `keep_mask` 可共同删字；单端口扫描下 gate PED 与 continuous PED 仍能从保留字正确重构；剪枝路径 emit 少于稠密；控制与互斥可综合（无 latch）。

**不声称**：F1 净服务优势；物理源字在真实 r1/双 PED 上的可删性；相对 HiNM/窄稠密的胜负；任何标题 X。cell 数只是 generic techmap 代理。

## MP4 通过了什么 / 不声称什么

**通过**：末 N=5 部分合并 + 一拍证书 + 失败回退全量重算的控制通路可表达；接受路径短于全合并；回退路径显著更长；`accepted`/`rolled_back` 互斥。

**不声称**：F4（或复活全合并）有硬件净收益。survey 已记**全合并软件布局为负**——本探针只对照控制表达力，禁止写成「部分合并已证明创新」。

---

## 纸面设计回写

已写入 `/workspace/overnight_20260911/f1_f2_paper_design.md`：

- **§1.8**：MP3 结果边界（cells / 周期 / 仅可表达）
- **§2.6**：MP4 结果边界（作 F2 杀门「改 F4」旁路粗验；仍 ≠ service%）
- §4 勾选「MP3/MP4 回写」已勾

并同步：

- `/workspace/survey_ab_fusion_20260910_stage/f1_f2_paper_design.md`
- **桌面同名文件** `C:\Users\ZMD\Desktop\f1_f2_paper_design.md`（本轮从 box 覆盖更新；若桌面另有手改请先备份）

---

## 下一探针想法（仍只做表达力，不抢 Stage B）

1. **MP5（可选）**：同资源槽数下「通用指令融合」vs「lifting 结构化接口合法融合」两臂计数器——用于口头说明为何 −13.56% 通用融合**不得**写成标题 X（两臂 fuse legality 不同）。
2. **MP3b**：删字后的索引/对齐税显式计数（压缩流重映射拍数），仍非净服务。
3. **MP2×RNE**：半步边界触发与 group accept 对齐的更小 FSM（已有 half-step RNE + MP2，可拼粘）。
4. **不要做**：生产 RTL、ismd/nts07、`main.tex`、Stage B 替代实验、把 cell 下降写成 PPA。

---

## 绝对路径速查

```
/workspace/rtl_microprobes/mp3_shared_source_prune/
/workspace/rtl_microprobes/mp4_last_n_merge_cert/
/workspace/rtl_microprobes/OVERNIGHT_RTL_SUMMARY.md
/workspace/overnight_20260911/rtl_probes/FUSION_RTL_ROUND2.md
/workspace/overnight_20260911/f1_f2_paper_design.md
```
