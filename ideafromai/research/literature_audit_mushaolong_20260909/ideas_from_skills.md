# Ideas from research skills（幕僚长 · 2026-09-09）

本文件用 **brainstorming-research-ideas** 与 **scientific-brainstorming** 挖候选；每条一页级：B / 最强对照 / A / X / 杀门。  
标注：有上下文的 AI 辅助复核，**不是**盲法 human workshop。scientific-brainstorming 引用：Kassis et al., arXiv:2609.00065。

镜头使用（brainstorming-research-ideas）：Problem-First、Tension hunting、Abstraction ladder、「改变前提后重开」（固定 BN / 允许训练 / 多口）。

---

## Idea 1 — Patch 条件少生产（主线）

**镜头**：Problem-First（对准剩余最贵 patch ~34.8%）；Tension（不可廉价删 vs 2:4 极强）。

- **B**：`resblocks` 删分支 AEE 爆；半宽/2:4 可过门但 2:4 是强控制；零列跳过是普通底座非新电路（`algorithm/patch_probe/README.md`）。
- **最强对照**：固定 BN（四处 patch full825 AEE 1.203184）+ 逐行 2:4 + 一次严格生产许可 + 普通压缩 Y/零掩码。
- **A**：DynConv 前驱生产 mask；TermiNETor 共同终止训练；Graham/SBNet/SSCN 索引与 halo 边界（`literature/patch_*`）。
- **X**：在**消费者完成反馈**下减少仍被需要的生产（不是持续检查已败的版本）；训练或校准生产分数，同资源比一次许可净收益。
- **杀门**：不胜一次许可/2:4；或 valid825>1.259 / 劣对照>0.005；或只删便宜工作。

---

## Idea 2 — PSN 真洞检验（先完 A，再谈 X）

**镜头**：Abstraction ladder（服务模型占比 → 物理位宽/扇出）；Tension（「贵」vs「完整 A 未测」）。

- **B**：旧学生 PSN/输出 73.81%、MAC 发射 68.517%（`bn_state/support_service_result.json`）；**尚未**同学生常矩阵编译对照。
- **最强对照**：da4ml 完整常矩阵编译 + 跨 p 流水；CompRRAE 严格 MSB 界；合法缩位；MFPSN 完整 STE（有损时）。
- **A**：da4ml §2–4；CompRRAE §IV；SPARK/USEFUSE 动态 CSE 权限对齐；Gustav 局部 S 消费链。
- **X**：仅当对照后仍有可命名的物理可省量（位宽/扇出/RF/反馈损失）时，提出区别于「界早停+按位按需」的机制。
- **杀门**：无具体差异；净服务 <15% 或不增状态位数失败；扫参续命禁止（`psn/psn_decision_one_page.md`）。

---

## Idea 3 — 完整结构稀疏迁到敏感挂点

**镜头**：改变前提后重开（离开可全删的 s2b3）；Tension（广播并集 vs hidden50 删消费者）。

- **B**：s2b3 全隐藏删仍 AEE 1.183003；Gyro 联合不胜 hidden50（`group_pruning_probe/README.md`）。
- **最强对照**：hidden50；row2:4；**完整** HiNM 二阶（含输入重排）。
- **A**：HiNM + VENOM/CRISP 两级结构 + Gustav 请求前跳块（AB_STACK P0-1）。
- **X**：在固定广播组 G 内用完整 T10 门+真实 FC2 损失选可共同删的物理字，且挂点先过敏感性。
- **杀门**：不胜完整 HiNM/hidden50；并集读满无周期余量；只在廉价块扫参。

---

## Idea 4 — 有损共同完成（并集损失）

**镜头**：Tension hunting（单 (p,h) 可免 ~5% vs H8 最慢后 ~1%）；改变前提（允许有损）。

- **B**：共享源字被最慢消费者钉死（`psn/group_completion_prefix_probe.json`：静默 66.91% oracle ≠ 可接受跳过）。
- **最强对照**：完整 Gustav；**独立**每神经元预测 + 相同组关闭；SparseInfer；hidden50。
- **A**：SparseInfer 预测后跳权；BitFair 学习终止与共享输入终止汇聚。
- **X**：固定检查点预测整组 T10 门字（含预测非零），一组接受/继续；训练目标用真实请求并集损失。
- **杀门**：不胜独立预测同权限；预测开销≈再做一次 PSN；AEE 爆。

---

## Idea 5 — 挑战探针：运动唤醒上界 / F_live>1

**镜头**：改变前提（允许训练小预测器；多活上下文）；Problem-First（旧固定 tile 差分负费用不能否证可训练唤醒）。

- **B**：注意力份额小；旧精确差分负费用；类别路径在 F_live=1 下可能被低估。
- **最强对照**：无唤醒基线；F_live=1 强 Gustav；禁止最终 flow oracle。
- **A**：ERAFT/TMA 预测调度；Gustav 多输出上下文。
- **X**：一日级上界——可训练事件/TDE 小预测器的乐观跳过率；或合法 F_live 搜索下类别 vs 时间。
- **杀门**：相对强对照 <5% 或不稳；把旁路注意力恢复成系统主加速。

---

## Scientific-brainstorming：三视角缺口与反对（再合并）

**声明**：此为有上下文的 AI 辅助复核（Kassis et al. arXiv:2609.00065），非盲法人类工作坊。

### 视角 A — 稀疏执行 / 架构

- **缺口**：完整 HiNM/VENOM 未迁；Gustav 64PE/源 SRAM 未闭；同资源净收益门未过。
- **反对**：继续在 s2b3 上加联合损失是在可删块上过拟合；2×4 mount-only 不能作文。
- **赞成推进**：Idea 1/3；先敏感性再完整 A。

### 视角 B — SNN / 光流机制

- **缺口**：θg 幅值路径使二值 SNN 加速器对照部分失效；运动唤醒未训；TDE-3+残差共设计未做。
- **反对**：把 ATLIF/Motion-XOR 换名当电路创新；用最终 flow 当 oracle。
- **赞成推进**：Idea 5 仅作上界；主贡献仍须落在 patch/PSN 算术消费。

### 视角 C — 电路 / CMVM / RTL

- **缺口**：da4ml/LUT-DLA 完整 A；PSN 位域费用；索引头共享等普通强对照未给足。
- **反对**：CPU 服务拍当 RTL；未过门扩 EDA；把 CSE/界早停换名。
- **赞成推进**：Idea 2 的「先对照后 X」；RTL 维持 mount-only。

### 合并结论

优先 **Idea 1 → Idea 2 的 A 补齐 → Idea 3/4 择一**；Idea 5 降级为轻量探针。任何候选必须通过 grilling：完整 A？最强简单控制？局部≠整法？是否只换名？
