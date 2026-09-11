# Round 2 报告：按 AT-LIF \(\{0,\theta\}\) 且 \(\theta\) 吸入 \(W\) 重挖融合

**日期：** 2026-09-11  
**性质：** 提案。独立构想未见 round1 fusion 目录。  
**身份：** `../IDENTITY_ATLIF.md`（用户锁定）。

## 身份改了什么

吸完之后，**脉冲路径就是二值 GeMM**。  
Prosperity / GustavSNN / FireFly-S / Phi 对这条路径从“身份不合”变成 **必须抄全的 A**。  
不能再写“我们不是二值所以那些论文不算”。  
也不能写“不可吸收 int8 θg”。

PSN 自己就切开了：阈值前 \(H=WX\)（连续混叠），阈值后 \(S\in\{0,1\}\) 送给下一层。da4ml 编前半，Prosperity 加速后半，这是 **A+A，不是 X**。

残差 / PED / I24 是**另一条连续张量**，不是 AT-LIF 带着模拟幅值被两个 MAC 共用。SDT 的 membrane-shortcut 已经回答“加法放在 SN 前所以脉冲保持二值”——那也是 A。

网上没找到“Prosperity 用在事件光流”或“同一生产者的二值脉冲 + 连续残差双消费者加速器”。没有证据 ≠ 从未研究。

## 融合岛与对抗

| 岛 | 一句话 | 对抗 |
|---|---|---|
| **G1** 吸收切口上的类型化 last-use | 二值 GeMM 用完 ≠ 残差/PED 用完；8088 可能是连续张量还活着 | **revise**；新颖 1、性能 1。并集可能更密；也可能只是 FIFO 伪影。先拆等待直方图 |
| **G2** 光流几何打包吸收后的二值支持 | 上一帧/粗尺度平移 NRV | **停标题**。ExSpike 已做邻位 AND；本地运动差分已是额外功 |
| **G3** 全幅投影 BN | live `10×96×120×160` batch 统计 | **hygiene**，不是 X |
| **G4** 训练二值支持密度 | 同 \(\rho_S\) 下胜过 shuffle | **停标题**。那是 FireFly/Phi/Bishop 的软件半边；不要把仅有的 paired recovery 花在这里 |

## 现在硬件从哪入手（按这个身份）

1. **脉冲 GeMM：** 把 Prosperity（或 Gustav NRV 当 GeMM 部分和，不是 LIF 膜）当对照底座抄全。这不是贡献句。  
2. **阈值前 T10：** da4ml/CSE 当对照。lifting 相对 AEE 仍红，不要当标题。  
3. **唯一还对准已测洞的研究问题：** 为什么 always-ready 能少、长背压仍是 8088。用 **G1 的问题**去拆等待（FIFO / 残差仍活 / BN 屏障），不要先画新 RTL。这就是 Codex `full_chain` 该交的分母。  
4. 若直方图显示残差 last-use 才是 8088，再修订 G1 的一页机制；若是 FIFO 或 BN，就停 G1 布局。  
5. 15% 过了 AEE 仍 +0.013 才允许一次 paired recovery，对象是精度。

## 工作流核对（Partial）

`deep-research-2` 全文：`web/DEEP_RESEARCH2.md`。与本轮身份一致：官方 AT-LIF 是 \(o\in\{0,\theta\}\)，冻结 \(\theta\) 折进下一层 \(W\)，部署不需要 graded-spike 硬件。同名 AT-LIF（KBS 放电后抬阈值、I-LIF 整数分级、本地三值、HBG 不可吸收 int8）都**不是**这个身份。

FireFly-S 把**量化尺度**折进膜阈值，和 \(\theta\to W\) 同类，仍是脉冲路径上的 A。ESTU 虽说二值可把 MAC 换成加法，正文仍有 Dense(int) 乘法器，不是完整无乘 GeMM。PED 这个名字在 Prosperity/ExSpike/LoAS/Phi/Bishop/FireFly 检过的正文里没有；连续残差双消费者仍是本地架构观测。

## 明确不要做

- HBG-RP 不可吸收 int8  
- “我们跑了 Prosperity”当标题  
- 光流几何当标题（G2）  
- 为 FireFly 式剪枝单独开训练当标题（G4）  
- 删 35 次 RNE 当标题  
- CIM、OpenROAD-as-PPA、模块动物园

独立构想 6×8 条在 `independent/`。文献重映射 `literature/R2L*.md`。对抗 `adversarial/`。决策 `DECISION_LOG.md`。

本轮使用 Scientific Agent Skills 的 scientific-brainstorming 与 hypothesis-generation（Kassis et al., 2026, arXiv:2609.00065）。
