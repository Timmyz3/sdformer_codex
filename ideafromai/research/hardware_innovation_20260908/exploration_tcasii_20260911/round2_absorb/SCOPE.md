# Round 2 — identity-locked fusion exploration (2026-09-11)

**Focal question:** AT-LIF 已锁定为 \(\{0,\theta\}\)，推理时 \(\theta\) 吸进下一层 \(W\)。在这个身份下，脉冲路径就是二值 GeMM。什么样的**完整先验（Prosperity/Gustav/FireFly/LoAS）+ 真增量 X**，能同时满足 TCAS-II 新颖性与同资源性能？

**本轮禁止：** 把“连续不可吸收 θg”当贡献；HBG-RP 不可吸收 int8；CIM。

**本轮允许：** 吸完后的 0/1 路径上完整抄 Prosperity/Gustav/FireFly；X 只能来自这些论文盖不住的对象（非因果 T10 混叠发生在阈值前、残差/PED 连续路径、全幅 BN、双路径 last-use、光流任务结构）。

独立构想只读 `PROBLEM.md` 与 `../IDENTITY_ATLIF.md`，不读 round1 的 fusion/independent 目录。
