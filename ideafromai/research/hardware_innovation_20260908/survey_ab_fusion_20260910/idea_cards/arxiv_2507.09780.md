# arXiv:2507.09780 · BitParticle

- uid/来源: `MAIN-R048`｜arxiv_2507.09780+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: BitParticle: Partializing Sparse Dual-Factors to Build Quasi-Synchronizing MAC Arrays for Energy-efficient DNNs
- 精读深度: 方法级（仅方法摘录窗口）+依据：双因子比特稀疏 particlization（1/2/2/2 粒子）；IR 拼接压到≤7 PP；近似丢次要 IR；组内/组间准同步弹性；可切换数据流

## 可继承 A
双操作数比特粒子化 + 拼接减 PP + 准同步弹性 MAC 阵列——作「比特级双稀疏执行与弹性同步」强对照（DNN 比特稀疏族；借入≠X）。

## 强对照 B
单因子 bit-serial；EBS 式 16 IR 爆炸；Laconic 组内刚性同步。

## 可差分 X线索
DNN 比特粒子≠SNN/T10 源字删字 X；可作普通压缩/近似乘对照轴，不作标题。

## 与 F1–F7 / Stage B 关系
旁路（DNN MAC）；与 FLARE/BitSift/BBS 比特族并列阅读。不抢 Stage B。

## 不可搬用边界
ResNet/MobileNet 精度≠AEE；近似变体误差勿外推光流；仅方法摘录窗口。

## 可复用 idea 点
- 双因子比特稀疏暴露作普通压缩上界
- IR 拼接「零开销拼接代累加」技巧对照
- 准同步弹性↔有限服务背压下利用率叙事
- 与已有 BitVert/BitSift 卡成谱系

## 杀门建议
仅换比特 MAC 无 lifting 结构，相对 ordinary 无净服务/精度门 → 停「当标题」。
