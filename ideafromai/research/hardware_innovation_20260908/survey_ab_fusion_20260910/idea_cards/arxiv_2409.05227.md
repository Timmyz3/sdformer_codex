# arXiv:2409.05227 · BBS / BitVert

- uid/来源: `MAIN-R047`｜arxiv_2409.05227+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration
- 精读深度: 方法级（仅方法摘录窗口）+依据：双向比特稀疏（可剪 0 或 1 列）；免重训 bit-level binary pruning；张量编码减足迹；BitVert bit-serial PE；相对值稀疏与量化张力分析

## 可继承 A
比特列级结构化稀疏（保证>50%）+ 免重训二进制剪枝 + bit-serial 执行——作 F1/F4「位宽/列取消」与量化后值稀疏失效问题的强对照（借入≠X）。

## 强对照 B
仅剪零比特的单向 bit-serial；值稀疏加速器（量化后稀疏≈0）；需重训的比特剪枝；Microscaling 浮点共享指数流水线。

## 可差分 X线索
BBS 比特列≠ r1 物理源字删字 X；差分须源活动+双 PED，而非搬 BitVert PE。

## 与 F1–F7 / Stage B 关系
F1/F4/F6（取消列）对照；Stage B 后可挂量化-稀疏轴。不抢 Stage B。

## 不可搬用边界
七个 DNN 分类/检测；bit-serial≠本地 MAC 阵列假设；勿搬 3.03×/2.44×；仅方法摘录窗口。

## 可复用 idea 点
- 双向剪 0/1 列改善 bit-serial 负载均衡——对照端口位列取消
- 免重训 binary pruning 作 PTQ 后稀疏化权限边界
- 「量化杀值稀疏」问题陈述对齐本地 fixed-BN 后压缩
- 负结果只停该比特列布局

## 杀门建议
迁到源字/系数后精度破门或服务不胜普通均匀量化/PoT → 停 BBS 布局。
