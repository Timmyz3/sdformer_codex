# arXiv:1708.04485 · FireFly→正文SCNN

- uid/来源: `MAIN-R057`｜arxiv_1708.04485+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: FireFly（批元数据）/ 实文 SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：摘录含 SCNN PE/笛卡尔积数据流、非零W&A、scatter 累加、输入驻留与压缩编码；批名 FireFly 与 arXiv 1708.04485 实文不符（该 ID 为 SCNN），FireFly 真方法不在本窗口

## 可继承 A
SCNN 压缩稀疏卷积数据流底座：仅输送非零权重/激活、笛卡尔积乘、坐标 scatter 累加、激活端到端压缩与 PE 本地 tile 驻留——可作「非零源字供数+稀疏执行计费」对照分母（借入≠X）。

## 强对照 B
同资源稠密 CNN；只压 DRAM、仍做零乘的弱稀疏加速器；无激活稀疏的仅权稀疏方案。

## 可差分 X线索
SCNN/零跳过本身非标题级 X；差分须落到 r1 物理源字删字+非因果 T10×双 PED 消费者接口，而非复述笛卡尔积。

## 与 F1–F7 / Stage B 关系
F1/F5 数据流对照；旁路主岛。不抢 Stage B；与已有 SCNN.md/MAIN-R031 核对，本卡记录批错绑。

## 不可搬用边界
元数据错绑：不得当 FireFly 精读；CNN ReLU 稀疏≠SNN/T10；勿搬原作 2.7×/2.3× PPA；仅方法摘录窗口。

## 可复用 idea 点
- 非零源字广播域与 F1 可共同删物理字对齐讨论
- scatter 累加坐标跟踪作为并集费用口径对照（非搬 RTL）
- 端到端压缩激活本地化 ↔ Stage B same-port 驻留分母
- 负结果只停「照搬 SCNN 到 PSN」布局，不杀稀疏家族

## 杀门建议
计费无法对齐 same-port/same-state → 仅文献对照，不停 Stage B；若当 FireFly 引用 → 废本卡改绑。
