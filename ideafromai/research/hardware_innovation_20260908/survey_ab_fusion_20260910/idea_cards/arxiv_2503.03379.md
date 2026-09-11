# arXiv:2503.03379 · Prosperity

- uid/来源: `MAIN-R001`｜arxiv_2503.03379+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: Prosperity: Accelerating Spiking Neural Networks via Product Sparsity
- 精读深度: 方法级（仅方法摘录窗口）+依据：ProSparsity 组合子串复用（exact/partial match）；线性复杂度启发式识别；流水线重叠识别与 GeMM；算法无关 spiking GeMM；摘录止于 Fig.2/背景，完整 RTL/森林细节可能截断

## 可继承 A
Product Sparsity（行间公共二进制子组合只算一次再复用）+ 运行时识别/调度流水线——作「多消费者共享部分和/模式复用」强底座与 F3 联合图对照（借入≠X；本地已有 Prosperity 探针负结果边界）。

## 强对照 B
仅 bit-sparsity 零跳过（PTB/Stellar 式）；稠密 Spiking GeMM；无执行序优化的朴素行扫描。

## 可差分 X线索
ProSparsity/部分和复用本身非本地标题级 X；差分须落到 lifting 源 DAG 上的公共虚节点/部分 lane（F3），而非再扫整 W 产品 mask 或复述 0.09% 级 mask 目标。

## 与 F1–F7 / Stage B 关系
F1/F3/F7 相关；第二队列底座。不抢 Stage B（Stage B 仍是无 Prosperity 的 pure schedule compare）。

## 不可搬用边界
勿搬 7.4×/193× 等原作 PPA；摘录未含完整架构§细部；本地 G4/产品费用负结果只停已列布局，不杀家族；仅方法摘录窗口。

## 可复用 idea 点
- exact/partial match 作共享部分和语义对照
- 执行序对复用率的敏感性→对照 lifting 因子组打包序（F7）
- 识别与计算重叠作「元数据开销须被掩盖」杀门叙事
- 与已有 Prosperity 三探针负结果联读，未试联合图挂源 DAG 仍保留

## 杀门建议
挂到 lifting 源 DAG 后同端口计划服务不优于 da4ml CSE+普通部分 lane，或相对无图端点净省过薄 → 停该联合图布局，保留家族。
