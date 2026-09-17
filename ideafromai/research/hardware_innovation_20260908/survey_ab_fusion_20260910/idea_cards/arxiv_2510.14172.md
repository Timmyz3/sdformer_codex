# arXiv:2510.14172 · DIAMOND

- uid/来源: `ARX-061`｜arxiv_2510.14172+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Systolic Array Acceleration of Diagonal-Optimized Sparse-Sparse Matrix Multiplication for Efficient Quantum Circuit Simulation
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III对角分解+offset加性；§IV DPE比较转发；对角累加器；二维对角分块；DiaQ存储；完整PPA表可能截断

## 可继承 A
对角格式SpMSpM：DPE按索引对齐比较/转发 + 对角累加器收Minkowski和——结构化对角稀疏乘法与局部复用合同对照（借入≠X）。

## 强对照 B
CSR/COO通用SpMSpM；SIGMA/外积/Gustavson无对角特化；DIA强制等长填充；稠密GEMM。

## 可差分 X线索
DIAMOND/量子对角核≠lifting X；同构稀疏核旁路，差分不在量子仿真×速%。

## 与 F1–F7 / Stage B 关系
F1/F5相关（结构化源字活动、对角中间量驻留/分块周转）。可挂第二队列线索。不抢 Stage B。f_candidates含F1,F5。

## 不可搬用边界
量子仿真加速比≠valid825；仅摘录窗；对角规律≠lifting任意T10稀疏图。

## 可复用 idea 点
- offset加性dC=dA+dB作对角乘积合同
- DPE比较jA≟iB选择性转发作不规则对齐模板
- 对角累加器按输出对角并行收集作局部复用旁证
- 二维对角分块界住网格作驻留/周转语言 | 与Gustavson/SIGMA成簇；负结果只停对角核替换

## 杀门建议
对角假设不成立或索引税吃掉增益挤占 Stage B → 保持旁路/降为F1线索。
