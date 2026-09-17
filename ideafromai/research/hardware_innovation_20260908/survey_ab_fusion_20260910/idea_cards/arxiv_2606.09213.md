# arXiv:2606.09213 · SNN-MLIR NIR→MLIR→C

- uid/来源: `ARX-032`｜arxiv_2606.09213+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: SNN-MLIR: An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C
- 精读深度: 方法级（仅方法摘录窗口+补arxiv HTML §III–V）+依据：类型多态snn.*方言；CUBA-LIF/LIF态就地；功率2权重量化+Q12态+自动rescale；降到linalg/arith→无依赖C11；原摘录偏局限已补方法

## 可继承 A
NIR→snn-mlir→裸机C：类型多态神经元/突触op + 自动尺度对齐rescale + 单IR服务浮点仿真与i8部署——可审计SNN编译桥与手写量化对照（借入≠X）。

## 强对照 B
每后端手写部署；float/int分裂两套op；无NIR的框架锁死；仅Python运行时逐步仿真。

## 可差分 X线索
snn-mlir编译桥≠lifting X；工具链旁路，差分不在213–266×相对框架。

## 与 F1–F7 / Stage B 关系
F7弱相关（IR层打包/量化合同语言）。工具旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
嵌入式吞吐≠valid825；线性链/无卷积/batch=1限制；补HTML≠原gap窗全文。

## 可复用 idea 点
- 类型多态snn.cubalif作浮点/量化同op合同
- snn.rescale自动插层作尺度对齐模板
- NIR前端→linalg/arith→C11作可审计降级旁证
- 功率2权+Q12态作嵌入量化边
- 负结果只停该编译桥挂接

## 杀门建议
拓扑覆盖不足或量化误差不可接受挤占 Stage B → 保持旁路。
