# arXiv:2607.05445 · BitFair

- uid/来源: `MAIN-R129`｜arxiv_2607.05445+本地excerpt（`p0_excerpt_batches/batch_04.json`）+方法窗补读 §III–IV
- 题名: BitFair: A 12-nm Bit-Serial CNN Accelerator with Learnable Early Termination and Adaptive Bit Ordering for Ultra-Low-Power XR Vision
- 精读深度: 方法级（方法摘录窗口+§III-A/B/C、§IV-A/D）+依据：bit-plane 跨输入归约后按学习 θ^l 早停；软阈值+温度退火+ℒ_bit 生存正则；Algo1 贪心层内位序；16×16 OS PE 本地比较与 FSM 汇聚终止信号

## 可继承 A
「学习提前终止 + 自适应位序 + PE/组级终止汇聚」条件计算底座：按 bit-plane 完整输入归约，Pk≤θ^l 则预测 ReLU 置零并跳剩余位；软门 G_k=σ(-(P_k-θ)/T) 与前缀生存 S_k；FSM 汇聚 terminate 抑制后续取数——可作 F2「有损共同完成/组关闭」强普通对照（借入≠X）。

## 强对照 B
无早停的 vanilla bit-serial；BitSET 静态 BN/偏置阈值+刚性 MSB-first+定制编码；SnaPEA/PredictiveNet 值级或固定 MSB 预测；SparseInfer 式预测后跳权但无学习位序。

## 可差分 X线索
单输出 ReLU-零预测≠ lifting 完整 T10 门字/共享请求并集共同完成；差分须落到「同组关闭逻辑下换独立误差→并集费用目标」且过 AEE，而非复述可学习 θ 或 ABO。

## 与 F1–F7 / Stage B 关系
F2 第二队列强对照（学习终止×组结束控制）。旁路主岛。不抢 Stage B / schedule_compare。

## 不可搬用边界
CNN ReLU bit-serial≠SNN/非因果 T10；勿搬 4.0–22.1×/0.07 pJ/SOP/12nm PPA；有损提前置零≠证书后缀界；仅方法级，未迁 RTL。

## 可复用 idea 点
- 生存概率 S_k / ℒ_bit 作「预期处理位分数」费用正则模板 → 对照并集未决消费者计费
- PE 本地终止 + FSM 汇聚作 MP2 组 accept/continue 的普通控制骨架
- ABO 贪心位序作「先算信息量大的前缀」对照，非标题
- 负结果只停「BitFair 式单点预测挂 r1」布局，不杀条件计算家族

## 杀门建议
同组关闭逻辑下相对 SparseInfer/固定θ 无并集费用优势或破 AEE → 停该预测目标，保留 bit-serial 早停文献对照。
