# arXiv:2607.25504 · Ventaglio Sparse Tensor RVV

- uid/来源: `ARX-059`｜arxiv_2607.25504+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: At-the-Roofline Sparse Tensor Contractions on Vector Processors for Transformer Inference
- 精读深度: 方法级（原摘录仅参考文献；补arxiv HTML全文方法）+依据：Gustavson数据流；Ventaglio多通道SCM gather/scatter；vfxmacc/vlx/post-inc ISA；Spatz集成；bitmap/N:M；GVSoC规模化；必须以HTML为准

## 可继承 A
向量机稀疏张量收缩：Gustavson激活触发∥权元数据索引累加 + Ventaglio多通道扩展态存储器 + 融合索引MAC/地址自增ISA——相对软件解码+L1索引访存的屋顶线对照（借入≠X）。

## 强对照 B
软件元数据解码+L1索引gather/scatter；仅矩阵瓦片的IndexMAC/SCG；无激活稀疏的2:4 Tensor Core。

## 可差分 X线索
Ventaglio ISA≠lifting X；RVV扩展旁路，勿搬6.9–7.4×核加速当净服务%。

## 与 F1–F7 / Stage B 关系
F1/F5相关（细粒度双稀疏执行、供数/累加组织）。ISA旁路。不抢 Stage B。f_candidates含F1,F5。

## 不可搬用边界
Spatz/GVSoC周期≠valid825；摘录失效已补HTML；索引累加≠lifting并集调度合同。

## 可复用 idea 点
- Gustavson激活×权元数据作双稀疏合同
- 多通道SCM扩展态作VRF解耦模板
- vfxmacc+post-inc作索引MAC旁证
- CSR选N:M/bitmap datapath作格式边
- 负结果只停该向量稀疏挂接

## 杀门建议
中等稀疏屋顶线不可达或挤占 Stage B → 保持旁路。
