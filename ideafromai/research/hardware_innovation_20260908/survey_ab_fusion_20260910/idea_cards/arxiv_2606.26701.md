# arXiv:2606.26701 · SegFold Fine-Grained Dynamic SpGEMM

- uid/来源: `ARX-060`｜arxiv_2606.26701+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
- 精读深度: 方法级（仅方法摘录窗口+补arxiv HTML提纲）+依据：Segment数据流；SelectA行交∩重排K；SegmentBC即时重分配；IPM LUT；空间折叠；向量组播；相对Spada/静态；RTL敏感度

## 可继承 A
细粒度动态SpGEMM：SelectA行交∩ + SegmentBC即时重分 + IPM LUT映射 + 空间折叠——不规则稀疏下动态数据流与静态/Spada对照（借入≠X）。

## 强对照 B
静态数据流；仅tile级适应的Spada；零偏移映射；无动态重映射的固定PE分配。

## 可差分 X线索
SegFold SpGEMM≠lifting X；稀疏算子旁路，勿搬SuiteSparse加速比当净服务%。

## 与 F1–F7 / Stage B 关系
F5相关（细粒度动态供数/折叠）。算子旁路。不抢 Stage B。f_candidates含F5。

## 不可搬用边界
SpGEMM cycle/MAC≠valid825；仅方法窗；动态调度≠lifting并集调度合同。

## 可复用 idea 点
- SelectA行交∩重排K作复用合同
- SegmentBC即时重分作负载模板
- IPM LUT≈理想映射旁证
- 空间折叠+向量组播作带宽边
- 负结果只停该动态数据流挂接

## 杀门建议
极端稠密行压垮行级分配或调度税过大挤占 Stage B → 保持旁路。
