# arXiv:2606.30039 · Mega 22nm Conv-SNN Accelerator

- uid/来源: `ARX-167`｜arxiv_2606.30039+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: Mega: A 22 nm Convolutional Spiking Neural Network Accelerator Achieving 0.375 pJ/SOP for Efficient Edge Vision
- 精读深度: 方法级（仅方法摘录窗口）+依据：§II九簇3×3事件卷积；九库交错膜态；288并行CU四段流水+RAW转发；dense spike map→地址streamer；统一TCDM；LIF行阈；完整PPA可能截断

## 可继承 A
卷积SNN加速：九簇核偏移并行事件ΔV + 九双口SRAM交错膜态(32ch/字) + dense spike map转地址 + 统一尖峰/态/权存储——相对AER与输出中心稠密窗的供数/并行对照（借入≠X）。

## 强对照 B
高稀疏才有效的AER；输出中心滑窗取零；尖峰/膜态/权分离存储；无核偏移并行的串行事件更新。

## 可差分 X线索
Mega边缘SNN加速≠lifting X；加速器旁路，勿搬0.375pJ/SOP当净服务%。

## 与 F1–F7 / Stage B 关系
加速器旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
硅测PPA≠valid825；仅方法窗；288SOP/周期≠lifting并集合同。

## 可复用 idea 点
- 事件ΔV九偏移并行作核级复用合同
- 九库交错膜态+32ch字作同窗并发读模板
- dense map→地址streamer作相对AER旁证
- CU四段流水+RAW转发作无停顿边
- 负结果只停该事件卷积挂接

## 杀门建议
层间稀疏度不匹配或统一存储争用挤占 Stage B → 保持旁路。
