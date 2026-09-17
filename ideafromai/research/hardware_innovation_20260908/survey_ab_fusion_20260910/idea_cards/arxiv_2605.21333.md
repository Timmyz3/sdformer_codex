# arXiv:2605.21333 · SymbolicLight V1 Spike-Gated Dual-Path LM

- uid/来源: `ARX-038`｜arxiv_2605.21333+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: SymbolicLight V1: Spike-Gated Dual-Path Language Modeling at High Encoder Spike Sparsity
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3 SpikeEncoder LIF；Dual-Path SparseTCAM(衰减态+窗局部注意力)；门控融合；context MLP；chunk=64；~90%零尖峰探针；完整训练细节可能截断

## 可继承 A
尖峰门控双路径LM：LIF二值门控衰减/FFN + 连续残差上流式窗注意力与context MLP——高编码器稀疏与纯SNN/纯稠密对照（借入≠X）。

## 强对照 B
纯SpikeGPT/RWKV递推；全二值残差；BitNet权重量化无尖峰门；无辅助CE深监督。

## 可差分 X线索
SymbolicLight双路径≠lifting X；语言模型旁路，勿搬PPL当净服务%。

## 与 F1–F7 / Stage B 关系
F2/F7相关（有损共同完成/稀疏门控路径、打包友好投影）。表示旁路。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
PPL/零尖峰探针≠valid825；仅摘录窗；混合模型≠纯事件合同；位置掩码实测近失效。

## 可复用 idea 点
- s门控衰减态∥c连续注意力作双路径合同
- 学习α半衰期作短程记忆模板
- chunk递推膜态交接作长序列旁证
- AuxCE深监督作训练稳定边
- 负结果只停该LM混合挂接

## 杀门建议
质量差距大或稀疏不可兑现挤占 Stage B → 保持旁路。
