# arXiv:2510.26614 · Spiking Patches

- uid/来源: `MAIN-R214`｜arxiv_2510.26614+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III 把 patch 当脉冲神经元（电位累加、阈值出 token、可选不应期丢事件）；保异步与空间稀疏；O(n) tokenizer

## 可继承 A
事件→token 的脉冲式时空分词（patch 电位/阈值/不应期）——前端表示层保留异步稀疏，可作事件光流输入表示对照（借入≠X）。

## 强对照 B
固定时间窗帧化/体素；先帧化再筛活跃 patch；破坏异步的 ViT 事件管线。

## 可差分 X线索
分词器≠ lifting T10 X；本地主岛在 r1 残差链执行，不在 tokenizer。

## 与 F1–F7 / Stage B 关系
旁路（前端表示）；与事件光流输入有弱相关。不抢 Stage B。

## 不可搬用边界
手势/检测精度与 token 缩减≠AEE 合同；仅方法摘录窗口。

## 可复用 idea 点
- patch-as-neuron 作异步稀疏 token 模板
- 不应期 T 作序列长度旋钮对照 Transformer 二次费用
- 相对帧/体素的输入缩减作前端基线
- 负结果只停「替换本地事件前端」布局

## 杀门建议
替换前端后精度/延迟无增益或破坏现有合同 → 停该分词挂载。
