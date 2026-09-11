# arXiv:2306.06493 · RAMAN

- uid/来源: `ARX-064`｜arxiv_2306.06493+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: RAMAN: A Re-configurable and Sparse tinyML Accelerator for Inference on Edge
- 精读深度: 方法级（仅方法摘录窗口）+依据：双稀疏（W+IA）tinyML；IA/OA 同空间重叠降峰值激活存储≤50%；可重构 NoC；全片上；MobileNetV1/DS-CNN；摘录偏 Intro/Related+贡献摘要，完整 PE/数据流§可能截断

## 可继承 A
双稀疏 + 激活峰值重叠存储 + 可重构 NoC——边侧稀疏推理供数与存储周转对照（借入≠X）。

## 强对照 B
单操作数稀疏（仅 W 或仅 IA）；IA/OA 双缓冲翻倍峰值；固定拓扑不可重配。

## 可差分 X线索
RAMAN≠lifting X；借重叠存储与双稀疏，勿搬 GOp/s/W。

## 与 F1–F7 / Stage B 关系
F1/F5相关（表示/稀疏与激活驻留）。第二队列旁证。不抢 Stage B。f_candidates含F1,F5。

## 不可搬用边界
FPGA LUT/能效≠AEE；摘录偏相关工作禁止虚报已读完整微架构；tinyML≠残差光流链。

## 可复用 idea 点
- IA∩OA 重叠作 F5 峰值内存合同
- 双稀疏前端配对作利用率旁证
- 可重构 NoC 作层切换边界类比 F7 弱
- 负结果只停 RAMAN 名替换

## 杀门建议
无同端口存储/稀疏改善或映射不上 → 停类比，不杀双稀疏家族。
