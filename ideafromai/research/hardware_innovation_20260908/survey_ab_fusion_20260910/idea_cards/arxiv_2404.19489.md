# arXiv:2404.19489 · EvGNN

- uid/来源: `MAIN-R230`｜arxiv_2404.19489+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: EvGNN: An Event-driven Graph Neural Network Accelerator for Edge Vision
- 精读深度: 方法级（仅方法摘录窗口）+依据：有向边因果局部子图；时空解耦棱柱邻域搜索+级联事件队列；层并行复用邻域；KV260部署指标；§III双栏混排完整微架构可能截断

## 可继承 A
事件驱动GNN：新事件只更新有向局部子图（不存全边）+棱柱邻域搜索引擎+层并行——边侧微秒级事件图更新与内存足迹对照（借入≠X）。

## 强对照 B
静态整段构图再推理；无向全图存边；AEGNN类高内存；纯CNN事件帧。

## 可差分 X线索
EvGNN图加速≠lifting X；边侧视觉前端旁路，勿搬延迟/片上存储当净服务%。

## 与 F1–F7 / Stage B 关系
F1弱相关（事件图表示选择）。旁路/前端。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
N-CARS准确率/平台指标≠valid825；摘录偏背景+贡献，禁止虚报已读完整微架构；量化8b≠本地阈值声明。

## 可复用 idea 点
- 有向因果子图免存边作内存合同
- 棱柱时空解耦邻域搜索作事件队列旁证
- 层并行复用邻域特征作更新延迟模板
- 与AEGNN/事件GNN成簇 | 负结果只停EvGNN前端替换

## 杀门建议
前端替换无增益或挤占 Stage B → 保持旁路。
