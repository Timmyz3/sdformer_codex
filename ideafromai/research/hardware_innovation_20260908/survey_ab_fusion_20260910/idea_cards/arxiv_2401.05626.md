# arXiv:2401.05626 · ESDA

- uid/来源: `MAIN-R236`｜arxiv_2401.05626+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPGA
- 精读深度: 方法级（仅方法摘录窗口）+依据：统一token-feature接口；子流形稀疏卷积；可组合模块；完整端到端表可能截断

## 可继承 A
可组合稀疏数据流：坐标token+特征同流、子流形卷积抑膨胀、编译期并行因子——事件稀疏空间供数与模板化对接对照（借入≠X）。

## 强对照 B
标准卷积致密化中间特征；稠密帧CNN全图扫描；不可组合的固定硬件；无token的纯特征流。

## 可差分 X线索
ESDA/子流形≠lifting数字残差链X；借稀疏数据流接口，勿搬平台吞吐数字。

## 与 F1–F7 / Stage B 关系
F7相关（稀疏坐标打包/模块级联边界）。第二队列旁证。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
平台分类吞吐/资源≠valid825/same-port%；仅摘录窗；子流形限制≠通用稠密光流出口。

## 可复用 idea 点
- token坐标与end标志+ravel序作稀疏流合同
- stride=1子流形输入输出同位可复用队列旁证
- 可组合模块级联作按模型拼数据流模板
- 负结果只停ESDA名替换数字链

## 杀门建议
无同端口稀疏服务改善或映射不上残差链 → 停类比，不杀稀疏数据流家族。
