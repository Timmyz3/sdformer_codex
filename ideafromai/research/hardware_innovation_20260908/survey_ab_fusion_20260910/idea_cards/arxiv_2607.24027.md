# arXiv:2607.24027 · Sol-Attn On-the-Fly Sparse Attention

- uid/来源: `ARX-120`｜arxiv_2607.24027+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification
- 精读深度: 方法级（仅方法摘录窗口）+依据：块代理分块阈值；嵌套外循环路由+内循环精确稀疏；零阶Taylor近似校正；分数瓦片复用；Algo1在线softmax；完整视频评测可能截断

## 可继承 A
视频生成即时稀疏注意：分块流式阈值选块 + 代理分复用作跳过块零阶近似校正 + 嵌套在线softmax核——相对物化top-k/top-p路由图的系统对照（借入≠X）。

## 强对照 B
物化全行代理图再全局top-k/p；离线路由索引写HBM；无近似校正的硬丢弃。

## 可差分 X线索
Sol-Attn视频注意≠lifting X；生成/内核旁路，勿搬墙钟加速比当净服务%。

## 与 F1–F7 / Stage B 关系
F2/F7相关（在线打包阈值、稀疏注意）。内核旁路。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
VBench/PSNR≠valid825；仅方法窗；在线路由≠lifting源活动合同。

## 可复用 idea 点
- 分块流式阈值作无物化路由合同
- 代理分复用≈零阶校正作跳块模板
- 外路由∥内精确嵌套作在线softmax旁证
- 路由开销并入近似GEMM作带宽边
- 负结果只停该即时稀疏注意挂接

## 杀门建议
质量掉点或与Stage B同分母冲突 → 保持旁路。
