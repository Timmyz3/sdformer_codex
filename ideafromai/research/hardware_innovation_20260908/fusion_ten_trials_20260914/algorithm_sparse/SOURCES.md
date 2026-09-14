以下primary实际读到全文；venue/date与code状态逐项记录。对A的归属用于约束自评，不把论文的其他模型收益搬到本网络。

| 来源 | 日期/venue核对 | 精读位置 | 已有A与本次边界 | code核对 |
|---|---|---|---|---|
| [Bishop全文](https://arxiv.org/html/2505.12281v1) | arXiv 2025-05-18；正文标ISCA2025、DOI10.1145/3695053.3731063 | §3.2,4.1,5.1/Eqs9–10 | TTB结构组与误差约束剪枝是A。本网完成后的signed latent不是二值Q/K，I24校准proxy没有继承其attention误差界 | 本次primary+定向搜索未定位官方实现；不是断言不存在 |
| [Phi全文](https://arxiv.org/html/2505.10909v1) | arXiv 2025-05-16；正文ISCA2025、DOI10.1145/3695053.3731035 | §3.1–3.3、Alg1、§4.2 | pattern+weight结果与双向残差、训练集聚类均为A。本网仅保留一个signed latent残差是有损端点，未做其PAFT | 本次未定位官方实现；全文可得 |
| [LUT-DLA全文](https://arxiv.org/html/2501.10658v1) | arXiv 2025-01-18；[HPCA2025官方Session5C](https://hpca-conf.org/2025/main-program/)确认 | §IV-A/B，§V-2，§VI-B | encoder距离和LUT驻留已经覆盖；L1取代L2降低硬件成本也是A。本次Q1照付、仅Q2用原型+单rank修正 | 本次未定位官方实现；全文可得 |
| [DeltaCNN全文](https://arxiv.org/html/2203.03996v2) | CVPR2022；arXiv初版2022-03-08，读到v2 2023-09-02 | §3.1–3.3、§9.1；[官方实现](https://github.com/facebookresearch/DeltaCNN)§3.1/3.4 | 缓存、阈值丢弃更新、保留累计状态及稀疏Δ传播均为A。本次是同帧T10完成latent的局部线性接口，未跳后继PSN或沿用跨帧flow | 官方GitHub可访问；本次未运行其代码 |

对三项均不能以“首次组剪枝”“首次原型+残差”“首次阈值时间保持”为标题。只记录固定真实函数下，经训练集选择、完整RTL与真实光流质量共同检验的接口增量。
