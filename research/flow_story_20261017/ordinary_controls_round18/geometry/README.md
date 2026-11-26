# 原生 attention 的空间分组核验

2026-10-17。此处只读核验当前 checkpoint 所用代码的实际索引语义，未修改原模型、未运行 GPU、未读 GT。结果与源 SHA 见 [actual_v1.json](actual_v1.json)，可复查实现为 [audit_native_layout.py](audit_native_layout.py)。

原 `window_partition_v2` 先把 `(window,time)` 连续排列，再 `view(2,-1,9,9,C)`。因此后续 PSN 所见的两个时间槽，不必来自同一个原空间窗口。在目标 s2b2 的 `[B=1,T=10,H=30,W=40,C=384]` 输入上，padding 后为36×45，packed形状为 `[2,100,9,9,384]`；packed列0的局部(0,0)两个PSN输入分别来自原坐标 `(t,h,w)=(0,0,0)` 和 `(4,18,0)`。不能只凭配置名 `[2,9,9]` 假定其独立空间计算单元。

核验从原源 AST 提取 partition/reverse，在四种形状上与独立 NumPy 实现和逐标量索引反解比较，全部一致，且 partition→reverse 的无计算往返为恒等。再按原 q/k reshape、PSN、q归约、乘法、输出重排和proj的顺序传播潜在输入依赖。当前目标checkpoint的相关PSN时间权重、q/k/proj通道权重均非零。故意不加入 current BN 后，20个空间tile的无向潜在依赖图仍连通。原 current BN 还存在额外全域统计依赖。

**这个图不是激活敏感性测量，也不是“必须同时删20块”的证明。**它没有求解最小可删集合、条件常量传播或生产者调度；不能把连通性直接换算成周期。有效结论仅是：原生attention的可取消服务必须按真实数据依赖重新导出，名义空间窗口和最终输出mask不足以证明上游计算可以取消。硬件代理已独立读审索引对应，见 [硬件范围与复核](../hardware/README.md)。

本轮没有把这一发现定义为源代码错误，也没有为预训练checkpoint“修正”重排。R17/R18只干预attention之后的MLP，所以本核验不否定其已有质量数据。它约束的是下一步宣称整窗attention＋MLP收益的范围。
