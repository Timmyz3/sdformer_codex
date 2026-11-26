# H0/H1 纯 CPU 分析准备

先读 [固定协议](ANALYSIS_PROTOCOL.md)。`analysis_core.py` 无 I/O、无模型依赖、无真实数据加载；`test_analysis_cpu.py` 只用生成的假指标。尚无实际 H0/H1 质量结论，也没有完成真实源/原数组审计。

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_analysis_cpu.py
```

纯函数接口：

```python
analyze_h0(h0_frames, h8_members)
analyze_h1(h0_frames, h1_frames, h8_members)
```

`members` 为固定顺序 8 条 `{target_id, sequence}`，每序列仅一帧。每个 frame 有对应身份和 `arms`；arm 直接映射全部 16 区域，每个区域为 `{valid_pixels, error_sum, AEE}`。H0 的 arm 名固定 native/FF/CC/P/F12/T8/matched_N/matched_F。H1 有全部 78 个 `k1_r0c0` / `k2_r0c0_r0c1` 格式的动作；额外 `ordinary_scores` 为 row-major 12 个 `{tile, 7个固定score}`。

这些是**已审计指标的内存输入格式**，不接受分析函数本身验证了 SHA/实际前向的暗示。随后 adapter 从冻结结果格式抽取指标、先校全部身份/source/前缀/状态/数据支持并独立复算原始预测后，才能调用。结果已有数据的一部分来自未来 exporter，不代表现在已执行；不在本准备中补建训练、GPU 队列或新的数据选择。

H0 返回同 W fine 保护、四参考动作门、全部 16 区域/9 作用域摘要和有符号差。H1 返回全部 78 端点同口径摘要、各普通方法真实 pair、固定 pair 控制、两种有限 oracle、singleton top2 实际端点、全部交互以及 1% 投入筛查。空区域、零参考、非单调及无可行集合均显式处理。

当前 source freeze 是方法封存，不是实际数据已通过或许可门。正式 R13 的原数组独立复算仍使用原来已冻结的 reducer，优先执行，不受本后续准备影响。
