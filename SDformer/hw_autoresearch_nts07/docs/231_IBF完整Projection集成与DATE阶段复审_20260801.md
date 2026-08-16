# IBF 完整 Projection 集成与 DATE 阶段复审

本轮完整结果由以下脚本生成：

```bash
python3 scripts/summarize_single_head_ibf_integration.py
```

权威产物：

- `results/single_head_ibf_integration_20260801/report.md`
- `results/single_head_ibf_integration_20260801/report.json`

IBF 已从叶 accumulator 接入同一个 term-to-final 单头 projection 顶层，并在默认
参数不改变原 RMW 路径的前提下完成 Icarus、Verilator 动态 SVA、Yosys 结构检查和
同顶层 Nangate45 无约束映射。详细证据边界与 DATE 阶段复审以权威报告为准。
