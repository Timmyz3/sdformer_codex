# 规则空间合并 MLP 探针

固定9×9完整T10 tile 内做空间均值、原生 MLP 和 branch 复制；提供 stride3 与 whole-tile 两档，固定 fine/coarse mask，无训练或运动 gate。普通机制适配，不是官方 DGE/ToMe 复现。

- [机制与筛选合同](STRUCTURED_MERGE_REPORT.md)
- [独立语义和代码审查](INDEPENDENT_SEMANTICS_REVIEW.md)
- [可调用模块](structured_merge_mlp.py) 与 [受控筛选入口](run_structured_merge_screen.py)
- [模块 CPU 结果](cpu_results.json) 与 [runner CPU 结果](runner_cpu_results.json)
- [冻结源码与检查 SHA](freeze_v1.json)

CPU 共13项检查通过。2026-10-14 固定67次 GPU 筛选已完成，见[GPU结果与边界](GPU_SCREEN_REPORT.md)和[原始结果](screen_v1/results.json)。执行源码在本轮保持冻结；下述机制合同的15:46状态保留为历史。

CPU 检查解释器为 `/opt/anaconda3/envs/pytorch310_cpu/bin/python`；两个测试脚本均屏蔽 CUDA、禁止旧目录字节码写入，仅在本目录输出 receipt。筛选入口需要明确 `--execute` 和既有 GPU 锁，默认不会启动模型推理。
