# 规则合并的有限组合范围筛选

**完成状态（2026-10-14）：** 59次GPU完整前向已完成，结果见[组合筛选报告](RESULTS_REPORT.md)，实际服务核验见[追加审计](SERVICE_SCOPE_ADDENDUM.md)。[TCAS-II独立内部评审](../../reviews/TCASII_ROUND4_INDEPENDENT.md)已完成；stage02冻结为探索期普通底座候选，不代表动态A贡献或净硬件收益。下文保留执行前固定的协议。

本轮是阅读 ROUND3 后追加的探索：直接测量组合范围的质量，避免把单独 stage0 和 stage2 的质量变化相加。它使用同一固定 dev8，是普通静态范围控制；不是独立确认，也不构成 A 动态方法贡献。

固定设置为135930步 checkpoint、MLP_CMA176、完整T10、stride3、无训练、无 gate。预定7臂如下，八帧共56次前向；首帧三个 coarse 臂另各做一次 dual，共59次。

| 臂 | 替换范围 | MLP 数 |
|---|---|---:|
| native_dense | 原图 | 0 |
| fine_stage02 和 coarse_stage02 | stage0、stage2 | 8 |
| fine_stage23 和 coarse_stage23 | stage2、stage3 | 8 |
| fine_stage023 和 coarse_stage023 | stage0、stage2、stage3 | 10 |

每个 coarse 只和自己的 matched tiled fine 比较。原生到各 fine、三个 fine 之间的算术漂移另外记录。首帧 actual/dual 输入 SHA、每层 branch 和全部 flow 必须逐值相同；全12层 FC hooks 核查真实行数。全部既有质量分区、逐帧及序列汇总、预测 NPZ/SHA、原参数与 BN 不变检查继续保留。

[新 runner](run_scope_extension.py) 从冻结父 runner `f23667f986ce6f83d6a51a7056ff25e82432d05def3bbd174530f17afc2551bf` 派生，直接加载未改动模块 `26454de4009e20d45edc186eaf255ef1df918b2250f270a239713efb90fe72f0`。前向审计提取为 `forward_probe`，供新测试直接执行；新版本将外部 FC hook 注册放入 `try/finally`，并测试部分注册失败时清理。没有修改父目录六个冻结文件。

[新 CPU receipt](runner_cpu_results.json) 绑定新 runner、新测试和冻结模块，8项检查通过。测试用12个小型原生 MLP 合成组合，覆盖三组范围的 dispatch、actual/dual、多层真实行数、原生路径、输入/参数不变、异常恢复、hook注册失败、身份及来源检查。它没有运行完整光流模型、GPU或远端任务。CPU解释器为 `/opt/anaconda3/envs/pytorch310_cpu/bin/python`。

筛选仍默认 preflight、显式 `--execute`、共享旧 GPU 锁、输出新目录且墙钟上限30分钟。新增两个必填参数，明确绑定本轮选择依据：

```bash
--prior-screen-results <ROUND3_results.json>
--prior-screen-results-sha256 70208e3c0378bed2127d41e47745a373f1c02aad49b07da2a79af0ad9be1e9fb
```

其余参数沿用父 runner；`--module-checks` 默认父目录 `cpu_results.json`，`--runner-checks` 默认本目录 `runner_cpu_results.json`。部署需保持本目录位于已冻结 `structured_merge` 目录下一层。新前向会验证 ROUND3 完成状态、来源、数值图和 dev8顺序；只从旧 mask bank 读取身份，不读取旧 mask/u SHA，不基于结果重新选帧。

源码与检查 SHA 见 [freeze_v1.json](freeze_v1.json)。CPU检查、远端preflight及GPU执行均已完成；完成状态和独立评审见文首链接。
