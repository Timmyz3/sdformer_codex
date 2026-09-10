# C1/C2 深度重构研究 · 2026-09-06

主交付：[十页研究与筛查报告](C1C2机制重构研究与筛查_20260906.pdf)。目标仍只有 TCAS-II；冻结二值 Motion C12 ep34。

本轮用户已授权推进研究和筛查。已完成文献对照、独立审阅及 CPU 实验；未进入生产 RTL、EDA 或改写主稿。这里所有结果均为 `PPA_ADMISSION=0`、`RTL_SPEEDUP_ADMISSION=0`。

当前开发顺序：

1. C1：原子集森林 + 深度优先穿线 + 2/4 个可丢弃父槽，缺父时直接从零精确重算当前行。5,640 tile 上额外源项发射槽为 1.4161% / 0.0188%，尚未证明能实际移除九宏或得到物理收益。
2. C2：最多两个原始 INT8 **权重**槽与延迟投递。64 源窗口下，随权重返回捕获的更新意图上界下降 FC1 19.4891%、FC2 23.7257%；每次捕获另收一次服务时为 8.0689% / 8.0261%。不能当周期、能量或 RTL 加速比。
3. 新增：PSN 保守输出判定及 Motion-XOR K 配对驻留，仅是待统计假说。

## 结果与范围

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `results/evidence_summary.json` | 报告主要数值，逐项重算汇总 | 当前摘要 |
| `results/c1_forest_recompute_5640.json` | 120 个既有 K phase 的全部空间 tile，12 个比较轴 | C1 主证据 |
| `results/c1_forest_recompute_480.json` | 扩展前的固定小队列 | 初筛 |
| `results/c1_differential_480_r2.json` | 对齐子集/差分的平局及选图规则 | 差分对照 |
| `results/c1_differential_480.json` | 未拆开零根平局与负边增量 | 已被 r2 取代，不单独归因 |
| `results/c1_online_anchors_480.json` | 2/4 个在线锚点的有限状态代价 | 上下文对照，不与旧 cycle 值直接求比 |
| `results/c2_selective_2880.json` | 1/2 个部分和并显式计 scatter | 第一轮负证据 |
| `results/c2_merge_2880.json` | 任意空输入位置的乐观合流 | 已由固定端口强对照补充 |
| `results/c2_merge_controls_2880.json` | fixed01 / flexible；部分和、singleton 类及联合 | 当前部分和对照 |
| `results/c2_source_staging_2880.json` | 任意单个源的两槽强对照，含重复模式及 unicast | C2 主证据 |

C1 的 5,640 tile 只有三个预选 K 分区，不是完整 5,184 万行。C2 是已有的 2,880 个 reduced B4 模板，第一批 1,920 + 续批 960，按输出 tile 加权，不是 all-token。C1 数值验证覆盖诊断整数全部比较轴；C2 数值事件回放只覆盖定向子集，不能把 2,880 意图计数覆盖说成全量数值执行。

后验审阅补充：`fixed01` 仍要求两槽到两个输入的选择网络，不是各槽单独固定连线；capture0 必须保留 capture-only 权重请求并支持同组双槽接收。已计入报告与脚本边界文字。原结果数值未修改。

## 复现

从本目录执行，Python 必须 3.12。以下使用本机已有 NumPy 的解释器；输出到新路径，保留本轮结果。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 scripts/screen_reuse.py c1 --output rerun/c1_differential_480_r2.json
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 scripts/screen_forest_recompute.py --all-spatial --output rerun/c1_forest_recompute_5640.json
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 scripts/screen_merge_controls.py --output rerun/c2_merge_controls_2880.json
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 scripts/screen_source_staging.py --output rerun/c2_source_staging_2880.json
```

依赖本地封存描述符与现有 CPU helper；输入路径和 SHA256 见 `provenance.json`。不读取 GPU 检查点、不启动 EDA。`scripts/summarize_evidence.py` 重算汇总并生成输入/脚本/结果快照；它不会授予硬件证据准入。

## 报告与研究记录

`report-source.md` 是报告内容源，`scripts/render_report.py` 渲染 PDF。渲染使用 ReportLab 与系统微软雅黑字体；本轮 ReportLab 仅安装在临时目录 `/tmp/codex_c1c2_report_vendor`，未修改全局 Python 环境。PDF 本身已嵌入字体。

`source-ledger.json` 区分作者全文、官方摘要和未获全文的出版商材料；`WORKLOG.md` 记录研究收敛原因。主报告已逐页渲染检查，PDF 共十页。本目录是研究包，不是 TCAS-II 投稿稿。
