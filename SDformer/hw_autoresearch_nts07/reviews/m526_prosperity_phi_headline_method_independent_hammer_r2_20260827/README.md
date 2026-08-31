# M526 Prosperity / Phi headline 方法独立打铁 r2

日期：2026-08-27  
评审类型：对 r1 五项 P1 的增量关闭复核。  
评审边界：只读检查修订后的 M526 文档、r2 审计脚本与封存输出；校验官方仓库身份、结果双 seal，并对 exact module 做两个不落盘的负向 fail-closed probe。未修改被审文件，未运行 EDA，未修改 `docs/359_DATE终局冻结_20260813.md`。

## 结论

评分 **98/100**，`P0=0 / P1=0 / P2=2`。

裁决：`GO__FIVE_P1_CLOSED__METHOD_ADMITTED__H67_RESULTS_STILL_OPEN`。

r1 的五项 P1 均已在正文和机器证据的实质口径上闭合：

1. Prosperity `7.4x` 已明确限定为 16-workload、supported-operator-scope 的 PTB/Prosperity runtime headline，不再称严格全网、全算子、同面积 `7.4x`；
2. r2 同时报 arithmetic `7.461107x`、geomean `7.313885x`、ratio-of-sums `6.731836x`，并明确 `paper_aggregation_convention_identified=false`；
3. baseline ladder 已拆为 `B3=K1x8 replicated equal-service baseline` 与 `C2=typed K8 shared-state candidate`；
4. effective GOP/s 已改为跨配置固定的 dense-equivalent 和 original useful-nonzero numerator，architecture-reduced 数量只允许称 executed additions；
5. B0/B1/Ours 已明确只能称 `iso-lane`，主表要求 area-normalized throughput，并保留 iso-service B3。

因此，M526 可以作为 H67 性能实验与论文表格的**方法合同**使用。该 GO 不准入任何 H67 system speedup、energy/frame、paper PPA，也不把外部 Prosperity/Phi/FireFly-T 数字变成 ours。

## 五项 P1 关闭证据

| r1 finding | r2 closure | 裁定 |
|---|---|---|
| P1-01 supported-operator scope | 文档首段和 2.1 节同时限定 transformer baseline 只覆盖所支持 linear layers；A100 `1.79x` 单独标 end-to-end | `CLOSED` |
| P1-02 aggregation convention | script/JSON 增加 ratio-of-sums、三种距离，显式标记 convention unknown；正文规定多序列同时报告四类统计 | `CLOSED` |
| P1-03 K8/K1x8 混行 | baseline ladder 拆成 B3 K1x8 与 C2 typed K8；Ours 明确包含 typed-K8 C2 | `CLOSED` |
| P1-04 variable numerator | 正文禁止使用 architecture-dependent `num_ops` 做跨配置 throughput；定义两个固定 numerator 与 physical issue rate | `CLOSED` |
| P1-05 iso-lane 非 iso-area | 正文明确同 lane/SRAM/BW 仅为 iso-lane，主表增加 Fairness、Area、GOP/s/mm2，要求 area-normalized companion | `CLOSED` |

## r2 artifact 与 fail-closed 复核

- 官方 Prosperity repo 当前 HEAD：`6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`；`git status --porcelain` 为空。
- 脚本在生成输出前动态检查 repo 目录非 symlink、HEAD 精确相等、工作树 clean、workbook 非 symlink 且 SHA 精确相等，并拒绝覆盖 canonical output。
- 对 exact imported module 的负向 probe：伪造 HEAD 时得到 `official repository HEAD drift`；伪造 dirty status 时得到 `official repository is dirty`；两次均未创建输出。
- r2 `SHA256SUMS` 校验 `REPORT.md` 和 JSON 通过；`SHA256SUMS.seal.sha256` 对 manifest 的外层校验通过。
- r2 JSON 将 `prosperity_artifact_recomputed=true` 与 `paper_text_verified_by_this_script=false`、`phi_or_firefly_artifact_recomputed=false`、`h67_run=false` 分开，关闭了 r1 的证据层混淆建议。
- `python3 -m py_compile` 与目标范围 `git diff --check` 均通过。

## 复算一致性

| 指标 | r1 独立重算 | r2 JSON | 结果 |
|---|---:|---:|---|
| PTB/Prosperity arithmetic mean | `7.461106560018359x` | `7.461106560018360x` | match |
| geometric mean | `7.313884876012105x` | `7.313884876012107x` | 浮点尾差内 match |
| ratio of summed runtimes | `6.731836241361511x` | `6.731836241361512x` | 浮点尾差内 match |
| min / max | `4.975778700180036x / 11.466771758512127x` | 相同 | match |
| paper convention identified | unknown | `false` | match |

## Residual P2

### M526-R2-P2-01 — 摘要清单第 4 项仍把 strongest equal-bandwidth 对照简称为 K8

文档第 20 行写“保留最强等带宽 K8 对照”，而正式 ladder 已正确规定 strongest replicated baseline 是 K1x8、typed K8 是 candidate。详细合同不会误导 simulator，但摘要句可能在复制到论文计划时恢复旧混称。

建议：将该句改为“保留最强等服务 K1x8 baseline，并单列 typed K8 candidate”。这是文字一致性问题，不重新打开 P1-03。

### M526-R2-P2-02 — P0 多序列输出清单漏写 ratio-of-summed-runtimes

正文 5.1 节已强制 arithmetic/geomean/ratio-of-summed-runtimes/min/max，但第 151 行执行清单只列 arithmetic/geomean/min/max。方法原则已闭合，执行 checklist 仍可能造成漏列。

建议：在 P0 第 4 项补 `ratio-of-summed-runtimes`。这是清单同步问题，不重新打开 P1-02。

## 准入边界

M526 r2 允许后续：

- 用 B0/B1/B2/B3/C2/Ours 跑同 checkpoint、同 trace、同精度、同 SRAM/BW 条件的统一 simulator；
- 生成固定 numerator 的 Table A、直接重跑 waterfall Table B、cross-paper specification Table C；
- 使用 Ours/B0 或 Ours/B1 作为明确标注的 iso-lane headline，同时同页给 Ours/B3 和 area-normalized throughput。

仍然禁止：

- 把 corrected analytical envelope `1.794--1.823x` 写成 admitted result；
- 把外部 H67 mapping `2.459487x` 或 Prosperity/Phi headline 写成 ours；
- 乘 C1/C2/C3 局部倍率；
- 隐藏 K1x8、用 variable numerator 或 kernel-only scope 构造 full-network headline。

机器裁定见 `m526_prosperity_phi_headline_method_independent_hammer_r2.json`，核查回执见 `verification_r2.txt`。本目录以 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 双 seal 封存。
