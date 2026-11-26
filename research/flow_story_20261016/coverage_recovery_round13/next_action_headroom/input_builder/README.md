# H0 完整审计后的输入构建工具

2026-10-16。本目录只准备 CPU 工具，未读取真实 R13 正式质量、真实恢复权重或 H0 结果，未选择真实状态、SSH、GPU、部署或排队。6 项合成 CPU 检查通过不能替代正式审计。[源代码](build_h0_inputs.py)只用 Python 标准库，导入时不读取实验结果或导入 torch。

## 固定选择与用途边界

工具必须先验证冻结质量分析、完整 16 运行独立执行审计、全部 8 个实际恢复结果及共同数据/BN 来源，再按以下固定顺序构建输入：

1. 四类冻结质量门全部通过的首个状态：`reference/CC → reference/fine_s2b2 → diversity/CC → diversity/fine_s2b2`。此处 `fine_s2b2` 是 P 恢复臂，scope 为 `qualified_ordinary_base`。
2. 如果以上均未通过，但普通 N/F 稳定，则按 `reference → diversity` 取首个稳定条件的 FF 恢复状态。scope 为 `action_only_screening`，明确不宣称该 CC/P 底座已过可部署门。
3. 两条件普通 N/F 均不稳定：只写 `NO_QUALIFYING_STATE.json`，**不写** `inputs.json` 或 `admission.json`。

同一条件的 N/native、F/FF 权重始终作为匹配锚点。这里的固定分叉不替代 root 对下一项科学实验的判断；例如仅 shared-fine 失败时，既有 M 轴普通恢复控制仍是竞争解释。构建成功本身不会启动 H0，更不会启动 H1。

## 来源与路径

工具复用已独立审查并固定 SHA 的 H0 纯元数据校验函数；不调用 H0 `prepare/execute`，不调用历史 collector，不加载模型或旧 CMA。`source_pins.json`固定 H0 core、runner、H8 和两个模板。输出字段严格沿用这两个模板。

[mapping.template.json](mapping.template.json)为故意不可执行的占位模板，需要 root 在完整正式报告存在后填写：

- 七个共同/报告资产：冻结 `analysis.json`、完整 `ACTUAL_EXECUTION_AUDIT.json`、本轮 runtime、成员 manifest、新 BN NPZ、新 BN receipt、固定 H8 manifest。每个显式提供本地源路径、未来远端路径和同一文件 SHA。
- 全部 8 个 `condition/arm`：本地实际 `results.json`、远端结果和 `weights.pt` 路径、二者 SHA；可选本地 `weights.pt` 用于字节 SHA 核对。
- 一个全新远端输入 bundle 目录；生成的 admission/inputs 不得与任何源文件路径相同。

冻结报告中的原绝对路径保留不改。跨机身份由 `(condition, arm, mode)`、实际 result SHA、weights SHA 对应；显式映射允许本地报告路径与远端路径不同。工具核本地证据字节，不连接远端，不声称映射目的文件已经存在或字节正确；部署后必须由已审 H0 preflight 再核远端实际文件。

完整报告核验包括 40 状态布局、320 预测、16 区域、1964 正式前向，以及独立执行审计的 16 运行、8 权重、128 archive、2048 前向和 1640 更新。初始状态/预测配对、72 BN 值、最终/冻结补集状态独立核验必须通过；8 个实际恢复结果的 200 更新、共同 binding、原模块/hook/BN 闭合均绑定到上述报告。新 BN 明确为合法训练侧 176 次、78 个 BN 重置 CMA、保留 24 MLP BN，禁止旧 CMA。工具只检查这些证据及其来源，不重新计算 AEE 或重跑数学。

## 最小用法

本地 CPU 检查（已做，纯合成）：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_builder_cpu.py
```

root 后续填写真实映射后可执行以下 CPU 构建；当前尚未执行：

```bash
python -B build_h0_inputs.py \
  --mapping /absolute/completed_mapping.json \
  --mapping-sha256 ACTUAL_MAPPING_SHA256 \
  --cpu-checks /absolute/input_builder/cpu_results.json \
  --output-dir /absolute/new_local_h0_input_bundle
```

成功输出 `admission.json`、`inputs.json`、`BUILD_RECEIPT.json`；没有资格时只有一份 no-eligibility 记录。输出目录必须全新，不能覆盖旧 bundle。生成 admission 的本地字节 SHA 写入 inputs，路径指向显式未来远端 bundle。receipt 记录源代码/映射/CPU 检查 SHA、选择依据、可选本地权重字节是否检查、远端尚未核验、实验尚未启动。

root 将生成 bundle 和未改动的证据部署到显式路径后，仍需按 [H0 README](../implementation_h0/README.md)先 preflight，再在科学分叉成立时运行 69 次 H0。这个 builder 没有执行参数、GPU 初始化或自动队列。

6 项 CPU 检查覆盖固定优先序/门一致性、与 H0 模板字段完全一致的合成 bundle、跨机 SHA 身份、无资格不产可运行文件、源 SHA/缺臂/权重/远端配对/碰撞/不完整审计拒绝、新 BN 校准合同错误拒绝和全部入口全局依赖。所有“结果/权重”均是临时合成字节；只读取真实的固定身份元数据。没有 flow 特有增量或硬件收益结论。
