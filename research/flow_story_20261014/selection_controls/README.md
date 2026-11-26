# 固定轻分支下的均值、局部与旋转选择对照

2026-10-14。本目录是新一轮 **CPU 全网推理控制入口，尚未执行真实模型**。不训练、不改封存文件，不将新普通对照称作原创算法；资源由主会话统一安排。

复用 [原运行器](../fast_warmstart/run_fast_runtime.py) 的模型/数据加载、BN、reset、度量和实际/双生产 reference 路径。新 `run_selection_controls.py` 显式复制其主循环并扩充臂列表；方向分区函数直接导入原运行器。源/父合同/已有结果全部 SHA 绑定。现有 `RuntimeGroupPolicy` 已实现四种策略，因此没有改评分、配对、DP或轻分支公式。

| 本轮臂 | 用途 |
|---|---|
| native_dense、tiled_all_keep | 原图与合法分块数值锚点 |
| independent_trained、motion_trained | 复核上一轮同 trained fast 的选择结果 |
| motion_mean | 与 motion_trained/旧 motion_zero 同 mask；检测均值偏置能否解释改善 |
| horizontal_local_trained | 同 head/幅值分数/fast 的固定水平普通邻域组 |
| rotated90_trained | 同 head/幅值分数/fast，将配对方向旋转90度 |

保留旧135930步权重、MLP_CMA176/外部current-stat、full9/T10/B1和同500步warmstart。仍为 stage2/block0 单MLP。每次四套proposal共同确定可达预算，实际固定900/1200空间点走slow，两FC各9000行；trained各3000行LN+Linear；mean无fast FC、填充1,152,000个逻辑元素。**同slow工作不是同总成本**，控制器操作完整保留。horizontal/rotated可改变边界资格、组数和冲突图；水平控制不匹配斜向欧氏距离，不能只凭胜负断言运动因果。

四帧顺序固定01a0083、05a0165、01a0030、05a0352。它们来自训练侧开发池、backbone已见且被反复探索；没有新增独立确认或官方测试。7×4前向＋首帧五个局部臂reference，共33次完整CPU前向。按旧32次770.8秒粗估约13–16分钟，内部上限30分钟；这是预算估计，不是已运行计时。

每个target先完成推理，再以GT评估所有有效像素和既有全部分区，空区保持null。四个旧锚点的预测SHA、度量充分统计、输入/输出/掩码/执行行数必须与原结果逐值一致。Mmean掩码还须与旧Mzero一致；只有所有锚点通过后，才把旧Mzero加入明确标注来源的跨轮配对差。失败留下 `SELECTION_CONTROLS_INCOMPLETE`，不形成完整比较。原始预测NPZ和逐帧/逐序列统计保留；没有PE周期、能耗或ASIC收益结论。

最短判断：Mmean若解释Mtrained的提升，不归因为非均值特征保留；local/rotated若同样好，不归因为正确运动方向；motion若更好也只保留候选，须后续充分恢复、预封存确认集和实际总成本对照。所有结果并列，不为主线删掉更好的普通控制。

## 可执行命令与数据位置

旧合同的数据 `/data/flow_a/data/DSEC_native_v1`、135930 checkpoint、coarse head权重在专用机，本机这些路径不存在。仅本地回传的fast权重/结果不能运行完整网。将本目录入口部署至 `/data/flow_a/code/research_20261014/selection_controls/` 后，由主会话在专用机CPU执行；以下父合同路径若部署不同，只替换路径，内容与SHA不变。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/data/flow_a/envs/v2/bin/python \
/data/flow_a/code/research_20261014/selection_controls/run_selection_controls.py \
  --parent-contract /data/flow_a/code/research_20261014/fast_warmstart/runtime_contract_v1.json \
  --parent-contract-sha256 564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389 \
  --prior-results /data/flow_a/runs/fast_warmstart_20261014/runtime_v1/results.json \
  --prior-results-sha256 08ec5c7032dbfe99d33c159e09369c147514cfe3ae8e3889c248b48c52244ccf \
  --output-dir /data/flow_a/runs/selection_controls_20261014/runtime_v1 \
  --preflight-output /data/flow_a/runs/selection_controls_20261014/preflight_v1.json
```

默认只做来源与数据合同preflight，无网络前向。完整运行使用相同命令，去掉 `--preflight-output ...` 并加 `--execute`。输出目录必须不存在；不覆盖结果。无需GPU锁或CUDA初始化，不能同另一个重CPU作业无计费叠加。

本地验证命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/opt/anaconda3/envs/pytorch310_cpu/bin/python test_selection_controls.py
```

实际 **6/6 CPU checks PASS**，见 [cpu_checks.json](cpu_checks.json)：四策略完整T/tail/共同预算与共享信息、旧合同/成员拒错、评分源/成本/branch身份拒错、旧预测/GT覆盖/掩码拒错、null及配对差方向。测试不加载完整模型、GT数组、远程资源或GPU；真实全网执行仍待主会话。
