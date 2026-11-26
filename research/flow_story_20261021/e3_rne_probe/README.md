# E3：普通 RNE4 背景上的六种 Q8 升级探针

这是隔离的新准备目录。没有连接服务器或使用 GPU，不修改 E0/E2 冻结实现。是否运行由 root 在全 64 普通质量与 RNE 双表示硬件成本之后决定。

只复用 SHA 固定的 E0 `mixed_integer_core.py`，不复制修改量化数学。所有 28 层完整 tile 默认 RNE4、真实尾部 Q8；六个动作每次选一组，在四个 patch residual conv 中把同一段 151 个完整 9×9 tile 直接执行 Q8。其余 24 个 MLP 算子完整 tile 继续 RNE4。四层每层余下 4 个完整 tile 也是 RNE4。完整 T10、当前 BN、PSN 与空间重建均保留。

这是 6 个固定布局的有限可达性探针。家族按计算覆盖预选，不能引用旧截断 E1 的敏感性证明其最优；没有学习评分器，没有预测任务收益。独立 RNE4 与 Q8 使用不同表示和尺度，不能先算 Q4 再补低位。每动作相对全 Q4 名义增加 82,192,320 核 tick；等核工作不是等实际服务周期。

**实际运行合同为 112＋6＝118 个完整前向：**

- `background_Q4` 与 `upgrade_0` 至 `upgrade_5`，每状态相同 16 个 `e2_reserved_ids`：112 个质量预测。其中背景 16 次必须与正式 E0 `endpoint_Q4` 四头及输入来源逐字节/摘要一致。
- 六个升级动作各在首身份上额外跑一次 dense integer reference：6 个工程前向。比较真实 28 integer、全部 PSN、四 head；实际前向本身也计入上述 112，不重复计质量。
- 不包含训练、STE 前向、pair、额外层扫描或自动续跑。失败保存部分记录，不能充作完成结果。

## 文件与状态

- `probe_contract.py`：纯 CPU 构造几何、16 身份角色、全部 mask；绑定真实正式 step512 的结果/检查点/状态 SHA。`BoundCore` 直接代理冻结 E0，禁止 STE。
- `PREPARED_CONTRACT_V1.json`：没有 endpoint 的准备合同；collector 必须拒绝执行这个文件。它不是运行批准。
- `collect_probe.py`：实际模型与数据装载复用 SHA 固定 R28 helper，CUDA 初始化前获取同一全局 GPU lock。新输出目录、60 分钟硬上限、无断点继续。
- `CPU_CHECKS_V1.json`：7 项 CPU 检查通过，含 14 次微型 conv 前向、全部真实几何 mask/行域、六组等核工作、dense 对照、部分 endpoint 拒绝；没有完整模型或 GPU。
- `SOURCE_FREEZE_V1.json`：冻结本目录实现、准备合同和 CPU 收据。真实 endpoint 之后另写新的绑定合同，不能改准备合同或源码。

## root 绑定与运行模板

以下是模板，不是本目录的执行收据。路径由 root 核实。`E0_CORE` 指向未修改的 E0 core；`INVENTORY` 是 R27 `hardware/UNIFIED_SERVICE_V1.json`，只取几何，不取旧周期结论。

```bash
python -B E3_ROOT/probe_contract.py \
  --core E0_CORE --inventory INVENTORY --selections SELECTIONS \
  --endpoint-result COMPLETE_FORMAL_E0_RESULTS_JSON \
  --weights FORMAL_RECOVERED_STEP512_PT \
  --output NEW_ENDPOINT_CONTRACT_JSON
```

绑定会拒绝 RUNNING/部分结果、非512更新、非1353实际总前向、错误 E0 源，以及缺少 endpoint Q4 full-graph parity 的结果。会记录实际结果文件、checkpoint 文件、477 项模型 state 的 SHA。随后先 CPU 预检：

```bash
python -B E3_ROOT/collect_probe.py \
  --e0-core E0_CORE --r28-root R28_ROOT --r26-root R26_ROOT \
  --members MEMBERS_JSON --weights FORMAL_RECOVERED_STEP512_PT \
  --endpoint-result COMPLETE_FORMAL_E0_RESULTS_JSON --selections SELECTIONS \
  --contract NEW_ENDPOINT_CONTRACT_JSON --contract-sha256 ITS_SHA256 \
  --source-freeze E3_ROOT/SOURCE_FREEZE_V1.json --source-freeze-sha256 ITS_SHA256 \
  --runner-sha256 COLLECT_PROBE_SHA256 \
  --output NEW_PROBE_DIRECTORY --preflight NEW_CPU_PREFLIGHT_JSON --wall-minutes 60
```

root 决定运行后才在相同参数上加 `--execute`。不支持 W0 模式，避免混合合同；若将来需要，必须另开版本。完整正式 E0 结果是来源要求，不替代 root 对其质量、独立执行审计与预算的判断。

## 输出供独立 reducer 使用

`results.json` schema 为 `flow_e3_rne_probe_v1`，成功状态仅为 `E3_RNE112_AND6_DENSE_REFERENCES_COMPLETE`。

`evaluations.background_Q4`、`evaluations.upgrade_0`…`upgrade_5` 各有 16 个 `frames`。每帧含 `target_id/sequence/action`、`input_evidence`（原文件来源）、`actual_input`（实际网络输入摘要）、`heads`（四头）、head3 NPZ 路径及双 SHA、`actual_BN_calls`、`actual_affine_call_counts`、全部 `work`（28 层，含真实 `kernel_calls` 和 `logical_row_plane_work`）及逐域核验。`mask_layout` 保留每组选块数/摘要。`parity` 保存六个动作的 actual/dense 完整摘要与比较；中间 tensor 不长期另存，不冒充离线原数组复核。

独立质量分析只能在 source/计数/数值/输入/工作核验通过后进行。主 oracle 只在六个同核成本升级动作内选，不能把更便宜的背景混入同预算 oracle。所有策略选同一个动作评价全 GT 和全部固定 16 区；LOSO 固定布局须训练折选，uniform 要用六状态的精确平均而非一次随机抽样。96 动作标签来自16身份/8序列；不是96独立场景，也不能据此推 pair 效应或新颖性。
