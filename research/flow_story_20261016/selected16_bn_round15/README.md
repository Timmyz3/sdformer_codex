# R15 S8 执行准备

固定271482/self-own-CMA/check8，只跑8次selected16/native；零训练、新校准、FF、dual。当前代码交独立审查后由root执行，不自动进入下一轮。完整合同见[PROTOCOL.md](PROTOCOL.md)。

`run_selected16.py` 的CLI与root已有 `root_launch.py` 对齐：`--inputs-manifest`、`--inputs-manifest-sha256`、`--cpu-checks`、`--output-dir`、`--wall-minutes 15`；默认还需fresh `--preflight-output`，实际才加`--execute`。默认preflight不会建立未来run目录，并自行创建preflight的父目录。远端runtime_inputs_v1.json已只读核对schema、固定SHA和缓存角色；路径存在性由root远端preflight核。

本机执行准备命令为 `/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_selected16_cpu.py`，仅CPU。`source_state_cpu.py`另实际读取原PT及K own72重建525状态；`expected_state.json`包含真实48数组shape/dtype/payload摘要、完整状态SHA，不含质量。已与独立CPU来源审计的状态SHA一致。模型载入/安装路径再次计算该状态，不依赖口头closure。

结果schema固定为 `flow_selected16_bn_round15_v1`，完成状态 `ROUND15_SELECTED16_S8_COMPLETE`。根字段保留`binding`、`data_evidence`、`runtime`、`full_forward_count=8`、`forward_records`、零optimizer/calibration/dual、状态/BN/hook/module closure与资源。`binding.references.private24`引用K endpoint、`current78`引用R14 endpoint；各含硬pin result/audit SHA、缓存状态SHA及8条完整native预测descriptor。`binding.selected_state_cpu_audit`为真实525状态记录，`own_BN_buffers_sha256/own_BN_receipt_sha256`只指原K来源。

新质量记录在 `checkpoints['271482'].evaluations.endpoint.frames`（8帧）和`summaries.native`；每帧`arms.native`包含原16区域metrics、prediction SHA及execution，`paired_cached_native`含`private24/current78`两descriptor。每帧单独保存`selected16_271482_<target_id>.npz`，仅`native`数组。全记录actual FC每次558,000行，四头、8input/branch SHA；execution另含`all78_BN_observations`，每个真实BN的`calls=1`、`input_shapes=[...]`、`current_stat`。

该checkpoint记录包含 `initial_selected16_state_sha256/final_selected16_state_sha256`、真实`BN_scope`的16/62名称与签名、8次计数和closure。另一个`selected_bn_buffers.npz`保存48个实际保留数组，根`retained_BN_buffers`给file/SHA/keys/buffer_tensor_sha256以及原own72/receipt SHA。它是BN来源证据，不能计为第9个质量预测。预测8、保留BN1，共9个NPZ，零PT。完整state恒定，不保存重复大PT。

source_pins递归继承冻结R14 V2/K/H0数学，不调用任何历史prepare/execute。新增observer使用私有H0.forward globals，只对其末尾hook-empty精确检查严格放行本次已知78个观察hook；返回后全部清理。CPU测试保留这项真实接口回归，包含未知hook拒绝与历史函数/globals未改断言。

质量通过也仅支持这次native BN政策的门；S8不验证FF、coarse、动态区域评分、恢复训练或硬件收益。新增62current的统计屏障/成本范围见既有[selected16源码分析](../bn_conversion_round14/SELECTED16_BN_SOURCE_FEASIBILITY.md)。
