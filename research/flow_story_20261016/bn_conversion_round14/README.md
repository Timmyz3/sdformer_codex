# R14 BN转换执行准备

本轮仅做[固定协议](PROTOCOL.md)的16次native前向：两个原始checkpoint各同check8、所有78BN current-stat，零训练/校准/dual。原24private/native缓存用于同W配对，绝不作为待推理模型。入口：[run_conversion.py](run_conversion.py)；新增状态与缓存合同：[conversion_core.py](conversion_core.py)。

CPU检查：[test_conversion_cpu.py](test_conversion_cpu.py)，4项PASS；检查真实小型12MLP与78BN转换，直接原生forward和冻结H0 native helper的四头、八输入、八branch完全一致；全状态/BN affine不变；固定16几何、缓存审计与NPZ payload损坏拒绝、嵌套依赖和禁训练/校准调用。源PT477 tensor锚点另由独立真实CPU审计给出，[SOURCE_STATES_V1.json](independent_review/SOURCE_STATES_V1.json)。作者CPU检查不读取真实新质量、不运行完整网络/GPU/SSH。

[inputs.template.json](inputs.template.json)规定资产结构；root已维护 [runtime_inputs_v1.json](runtime_inputs_v1.json) 与 [root_launch.py](root_launch.py)。CLI为：

```
--inputs-manifest PATH --inputs-manifest-sha256 SHA
--cpu-checks PATH --output-dir FRESH_RUN_DIR
--preflight-output FRESH_RECEIPT_OUTSIDE_RUN_DIR --wall-minutes 15
```

默认只做CPU/实际文件preflight，不能提前创建执行目录。仅root在source freeze及独立审查满足后追加`--execute`。严格加载源checkpoint会先载入其234个BN buffers，然后全部移除；不加载外部CMA，不用源running统计推理。其余实现与输出格式见协议及源码；无自动后续实验队列。

结果schema `flow_bn_conversion_round14_v1`，完整状态 `ROUND14_BN_CONVERSION_COMPLETE`。`checkpoints['135930'/'271482'].evaluations.endpoint.frames`各8帧，`arms.native`保存16区域原始指标和执行记录，每帧一个NPZ/native最终预测，共16NPZ/16预测。`paired_cached_native`保留同帧既有部署结果来源。顶层16条`forward_records`，每W完整state/78BN signature不变；每次558000 FC行、合计每FC8,928,000行。新allcurrent是reference/分母，原24private是candidate；不能反转方向来淡化转换代价。

另有[加权全局BN候选](WEIGHTED_BN_CANDIDATE.md)，仅源码设计，未实现/未运行，未加入本轮预算或声明光流特有创新。
