# R28 独立质量补充

只新增两个 Q4 状态、128 个真实 head3。R27 原质量与 64 个共享 GT/分区档只读复用；不改变旧结果、训练合同、阈值或开发身份。规则见 [预结果协议](PROTOCOL_PRE_RESULT_V1.md)。

待完整 `round28/actual_execution_audit_v1.json` PASS 后：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B prepare_raw_inputs_v1.py \
  --result ../actual_v1/formal_Q4_v2/results.json \
  --audit ../actual_execution_audit_v1.json --output ACTUAL_INPUTS_V1.json
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B reduce_raw_v1.py \
  --inputs ACTUAL_INPUTS_V1.json --output actual_quality_v1
```

执行目录为本目录。必须先封存 `IO_FREEZE_V1.json`，其中绑定实际 auditor/freeze、原 R27 数学/质量和本轮入口修正。所有新预测与旧来源 SHA 通过后才读取共享 GT 数值。原 256 个 native/Q8 预测只重新核 archive 字节，标量来自已冻结的独立 R27 复算；不能声称本次重新复算了旧 384 个预测。

开发时首次 6 项合成测试中，3 项 IO 测试发现旧目录前插 `sys.path` 遮蔽新同名 CLI；已改为追加旧 helper 目录并全部 6 项通过。该阶段无真实 Q4 数组或 GT 读取，无模型/GPU；原失败说明保留在 CPU 收据中。不是一次实际结果失败或门放宽。

统一 Q4 双参考失门只判这个统一配置不合格，输出 `UNIFORM_Q4_NOT_QUALIFIED_MIXED_FAMILY_UNDECIDED`，不自动宣布全部成本受限混合策略失败，也不自动运行后续实验。
