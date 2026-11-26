# R29 新 64 个同权重 Q8 预测

规则见 [预结果协议](PROTOCOL_PRE_RESULT_V1.md)。实际状态字段是 `Q8_on_recovered_Q4`。R27/R28 的参考标量已经暴露，均以 SHA 固定；这里仅独立重算 R29 的新 64 份 head3。原 64 份共享 GT/16 分区档和 64 份同权重 Q4 参照 archive 的归属再次核验。

完整实际审计 PASS 后，在本目录运行：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B prepare_inputs_v1.py \
  --result /absolute/path/to/actual/results.json \
  --audit ../actual_execution_audit_v1.json --output ACTUAL_INPUTS_V1.json
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B reduce_same_weight_v1.py \
  --inputs ACTUAL_INPUTS_V1.json --output actual_quality_v1
```

`IO_FREEZE_V1.json` 固定实际 thin auditor 的源码 SHA 和它使用的旧来源；此 auditor 没有另一个 `auditor_freeze_sha256` 输出，不虚构该字段。原 +3/+5/+5% 比较和尾部警戒只作一致描述，不新增资格门，也不自动启动混合布局或评分学习。全 Q8 端点不是等成本压缩策略。
