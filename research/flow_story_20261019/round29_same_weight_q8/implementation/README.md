# 同一 Q4 末权重的完整 Q8 诊断

仅将 R28 实际审计通过的 step512 Q4 权重以 Q8 模式执行。复用冻结 R28 V2 整数 core、全图加载/捕获和 64 帧 event-only 评价函数。全部 477 参数按审计绑定的 PT 严格加载后才做前向。没有教师、优化器、训练或质量选择；安全 CPU 读取 PT 会反序列化其中已有优化器字段，但不恢复它。

67 次完整前向：首个固定训练身份的完整空间输入上 STE、部署、dense-integer 三路径各一次；随后同 64 dev 各一次 Q8 部署。前者逐字节比较 28 整数输出、105 PSN 和四个头；保存四头原数组，其余中间值仅在内存比较并留下 SHA。后者仅保存每帧 head3 原数组及四头 SHA、78 BN 和 28 affine 实际调用记录。GT 文件只按已有来源做字节核验，不读取数值。

结果 `schema=flow_r29_same_weight_q8_evaluation_v1`，`graph_id=flow_r29_same_Q4_recovered_weights_all_Q8_all78current_v1`，成功状态 `SAME_Q4_WEIGHT_Q8_EVALUATION_COMPLETE`。标准状态名 `state_label=Q8_on_recovered_Q4`，64 帧在 `evaluations.diagnostic.frames`，每帧字段沿 R28。`source_Q4_checkpoint` 绑定实际 PT、末状态、原 result 和实际 audit SHA。`final_training_deployment_parity` / `final_dense_integer_parity` / `final_parity_heads.npz` 沿原接口。

计数为 66 个 serial 图和 1 个 dense 参考图：serial row-plane 总数 2,211,264,000；dense target 逻辑行数另列 4,188,000；总 BN 调用 5,226，含诊断 PSN 的实际调用 7,035。软件 int8 plane 存储、padding、完整 BN 和其余算子均保留；这些数量不是 PE 周期或完整 DAG 延迟。

入口示例（路径由 root 部署，`AUDIT_SHA` 是真实 R28 actual audit 文件 SHA）：

```bash
/data/flow_a/envs/v2/bin/python -B run_same_weight_q8.py \
  --r28-root /data/flow_a/code/research_20261019/round28 \
  --r26-root /data/flow_a/code/research_20261017/window_precision_round26 \
  --members /data/flow_a/code/research_20261019/round27/independent_review/membership_v1.json \
  --r28-audit /path/to/actual_execution_audit_v1.json \
  --r28-audit-sha256 AUDIT_SHA \
  --weights /data/flow_a/runs/round28_20261019/formal_Q4_v2/recovered_step512.pt \
  --runner-sha256 6b730696940018d40412054cbba1c76434d1b1986342d2b0e4e62fb037ab37e3 \
  --output /data/flow_a/runs/round29_same_weight_q8_20261019/run_v1 \
  --preflight-output /path/to/preflight_v1.json --wall-minutes 20
```

默认仅 CPU preflight，核真实权重/audit/源/960 数据文件字节，输出目录保持不存在。正式加入 `--execute`，使用已有 GPU0 全局锁和 20 分钟 alarm；root 另加外层进程上限。原 R26 prepare 只复用数据/来源核验，不执行其模型。

作者四项 CPU 检查使用本轮真实 R28 PT、actual audit 和实际冻结 membership。没有完整模型前向、GPU、GT 或 dev 预测读取；完整 remote prepare 由 root 执行。独立评审在其他目录留单独收据。
