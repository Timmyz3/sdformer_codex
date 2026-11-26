# R20 实现交付

`run_action_domain.py` 是原 R19 runner 的窄编排适配；只更换 fixed44布局、真尾互补 mask、两个独立重放族、图/schema 标识和总账，数学 primitive 保持原 SHA。`PROTOCOL_V1.md` 给出执行与证据边界。

CPU 检查命令（输出必须不存在）：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_action_plan_cpu.py orchestration_cpu_v1.json
```

根代理远端使用的 CLI：

```bash
python -B implementation/run_action_domain.py \
  --inputs runtime_inputs_v1.json --inputs-sha256 INPUT_SHA \
  --freeze source_freeze_v1.json --freeze-sha256 FREEZE_SHA \
  --plan action44 --wall-minutes 20 \
  --output /data/flow_a/runs/action_domain_round20_20261017/run_v1 \
  --preflight-output /data/flow_a/runs/action_domain_round20_20261017/preflight_v1.json
```

默认只 preflight；正式命令使用同一显式输入/冻结 SHA，加 `--execute`。本作者不自行 launch。CPU receipt 不等于远端真实资产 preflight。

输入 schema `flow_r20_action_domain_inputs_v1`，字段与 R19 相同：`old_root` 指向远端 date16；`runtime/membership/h8/regions/checkpoint/primitive/independent_admission` 都是 `{path,sha256}`；`BN_mode=NATIVE_DENSE_CURRENT`。原 checkpoint、H8、训练侧 regions 与原 loader source 合同不变。不存在隐式本机 research 绝对路径导入。

根 sourcefreeze schema `flow_r20_action_domain_source_freeze_v1`，`files` 是实际部署路径/SHA列表，包含入口、原 primitive 和原 shared math。独立 admission schema `flow_r20_independent_admission_v1`，须 `status=ADMIT`、`permitted_plans=[action44]`，并绑定 `runner_sha256/core_sha256/BN_mode`。不能使用 R19 admission 代替。

结果 schema `flow_r20_action_domain_v1`，顶层与 binding 均含 `graph_id=r20_W271482_all78current_s2b2_R16V2_action_domain`。全部成功才 `ACTION_DOMAIN44_COMPLETE`；失败 `INCOMPLETE_OR_NUMERICAL_FAILURE`，保留前缀。`binding.forward_plan`、`forward_records`、`numerical_checks`、`target_execution`、NPZ键沿 R19。新增 replay kind `T8_exact_replay`；CC/T8的各自首引用是 forward 3/7。质量端点仅40；4重放禁止入质量平均。独立审计请直接绑定该 graph_id，不假定未来审计 JSON 自动导出同名顶层字段。

每次全部四 head 数据保存；292 原数组明细见 protocol。完整模型必须保持477-state（所有234 BN buffers移除）；runner由已审 original loader/state digest合同强制核，不在本轮重定义哈希。

来源：原 R19 runner SHA `ef65ba75518804a83a7faba60468b132ab551fe4a2e96c9d4af6c168142f58b7`。原 R16 V2及shared SHA见协议。新的实施冻结收据列当前入口/测试/协议 SHA及 CPU检查，不改任何旧文件。
