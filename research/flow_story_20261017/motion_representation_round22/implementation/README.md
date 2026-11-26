# R22 native26 交付接口

CPU检查：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_representation_cpu.py orchestration_cpu_v1.json
```

远端默认预检（实际加`--execute`）：

```bash
/data/flow_a/envs/v2/bin/python -B implementation/run_representation.py \
  --inputs inputs_v1.json --inputs-sha256 INPUT_SHA \
  --freeze source_freeze_v1.json --freeze-sha256 FREEZE_SHA \
  --plan native26 --wall-minutes 15 \
  --output /data/flow_a/runs/motion_representation_round22_20261017/run_v1 \
  --preflight-output /data/flow_a/runs/motion_representation_round22_20261017/preflight_v1.json
```

输入 schema `flow_r22_representation_inputs_v1`：保留原 `old_root/runtime/membership/regions/checkpoint/BN_mode`；用 `selector_roles={path,sha256}`代替H8，增加 `native_forward_helper={path,sha256}`指向冻结R21入口。没有primitive/background参数。

根sourcefreeze schema `flow_r22_representation_source_freeze_v1`，files须含本入口和实际native helper及相关冻结源；不改旧文件。独立admission schema `flow_r22_representation_independent_admission_v1`，`status=ADMIT/permitted_plans=[native26]`，绑定 `runner_sha256/native_helper_sha256/BN_mode`。

result schema `flow_r22_representation_v1`，graph_id `r22_W271482_all78current_native_fit16dev8`；完整status `REPRESENTATION_NATIVE26_COMPLETE`。顶层/binding图相同；binding新增 `selector_roles_sha256/selected_target_ids/selector_role_map/confirmation_array_accesses=0/native_helper_sha256/primitive_calls=0`。每record含selector_role、input_evidence(events/gt/mask SHA)、saved_payload_keys；其余FC/BN/target_execution/tensor_records/head_records沿冻结helper。两 `native_exact_replay` ref1，原数组八项。208payload/26archives/24quality；不与R21 domain122或未冻362混用。

本源只复用R21 native forward/source检查。已冻结 helper 所带“allowed8分别检查”的临时说明字段被当前入口明确替换为“allowed24”，并实际核本轮72文件；不重写旧收据、状态或科学门。完整源SHA与CPU记录见实施冻结。
