# 独立 H8 domain122 入口

`run_tail_fine_domain.py` 窄适配已冻R19编排；唯一新空间动作是保留八尾fine。原数学与源SHA不变。CPU只用合成原MLP/PSN fixture；实际来源和GPU数值仍需根预检/独立审计。

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_tail_fine_cpu.py orchestration_cpu_v1.json
```

远端真实Python为 `/data/flow_a/envs/v2/bin/python`。根模板：

```bash
/data/flow_a/envs/v2/bin/python -B implementation/run_tail_fine_domain.py \
  --inputs inputs_v1.json --inputs-sha256 INPUT_SHA \
  --freeze source_freeze_v1.json --freeze-sha256 FREEZE_SHA \
  --plan domain122 --wall-minutes 20 \
  --output /data/flow_a/runs/tail_fine_feasibility_round21_20261017/run_v1 \
  --preflight-output /data/flow_a/runs/tail_fine_feasibility_round21_20261017/preflight_v1.json
```

默认只preflight；正式同一显式SHA加`--execute`。输入字段与R20 H8输入相同，独立 schema `flow_r21_tail_fine_domain_inputs_v1`；sourcefreeze `flow_r21_tail_fine_domain_source_freeze_v1`；admission `flow_r21_tail_fine_independent_admission_v1`、`status=ADMIT`、`permitted_plans=[domain122]`并绑定runner/core/BN_mode。

结果 `flow_r21_tail_fine_domain_v1`、完成 `TAIL_FINE_DOMAIN122_COMPLETE`；graph_id顶层/binding为`r21_W271482_all78current_s2b2_R16V2_T8_k1_H8`。122计划与字段沿R19，CC改T8；8allfine_vs_native和2T8_exact_replay，ref3；FC67229280、BN9516、586原数组。原半成品362入口/schema不能用于此轮。

源来自 `compact_graph_round19/run_compact_graph.py` SHA `ef65ba75518804a83a7faba60468b132ab551fe4a2e96c9d4af6c168142f58b7`。额外来源预检组合显式只核元数据和H8真实文件，避免stat新角色数组，数学不变。所有source/CPU SHA列在实施冻结收据。
