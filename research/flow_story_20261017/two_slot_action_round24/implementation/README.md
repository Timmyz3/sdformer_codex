# R24 固定双代表执行图：CPU 完成，未启动 GPU

`run_two_slot_domain.py` 沿 R21 编排窄改；`two_slot_mlp.py` 新增固定 H1 pack，并将冻结 V2 `compact_mlp` 的私有 globals 中仅 `pack` 一项替换。旧模块/全域 BN/expand/gather 数学和原 globals 不变，实际导入路径、SHA 与函数身份均核验。来源：原 R21 runner SHA `7e650140ebd50a1af50aed0f46189a3c218683ad8c1e814e7c7a131326532411`，R16 V2 SHA `e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731`，shared SHA `ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec`。

首次固定 CPU 矩阵在 `PRE_CPU_SOURCE_FREEZE_V1.json` 和独立源短审之后执行：7 tests PASS，真实原生 MLP/PSN 的 5 个微前向，0 full model/0 GPU/0 GT。实际 target FC 行数依次 `12000/12000/4440/4440/5070`；FF 分支/PSN 与 native 同值，T2 重放逐字节相同。76 个附加真实 BN 加目标 2 BN 构成 78BN 捕获 fixture；两目标 BN 完整 T/H/W，状态/原模块/hooks 保持。该 fixture 不等于全模型预检。收据 `orchestration_cpu_v1.json` 由测试一次写入，未覆写。

## 输入和命令

输入 JSON 沿原 R21 `inputs_v1.json` 的路径/SHA合同：schema 改为 `flow_r24_two_slot_domain_inputs_v1`；`primitive` 指本目录 `two_slot_mlp.py`；新增 `base_primitive` 指远端 `/data/flow_a/code/research_20261017/current_bn_round16/implementation/native_dense_bn_v2.py`；`independent_admission` 指本轮独立源 admission。旧 runtime、W271482、membership、H8、regions 和 `BN_mode=NATIVE_DENSE_CURRENT` 保持。

root 的执行 freeze schema 为 `flow_r24_two_slot_domain_source_freeze_v1`，`files` 至少显式纳入当前 runner、新 primitive、旧 V2 以及旧 shared 依赖（均 `{path,sha256}`）。独立 admission schema 为 `flow_r24_two_slot_independent_admission_v1`，含 `status=ADMIT`、`permitted_plans=[domain122]`、实际 `runner_sha256`、新 `core_sha256`、`BN_mode`。由 root 根据最终独立收据制作，不用手填假通过。

远端预检命令模板（所有 SHA 由冻结实际文件填入）：

```sh
/data/flow_a/envs/v2/bin/python -B \
  /data/flow_a/code/research_20261017/two_slot_action_round24/implementation/run_two_slot_domain.py \
  --inputs /data/flow_a/code/research_20261017/two_slot_action_round24/inputs_v1.json \
  --inputs-sha256 INPUT_SHA \
  --freeze /data/flow_a/code/research_20261017/two_slot_action_round24/SOURCE_FREEZE_V1.json \
  --freeze-sha256 FREEZE_SHA \
  --plan domain122 --wall-minutes 20 \
  --output /data/flow_a/runs/two_slot_action_round24_20261017/run_v1 \
  --preflight-output /data/flow_a/runs/two_slot_action_round24_20261017/preflight_v1.json
```

预检仅读源、检查点、metadata 和 H8 的 24 文件 SHA，不读取 GT 值、不初始化 CUDA，不创建 `run_v1`；会创建 preflight 记录的父目录。只有 root 在预检和独立源审之后，用相同参数追加 `--execute` 才执行。fresh output、GPU0 全局锁、20 分钟墙时和失败 prefix 保存沿原编排。

## 稳定输出接口

`schema=flow_r24_two_slot_domain_v1`，`graph_id=r24_W271482_all78current_s2b2_fixedH1_two_slot_T2_k1_H8`，完成状态 `TWO_SLOT_DOMAIN122_COMPLETE`。120 个质量端点、2 个 T2 exact replay，首 T2 reference forward3；expected FC each `67335120`，BN `9516`。字段沿 R21 `forward_records/numerical_checks/target_execution/tensor_records/head_records/input_evidence`，保存方式沿 R21，122 NPZ/586 arrays。新增 `binding.base_primitive_sha256/partition_rule` 和每个 compact `target_execution` 的 pack-only 来源声明。actual 全模型计数、状态和原数组仍待独立执行审计。

本轮没有 q、方向择优、selector 或自动后续训练。新 H1 2slot 的表示成本高于原 1slot；名义 FC 减少不等于周期、全网或光流创新。失败/成功只对本次固定动作域解释。
