# R13 独立执行审计

本审计由未参与 R13 runner/数据协议实现的代理完成。代码不导入 runner，不调用 SSH/GPU，不计算或排序 AEE，也不按质量选择保留的运行。当前完成审计器准备与共同原状态独立重建；实际 R13 smoke/formal 结果需在回收后另行运行 CLI，不能把准备通过写成执行已通过。

[预结果 freeze](pre_result_freeze_v1.json)绑定审计器与 15 个只读协议/实现来源。审计规则在本审计方读取任何 R13 result 前写入；当时共同 BN 已完成、smoke 正在执行，因此称“审计方读结果前冻结”，不称“所有实验启动前冻结”。

已执行[五项 CPU 检查](pre_result_cpu_checks_v1.json)：独立按十二层几何及真实尾部推导 FC 行数，并核对旧 R12 的 42 条真实 smoke forward/service schema；故意破坏 FC hook、尾块形状、身份课程均被拒绝；没有读取旧质量字段。额外从回收的原始 checkpoint 实际字节与新 BN NPZ，在 CPU 独立重建 549 个初始 state tensors：

- 原始 checkpoint SHA：`d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30`。
- 新 BN NPZ SHA：`283ced9a6506ddcf22bd9a5369bd964558ef44575233735d4ea6d6c1e80b1ce1`。
- 共同初始 state SHA：`21fa99e93438b37a6e4f46d9a28a46ca737017d7697e4b0f343136fc69d937ea`。
- 除80个允许训练参数以外的冻结补集 SHA：`ececf54eca2c8440ec3f35829413775015aac0e3b3bea5843bf31dcd0079d579`。

审计器实际读取 NPZ 的72个新 BN buffer，核对 float32/int64、原生尺寸、finite、variance非负、counter176，并删除原 state 中外部54 BN的162个running keys；没有实例化模型或执行推理。

## 实际执行审计内容

1. 两条件固定40×5/200×1及逐步序列一致性；每臂完整200步或固定5步smoke顺序，禁止丢帧、补步和重复run覆盖。
2. 从协议产生每个 forward 的精确顺序，逐项匹配用途、身份、初始/终点布局和actual/dual模式。独立重算十二层两FC hook、八wrapper producer形状、完整T10、尾块、pool/repeat与PSN/BN元素；不以runner最终总数为唯一依据。
3. 每个训练step与对应forward原记录配对，独立汇总梯度覆盖；两次数值比较的四输出、八branch/input及dual记录完整。未保存的中间数值张量无法在审计中重算，源码与比较结果属于执行记录证据，文档不把它夸成独立重跑数值等价。
4. 每个NPZ文件的实际字节SHA、键集合、每个float32 `[2,480,640]`预测的字节SHA和finite检查；两条件同布局初始预测严格一致。AEE/分区排名完全不使用。
5. 每个正式PT实际字节SHA，CPU `weights_only=True`装载；独立重算完整state/冻结补集digest、实际72BN值、允许80参数和7,252,064元素、原始起点重建、更新参数集合、AdamW的80×3状态及step200和RNG快照。
6. 正式臂使用自己条件的四份真实smoke receipt；全部臂相同起点/BN/recipe。记录预测文件/权重/前向总量和来源，不换算硬件收益。

## CLI

以完整回收目录为输入，输出文件必须尚不存在：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B audit_execution.py \
  --stage audit --freeze pre_result_freeze_v1.json \
  --results-root ../raw/experiments_v1 \
  --output ACTUAL_EXECUTION_AUDIT.json
```

默认严格要求两条件、smoke和recover、每阶段四臂，共16个完整结果。仅工程阶段齐备时可加 `--modes smoke`；单条件可加 `--conditions reference` 或 `diversity`。输出明确记录所审范围，局部通过不能写成全轮完成。默认原始PT和新BN路径分别是 `../source_snapshot/checkpoint_step_000135930.pt`、`../preparation/bn_buffers.npz`。

已冻结审计器 SHA：`dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8`；freeze SHA：`7e5e9673c1bdf96d759e225f954af25e2684590af634722a9ac764a8b4518447`。若后续遇到审计器自身schema错误，必须保留本版/失败证据，单独记录更正，不能静默放宽实验规则。

限制：本审计不会重新读取数百个原始训练输入/GT文件，其SHA与已接受输入证据及实际步骤记录核对；不会重新训练或重新执行未保存的四头/八分支数值审计。上述证据的边界与独立权重/预测实际字节审计分别报告。
