# 新自训v2的独立全网评估

入口[evaluate_native_v2.py](evaluate_native_v2.py)只做推理，默认仅CPU计划检查；**本次没有GPU执行或真实checkpoint AEE**。25项CPU检查已通过，包括按帧/像素聚合差异、GT洞与空分区、成员泄漏、BN双臂和新checkpoint/schema/内容漂移；见[evaluation_cpu_tests.json](evaluation_cpu_tests.json)。这些不是网络精度结果。

与[新训练器](../baseline/train_baseline.py)约定`flow_a_native_v2_checkpoint_v1`：只读`model_state_dict`及可追溯新训练配置；显式文件SHA和同名JSON sidecar相符，`weights_only=True`，拒绝不受限pickle回退。核对random-init/no-old-checkpoint标记、正更新数、contract摘要、成员摘要、63文件原生source manifest及其实际文件SHA。工程子集权重仅允许显式train-side评估，不能冒充最终valid baseline。`canonical6000`表明训练成员范围；少量更新仍欠训练，不自动构成复现或杀A证据。

每次评估按相同新权重重建两臂：P-paper关闭BN tracking并将running buffers设None，使用当前统计；P-code保留训练得来的running buffers，要求各BN非负有限方差及正计数。两者均`eval()`关闭dropout/drop-path；每个target前reset，B1/T10/FP32、TF32关闭。P-code不是原B1训练入口里`model.train()`的验证行为。每臂结束核BN buffers未变，不训练、不改源码、不保存新权重。

支持三个明确标签：`crop_center9`（288×384中心crop，原坐标起点96/128）、`full9`（480×640部署适配）、`full15`（480×640/W15）。crop9 checkpoint可在原生full9构造严格载入；window9与15不匹配直接拒绝，**不静默插值**。full15需后续同schema的真实W15权重；其容量测试不能充作权重。本评估没有实现9→15微调。

必须先生成固定B/L阈值：`fit_region_thresholds.py`在gate-fit成员上只读GT，CPU计算。默认全部4573；探索可给明确`--target-id`子集，但这样的阈值只可用于train-side探索。B是四邻域全有效时的最大GT流差，GT洞/外缘记U；L是GT流幅值。固定95%/90%训练分位、每target最多4096确定性样本，严格`>`阈值，非论文门限。具体语义与限制见[实验卡](../idea_and_protocol/A_DISCRIMINATING_EXPERIMENT.md)。最终valid需全部4573拟合结果，且不能看valid后换阈值。

```bash
RESEARCH_ROOT=/data/flow_a/code/research
PY=/data/flow_a/envs/v2/bin/python
"$PY" "$RESEARCH_ROOT/flow_story_20261012/evaluation/fit_region_thresholds.py" \
  --data-root /data/flow_a/data/DSEC/saved_flow_data \
  --output /data/flow_a/results/regions_gate_fit_all4573_v1.json
```

拟合命令会读取全部训练gate-fit的GT/mask；完整迁移未验收时不要把缺数据改成静默跳过。它不需要torch/GPU。少量train-side探索可重复传`--target-id`，并更换输出文件名；不要用01_a/05_a gate-dev来拟合分区定义。

下面变量值来自已经核准的新checkpoint和阈值artifact（不自动读取可能变化的`latest.json`指针）:

```bash
"$PY" "$RESEARCH_ROOT/flow_story_20261012/evaluation/evaluate_native_v2.py" \
  --source-root /data/flow_a/code/SDformerFlow \
  --data-root /data/flow_a/data/DSEC/saved_flow_data \
  --checkpoint "$FLOW_CHECKPOINT" --checkpoint-sha256 "$FLOW_CHECKPOINT_SHA" \
  --regions "$FLOW_REGIONS" --regions-sha256 "$FLOW_REGIONS_SHA" \
  --scope train_gate_dev --target-id zurich_city_01_a_0002 \
  --target-id zurich_city_01_a_0003 --geometry full9 \
  --run-inference --output /data/flow_a/results/pilot_full9_dual_bn.json
```

正式独立valid采用`--scope valid_all`，删除所有`--target-id`，且必须提供`--reference-hashes "$FLOW_DATA_INVENTORY" --reference-hashes-sha256 "$FLOW_DATA_INVENTORY_SHA"`；清单为源/独立验收产生的CSV，列含`relative_path,sha256`且路径以`saved_flow_data/`开头。全部2152个target三件套必须存在，读取的**同一bytes**先核SHA再np.load，shape/dtype/finite检查。不提供reference的train-side探索只记录实际读取bytes的SHA，并明确未绑定源清单；正式valid不允许该模式。数据root注意是`saved_flow_data`目录，新训练器参数则是其上一级DSEC。

输出只用原生最后一级完整flow，原像素位移单位、不乘event mask。主AEE先在每帧全部有效GT求均值，再对帧平均；另报全像素加权、逐序列及B/L/U/活动交叉分区。B/L重叠与互斥子分区分别列，GT洞未知不归平坦；空分区为null并报缺失帧，整帧无有效GT则拒绝。分区在全幅GT确定后再crop，不引入crop伪边界。不计算或伪报误判gate指标：当前是dense P双臂，真正gate需额外mask/反事实标签证据。

JSON保留每帧原始误差分子和有效像素数、每臂摘要、P-code−P-paper差、输入文件SHA、实际包版本和训练进度。中途缺文件/数值/BN异常保存部分结果但标失败，不能当完整平均。运行时间包含读取、哈希和CPU指标，仅为评估工程用时，**不是纯网络吞吐或硬件加速证据**。

本地CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /opt/anaconda3/envs/pytorch310_cpu/bin/python \
  /home/zhumd/work/sdformer_codex/research/flow_story_20261012/evaluation/verify_evaluation_contract.py
```
