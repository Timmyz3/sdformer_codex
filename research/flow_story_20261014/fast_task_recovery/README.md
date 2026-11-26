# 固定轻分支的 R0/R1 有限恢复

2026-10-14。**真实GPU检查与两臂各200更新均已完成，见[结果报告](RECOVERY_REPORT.md)与[执行记录](RUN_LOG.md)。** 本轮是同一普通 DynamicSwin LN+Linear 底座的目标函数对照，不是完整原法复现或光流特定增量。下文保留封存协议，示例目录可替换为实际的prepare_v3与各臂_v3；来源与SHA不能改变。

同135930中期checkpoint、MLP_CMA176/外部current-stat、full9/T10/B1、stage2/block0，原始骨干全部eval且参数冻结。两臂从同一500步fast warmstart独立开始，seed20261014、AdamW lr.001/wd.0001/default betas/eps、FP32/TF32关闭、无AMP/梯度裁剪、无增广。固定fit20顺序×10，共200更新。不得使用新271482权重替换起点。

- **R0**：完整fast位置、全部T与通道的原生teacher分支MSE；teacher detach，不是原128点缓存。GT只用于共同的任务loss记录，不进入R0优化目标。
- **R1**：原生 `flow_loss_supervised(config,device)(output['flow'],label,mask,gamma=None)`；所有原生flow输出，实际flow_scaling/lambda_mod/shape/requires_grad写入结果。冻结参数不等于截断梯度：保留对fast输出的下游autograd。
- 两臂同样本、mask和更新数，不声称训练FLOPs或wall相同。记录前向、反向/优化耗时、teacher额外FC工作、grad norm和CUDA allocated/reserved峰值。

新 `trainable_fast_tile.py` 训练时按真实9×9尾块双生产同形状teacher/fast；teacher无梯度，loss只覆盖fast位置。输出仅MLP分支，外部原生残差加一次。评价只执行所选分支；每臂更新前后对首fit样本核actual/dual所有flow输出及目标分支精确一致。骨干完整state/BN前后核验，非finite、首步零梯度或输入SHA漂移立即标实现/合同未完成，不记算法失败。

## 固定mask及数据

`prepare` 在相同GPU数值图上对封存fit20＋既有dev8各做一次原生全网前向；pre-MLP u送入**同一冻结CPU粗流头/幅值评分/四策略共同预算程序**，只保存independent mask、u SHA、来源和计数。选择不读取GT数组或最终流；prepare虽然完成完整网络，最终流只做finite检查。当前版本固定cap900，保留每个目标由原common_target计算的正共同可达预算（≤900）；不筛样本，不改变原planner。两臂各目标使用同一已封存mask。实际slow点、target_cost与selection/common_budget须一致；MSE分母为(1200−实际slow点)×T10×C384。

同一 `mask_bank.json` 两臂共用，每次训练/评价复核u SHA。fit20是10个gate-fit序列各2；dev8是01a/05a各4，与fit按序列分离，但均backbone-seen。旧SVS fit20文件只借**成员身份**，不继承其P_paper计算图。dev8只在第200步终点评价warmstart和recovered，各8次actual前向；不按dev挑step。所有GT/B/L/unknown/voxel分区、覆盖数、帧均与像素加权、序列及预测NPZ保留。无final valid2152介入。

`smoke_R0/smoke_R1` 各仅首fit目标1更新、独立新输出；不读取dev图像，不作为正式起点。完整R0/R1必须提供**同臂、同mask/source绑定的一步烟测结果SHA**，仍重新载入原500步warmstart，绝不接着烟测权重训练。没有resume/额外轮数接口。每阶段内部上限30分钟；超限输出未完成，不当充分训练失败。

## 专用机命令

先将这两个新 `.py` 部署到 `/data/flow_a/code/research_20261014/fast_task_recovery/`；已有旧依赖位置全由封存父合同给出。默认只作preflight；加 `--execute` 才开始实际模型工作。所有实际CUDA工作先非阻塞独占原锁 `/data/flow_a/runs/.flow_a_gpu0.lock`；锁忙退出，脚本不杀/接管其他任务。由主会话统一排程，不同时运行R0/R1。

以下只定义命令参数；fit manifest部署路径如不同仅替换路径，文件SHA不可变：

```bash
flow_recovery_python=/data/flow_a/envs/v2/bin/python
flow_recovery_entry=/data/flow_a/code/research_20261014/fast_task_recovery/run_fast_recovery.py
flow_recovery_root=/data/flow_a/runs/fast_task_recovery_20261014
flow_recovery_common=(
  --parent-contract /data/flow_a/code/research_20261014/fast_warmstart/runtime_contract_v1.json
  --parent-contract-sha256 564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389
  --fit-manifest /data/flow_a/code/research_20261013/motion_groups/coarse_fit20_membership.json
  --fit-manifest-sha256 29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915
)
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$flow_recovery_python" "$flow_recovery_entry" \
  "${flow_recovery_common[@]}" --mode prepare --output-dir "$flow_recovery_root/masks_v1" \
  --preflight-output "$flow_recovery_root/preflight_v1.json"
```

preflight通过后，同一命令去掉 `--preflight-output ...`，改为 `--execute` 生成28目标mask bank。然后：

```bash
flow_recovery_bank="$flow_recovery_root/masks_v1/mask_bank.json"
flow_recovery_bank_sha=$(sha256sum "$flow_recovery_bank" | cut -d' ' -f1)
for flow_recovery_arm in R0 R1; do
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$flow_recovery_python" "$flow_recovery_entry" \
    "${flow_recovery_common[@]}" --mode "smoke_$flow_recovery_arm" \
    --mask-bank "$flow_recovery_bank" --mask-bank-sha256 "$flow_recovery_bank_sha" \
    --output-dir "$flow_recovery_root/smoke_$flow_recovery_arm" --execute || break
done
```

两臂真实烟测均通过并审查输出后，按R0、R1顺序分别执行下面命令，不在未核烟测时自动接正式恢复：

```bash
flow_recovery_arm=R0  # 第二臂改为R1；其他参数不变
flow_recovery_smoke="$flow_recovery_root/smoke_$flow_recovery_arm/results.json"
flow_recovery_smoke_sha=$(sha256sum "$flow_recovery_smoke" | cut -d' ' -f1)
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$flow_recovery_python" "$flow_recovery_entry" \
  "${flow_recovery_common[@]}" --mode "$flow_recovery_arm" \
  --mask-bank "$flow_recovery_bank" --mask-bank-sha256 "$flow_recovery_bank_sha" \
  --smoke-result "$flow_recovery_smoke" --smoke-result-sha256 "$flow_recovery_smoke_sha" \
  --output-dir "$flow_recovery_root/$flow_recovery_arm" --execute
```

输出目录必须不存在。每正式臂固定200训练前向＋更新前后actual/dual四次＋终点dev8两臂16次，共220次完整前向；反向各200次。prepare28前向、各烟测5前向＋1反向单列。R0与R1反向范围不同，实际时长/容量需从GPU烟测取得，当前不编性能预算。输出`weights.pt`仅fast与优化器状态，不是全模型checkpoint；schema `flow_fast_task_recovery_weights_v1`，stage2 `state_dict`，`smoke_only`/200步、原权重/source/mask/recipe均绑定。

本地检查使用CPU环境运行 `test_fast_recovery.py`，结果见 [cpu_checks.json](cpu_checks.json)。它验证真实autograd边界与小型数值参考，不冒充原生GPU全网烟测。没有硬件周期/ASIC结果。

版本记录：旧输出路径解析与CPU拟合模块导入的设备副作用已分别修复，前者使用解析前替换输出参数，后者仅直接加载已pin的FastBranch、不导入CPU拟合入口。prepare_v2在第六fit目标因共同可达点数非900按旧guard停止；该失败保留，不记算法失败。当前v3合同使用固定cap900和逐目标共同可达预算，完整重建28目标maskbank，不复用旧binding。
