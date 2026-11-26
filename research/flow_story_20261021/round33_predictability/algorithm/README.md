# R33 CPU 小预测器

设计与边界见 [DESIGN_CONTRACT_V1.md](DESIGN_CONTRACT_V1.md)。本目录只完成实现及合成核验，未运行真实拟合、GPU 或读取确认结果。root 统一调度实际数据、模型冻结和确认。

- `predictors.py`：公共 `voxel_features` / `make_labels`、固定七方法、序列 OOF 堆叠、可选完整 nested LOSO、只接受 X 的预测接口。
- `run_predictors.py`：强制 frozen membership 的 fit-only CLI；模型 hash 固定后才预测 tune/confirm。不会自动改配置、续训或扩大搜索。
- `test_synthetic.py` / `SYNTHETIC_CHECKS_V1.json`：5 组数值/泄漏检查全部 PASS；所有拟合数据均为生成的合成数组。
- `SOURCE_FREEZE_V1.json`：本目录源和设计/自测绑定，供 root 的 pack/模型 freeze 引用。

NPZ 每角色分文件；root `make_pack.py` 已与接口对齐：`ids`, `groups`, `roles`, `x[N,128]`, `delta_aee[N,6]`, `valid_pixels[N]`, `aux_g1[N,14]`, `aux_g4[N,224]`。其余离线 ideal/support 字段不作为可部署模型输入。预测 CLI 只打开身份/序列/角色元数据和 x 数组；身份和序列只作防串集验证，`predict_suite` 仅接收 x。

以下是 root 的模板命令，**不是本目录已经执行的真实实验**。将 `FIT_TUNE_PACK` 和 `OUTPUT_FIT` 替换为本轮实际新路径：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python -B \
  research/flow_story_20261021/round33_predictability/algorithm/run_predictors.py fit \
  --fit-pack FIT_TUNE_PACK/fit.npz \
  --membership research/flow_story_20261021/round33_predictability/data/MEMBERSHIP_V1.json \
  --output-dir OUTPUT_FIT --include-loso
```

输出 `MODEL_V1.json`（schema `flow_r33_voxel_small_predictors_v1`）、`FIT_RECEIPT_V1.json`（schema `flow_r33_fit_receipt_v1`），以及显式开启时的 `FIT_NESTED_LOSO_V1.json`。闭式训练无随机 seed；固定 alpha=0.1，不提供 CLI 超参数扫描。receipt 绑定 64 fit 身份/序列、模型、fit pack、membership 和程序源 SHA。root 再用 receipt/模型建立 `HEADS_FREEZE_V1.json`，确认 collector 验证后才允许新确认采集。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python -B \
  research/flow_story_20261021/round33_predictability/algorithm/run_predictors.py predict \
  --model OUTPUT_FIT/MODEL_V1.json --model-sha256 FROZEN_MODEL_SHA \
  --input-pack FIT_TUNE_PACK/tune.npz --role tune \
  --membership research/flow_story_20261021/round33_predictability/data/MEMBERSHIP_V1.json \
  --output OUTPUT_FIT/TUNE_PREDICTIONS_V1.json
```

确认时只将输入换成单独的 `confirm.npz`、role 改为 `confirm`、输出换成新文件，模型/hash 不变。`fixed_six_choices` 与 `allow_Q4_choices` 分开，索引 **−1=Q4，0…5=升级宏**。所有 7 方法固定输出，不根据结果选择较弱普通对手。最终质量与 16 区/逐序列/尾部由独立 reducer 使用已冻结选择复算；本 CLI 不读确认质量或产生论文收益结论。

模型中提供 float64 系数与标准化参数，尚无硬件定点/周期部署合同。辅助头、字段乘积、feature 读取、控制和双表示供数不能视为免费。
