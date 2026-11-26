# Native 小学习率稳定性诊断

2026-10-16。root 在完整 R13 八臂质量与 raw-GT 复算一致、16 次执行审计通过后，选择计划 §11 的唯一小学习率分支。本目录只实现和 CPU 检查；作者未运行 GPU/SSH、读取真实检查质量或修改冻结 R13/H0/H1。此入口不会启动 checkpoint 分支或评分器实验。

固定 `diversity/native`，学习率 `2.5e-5`，原值 `2.5e-4`。除 LR 外，`RECIPE` 每个字段逐项与冻结 R13 相同：同原生 checkpoint135930、共同新 BN、D200 一次顺序、200 更新、80 个参数共7,252,064值、四头全有效 GT loss、AdamW其他设置、seed、FP32/noAMP/noTF32、clip100、无augmentation。原R13检查8帧已经用于开发，不能再称独立确认集。

## 入口与来源

`run_native_lr.py`固定条件、臂和学习率，没有开放搜索参数。`execute_recovery.py`是固定 R13 执行循环的窄适配；新增独立 schema/status、同初始状态比较、native/FF 终点范围和参数实际变化统计。原 R13 数学 helper（完整反向、原参考前向、梯度/状态检查）、原生 loss 和新 BN 安装语义保持不变。

元数据 preflight 复用冻结 R13 `prepare` 的私有 globals：只替换本目录 `HERE/OWN/__file__`、新 recipe 和单 native smoke 校验。R13 prepare 无模型前向、无 BN 重校准，不调用历史旧 prepare/CMA。最终 binding 记录**新入口、新 execute、新 CPU receipt、新 recipe**，另明确保留原 prepare/execute 来源 SHA。原 R13 文件不变；入口拒绝使用旧四臂 smoke 代替新配方 smoke。

初始模型由原生 loader 加原 checkpoint，随后加载完全相同的合法新 BN。`baseline-result`必须为已完成 R13 `diversity/native` 原 LR 的 results，实际 SHA、原源代码/recipe、全部共同输入、D200顺序、8检查身份和闭合状态均核验。新模型第一次前向前，其完整 state SHA 必须与该原运行初始状态完全相同；正式运行每帧初始 native 预测 SHA 必须与对应原 R13 初始预测相同。

运行中不读取或装载 R13 恢复后的权重，也不恢复 optimizer。5步 smoke 丢弃权重；正式入口重新加载共同 W0、共同 BN，重新创建 AdamW。新末 PT 使用 `flow_native_lr_stability_recovered_state_v1`，含完整 model、optimizer、RNG、binding、初末 state SHA、允许参数名和 BN scope，不能误写为 R13结果。

## 固定预算

| 段 | 更新前向 | 初末数值检查 | 检查集预测 | 完整前向 |
|---|---:|---:|---:|---:|
| smoke | 5 | 4 | 0 | 9 |
| 正式 | 200 | 4 | 初native8＋末native/FF各8 | 228 |
| 总计 | 205 | 8 | 24 | 237 |

初末数值检查各为新可微 native 和旧原参考 native 两次，核四头、八选定MLP branch及输入逐位一致。native没有tiled dual，不另加FF dual。没有176次新校准，也不重跑原R13基线。

native和FF两种执行的各FC总行数均558000；smoke每FC合计5,022,000，正式127,224,000，总132,246,000。这是逐层观测行数需满足的几何合同，**不是周期、时间或能效结论**。每段共享 GPU0 非阻塞锁，SIGALRM与每次deadline上限30分钟；输出全新目录，失败保留 `INCOMPLETE_NO_SCIENTIFIC_CONCLUSION`，不自动续跑、换帧或增加预算。

全部初末24个最终预测保存16份NPZ；保留16区逐帧、总体、两序列指标和原数组SHA。每次前向独立记账；所有200步保留真实loss、preclip范数、梯度覆盖、同步时间和内存。参数变化额外以CPU FP64按FC/PSN/BN affine和每MLP归约初始范数、变化范数、相对L2、最大绝对变化，不增加模型前向。

## CPU检查与实际使用

7项检查通过：固定单轴recipe/237预算，原LR来源和8初始预测合同拒绝8类错误，单新smoke准入拒绝8类错误，私有prepare绑定本目录源码与24评价数，实际小原生MLP五步低LR反向及native/FF初末与旧实现一致，freshW0/freshoptimizer及BN/冻结补集/hooks闭合，独立参数漂移标量验证、嵌套globals完整性。测试只用合成元数据和小原生MLP；没有完整真实模型/数据，实际四头及完整执行由root的preflight/smoke继续验证。

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_native_lr_cpu.py
```

[launch.template.sh](launch.template.sh)列出 root 需填的实际 source path/SHA。默认只 preflight；运行须使用同一组SHA、新输出目录并加`--execute`。正式 preflight 还需唯一新 native smoke 的实际结果路径；其五步/九前向、同配方源SHA、梯度覆盖和闭合必须通过。模板故意使用不可执行占位符，没有自动队列。

这项诊断最多证明更小 LR 是否减少普通恢复漂移。即便native恢复稳定，仍需公平验证FF/粗分支；不能直接准许H0，不能宣布光流信息增量或硬件收益。判断与对照规则见 [固定分析边界](PROTOCOL.md)。
