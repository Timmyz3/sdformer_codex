# H0：恢复状态动作空间诊断的CPU准备

2026-10-16。只实现H0的69次前向，没有运行GPU、SSH、读取R13正式质量、读取真实恢复权重、排队或改任何冻结R13文件。[CPU记录](cpu_results.json)为9项合成检查通过，不能替代真实模型、数据与权重运行。H1、k4、预测器、训练和新的BN校准均不在此入口内。

## 固定问题与动作

[h8_manifest.json](h8_manifest.json)完全来自冻结R13的40个selector身份：按训练序列顺序，每序列取最小`(SHA256("flow-headroom-v1:"+target_id), target_id)`；8条序列各1帧。身份与D200恢复、176 BN、R13检查成员不交叠。选择未读取事件、GT或质量。骨干恢复见过这些序列；这里只是训练侧动作筛查，不能作跨序列确认。

| 状态/锚点 | 固定动作 | 全12 MLP的FC1/FC2各执行行数 |
|---|---|---:|
| W/native | 原生完整MLP | 558000 |
| W/FF | stage0两层+stage2六层全部fine tile | 558000 |
| W/CC | 同八层全部3×3代表coarse | 153600 |
| W/P | CC背景，stage2/block2全部20真tile fine | 164200 |
| W/F12 | CC背景，block2的12完整tile fine，8尾coarse | 162240 |
| W/T8 | CC背景，block2的8尾fine，12完整coarse | 155560 |
| matched_N | 同条件R13 native恢复权重，native布局 | 558000 |
| matched_F | 同条件R13 FF恢复权重，FF布局 | 558000 |

8锚点×H8=64次实际前向，第一帧CC/FF/P/F12/T8各加1次dual，共69。每个dual的两条producer都执行，各FC共609600行。以上是源码几何和CPU检查，**不是GPU实测时间、周期、能效或整网收益**。完整tile fine增量720行；8尾增量为`[300,300,300,240,240,240,240,100]`，合计1960行。

每帧CC先于P/F12/T8，核目标输入和前四个已选MLP prefix完全一致。FF改变上游动作，不误要求它与CC同prefix。全部4个flow头的shape/finite/SHA、8个MLP完整branch及输入SHA、所有12层独立FC hook行数、目标20tile真实几何记录。首帧5对比逐位核全部flow与branch，不仅最终AEE。预测保存全部8锚点，指标保留全16分区、逐帧及每序列；GT只用于评价，静态动作接口只接收几何/锚点。

## 恢复权重与共同BN绑定

执行需要root在完整R13审计之后提供明确`condition`和`W_arm`，不自动挑优：

- `qualified_ordinary_base`：W为CC或P恢复状态，冻结质量分析的普通底座及shared-fine门均通过。
- `action_only_screening`：N/F已稳定但当前动作底座不具备可部署门资格，W明确为同条件F恢复状态。只筛动作，不冒充底座通过。

这是科学前置，不是再次向用户请求许可。用户已经授权持续实验；选择该分支仍须由已完成证据决定。若N/F不稳，本入口拒绝，先普通稳定性诊断。若主要是shared-fine失败，既有M轴100自身+100FF普通控制仍是竞争解释，H0不是唯一出口。

`admission`必须同时绑定：

1. 冻结`analyze_coverage.py`产生的`analysis.json`：schema、脚本/协议/数学SHA、40状态布局、320预测、16区域、1964正式前向、8个结果来源和预测artifact审计均核验；普通参考稳定性和选定合格门由此读取，不能由admission口头布尔代替。
2. 冻结`independent_execution_audit/audit_execution.py`产生的完整`ACTUAL_EXECUTION_AUDIT.json`：PASS、两条件×smoke/recover四臂共16运行、8权重、320预测、128archive、2048前向/1640更新；独立重算初始/最终/冻结补集及72 BN值的核验必须存在。

本地报告中的来源绝对路径可与GPU服务器不同。`all_eight_results`显式列当前8个结果路径/SHA，按`condition/arm`和SHA对应原分析；独立执行审计按condition/mode/arm及result/weights SHA绑定。保留原证据路径，**不改写冻结报告**。W/N/F只能取自该同条件已审集合。

原`loader.load_model`只用于构造已核对的原生模型、加载原checkpoint的初始结构。随后先安装R13合法新BN，移除54 external BN的162运行buffer，再严格装入明确指定的R13完整恢复`model_state_dict`；在此之前没有推理。逐tensor核keyset/shape/dtype/finite、72共同BN完全相等、80允许参数名、200步及全部binding，最终完整state SHA须与原result及artifact一致。**旧CMA176 payload从未加载，也不运行任何旧prepare/execute。**

一次GPU模型依次使用W（53次）、N（8次）、F（8次），避免常驻三个GPU模型。切换前后核state与共同BN不变、原模块恢复、hooks清除；只加载模型状态，不恢复optimizer或继续训练。原artifact含optimizer，但读取后立即丢弃未使用payload。

## 可运行入口与仍缺的真实材料

[run_h0.py](run_h0.py)是完整入口，[h0_core.py](h0_core.py)提供可单测编排和冻结数值adapter。`source_pins.json`核历史数值代码；probe函数通过私有globals替换静态mask wrapper，原历史模块不被修改。原生参考走R12→R9冻结函数，FF/CC/P在CPU上另与旧对应参考逐位比对。

现在缺项是**实际入参证据与科学分叉**：完整正式R13质量/独立执行报告、选定条件/W、该W及匹配N/F真实权重的路径/SHA、当前共同BN/runtime路径。尚未读取或填写它们，也未做完整真实模型CPU/GPU预演。提供的[inputs.template.json](inputs.template.json)、[admission.template.json](admission.template.json)使用不可执行占位符；不能把模板当PASS材料。还需要root完成运行环境的数据/模块可达性preflight，以及独立读审闭合；本目录并不宣称这些真实条件已具备。

CPU复核（本地已执行，无GPU）：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_h0_cpu.py
```

root补齐已审证据后先运行默认preflight，`--inputs-sha256`填写实际JSON SHA，输出必须是全新目录：

```bash
python -B run_h0.py --inputs /absolute/h0_inputs.json \
  --inputs-sha256 ACTUAL_INPUT_MANIFEST_SHA \
  --cpu-checks /absolute/implementation_h0/cpu_results.json \
  --output-dir /absolute/new_h0_preflight --wall-minutes 30
```

实际运行仍用同一inputs SHA，另选新输出目录，加`--run-inference`。入口持共享`/data/flow_a/runs/.flow_a_gpu0.lock`非阻塞锁，CUDA在拿锁后初始化，硬SIGALRM与逐次deadline均上限30分钟。只有完成全部69次、8个全锚点预测和状态闭合才写`H0_69_FORWARD_COMPLETE`；失败标`INCOMPLETE_NO_SCIENTIFIC_CONCLUSION`，不自动换帧、续跑、加预算或接H1。默认preflight也不创建可自动执行队列。

9项CPU检查包括：H8角色与身份；生产真实尾几何与各arm行数；小原生MLP上5 actual/dual和4历史锚点参考完全一致；异常hook/模块清理；三个合成恢复state的严格序列装载与21类拒绝；实际编排函数的69调用mock；未完成/不稳准入拒绝；冻结完整分析、独立执行审计与跨机SHA绑定的14类拒绝；所有入口嵌套全局依赖已绑定。fixture只有两个人造输出头，因此只证明适配器数学/控制，真实4头另由运行时严格核验。

未实现损伤/边际收益分析或论文图，也未选择最终故事成功与否。该collector只为root后续完整审计提供动作空间证据，不输出flow特有增量或硬件收益结论。
