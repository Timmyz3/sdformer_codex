# CODEBUDDY 工作交接报告（2026-09-17 ~ 09-19）

> 本文档汇总本会话在 `/root/private_data/work/sdformer_codex/SDformer/` 上的全部工作，供 codex 统一接管。
> 每节末尾给出【交接状态】：已完成 / 待接管 / 已封存勿动。
> 配套详细文档均在 `neuron_experiments/` 下，本文是索引+结论层。

---

## 0. 一页总览

| 工作线 | 状态 | 一句话结论 |
|---|---|---|
| ATLIF 阈值不更新根因 | ✅ 已定案+修复 | 4 个 backward 恒负梯度缺陷，H12 副本已修+单测全过 |
| 死神经元问题 | ✅ 已定案 | binary 线 19 个真死（跨模型一致）；sn_k 死块旁路 bit-exact 验证通过 |
| 三元 warm start | ⛔ 已封存 | zero-shot 转换即碎（firing 6%→46%），全线实验失败有档 |
| stage0-local5 / motion 简化 | 📐 数学证明完成，代码未落地 | S1 popcount 恒等式 bit-exact（9 组对拍）；用户已决定**注意力降级不管** |
| 顶会调研（算法+硬件） | ✅ 完成 | 硬件 Top10 排行 + TCAS-II 投稿画像 + 对比锚点 |
| 硬件路线规划 | ✅ 完成 | 四阶段（A 死块旁路 DC 交付 → B 补 RTL → C 学新文 → D 对比表） |
| 服务器实验 | ⚠️ 有并行会话 | t49-t52 是另一会话的队列，勿动；我方实验全部在 H12 目录前缀 h12* |

**当前基线**：local5 ep44，DSEC valid825 AEE **1.2819**（硬件定点 1.2786-1.2804），优于 H67 ep35 的 1.3287。

---

## 1. ATLIF 阈值不更新（"阈值都约等于 1"）—— 已定案 + 已修复

### 1.1 根因（双层）

**代码层**（`H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`）：
四个 Surrogate 变体（Ternary:30-38 / Binary:52-58 / SymmetricBinary:75-83 / **OfficialATLIF:105-112，H67 主线用的就是它**）的 backward 全部有同一缺陷：

```python
grad_input = grad_input * tmp          # 先覆盖上游梯度 g
grad_thre = -(grad_input.abs() * tmp).mean()   # tmp 乘了两次(tmp²)；.abs() 毁符号 → 恒 ≤0
```
缺失恒等项 `d(out)/d(thre) = ternary`（out = ternary×thre）→ Adam 通道只会单调推高阈值，无闭环。

**配置层**：主线部署配置 `threshold_init: 1.0` + `threshold_eta: 0.0` + `activity_eta: 0.0`（如 `date11_allbinary_tx_ft_ep19_deploy_score_int8_gate_int8.yml:67-89`）→ θ 从未被训练，恒为初值 1.0。

另有独立 bug：E2 `entrypoints/train.py:179-190` 的字符串 anchor 补丁中 `LOSS_REGULARIZATION_ANCHOR` 在当前基线 count=0，正则静默失效（修复规范已写入设计文档 §3.4，**未实施**）。

### 1.2 修复（已完成，在 `neuron_experiments/H12_atlif_gradfix/`）

- overlay 整体复制自 H9（**遵守 `hw_autoresearch_nts07/autoresearch.md:42-44` 禁改 overlay 约束**，先例 H11→H9→H12）
- 四个 backward 统一改为：`grad_thre = (g·code).mean() − (g·tmp).mean()`，forward 增存离散码（ternary/active/out）
- 单元测试 `tests/test_gradfix.py` 四判据全过：
  1. 无发放场景 grad_thre 翻正（缺陷版恒负死锁）
  2. 饱和场景恢复负反馈（缺陷版恒 0）
  3. 恒等项手工微分对照精确一致（1/30）
  4. **闭环动力学：修复版 thre 3.00→1.86、firing 0→5.3%；缺陷版死锁 2.99/0**

### 1.3 实验证据链（都跑过）

| 实验 | 结果 | 位置 |
|---|---|---|
| t52_eta_on/off（并行会话跑的） | 缺陷版开 eta：0.008→阈值仅爬到 1.097；0.1→仅 1.02，只升不降 | `H9.../results/t52_*` |
| h12base probe（θ=1.0 原样+修复 backward，600 步） | AEE 7.76 —— 但这暴露的是三元转换问题（见 §3），**修复无罪**（θ 静态时修复不生效，行为与 H9 等价） | `H12.../results/h12base_probe/` |

**【交接状态】**：修复完成。**待办**：binary 线 eta>0 短训练（2-3h GPU）给修复做完整训练验证；E2 anchor fail-fast 修复未实施。

---

## 2. 死神经元 —— 已定案，方向反转成贡献

### 2.1 二元线取证（2026-09-18，双模型交叉验证）

探针：`/tmp/dead_neuron_probe.py`（stub sys.modules 加载，复用 H9 train.py install 链）。
数据：`/tmp/deadprobe_A/B_theta_*.json`（h67ep35 / local5ep44）。

| | H67 ep35 | local5 ep44 |
|---|---|---|
| mean firing | 0.0660 | 0.0687（健康） |
| 死模块 r<0.001 | 9 | 10（全在 attention sn_q/sn_k/proj_sn） |
| **θ×0.1 反事实复活** | **0/9** | **0/10** |

**结论**：binary 线死模块是**真死/结构性静默**（与三元线的"尺度死"6/7 复活完全相反）；两个独立训练 ckpt 名单几乎重合（layers.1.blocks.0 的 sn_k/sn_q/proj_sn、layers.2.blocks.3 的 sn_q/sn_k、layers.0.blocks.0.sn_q、layers.2.blocks.1/5.sn_q、layers.3.blocks.1.sn_q）→ **架构级性质，可泛化**。

### 2.2 静默块旁路（bit-exact 已验证 ✅）

推理链：sn_k 死 → K≡0 → attn = K×gate ≡ 0 → 该 block attention 分支输出恒零 → 可整段删除。

位级验证（`/tmp/verify_deadskip.py`，输出 sha256 逐 batch 对比）：

| 模型 | 跳过的 block | 结果 |
|---|---|---|
| local5 ep44 | layers.1.blocks.0 / layers.0.blocks.1 / layers.2.blocks.3 | 3 batch digest 全等 ✓ |
| H67 ep35 | layers.1.blocks.0 / layers.2.blocks.3 | 3 batch digest 全等 ✓ |

mask：`/tmp/binary_deadmask.json`。文献背书：LoAS (MICRO'24) silent-neuron 剪枝。

**【交接状态】**：取证+验证完成。**待办**：①mask 固化为配置选项 + valid825 全量复测（半天）；②DC/PTPX 交付包加"死块数据通路删除"对比行（远程 Synopsys 流程）；③近死模块（7-9 个 r 0.001-0.01）跳过收益评估（有损，非 bit-exact，需验证门）。

---

## 3. 三元 warm start —— 封存（NO_GO，勿再投入）

完整证据链（`CLOSURE_TERNARY_WARMSTART_20260918.md`）：

1. **决定性**：t49 三元 zero-shot（不训练直接评）：firing 6%→**46.5%**（7.3×）、spikes 597G（基线 82G）、AEE 20.19/84.4/82.9、outlier 99.8% → **转换本身破坏网络**（正阈值不变，发放率不该变；源头未定案——时间混合器 W/bias 语义、center、neg_thre=4 不对称是候选）
2. t49 三元 5-epoch：val loss 7.72→**8.41 发散**；并行会话已悄悄退回 binary（t52 全部 output_mode: binary）
3. 我的 h12cal（一次性 θ 标定）AEE 7.78；h12s0（+stage0-local5）7.78；h12cal2（v2 迭代闭环标定，firing 已收敛 0.32）**8.39**；h12base（θ=1.0 原样）7.76 → 排除标定/backward 嫌疑，锁定"三元转换本身"
4. **根因**：ep34 权重与 binary 语义共同训练适配；负脉冲与任何 θ 重缩放都打碎协同（残差路径不同尺度直接相加），5 epoch 无法恢复

**教训**（已记忆）：死神经元 ≠ 需要激活；warm start 下改神经元取值域 = 重训练级工程；θ 重标定只适用于从头训练。

**遗留未定案**：zero-shot 46% 发放爆炸的精确源头（差分探针半天可定位——同一输入过 binary PSN vs 转换后三元模块对比 h_seq 分布）。**若将来重启三元（从头训练），必须先做这个定位。**

**【交接状态】**：封存。t49-t52 是并行会话的实验目录，不要清理，其结论已引用进封存文档。

---

## 4. 注意力侧 —— 数学证明完成，用户已决定降级不管

（用户 2026-09-19 决策：attention 只占 0.59% 算子份额，不影响硬件主线，不再投入）

已完成、可供将来捡起的资产：
- **S1 popcount 恒等式**（bit-exact 已证，9 组 GPU 对拍，`/tmp/motion_parity.py`）：
  `64·D·s = 65·o − q − k + opp + 16·m + D`（o/opp/q/k/m 全是 popcount，α₀=1/64, motion α=0.25）
  实现 spec：H12 overlay `bsa_attention.py` 新增 mode `h60_pc` 替换 `_tx_sc_fusion_score_pair`（:3411）调用点，~80 行
  顺带删除 SC/opposite/single_active/K_mag 四个死分支（约 40% 前端算力浪费，行号证据在 `RESEARCH_MOTION_HW_IDEAS_20260918.md` §1）
- **stage0-local5**：`bsa_attention_stage0` 双段安装已实现进 H12 train.py（`stage_selection: stage0` + 主段 `target_blocks` 覆盖 stage1-3，配置生成器 `build_stage0_config.py` 就绪）。binary 线 + θ=1.0 原样的干净版**未跑**（h12s0 被坏标定污染）
- local5 全 12-block 版就是当前最好模型（AEE 1.2819），RTL 四件套与软件逐位一致

**【交接状态】**：暂停。若重启，先跑 binary 线 stage0-local5 干净实验。

---

## 5. 顶会调研结论（算法+硬件，全部 WebSearch 核实出处）

文档：`RESEARCH_MOTION_HW_IDEAS_20260918.md`（硬件 Top10+motion）、`ACTION_PLAN_DEADNEURON_MOTION_IDEAS_20260917.md`（死神经元+早期调研）。

### 5.1 硬件要抄的（按"打到 99.4% 主体算子"排序）

| # | 来源 | 抄什么 | 落点 | 原文数字 |
|---|---|---|---|---|
| 1 | LoAS (MICRO'24, 开源 RTL) | 静默神经元剪枝 | §2 已做，只差 DC 交付 | +20% 叠加收益 |
| 2 | MM2IM (arXiv 2507.07683) | deconv→col2im 矩阵乘，跳 cropped 无效区 | decoder deconv 栈（主体大头） | TCONV 2.7×、能耗 1.7× |
| 3 | NJUST 稀疏位置编码 (2501.07825) + COMPASS (MICRO'24) | conv 输入行全零跳整行 | conv 主体 | 13.2×/26.7× |
| 4 | MI-TRQR (NeurIPS'25, 开源) | 相邻时间步冗余掩码 | conv/PSN 时间维 | 能耗 −37.5% 且精度 +1.7% |
| 5 | ESDA (FPGA'24, 开源) | 稀疏 token dataflow | **patch embed 新 RTL 直接按稀疏形式写**（补缺口） | 10% 非零输入 4.5-11× |
| 6 | FPGA Spikformer (开源 RTL) | conv+BN 融合、多时间步并行 LIF | 补 MLP-FFN RTL 抄结构 | — |
| 7 | Nebula (ISSCC'25, 28nm 硅) / STI-SNN | 按稀疏度跳层 / 时间轴裁剪 | 逐层跳过 | 109.8 TOPS/W |
| 8 | ICG 时钟门控 (ISCAS'21 锚点) | 无发放 lane 门控 | 所有累加行 | 动态功耗 −20~40% |
| 9 | Voyager (Phil Trans 2026) | iso-resource 公平对比口径 | TCAS-II 对比表 | 否则一票否决 |

**对比表锚点（28nm 硅实测主线）**：ReckOn 5.3 pJ/SOP（0.45mm², JSSC'23）/ ANP-I 1.5 pJ/SOP（1.628mm², JSSC'24）/ **NYCU ISCAS'25 38.3 TSOPS/W（同工艺同网络族，最高审稿风险，必须引用并差异化：事件光流任务/三元神经元/精度闭环）**。FireFly（FPGA）不入主表。

### 5.2 硬件四阶段规划

- **Phase A（证据在手只差交付）**：死块旁路 → DC/PTPX 对比包
- **Phase B（TCAS-II 硬门槛）**：补 patch embed/MLP-FFN/PSN 三块 RTL + 收 m931 WNS −4.91ns（不做完 PTPX 6.25mW 就是系统性低估）
- **Phase C**：MM2IM 重写 deconv + 跳零统计 + MI-TRQR
- **Phase D**：Voyager 口径对比表 + NYCU 差异化

### 5.3 算法候选（暂缓，用户指示等神经元/注意力收尾后再做）

TP-Spikformer training-free token 剪枝（ICLR'26）、SPEAR 通道剪枝（NeurIPS'25）、特征级 KD 补偿；Ternary Spike (AAAI'24) 梯度公式对照 H12（引文背书用）。

**已证伪勿碰**：全栈三元换 carrier（AAE 71.6）、官方 TSN 全局替换（29.77）、三元扩 FFN（SOPs 变密）、neg_scale=30、Winograd、attention 内部优化（份额天花板）、XNOR-popcount 全栈二值。

**【交接状态】**：调研完成，待按四阶段执行。

---

## 6. 服务器/环境注意事项（给 codex 的重要交接）

1. **并行会话**：`/root/t49_ternary_retrain.py` 及 t49-t52 队列是**另一个会话**驱动的，**不要动它们的目录**；其结论已被我的封存文档引用。GPU 抢用规则：双方都用"等显存 <2GB 连续 3 次"的排队脚本（我方模板 `H12_atlif_gradfix/queue_*.sh`）
2. **环境**：训练用 `/opt/conda/envs/sdformerflow/bin/python`（有 mlflow/spikingjelly/cupy）；`env312` 缺 mlflow 不能跑训练入口；GPU A800 80GB
3. **DSEC 数据**：`/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC/saved_flow_data`；ckpt 加载的版本漂移坑（OfficialATLIFSurrogate/_qk_shiftmax_gate_forward 缺失）用 stub sys.modules 绕过，模板见 `/tmp/dead_neuron_probe.py`
4. **评估器**：`H9.../entrypoints/profile_checkpoints.py --config X --run-dir Y --epoch N`（默认 valid40，出 `AEE=/AAE=/SOPs=/firing=` 行）；valid825 全量用 `run_h9_standard_valid825_eval.py`
5. **H67 评估配置**：`configs/generated/dsec_fullres_w15_H67_ep35_score_qf8_gate_q17_sensitivity.yml`（deploy float_ref 配置分辨率不匹配会 shape 报错）
6. **overlay 冻结**：`H9/H11 overlay 禁改`；实验一律复制到新目录（H12 先例）
7. **本机无商业 EDA**（无 dc_shell/vcs/pt_shell，无 license）；ASIC 全流程走远程 `ic.ismd-nemo`（Synopsys，流程见 `hw_autoresearch_nts07/dc_handoff/SERVER_RUN.md`）；本机只有 iverilog/verilator/yosys/openroad
8. **API 配额**：昨天（09-18）15:20 左右 429 重置于 19:35；agent 并发 >4 容易触顶，建议串行或 ≤3 并发

---

## 7. 待办清单（按优先级，GPU 当前空闲）

| # | 任务 | 成本 | 产出 |
|---|---|---|---|
| 1 | 死块旁路配置化 + valid825 全量复测（local5 ep44，预期 AEE 纹丝不动） | 半天 | 论文第一组"零精度损失省算力"数字 |
| 2 | H12 修复 backward 的 eta>0 完整训练验证（binary 线） | 2-3h | 神经元修复闭环 |
| 3 | Phase A：死块旁路 DC/PTPX 交付包（远程 Synopsys） | 1-2 天 | 第一个硬件数字 |
| 4 | Phase B：补 patch embed/MLP-FFN/PSN RTL + 收 m931 WNS | 数天 | TCAS-II 硬门槛 |
| 5 | 12 个 sn2_q 结构死参数剔出 installer 统计 | 零成本 | 统计净化 |
| 6 | （暂缓）training-free 剪枝评测、S1 h60_pc 实现 | — | 用户指示暂缓 |

---

## 8. 我产出的全部文件索引

**仓库内（`neuron_experiments/`）**：
- `CLOSURE_TERNARY_WARMSTART_20260918.md` —— 三元封存 + §5 binary 线取证/旁路验证
- `H12_ATLIF_GRADFIX_TERNARY_DESIGN_20260917.md` —— ATLIF 修复规范（行号级）+ 三元防护设计（历史参考）
- `ACTION_PLAN_DEADNEURON_MOTION_IDEAS_20260917.md` —— 死神经元定案 + 一周排序
- `RESEARCH_MOTION_HW_IDEAS_20260918.md` —— motion 简化 + 硬件 Top10 + TCAS-II 数据包清单
- `H12_atlif_gradfix/` —— 修复 overlay + tests + entrypoints（train/calibrate_and_train/build_stage0_config）+ queue 脚本 + results/h12*（含失败的标定实验，留档）

**/tmp（重要临时资产，建议 codex 收编）**：
- `dead_neuron_probe.py` —— 死模块取证探针（stub 加载模板）
- `verify_deadskip.py` —— 旁路位级验证
- `binary_deadmask.json` —— 剪枝 mask（H67/local5 两份名单）
- `deadprobe_*.json` —— 取证原始数据
- `motion_parity.py` —— S1 恒等式对拍（bit-exact 证据）

**未跑的干净实验 spec**：binary 线 stage0-local5（`build_stage0_config.py` 生成配置即可跑）；h12base 全量 5-epoch（θ=1.0 + 修复 backward，对照 t49 缺陷版）。
