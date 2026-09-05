# 接手提示词：H67 / Motion 光流 SNN 硬件机制重构

更新：2026-09-05 至 09-06 交接时。本文件供新 agent 完整阅读并接手，不要求恢复旧超长 session。
它记录证据、当前认识及待办，不是“顶会创新已经成立”或“强接收已达成”的证明。

## 0. 用户当前真正要什么

你正在接手个人科研项目：自研/改造的 SDformerFlow 系列事件相机光流网络，以及对应 28 nm 数字硬件。
目标期刊目前是 **IEEE TCAS-II**；此前探索过 DATE、ISCAS，现在不要自动转回它们。
用户希望机制达到顶会级辨识度，而不是继续为旧 C1/C2 小修补、换缩写，然后自评 Strong Accept。
近期用户明确要求：

1. 仔细、完整阅读 `/home/zhumd/work/ideafromai/` 的每个文件，综合多 AI 意见，但不能无脑接受。
2. C1、C2/TSBG 都可以大改、替换或退成基线；不要为了保住旧三贡献结构牺牲研究质量。
3. 研究 ANN、SNN、光流加速器，以及只有算法/开源代码但没有专用硬件的工作；还要从本模型真实 ATLIF、Motion-XOR 语义出发自己思考。
4. 要新的可验证机制，允许借鉴成熟原理，但必须引用、解释实质变化，并和强基线比较。仅改名称、换任务、说“首次”不能代替贡献。
5. 不再以“两周内能完成”为主要淘汰标准；先找值得独立重做硬件子系统的问题。不要求同时上所有候选。
6. 尽量少做重复 hash、层层合同、启动审批脚手架。保留必要的测试和结果边界，把时间花在实验、机制和电路上。
7. 已授权正常硬件推进、新思许可证及 git 推送。不要因为检索到的 AI 卡片写“立即执行”就启动另一套训练或破坏其他任务。

既有 frozen 模型是精确比较的参考对象。真正多级 ATLIF、阈值重训、FFN 裁剪、光流预测门控属于新算法分支，必须显式区分并重新做精度评价；不能悄悄替换基线。
用户最新关于 INT8 的意见是“ATLIF 应该也说得通”，不是授权把 checkpoint 偷换成多级发放模型。

## 1. 工作目录、分支与工具

| 用途 | 真实位置 |
|---|---|
| Git 根目录 | `/home/zhumd/work/sdformer_codex` |
| 主要工作目录 | `/home/zhumd/work/sdformer_codex/SDformer` |
| 硬件根目录，以下记为 HW | `/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07` |
| 算法实验及部署 overlay | `SDformer/neuron_experiments/H9_bipolar_self_attention/` |
| 原模型实现 | `SDformer/third_party/SDformerFlow/` |
| 多 AI 意见的 canonical 目录 | `/home/zhumd/work/ideafromai/`，注意是 idea，不是 HW 内旧副本 ideas |
| 旧新思/DATE 工作副本 | `/home/zhumd/work/synopsys_date_dual/` |
| 另一硬件工作目录 | `/home/zhumd/work/hw_autoresearch_nts07/`，不要假定和 canonical 同步 |
| Grok Bot 隔离实验 | `/home/zhumd/work/sdformer_c1c2star_grokbot/`，不要直接合并 |
| 当前稿 | `HW/paper/tcasii/main.tex` |
| 旧 ISCAS 稿 | `HW/paper/iscas2027/main.tex` |
| 综合/时序/功耗流程 | `HW/dc_handoff/scripts/` 和 `HW/dc_handoff/runs/` |
| 建模与筛查代码 | `HW/system_simulator/scripts/` |
| 小实验结果 | `HW/results/`；多数结果被 gitignore 忽略，不代表不存在 |

分支 `autoresearch/neuron-ops-20260507`。本交接稿写入前 HEAD 为 `7e3d3030`，后续提交会推进 HEAD，以 `git log -1` 为准。
这个提交包含 C1 调度/压缩筛查和 cofill 映射结果，不是新的顶会机制已实现。

解释器请使用实际验证过的环境：

- NumPy CPU：`/opt/anaconda3/bin/python`。
- PyTorch CPU：`/opt/anaconda3/envs/pytorch310_cpu/bin/python`。
- 系统 `python3` 为旧 3.6；`/usr/bin/python3.12` 没有 NumPy。不要照旧交接稿随意换解释器。

新思共享锁：`/tmp/date_dual_synopsys_same_uid_eda_queue.lock`。历史许可证服务 `27030@ic.ismd-nemo`，先看现有 runner 配置。
只用 Synopsys 做本项目正式 HDL 验证/PPA；不要把开源库工艺结果当 28 nm 证据。
当前主 agent 未启动新 EDA。机器上有其他人的 simv/VCS 进程，不能全局 kill。
宏目录 `/opt/tech/tsmc28/Memory/`；双端口编译器包确实存在，但未等于已经生成、验证所需实例。
工艺库、宏编译器和训练服务器口令不放公共 Git。

**必须保留**：`docs/359` 不动；其他人的未提交修改不覆盖。
本轮开始时已有他人修改 `HW/reviews/tcasii_accelerator_story_and_next_ideas_20260905.md`，不属于本轮成果，不顺手提交。

## 2. 网络对象和 ATLIF：先纠正最重要的前提

主对象为 **Motion C12 ep34**，H67 配置使用 h60 分支、Motion-XOR alpha=0.125、K 兼作 V。
历史 valid825 AEE 约 1.199514，发放率约 5.6709%；这是该验证协议，不是任意 capture 的统计。
冻结软件配置 `hardware_quant_enabled=false`。已有 Q7/Q1.7 Shiftmax 或 INT8 导出是部署候选，不能直接称 frozen 软件路径。

数据在：

- `HW/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth`
- `HW/system_handoff/incoming/m2041_ep34_quant_binding_inputs/dsec_c12_alpha0125_ep29_resume5_20260830.yml`
- `HW/results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/`

### 2.1 105、93、81 是三种口径；没有 85

| 数量 | 含义 | 可核查依据 |
|---|---|---|
| 105 | checkpoint 中安装的 ATLIF/PSN 模块实例；不是 105 个标量神经元 | 105 组 `.thresh` 及其模块参数 |
| 12 | 当前 h60 分支不调用的 `attn.sn2_q.spiking_neuron` | checkpoint 中存在，40-sample capture 不出现 |
| 93 | 实际被调用的模块，48 个 T2 + 45 个 T10 | `atlif_activity.json` 的 93 个唯一名字，每个 `calls=40` |
| 12 | 被调用的 `attn.attn_sn.spiking_neuron`，都是 T2；其输出仅是注意力诊断支路 | attention forward 与 caller 源码 |
| 81 | 以光流预测为功能输出的推理图中，结果继续参与计算的模块：36 个 T2 + 45 个 T10 | 93 减去上述第二组 12 |

两组 12 互不重叠，不能写成“93 中有 12 个从未调用”。模型也没有被我们删成 81 个。
**85 没有本轮源码或 capture 支持。** 不猜测它来自哪次口误；若其他稿有 85，按具体名单追查。
12 个 attention block 分布为 stage 0/1/2/3 的 2/2/6/2，不是每层两个。

源码证据：

- `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`，约 6083 行 h60 分支不调用 sn2_q；约 6620 行是：

  ```python
  attn = self.attn_sn(x)
  x = self.proj(x)
  # ...
  return x, attn
  ```

- `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py` 约 805 行拿到 `attn_windows, attn_score`，普通 forward 继续使用前者；只有 `return_attention=True` 才返回后者。
- capture 的 `deployment_dead_result` 是按模块名字设置的标签，不是自动数据流证明；本轮额外核对了上述调用关系。

因此：统计实际 PyTorch 运算应计 93；描述 installed 应计 105；只优化光流输出推理图可用 81。若需要导出 attention 诊断输出、统计副作用或未来训练辅助损失，不能直接删掉那 12 个调用。
本轮没有删除任何模块，也没有进行完整“移除 12 个诊断分支”的网络 AEE 对照。

### 2.2 二值不等于 ATLIF 无用，但也不是任意 INT8 幅值

实际模块文件：
`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`。

当前公式是：

```text
h = A x + bias
y = theta_module * 1[h >= theta_module]
```

`OfficialATLIFSurrogate.forward` 返回 `{0, theta}`；theta 为模块共享标量。
PSN 前向通过 `torch.addmm` 进行完整时间矩阵混合，**不是传统 LIF 的逐步 leak/reset 递推**。
配置 `factor_rank=0` 表示不用低秩分解，**不是矩阵的数学秩为零**。

M2270 已直接加载实际 checkpoint、实际 neuron class，做 93 模块随机输入 CPU 测试：

- 140,322 标量对 `theta * (h >= theta)`，0 mismatch。
- 105 个 theta 中 95 个严格等于 1，10 个略低；范围 `[0.9998828172683716, 1.0]`。
- 93 个调用点中 83 个 theta=1；81 个光流图点中 71 个 theta=1。这些 **83/71 是参数统计，不能和模块数量混用**。
- 参与光流的 45 个 T10 矩阵均为 rank10，36 个 T2 均为 rank2；有非零跨时刻权重。这说明有实际时间混合，不证明其 AEE 增量。

检查到的 ep29→ep34 resume YAML：threshold_eta=0、activity_eta=0、target_rate=None、target_rate_eta=0；手动阈值增长在该配置关闭。
但 `threshold_lr=5e-6`，`trainable=all`；`threshold_freeze_after_step` 只冻结手动增长，任务梯度冻结另有开关。
独立 surrogate 小例子证明 eta=0 仍可能有 theta 梯度；不等于每一条网络支路都有非零梯度。
不要由这一份 resume YAML 推断全部历史训练都没用过活动正则，也不要断言那 10 个非 1 参数必然在最后五轮产生。

**INT8 的正确讨论**：权重、ATLIF 输入电流、部分模拟残差可以用 INT8/定点作为部署精度；`{0,theta}` 也能编码。但格式为 INT8 不代表每个非零发放拥有独立的 8-bit 幅值。
theta 可以在实数线性代数中提到外部或折入权重，但 FP32/TF32 与定点的重新结合/舍入不自动 bit-exact，不能据此删掉数值绑定。

待验证的算法消融：

1. 保持比较阈值 theta，只把发放幅度改成 1，单独测幅度作用。
2. 比较阈值也改成 1，单独测阈值作用。
3. 有/无自适应活动正则的匹配训练，而不是拿当前二值输出直接判 ATLIF 无效。

尚未做这些全网 AEE/训练消融。没有授权它们自动占用训练服务器。

## 3. 现有架构和完成度：有组件证据，没有强接收保证

### C1：四层 bottleneck Conv3x3 的 subset/product-capture 岛

- 核心 RTL：`HW/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv`。
- 64-row × 16-source，96-lane；二值 source、signed 权重；parent 结果加 residual。
- 最大 popcount 合法 subset，平局按原 index；稳定执行顺序、parent scratch、保留结果写回、未被选为 parent 的结果免写。
- **当前 task 全部 64 行可驻留，没有 task 内驱逐。** parent_live 是静态被选作父行标记，不是动态最后消费者释放。
- 九个 128×128 1RW 宏是 1152-bit 宽度切片，不是九个可免费并行访问不同 row 的独立 bank。
- parent 向量为 96×12bit=144B；逻辑64行9KiB，物理128行18KiB。不要和 C2 的 Acc24 向量混淆。
- 旧可用组件数字：同 trace CPU 账本约 1.6945× vs strongest-zero；九宏映射岛 area 约166,514 µm²，并有对应 VCS/DC/PT/FM 工件。CPU 倍率不是完整 RTL 或全网速度。

创新性核心问题：subset、prefix、residual、forest、稳定依赖排序都与 Prosperity 高度重合。Prosperity 已有有限 buffer、16-wide tile、时间维展开，不能说它没有这些。
引用并做实现不等于抄袭；但只剩 1RW 适配不足以自动支撑强创新。现有约1.69×也不是我们相对 Prosperity 的提升。
“首选 parent 被驱逐、Top2 救回”的建议不符合当前 RTL；如果重做更小容量才可能成为新实验对象。

### C2：typed K8 / Acc24 + B4 TSBG + selective cofill

- 基础 scheduler：`HW/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv`。
- 新 selective/cofill：`HW/rtl_m2249/m2249_c2_consumer_scoped_bank_fill_frontend.sv`。
- 同一计算值，四个独立输出 context，按非零 source 供权重；token-major/group-major 改变权重请求共享。
- 旧 K8 对等带宽 K1×8：1913 vs1945 cycles≈1.0167×，logic throughput/area约4.54×、面积约−77.6%。这是并行累加/面积组织，不是额外4.76×稀疏。
- 各版本面积不能混用：后来的完整门控 ordinary/TSBG 为234,537/235,701 µm²，增量0.496%；两边 setup/hold 在对应理想时钟约束下非负，极小 hold 余量不是 post-CTS 签核。
- 后来 cofill 的 group-demand/union 配对 area190,949/191,635 µm²，增量0.3592%；setup +306.406/+5.909ps，hold +0.052/0.000ps（报告舍入）；两轴 mapped→mapped FM 各77,247 PASS。不要写成 RTL→mapped 等价。

cofill 最新同样 group-major 强基线、同 bank reads，4320 个冷启动 G48 chunk，1-cycle SRAM 模型由9个 VCS pilot 校准：
局部服务约 **1.23276×**，refill事务−36.16%，**bank read数量相同**。24块略慢。旧1.3097×来自旧服务假设，不应顶替最新模型。
这些4320块不是所有网络 token，不是整机速度。

已测旧 M2018 配对 gate-SAIF/PTPX：low reuse 能量**增加11.53%**，median减少26.15%，high减少62.12%。
这是候选 INT8 FC 权重的选窗、TT0.9V25C、零延迟 gate activity、logic-only；无 SRAM/CTS/SPEF。不是精度准入、总体平均或 frame energy。
**新 cofill pair 的 matched power 尚未完成，不能借旧功耗点。**

创新性问题：广播、Gustavson、请求合并、cache partial-valid、消费者计数已有 ELSA/Bishop/SpikeX/OuterSPACE 等直接先例。
只改乘法/供权重顺序、新缩写、typed signed 协议，不足以当新计算机制。
暖缓存下全零拷贝841→1094cycles的退化必须保留；当前 bridge_ready 是 bundle级，不是四个独立 ready，所以“一个慢consumer堵一个bank”的例子不能直接套用。

### C3 / attention / decoder

C3 是 Fixed-T10 ATLIF/PSN 整数服务覆盖，已有工具链组件证据；没有已成立的独立加速贡献。
attention 的 Motion-XOR / Shiftmax / RQTB 有局部实现与历史工作，不能靠0.59%历史工作份额推出新的全网大倍率。
decoder 有捕获、计算映射和旧长期重放工作，但本文交接未重新审计其全部结果；不能写“完整 memory-inclusive 全网已闭合”。
旧 Table-A、选窗、CPUsim、映射岛、完整系统是不同层级。不要为了期刊短文强行补一个假的系统 FPS；也不要自称物理系统完备。

## 4. 最新真正推进过的研究实验

### M2260 / M2266 / M2268 / M2269：C1 调度、生命周期和端口

入口文档：

- `HW/reviews/m2261_c1_prosperity_boundary_and_tcasii_progress_20260905.md`
- `HW/reviews/m2267_c1_conv_redesign_and_port_choice_20260905.md`
- `HW/reviews/m2268_c1_reuse_driven_dispatcher_design_20260905.md`

M2260：同一个 forest，DFS线程顺序+2热槽，相对稳定序局部phase约1.06724×，parent backing访问−97.71%；但不重叠约0.98585×，新增2R1W槽/metadata，原九宏没删除。不是整机节能97%。
M2266：3样本×4Conv×3空间chunk×全部432K=15,552tiles，同依赖/credit下只换端口：

| 端口 | 服务cycle合计 | 局部phase相对1RW |
|---|---:|---:|
| 1RW | 713655 | 1.0000× |
| 两独立1RW/parity | 706201 | 1.0095× |
| 1R1W | 701079 | 1.0162× |

不是1R1W所有新架构的上限，但已经否定“只换口就把当前结果轻易推到1.9×”的承诺。
M2269 看见FIFO限制和已就绪root机会；M2268是 ready-window、parent-owned slot、prefetch、retire/spill 的设计研究，**尚未成为新 RTL**。
应研究图/调度/物化共同选择，而非单独再加 comparator；但寄存器分配、重算、DFS都非新原理。

### M2262–M2264：稀疏随机访存感知无损压缩

综述/数字：`HW/reviews/m2265_subsystem_research_screen_20260905.md`。
实际统计32张量、45.626M INT8 code：8Conv/decoder来自已有M2042数组，24FC来自M2251量化候选。
**字节无损≠对FP frozen网络精度无损；FC AEE没有完整准入。**

- 简单位宽/offset编码含目录：Conv容量约−7.87%，decoder−10.77%，FC反而约+1.85%。
- 96值restart Huffman含目录：Conv−16.92%，decoder−22.00%，FC1−10.37%，FC2−9.83%。这些不是宏面积。
- 4320真实FC冷G48请求、目录与payload共用同样字缓存：raw1,604,430宏读；compressed为1,818,652（4word/bank）或1,730,101（16word/bank），**反而多13.35%/7.83%**。
- 内容有压缩性，但目录/对齐/访问粒度抵消收益；这是值得重做子系统的问题，不是现成EBPC接线即成功。

下一步可研究物理sector内索引/定长上界包、按bank/consumer可独立解码、读出即复用；必须和普通BPC/codec、相同缓存/目录收费比较。

### M2270：ATLIF 数量和数值语义诊断

代码 `HW/system_simulator/scripts/m2270_atlif_semantics_probe.py`。
结果 `HW/results/m2270_atlif_semantics_probe_20260905/result.json`。
详见第2节；实际 neuron class/实际checkpoint 的140322标量检查已通过，不是完整网络 AEE。

### M2271：按消费者签名共享部分归约，已经测过，不要只看理想算术

代码 `HW/system_simulator/scripts/m2271_destination_signature_screen.py`。
结果 `HW/results/m2271_destination_signature_screen_20260905/result.json`。
来源为 M2051 + M2067 的40sample×24FC层×first/middle/last B4，共2880个workload；不是所有token。
同一source的4bit消费者签名，最多15类；类内先归约再送各Acc24，单个source/单消费者旁路。
近邻是 Mailman/CSE、Comperity、Phi、Transitive Array，不是新数学。

已经纠正：跨window累加不能免费、两边第一项都允许直接赋值、按真实Cout/96加权、C1必须对64row forest而非只对4row zero。
实际执行被计费的整数图，定向 signed INT8 权重共106752个输出0mismatch；不是网络原权重量化精度证明。

| 目标/窗口source数 | 完整dot二元加法减少 | 保持原K8 bank分组后的更新意图数/ordinary |
|---|---:|---:|
| FC1/16 | 8.38% | 1.1638（更多） |
| FC1/64 | 20.26% | 1.1974（更多） |
| FC1/768 | 36.45% | 0.9838 |
| FC2/16 | 7.28% | 1.0954（更多） |
| FC2/64 | 16.54% | 1.0901（更多） |
| FC2/768 | 31.02% | 0.9110 |

更新意图计入class更新、旁路和每个消费者的partial scatter，仍没计完整分类/端口/flush/延迟，**不是cycle**。
大窗口算术很漂亮，小窗口却增加服务；不要只引用36%加法减少就写RTL。
满15类×96lane×Acc24=4320B，原4个输出只有1152B；状态面积也要计。
C1同cohort，signature完整dot1,044,594adds vs旧forest843,274adds，**多23.87%**，目前不适合替代C1。

这条只能继续研究“选择哪些partial值得建立+bank-local归约/分发”的联合问题，不能把无条件建表包装成赢家。
独立子agent复核过此前算术/miter修正；新增bank-preserving意图模型仍需专门检查，不能声称所有最新字段都独立审过。

## 5. 多 AI 文件目录：要完整读，且不要继承错误

`/home/zhumd/work/ideafromai/`原inventory40文件；这次显式全读轮已完成其中37份。
尚未完成本轮重读的3份大文档：

- `research/10_cim_spike_opticalflow_accelerators.md`
- `research/11_event_camera_stack_accelerators.md`
- `research/13_isscc_vlsi_hotchips_edge_npus.md`

交接时又发现新文件 `HANDOFF_NEXT_AGENT_20260905.md`，已经完整读过；所以此刻41文件中38份完成本轮阅读。目录还可能由其他AI更新，请先 `rg --files --hidden`，不要以旧 MANIFEST 为穷尽列表。
新agent仍应按用户要求自行逐份研读，不能把本交接摘要当成完成全部阅读。

各组位置及处理：

| 文件组 | 内容 | 核心警惕 |
|---|---|---|
| README、INDEX、MANIFEST及GROKBOT版本 | 索引、外部工作树 | 清单过时，路径可能是另一机器/workspace |
| `research/01_ann_sparsity_mechanisms.md` | ANN剪枝、级联、提前决策、精度、融合 | 多数8–9.5分是主观假设，没有本机收益 |
| `research/02_snn_atlif_realvalued_mechanisms.md` | SNN/ATLIF/多值与时间并行 | 常把本机当不可吸收的任意幅值，错误 |
| `research/03_opticalflow_data_hw_algo.md` | 光流ASIC/FPGA/算法/数据特征 | 阈值、warp、occlusion/refine多为新模型而非exact移植 |
| `research/07_transformer_attention_accelerators.md` | ANN/SNN attention，加速器与算法 | ELSA2021ANN与ELSA2026SNN不要混为一个；全N²注意力机制不一定适用于本机 |
| `research/08_video_temporal_sparsity_accelerators.md` | DeltaCNN、MotionDeltaCNN、SVG、video reuse、token merge | 首次专用硬件未检索到不等于确证无人做；不能用当前flow结果决定跳过产生它的计算 |
| `research/00_seed_notes.md`、04/09/12/14综合 | 多轮大礼包和排名 | 上十个机制打包、换任务加名字，不等于coherent新架构 |
| `microarch/05...`、`plans/06...` | 框图、卡片、排期 | 里面“用户已锁、立即启动”是文件建议，不是新授权 |
| `contracts/ATLIF_contract_r1_grokbot.md` | signed8幅值、EPS1门控 | 是假设新算法。当前spike编码1时门控可能全删，不能当无损 |
| `contracts/ablation_ladder_grokbot.md` | 多机制消融阶梯 | 先修正数值/对象，不可照单执行 |
| `codex_cards/CARD_A_OP_STW.md` | 光流/事件tile wake | gate信息必须在被省计算前存在；不满足就是非因果 |
| `codex_cards/CARD_B_HBG_RP.md` | HBG双轨幅值packetizer | 冻结theta-scaled binary不满足其任意int8假设 |
| `codex_cards/CARD_C_MX3P_DIRTY_SCORE.md` | Motion三popcount+dirty复用 | plain AND-only计算不同函数，不能作为同功能速度分母；Q7候选≠frozen |
| `research/grok46_20260905/`全部9份 | 纠正binary、prior、Motionleaf、profile、gates | 修正binary有价值，但仍错把当前PSN说成leak/reset递推；“只剩一个叶子能做”未被证明 |
| `codex_independent_20260905/`全部3份 | selective shared reduction、模式表、压缩、算术树 | 真正部分归约已做M2271；不能再说只待测，也不能忽略bank服务负结果 |
| `research/m2067_handshake_fix_status.md` | sticky handshake历史TB诊断 | 是历史失败/修复说明，不是当前960生产VCS成功，也不是新机制 |

**另一份 HANDOFF_NEXT_AGENT 文档也不是权威**，不要覆盖它，但本轮发现需纠正：

- 它用ep35、calls=1介绍93，而当前ep34 S40实际calls=40。
- 说“自适应训练产出此网”过度归因，M2270显示resume手动增长/活动正则为0，需做训练消融。
- 把freeze字段当全部theta训练冻结不对；theta折权重也不是自动FP/量化exact。
- “换1R1W接近1.90×”未被最新单变量实验支持。
- “只做Motionleaf，等待用户三选一”是另一AI建议，不是当前用户已选择；当前仍要求C1/C2重大机制研究。
- 固定投稿期限、期刊格式/政策如要用于决策请查最新官方来源；此交接不沿用未经本轮核实的日程。

## 6. 研究候选：允许推翻，不要再按新名字个数打分

### A. C1：算术关系图、执行顺序、中间结果物化联合设计

问题不是“有没有subset”，而是某个共享中间结果在有限端口/缓存下是否值得建立、保留、重算，以及如何安排消费者。
可探索非实际parent的小basis/局部intersection，但 Phi exact模式和Transitive Array已覆盖大量类似关系；要识别真正未解决的资源/控制问题。
与稳定forest、普通DFS+forwarding、相同buffer、DP强实现、zero baseline全对照。若只比人为弱1RW点，不合格。
卷积原生tap/input/output布局可加入；相邻输出共用输入不意味着共用乘积，因为kernel offset权重可能不同。

### B. C2：从“共享权重读取”转向“选择性共享归约、少做独立Acc更新”

M2271已经看到算术机会与执行代价不一致。下一步不是固定15桶，而是只物化收益足以覆盖状态/散射代价的类，保持bank可执行、其余旁路。
不能默认跨bank任意8source免费重打包，也不能新增多路acc口不收费。
这是研究路线，不是已经新颖或已加速。若赢不了现有强K8，保留为负结果。

### C. 独立主线备选：稀疏访问感知的无损压缩存储子系统

联合压缩包、bank/sector布局、索引、随机解码、consumer复用。真正要解的是M2264“容量省但宏读增加”的冲突。
小模式表是另一备选，不要与压缩/重组全叠到一篇brief。融合signed压缩树仅作配套电路/消融。

### D. 算法原生：只计算足以确定ATLIF阈值判决的信息

若有严格范围，可做位平面逐步揭示/区间收紧，避免全精度计算后才发一个bit。
但旧G12/M366已做双侧bound、系数排序和context内lane压紧：term约省6.6%，issue约省0.068%，且旧量化有翻转；**不能重命名它当新idea**。
SnaPEA、digit-plane early termination、activation-deterministic binary/ternary CNN已有直接prior。
可能的研究差是满秩PSN时间混合的共享范围服务/位平面调度，但需要真实输入分布和数值协议，尚未证明。
现有S40的3720条ATLIF ordered输入均未保留payload；只有统计/采样，不能直接从它做完整bitplane/AEE重放。

### E. 新算法—硬件分支：光流粗基底 + 边界/遮挡局部修正

相比给通用broadcast贴“光流”标签，更任务原生的问题是：平滑运动区域用coarse baseflow表示，昂贵计算只细化边界/遮挡不确定区域。
EEMFlow/EEMFlow+的meshflow、confidence-induced detail completion等提供算法线索；需要cheap预测先于被跳过的decoder，不使用未来flow oracle。
这是新模型/精度Pareto分支，不是frozen H67免费exact优化；未训练未实现。
用户若选这条，应明确新数据/训练/模型身份与旧精确硬件基线共存。

## 7. 优先读的原始文献与实现

这些是研究索引，不把论文里的倍率转述为本项目结果；重要的新颖性结论请读机制段和反例。

- [Prosperity HPCA2025](https://arxiv.org/html/2503.03379v1)，[官方模拟器](https://github.com/dubcyfor3/Prosperity)：有限buffer、16宽、时间展开、forest/dispatcher，不能弱化描述对手。
- [Phi ISCA2025](https://arxiv.org/html/2505.10909v1)：exact模式+signed residual与PAFT分开；建表/PWP读取不免费。
- [Transitive Array ISCA2025](https://arxiv.org/html/2504.16339v1)：结果关系/Hasse图、虚拟节点、依赖调度；不能把“添加虚拟parent”当空白。
- [Mailman作者论文](https://www.cs.yale.edu/homes/el327/papers/matrixVectorApp.pdf)：有限字母表分组/共享归约，是M2271直接数学邻居。
- [Comperity DOI](https://doi.org/10.1145/3828526)：其他AI核到摘要shared base/delta；本agent未取得全文，保留重合风险，不能宣称已排除。
- [ELSA ISCA2026](https://arxiv.org/html/2605.20802v1)，[实现](https://github.com/Intelligent-Computing-Research-Group/ELSA)：bundled事件、Gustavson、状态访问，含符号不能说只有我们支持signed。
- [Bishop](https://arxiv.org/abs/2505.12281)，[SpikeX](https://arxiv.org/abs/2505.12292)，[LoAS](https://arxiv.org/abs/2407.14073)：打包/权重共享/时间组织/剪枝的直接比较。
- [EBPC](https://arxiv.org/html/1908.11645v2)，[SV](https://github.com/pulp-platform/stream-ebpc)，[BPC ISCA2016](https://lph.ece.utexas.edu/merez/uploads/MattanErez/isca2016_bpc.pdf)：featuremap与weight区分，随机读取已是prior，不借featuremap压缩率。
- [Activity-Pruning-SNN](https://github.com/putshua/Activity-Pruning-SNN)，[PSN](https://github.com/fangwei123456/Parallel-Spiking-Neuron)：先对照本地hybrid源码，不把原ATLIF的递推照搬进本机。
- [SnaPEA作者ISCA2018 slides](https://iscaconf.org/isca2018/slides/8A2.pdf)：early activation prediction已有。
- [EEMFlow/EEMFlow+](https://arxiv.org/html/2510.04111v1)，[开源](https://github.com/boomluo02/EEMFlowPlus)：光流mesh/coarse与细节完成；不是frozen模型。
- [MotionDeltaCNN ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)，[TMA ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_TMA_Temporal_Motion_Aggregation_for_Event-based_Optical_Flow_ICCV_2023_paper.html)：moving-camera复用/时域光流模型边界。

## 8. 接手后按什么顺序推进

1. 检查工作树和本轮结果文件；不要读百万行旧session恢复聊天，也不要重跑已完成全部实验。
2. 核对第2节ATLIF，首先向用户明确105/93/81与二值含义；85不用。需要更强证明时做普通光流forward与诊断支路剥离的配对测试，别只信名字标签。
3. 完整读idea目录全部文件，尤其前agent未完成本轮重读的3份；输出“建议—真实前提—直接prior—可检验增量”，而不是照抄AI排名。
4. 复核M2271最新bank意图计数，明确为何少加法不一定少K8服务。若继续这条，优先做选择性partial+真实bank服务的最小模型。
5. 同时研究C1图/物化/调度，以及压缩子系统的读放大根因；最多挑一个明显有潜力的主线进入大改RTL。
6. 要继续旧线收口，做新cofill matched power、真实存储/hold边界，不再反复给同一合同打100分。
7. 如需新增神经元输入capture，先确认训练端运行情况、最小统计/抽样方案；不声称现有全部payload已够，不抢别人GPU训练。
8. 候选通过数值、强基线、资源/服务模型后再用VCS/DC/PT/FM/PTPX。CPU operation-count不是cycle，VCS不是postlayout，logic energy不是总能量。
9. 新机制足够强时可以重切TCAS-II题目，让旧C1/C2成为基线。不要预写Strong Accept概率，也不要用过时deadline迫使选择低价值补丁。

本轮新增、交接前应保留的四个小工件：M2270/M2271两份脚本和各自result.json，加本交接稿。
结果没有新封存审批层；可正常git版本管理。提交时只选自己的文件，不要 `git add -A` 把其他AI修改一并打包。

最后，用户希望你主动推进，但遇到真正需要选新算法/训练对象的分岔应说明选择与代价；不要因为一个AI文档说“立即开工”就擅自替换模型。
