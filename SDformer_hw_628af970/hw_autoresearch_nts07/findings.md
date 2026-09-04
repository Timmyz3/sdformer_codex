# H67/H68 ASIC 研究发现

## 当前理解

- H67 与 H68 的真实事件分布近似，适合 H67 功能超集加 H68 编译期特化，不适合分别做两套物理核。
- 100 样本显示 pair-empty 约 74%、K-zero 约 83%、active-entry 约 18.4/162，但 block 间差异远大于模型间差异。
- S1B0 和 S2B3 接近全静默，S0B0、S2B5、S3B1 明显更活跃，调度和表示配置至少需要到 stage/block 粒度。
- Delta=0 与 pair-empty 仅相差约 0.1 个百分点，非空 exact temporal reuse 不是当前主要收益来源。
- 现有 row RTL 是单 context、162-token serial front end；架构主线应升级为 temporal-pair 驻留、class-stationary、多 row context 和共享 SCS。

## 形成中的架构主线

全Encoder工作名HIT-Flow：

1. 四个stage恒定`head_dim=32`与`9×9`窗口形成统一Head-Time Tile；
2. `32×10`除数打包时间矩阵阵列在T=10模式处理一个位置，在T=2模式五路打包，两个模式保持高利用率；
3. 105安装、93动态调用、12调用结果死亡、81固定部署功能活跃ATLIF分列；
4. TESSA作为attention子系统，直接连接factorized gated projection；
5. binary event与ADD residual/S0-S2 skip使用独立precision island。

TESSA子系统：

工作名 TESSA：

1. 128-bit `{Q0,Q1,K0,K1}` temporal-pair interface；
2. bitmap/event membership 两种 exact 表示的同构充分统计量单元；
3. pair-coalesced K-zero class commit；
4. context 参数化 1/2/4，首版启用 2；最终物理数量与 PCCC 合并率和 commit stall 联合决定；
5. block-aware descriptor 和精确 completion；
6. H67/H68 编译期模式冻结；
7. full-encoder skeleton 连接 projection/ATLIF、两次 block residual 和 S0-S2 三条 skip。

## 文献迁移边界

- Bishop 已有 TTB、stratifier 和 dense/sparse cores，不能把异构双核本身当原创。
- LoAS 已有 fully temporal-parallel、spike compression 和 inner join，不能声称首次时间并行。
- 复旦 ISSCC 2023 已有 in-memory butterfly zero skipper；只有动态 four-vector membership、充分统计量归约融合和 exact fallback 的新组合才可能形成增量。
- C-Transformer 已有 homogeneous reconfigurable SNN/ANN core，支持但不独占我们的同构表示切换思路。
- ISSCC 2022 已有 speculative OOO，TESSA 只能强调 metadata 可证明、无近似的独立 row 调度。
- D3TA 已有相同数据的 charge-recycling；数字切换抑制只能作为有 SAIF 证据的子贡献。

## 已完成证据

- H67/H68 profile100 重新分析：stage/block 分位数、每样本周期代理、TTB 覆盖、Delta 分桶。
- 原两阶段 flowshop 中 4/8-context 相对 2 的额外收益低于 0.1%；加入 commit 后，无合并时 4 相对 2 改善约 3.7%，PCCC 全合并上界下约 12.0%-13.5%。
- 独立架构审阅指出 81-cycle 上界未计 128-bit/拍供数和同一 pair 双 commit 端口冲突；fixed-bitmap pair 与可退化 2-context 仅为受控准入，完整 TESSA 仍未签核。
- 真实 pair 类别为：双 K-zero 约 83%、单 K-zero 约 11%、双 active 约 6%；分 bank 单写口无合并的 commit 约 153 cycle/row，PCCC 全合并上界约 86 cycle/row。
- TESSA探索性RTL前合同已冻结：encoder-attention subsystem边界、64-bit descriptor、128-bit pair、原子双commit、1/2/4-context逻辑memory map、独立completion和13类性能计数器；早期NTS07/98-token接口已废止。
- 全Encoder审计确认ATLIF是`T×T` PSN时间矩阵而非递归LIF；81个活跃点每帧约44.244亿时间MAC，但8-bit参数容量仅约5.12KiB。
- S0-S2三条长skip共11,612,160元素；旧profile只证明接近全非零，不能证明可用1 bit保存。
- HIT-Flow候选形成：HTT、DP-TME、TESSA/FGP和RPI；单320-MAC DP-TME的ATLIF-only上界仅36.16 FPS，平衡候选至少从双阵列开始。
- 新 collector：时间对充分统计量、表示 DSE、sample-flow 相关性和 9×9 空间/bank mapping 统计。
- CPU 单元测试 56 项通过。
- 中文文档 53/54/55 已形成。

## 仍需验证

- 新 profile100 ordered trace 和 finite-FIFO replay；
- four-vector union packet 的真实流量；
- pair-coalesced class merge 比例；
- 1/2/4/8 context cycle/occupancy；
- row-major/diagonal/XOR bank mapping；
- BMRF 与简单 prefix/bitmap 的 DC PPA；
- 128-bit 直读与 `2x64-bit` assembler 的带宽/冲突对照，以及 active/histogram 双提交队列的 stall；
- encoder-level Amdahl、DP-TME/Spatial Engine、RPI/skip SRAM、SAIF、Formality和端到端吞吐。
- ATLIF 4/6/8-bit参数量化后的margin、事件翻转率和valid825。

## 不能重复的错误

- current-empty 不能当 pair-empty；
- K-zero 不能从 Shiftmax denominator 删除；
- 105个ATLIF是安装口径、93是动态调用、81才是固定正常推理功能活跃候选，三者都不等于物理实例数；
- skip 只有 S0/S1/S2 三条，S3 是 bottleneck-local；
- row-kernel cycle、Yosys cell 和 spike energy 不能冒充芯片端到端 PPA；
- ANN 机制迁移到 SNN 不是天然原创，必须引用和做增量对照。
# 2026-07-13 门类驻留多播投影

> **历史结论，已由后续代码复核否定**：量化score class只可用于同一Shiftmax row内的前端归约，
> 不能直接作为跨row/window投影复用码。跨窗口必须使用RTL最终9-bit Q1.7 gate code。

- 同一row内可用score class辅助SCS归约；投影后端只对同一block内的
  `(final gate code, global K channel)`生成一次`gate×weight`，再按token bitmap多播。
- 该重排保持每个token输出独立，减少乘积和权重读，不减少活动目的累加；因此必须联合统计class-channel项和多播事务。
- 复旦蝶形zero skipper、UCNN重复权值、Prosperity product sparsity、Eyeriss多播和FuseMax算子融合均构成明确先例。可辩护增量是复用Shiftmax已有class metadata，不做通用模式检测，并与事件K bitmap和class-stationary SCS融合。
- 旧数据表明纯K0/K1时间复用仅约10%，不支持复杂全局蝶形。GCM-P是否晋级等待真实ordered
  profile的最终门码压缩比、活跃门码、fanout和多播宽度DSE。

# 2026-07-13 最终gate码与窗口组架构修正

- 代码复核确认H67/H68为token-wise `K乘gate`后接跨head拼接的`C乘C projection加BN`；
  `attn_sn`结果不进入functional projection，但projection后的attention ADD和MLP ADD必须保留。
- 归一化前score class只能在同一row内映射到相同gate。跨window时分母不同，唯一合法复用键改为
  RTL Shiftmax最终9-bit Q1.7 gate code；文档66已加入语义修正。
- ordered profiler新增最终gate码、G等于1/2/4/8/16窗口组、fanout和M等于1/2/4/8/16多播事务。
  H67/H68排队配置已修正为`*_rtl_exact.yml`并增加硬失败审计，避免浮点Shiftmax门码污染统计。
- 形成HIT-Flow-WG候选：SCS归一化元数据前推NMF、窗口组门码乘积驻留WG-GPS、分层分段多播，
  同时保留G=1 direct fallback和RPI残差精度岛。
- Prosperity官方代码复核表明其复用二值脉冲行子集及prefix输出；FuseMax artifact按Einsum、buffer
  和functional action count评价；Transitive Array、SWAT和复旦蝶形进一步收紧泛化创新表述。
- 当前仍为条件架构：G大于1是否晋级必须同时满足真实product减少、完整状态/端口、projection EDP
  和全encoder Amdahl门槛，不能用合成乘法减少或“ANN迁移到SNN”代替证据。
- 跨窗口组必须限制在同一样本、同一block/head内。原扁平`batch_windows`分组存在跨样本虚假复用
  风险，现已按`windows_per_sample`修复，并为每个G记录有效窗口数和尾组slot利用率。

# 2026-07-13 G1普通乘法后端

- 9-bit无符号最终gate码乘signed int8权重的完整65,792种组合均可由signed 17-bit精确表示；
  普通8-lane乘法后端通过Icarus、Verilator lint、绑定SVA和Yosys结构检查。
- 最终gate目录只减少`gate×weight`和对应权重读，不减少每个目的token的独立累加。后续架构成败
  更可能由bitmap展开、accumulator bank冲突和写回流量决定，而不是局部乘法器数量。
- 蝶形或复杂多播网络不默认实现。先用简单分段banked multicast得到真实trace的stall、fanout、
  线长代理和完整projection EDP，再按文档69门槛决定是否晋级。

# 2026-07-13 当前段驻留多播

- 全162-bit每拍扫描的多播选择器虽然正确，但结构代价过高；改为18-token当前段驻留后，Yosys
  generic cells由3,122降至294。该差异证明RTL表达和数据布局本身必须参与架构DSE。
- 18-token/2-bank利用162=9×18且保持bank相位固定，是待真实时空布局验证的探索点；它不是冻结
  参数。非法的segment/bank非整除配置必须fail-closed。
- 简单网络已经形成可测基线。没有真实trace的低交付效率、显著互连stall和完整projection EDP
  收益，就没有理由实现或声称蝶形网络创新。

# 2026-07-13 Bias提交输出

- accumulator的最后bias更新可以直接产生最终整数输出，理论上消除一次完整acc SRAM读出；该
  BCOD重排保持exact，但必须与传统readout在相同SRAM和输出反压下比较后才能作为贡献。
- 默认累加状态明确映射为两块81×256-bit同步1R1W SRAM。强制门级展开未初始化但valid保护的
  memory会产生大量伪结构问题，后续综合必须使用SRAM wrapper或`memory -nomap`做前置审计。
- 当前每bank两拍一更新是正确性基线。是否增加流水旁路由真实bank stall决定，不能仅为提高理论
  峰值扩大控制和验证复杂度。
