# H1：同一H0状态的12单块与66双块实际干预

2026-10-16，**CPU准备；未运行、未排队**。[CPU记录](cpu_results.json)为8项合成测试通过。未读真实H0或R13结果、质量、权重、事件/GT数组；未GPU/SSH，已审H0文件均未改。只有root依据完整H0证据决定该窄动作集合值得投入时才执行本入口。即使H0已完成，本目录也不会自动启动。

## 唯一新增工作和固定预算

承接实际H0同一W、BN、runtime、H8、输入及保存的CC前缀。在stage2/block2的12完整9×9 tile上，逐帧执行12 singleton与全部66个不同pair；其他stage0/2 MLP及8个真实尾tile保持coarse，stage1/3保持原生。T10与全部通道不改，attention已执行。

mask完全由坐标tuple枚举，先单块、再按坐标字典序枚举pair。GT、H0误差、事件量、分数、预测均不参与生成mask。第一个pair固定为`((0,0),(0,1))`；仅H8第一帧对此追加1次dual。总计`8×(12+66)+1=625`个**真实完整网络前向**，不是把单块损伤求和。H0的CC、native、FF等已有证据不重跑，BN不重新校准，也不训练/拟合任何参数。

| 固定段 | H8成员 | 新actual | 新dual | 合计 |
|---|---|---:|---:|---:|
| 1 | 顺序第1–2帧 | 156 | 1 | 157 |
| 2 | 第3–4帧 | 156 | 0 | 156 |
| 3 | 第5–6帧 | 156 | 0 | 156 |
| 4 | 第7–8帧 | 156 | 0 | 156 |
| 全部 | 固定8帧 | 624 | 1 | 625 |

H8顺序来自已审H0清单；序列顺序也等于序列名SHA升序，前两帧与设计的固定pilot一致。不能按前段质量换帧、挑后续成员或用不完整段声称全H8。

每次singleton/pair的全12 MLP FC1/FC2各执行154320/155040行；唯一dual为609600行；625次每FC总97285440行。这些是源码/CPU几何服务，不是周期、访存、GPU时间或能效。每段复用同一共享GPU锁并硬封顶30分钟；四段预算上限120分钟，仅为中止界限，尚无本实现GPU耗时实测。

624个最终预测全部保留为float32 `(2,480,640)`，未压缩数值内容合计1533542400字节，约1.43 GiB；实际NPZ压缩大小未实测，还需要结果JSON和原H0 artifact空间。每帧一个NPZ，含全部78个mask；其他flow头保留shape/finite/SHA，first-pair actual/dual在内存逐位比对4个完整头及8条完整MLP branch。不能只检查AEE或最终一个头。

每帧**首个已有singleton**同时抓取predecision `u`，保存原生float32 `[10,1,30,40,384]`及原始payload/NPZ SHA；其SHA必须等于保存H0 CC的目标输入。8份u原始内容再需147456000字节。随后只将u与同W原生MLP传给冻结`probe.feature_scores`，保存12完整tile×7普通分数：amplitude、RMS、pooling residual、空间/时间邻域差、PSN1活动、PSN1空间Hamming。捕获与评分不读取GT或已有flow输出，不改变枚举mask。

评分额外调用PSN1，每帧12次，临时hook逐次核shape与实际elements并finally清理；全部8帧为**96次PSN1、29859840元素、298598400名义temporal MAC**。它不新增完整前向，但绝不免费：归约、读取、u缓冲和实际延迟尚未计入周期模型。记录`ordinary_score_cost`与`ordinary_score_actual_audit`单独分账，完整前向预算仍625。PSN没有持久膜态；W/BN与hook闭合检查仍保留。调用冻结评分原语不等于完整复现ADAPT。

## 严格承接H0，零额外基线前向

直接复用**已审的新H0.prepare**做共同数据/runtime/BN/admission校验；该函数没有前向和BN校准。绝不调用历史collector的prepare/execute，也不调用H0.execute。来源固定在[source_pins.json](source_pins.json)。

此外要求H0实际收据为`H0_69_FORWARD_COMPLETE`：完整69 ledger、8成员×8锚点、5个全头/branch dual、共同状态/BN不变、模块与hooks闭合，且W/N/F装载状态、数据/timestamp和输入manifest与当前来源一致。核所有8个H0预测archive及64份最终预测的原始SHA、dtype/shape/finite。H0的source/CPU/全8 R13质量与独立执行审计绑定全部沿用。H0的`qualified_ordinary_base`或`action_only_screening`范围继续保留，不升级科学资格。

每段只构造一个模型：原生loader建立结构→合法R13 BN→严格加载H0同一恢复W，随后才推理。不会用原135930权重或旧BN做推理，也不恢复optimizer。实际W state SHA、72 BN值、54 external current-stat语义、输入数组SHA必须匹配H0。逐次检查本次干预对已保存H0/CC的目标输入SHA与四个前置MLP输出SHA完全一致。模型每次reset，未对H0基线添加任何新前向。

单块/双块均走冻结probe及merge数值数学，仅通过私有wrapper namespace设置固定tile mask。每次保留全12层独立FC hook观察值、8层branch/input记录、20目标tile真实几何与mask。所有尾块必须coarse；实际行数独立公式再核。GT只进入全16分区评价，各帧全部有效GT保留，不因无事件、误差大或分区为空删帧。

## 入口

[run_h1.py](run_h1.py)默认只preflight。root需先有真实H0收据、同一份H0 input manifest及其SHA、H0与H1 CPU记录；此处尚未提供这些实际运行入参，也未选择W。

```bash
python -B run_h1.py \
  --h0-result /absolute/H0/results.json --h0-result-sha256 ACTUAL_H0_RESULT_SHA \
  --h0-inputs /absolute/h0_inputs.json --h0-inputs-sha256 SAME_H0_INPUT_SHA \
  --h0-cpu-checks /absolute/implementation_h0/cpu_results.json \
  --cpu-checks /absolute/implementation_h1/cpu_results.json \
  --segment 1 --output-dir /absolute/new_h1_segment1_preflight --wall-minutes 30
```

实际执行为同一命令换新输出目录并加`--run-inference`。其余段显式指定2/3/4，入口不自动排队。任何段失败标`INCOMPLETE_NO_SCIENTIFIC_CONCLUSION`；不存在跳过失败帧、扩预算或自动续跑。完成一段只标`H1_SEGMENT_COMPLETE`，不能当625次全部完成。

四段齐备后，将各自`results.json`路径/SHA填入[segments.template.json](segments.template.json)，使用只读CPU合并入口：

```bash
python -B finalize_h1.py \
  --segments-manifest /absolute/h1_segments.json --segments-manifest-sha256 ACTUAL_MANIFEST_SHA \
  --h0-result /absolute/H0/results.json --h0-result-sha256 SAME_H0_RESULT_SHA \
  --cpu-checks /absolute/implementation_h1/cpu_results.json \
  --output-dir /absolute/new_h1_complete
```

[finalize_h1.py](finalize_h1.py)只验四段共同binding、同W/BN/H8/H0/源码、625 ledger覆盖、actual mask与row、全部78×8预测文件及payload SHA和16区记录，完整才给`H1_625_FORWARD_COMPLETE`。它重新读取显式SHA绑定的真实H0结果与64个预测artifact，从每次actual/dual的原始8 branch SHA/形状重算相对该H0 CC的prefix，不只信PASS布尔；首pair全头和8 branch SHA也重比。全部8份u archive及payload、7分数的canonical JSON SHA、实际/名义评分work均核验。预测和u不复制到合并目录，最终每帧新增`prediction_source.path/sha256`、`predecision_u_source.path/sha256`指向原段artifact；原basename也保留。合并器不算最优mask、不拟合、不比较质量门、不生成singleton加和oracle。

CPU复核：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B test_h1_cpu.py
```

测试覆盖独立坐标枚举与625预算、固定四段实际编排函数mock、两种mask的小原生MLP actual/dual和同CC前缀、tail拒绝、H0实际完成/输入/状态/ledger继承的9种拒绝、四段合并14种拒绝（含删除全部625条branch记录）、真实合成NPZ字节/shape核验、实际u capture与PSN1 hook/零额外完整forward、u/评分artifact的9种拒绝、入口全局依赖及无历史prepare/训练调用。fixture只有两个合成输出头，不替代真实网络四头验证；真实全模型、H0输入与恢复状态可达性仍未运行。

后续普通控制分析须让每种方法在全部H8使用同一种降序top2和坐标tie规则，7种分数完整保留；不能逐帧按GT切换“最佳普通方法”。H8拟合的最佳fixed pair与逐帧有限枚举最优必须单列为探索性诊断，不能当确认性能。这是分析边界，本collector不实施任何排名。

这批证据将来最多说明：固定W/BN/Q下，这8条训练侧帧的12完整tile有限动作集合有何可用余量。它不代表所有层/尾部/跨帧动作的全局最优，也不提供可部署预测器或净硬件收益。H0若显示不值得继续，H1就维持准备状态。
