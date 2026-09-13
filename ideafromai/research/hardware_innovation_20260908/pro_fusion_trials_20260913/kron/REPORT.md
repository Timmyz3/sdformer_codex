# 当前连续 PED 后因子的 Kronecker 融合实试

2026-09-13。**已完成真实 96×24 矩阵、真实连续输入、固定一项/两项拟合及同资源 SV/Verilator；本轮结果是“局部执行有效，当前免训质量不占优，X 仍为 0”。** 没有训练、布局扫描、GPU、AEE、EDA、生产文件或主稿改动。

## 半页 A–B–X 与筛选结论

**完整强 A。** [GKPD，AAAI 2022](https://cdn.aaai.org/ojs/19958/19958-13-23971-1-2-20220628.pdf) 的多项 Kronecker 表示、重排 SVD 拟合和不重建大矩阵的收缩是已有方法；[FastKron，PPoPP 2024](https://arxiv.org/pdf/2401.10187) 的直接按目标布局写出、选择收缩方向、因子间保留中间状态是已有执行权限。本轮实现其中适用的局部部分，并给普通矩阵同 8 MAC、同系数/状态/输入输出端口预算。作者完整训练和 GPU 系统未复现。

**本网余洞 B。** 当前 dense 学生已是 U24×96 → RNE16/sat24 → V96×24 → RNE15/sat24 → bias/sat24；V 输入完全连续，本窗口非零率 100%。已有 U/V 分解没有继续压缩 V 的 2,304 个系数及每位置 2,304 次乘加。其 profile 实调用一次、192,000 个位置、442,368,000 名义 MAC，占既有 596.546 G 分母仅 **0.0741548%**；适合验证完整小算子，不能包装成当前全网主要瓶颈。

**本轮尝试 X。** 固定 96=(4,24)、24=(6,4)，先收缩 A4×6，再收缩 B24×4，用 16 个连续中间数替代大矩阵读流；保留原 U、V 的舍入位置，中间只增加精确整数状态。这是 A 在当前定点接口上的迁移与闭环验证，**尚没有足够理由称为新表示或新架构 X**。两项比原直接矩阵少 41.55% 完整周期，但后 V 局部 NRMSE 为 77.48%，比同参数普通 rank2 的 73.65% 更大。当前固定免训点不升为网络精度候选或论文主标题；Kronecker 家族、恢复训练和别的已授权挂点不据此关闭。

## 1. 真实参数和输入身份

唯一参数源为 [当前 dense 导出](../../open_fusion_execution/breadth_20260912/algorithm/hardware_exports/dense/deployed_constants.npz)，输入为同目录 [首帧捕获](../../open_fusion_execution/breadth_20260912/algorithm/hardware_exports/dense/000_zurich_city_09_a_0001.npz)，不是 Pro 中的旧 96×32，也没有从原学生归档误取 R32。

| 当前导出 | 形状 | 指数 |
|---|---:|---:|
| U_conv2_theta_q16 | 16×864 | 17 |
| F_q16 | 96×16 | 14 |
| U_ped_q16 | 24×96 | 16 |
| V_ped_q16 | 96×24 | 15 |

从 `corner_updated_I24` / `interior_updated_I24` 的真实 even/even 更新锚点取数，按导出 U 和原 RNE16/sat24 重新生成 V 的 latent24 输入。原 V、原 bias 产生的每窗 15,360 个连续 PED 输出都与捕获逐值一致，合计 **30,720 值零差**。两窗各 4×4、T10，共 320 个向量；输入范围 −377,353…195,659，7,680 个标量全非零。`continuous_q24` 是目标输出，未误当作 V 输入。

结构/边界参照 [旧 full_chain 数值执行](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/numerical_reference.py) 与 [最终组合对齐](../../open_fusion_execution/stage_20260912/hardware/final_combo_alignment/README.md)。后者说明当前真实 R24 接口的来源，但没有借 ordinary/lifting 的历史 AEE 作为本次近似网络精度。dense 目录 metadata 的部分旧 bias/range 描述与实际 q 数组不一致；本试验以实际数组和零差捕获回放为准。

调用份额来自 [当前 profile](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json) 的 `patch_embed.proj.conv_res` 两条 mm：U、V 各 442,368,000 MAC；没有按旧 FFN 份额外推。该分母包括实际软件名义算术调用且不计全部 elementwise/访存，0.074% 不是 ASIC 时间占比。

## 2. 唯一布局、公式和舍入边界

定义 i=24a+b，j=4u+v，其中 a∈[0,4)、b∈[0,24)、u∈[0,6)、v∈[0,4)：

\[
W_{24a+b,4u+v}\approx\widehat W_{24a+b,4u+v}
=\sum_{r=0}^{K-1}A_{r,a,u}B_{r,b,v},\quad K\in\{1,2\}.
\]

将原实效 V=`V_ped_q16/2^15` 的 `[4,24,6,4]` 张量重排为 `[4,6,24,4]`，再作 24×96 的截断 SVD；每项 4×6+24×4=120 参数。普通矩阵 rank1 / rank2 也各用 96+24=120 参数/项，固定等参数比较。没有调通道排列、搜索形状或根据窗口误差选布局。

先算 \(z_{r,a,v}=\sum_u A_{r,a,u}x_{u,v}\)，再算 \(y_{a,b}=\sum_{r,v}B_{r,b,v}z_{r,a,v}\)。每项为 16×6+96×4=480 MAC，原矩阵为 2,304 MAC。反方向先算 B 会需 1,152 MAC/项，本轮直接用较强的 A-first 次序。**480 是推导的工作量；最终周期取自下面的真实 RTL。**

固定量化 `Aq=RNE(A·2^7)`、`Bq=RNE(B·2^8)`，二者均存 signed16；不在中间 z 后加 RNE。整数有效矩阵 \(\widehat W_q=\sum_r A_q\otimes B_q\) 仍可存 signed16，本次 K1 为 −14,125…14,238，K2 为 −17,774…20,775。因此可用完全相同直接矩阵硬件执行 \(\widehat W_q x\)，作为 **同函数强对照**。

两收缩合并后指数仍是 7+8=15，原 U 的 RNE16/sat24、V 的 RNE15/sat24、bias 后 sat24 均保持；新的因子截断和系数量化改变 W 的函数，不能称对原网络无损。分解收缩与其本身展开矩阵在这里为精确整数恒等，不依赖浮点结合律。最终 V 单点误差的数学上界可由 \(\|\Delta W\|\|x\|\) 加至多 1 LSB 的两次 RNE 差界限制，sat24 非扩张；这不推出光流 AEE。

## 3. 固定拟合的真实局部误差

NRMSE 为 \(\|y-\widehat y\|_2/\|y\|_2\)，分母为原 V 输出；“V 后”包括原 RNE15/sat24，全部使用相同 320 个真实输入，未用输入做拟合。本表只是一个首帧双窗口局部观察。

| 表示 | 参数 | 权重 FP NRMSE | V 后整数 NRMSE | bias 后 NRMSE |
|---|---:|---:|---:|---:|
| 原 V | 2,304 | 0 | 0 | 0 |
| Kron1 | 120 | 94.3800% | 89.9459% | 89.9304% |
| 普通 rank1 | 120 | 94.8891% | 91.7496% | 91.7338% |
| Kron2 | 240 | 88.6745% | 77.4754% | 77.4619% |
| 普通 rank2 | 240 | 90.2577% | 73.6455% | 73.6327% |

Kron2 的权重 Frobenius 误差虽更小，真实输入上的误差却更大，说明不能只拿 W 重建排序替代消费者输入误差。因数量化新增的权重误差很小，主要问题是此固定免训结构截断。原 U 输出含饱和/舍入，不能用一个未保留中间边界的 `VU` 浮点乘积替代本表参考。普通 rank1/2 只做等参数数值控制，未在本轮实现其 RTL；故没有称 Kron 比同参数 LR 更快。

原网络的完整 AEE 不能继承给这些近似臂；本轮 **AEE 未评价**，既不是精度 PASS，也不是按局部误差宣布整个家族精度 FAIL。

## 4. 隔离可综合 RTL 的共同资源合同

源码 [ped_kron.sv](ped_kron.sv)，C++ TB [tb.cpp](tb.cpp)，机器可读 [resource_contract.json](resource_contract.json)。所有模式实例化**同一模块、同一物理数组、同一个 8-lane 乘加 datapath**，运行时只选调度模式。

| 资源 | 两臂共同上限/实现 |
|---|---|
| 算术 | 8 个 signed32×signed16→signed48 乘法器，8 个 signed48 accumulator；一周期组合乘加，II1 |
| 系数 | 288×128bit local array；每 MAC 周期只读一个 128bit 词，至多 8 个 signed16 |
| 静态整词跳零 | 共同288bit词live mask，由每个cfg_data的OR在原装入拍生成；共同24位next-live选择器，保持升序 |
| 源读取 | 本地至多 4 个不同 signed32 源词广播到 8 lane；直接矩阵同读口/mux 预算 |
| 中间/结果写 | 每 COMMIT 周期 8 个 lane 写32bit latent 或48bit结果，分时共享写资源 |
| 外部输入/输出 | 相同 192bit ready/valid 总线，每拍 8 个 signed24；不重叠装入、计算、最后输出 |
| local 数据/状态 | source24×32、latent16×32、result96×48、acc8×48、bias96×24；模式均保留相同数组 |
| 全部数组容量 | 含系数/live mask共 **45,728bit / 5,716B**，另有少量相同控制寄存器 |

这是新的隔离核合同。它**没有复用旧 full_chain 的 SR64/SW64、两周期整数 MAC 或共享 DMA 时序**，不能把本表直接加到旧服务槽或宣称旧原机提速。可综合结构使用 flop/mux 和组合 local read；没有假称已做 SRAM 宏映射、综合面积或频率等价。两臂共同预算控制了核内比较；实际 ASIC 还须支付端口、乘法器与 mux 的物理延迟。

解析的全 signed24 输入域位宽界已固定到实际系数：K1/K2 中间绝对值分别 ≤1,509,949,440 / 1,694,498,816，均 <2^31；两项末级绝对和 ≤703,602,884,608，原 V 为 ≤1,223,679,803,392，均 <2^47。因此共同用 signed32 latent、signed48 partial sum，不丢位、不新增中间舍入。初版较宽 datapath 后按此界共同缩窄并重跑，未扫多个硬件点。

系数物理装入含 padding：K1/K2 分别 18/36 个 128bit 词，即 288/576B；逻辑参数本体为 240/480B。A 六词中每词只占四个16bit系数，其余四槽不免费消失。B 十二词跨 a 重用。原/展开矩阵 288 词即4,608B；所有模式再装入96个原 bias。没有运行时展开 Kronecker 大矩阵。

独立复审发现旧直接控制漏掉 Kron1 展开矩阵的36个全零128bit词，现已补齐共同硬件。mask由cfg同拍生成，零系数词仍支付原装入；direct每组24词、因子A/B每组6/4词都由同一24位选择器找下一个存活词。全零组仍付CLEAR/COMMIT，跳过MAC；活组只读/计算存活词。该选择器是组合逻辑，已经实际编译/执行，但其关键路径延长未做物理测量，不能把少周期直接称同频提速。[静态词清单](static_zero_words.json)：只有expanded_k1有36个全零词；原V、expanded_k2、Kron1/Kron2因子装入词均无全零词。因子A同词含4个a的系数，保守地只跳整个物理词，不按其中某两行的逻辑零额外跳读。

[flow代理复审](../interval/REVIEW_KRON.md) 已确认共同权限、cfg/live-mask、源/CR同步地址和算术边界，关闭此强控制缺口；10行周期/CR读账均闭合。固定模型没有整个loop全零，空loop分支只有源码审阅覆盖；所有将被调用的系数地址须先完整cfg。selector→异步CR→MAC的组合链未测Fmax，这些边界保留。

## 5. Verilator 完整输入到最后输出的实测

实际环境 `/opt/anaconda3/bin/python3.12`、Verilator 4.028、g++。加入共同整词跳零后编译无 lint warning，并重新运行全部10个case。每模式运行320向量；测试正常接口和固定压力接口，合计**307,200 个输出零差**。原矩阵与捕获一致；Kron1/Kron2 与各自展开矩阵一致。压力 TB 还检查 stalled output 的 valid/data 保持稳定，CR读事件与实际MAC发射逐拍计数。

计时从该模式开始装入系数/bias，到第320个向量最后一个输出被接受；包含每向量 start、全部输入接受/等待、清 accumulator、全部实际乘加、中间/结果写回、最后外部输出/背压。复位拍不计；一次系数装入被320向量摊销。没有把 Python 算子数改名成周期。

| 真实执行，已共同跳整零词 | 无背压周期 | 压力接口周期 | 相对同函数直接减少，无背压 / 压力 |
|---|---:|---:|---:|
| 原 V 直接矩阵 | 105,344 | 107,210 | — |
| 展开 Kron1 同函数直接矩阵 | **93,824** | **96,103** | — |
| Kron1 两级收缩 | **33,394** | **35,321** | **64.4078% / 63.2467%** |
| 展开 Kron2 同函数直接矩阵 | 105,344 | 107,210 | — |
| Kron2 两级收缩 | **61,572** | **63,492** | **41.5515% / 40.7779%** |

93,824是本次修改后Verilator的实测值，不是把复审给出的乐观下界直接写进结果。Kron1的同函数直接控制减少了36×320=11,520次CR读/MAC，真实CR读为80,640；控制、装入和写回均未省略。相对不同函数的原V，Kron1仍少68.3000%无背压周期，但同函数执行增益应采用上表64.4078%。

| 无背压分项，320向量 | 原直接 | 展开Kron1直接 | Kron1 | Kron2 |
|---|---:|---:|---:|---:|
| 系数装入，同时生成mask | 288 | 288 | 18 | 36 |
| bias 装入 | 96 | 96 | 96 | 96 |
| 每向量 start 合计 | 320 | 320 | 320 | 320 |
| 全部输入装入 | 960 | 960 | 960 | 960 |
| accumulator 初始化 | 3,840 | 3,840 | 4,480 | 8,960 |
| 真实 MAC 发射 / CR词读 | 92,160 | 80,640 | 19,200 | 38,400 |
| latent 写回 | 0 | 0 | 640 | 1,280 |
| 结果写回，包括跨项累积 | 3,840 | 3,840 | 3,840 | 7,680 |
| 最后输出写出 | 3,840 | 3,840 | 3,840 | 3,840 |
| **总计** | **105,344** | **93,824** | **33,394** | **61,572** |

旧控制表、源码、资源与日志原样归档在 [old_control/REPORT.md](old_control/REPORT.md) 和 [旧RTL结果](old_control/rtl_results.json)，其中Kron1展开direct为105,344/107,210拍、对该旧控制节省68.3000%/67.0544%。这是明确标记的旧控制，不再作最强同函数分母；归档报告内相对链接仍以原kron目录为基准。

压力接口：每7个系数词前插入一个装入空拍；input valid 在全局 cycle mod11 的0/1/2关闭，output ready 在 mod13 的0/1/2/3关闭。相同确定性时间规则下不同调度遇到的空拍数不同，所有等待逐拍计费；不是对候选赠送同一人工常数。原/Kron1/Kron2 输入等待9/197/317拍，输出等待1,815/1,727/1,597拍，另有系数装入空拍。

全部模式现在都有整128bit零系数词跳过权限；本捕获原V系数词和输入全部非零，原V没有额外可省的整词。初始化和 COMMIT 分拍让短收缩相对承担更多控制开销。对原密集矩阵，8 MAC 光做完整乘加也至少需92,160拍，仍高于本Kron2完整61,572拍；收益不依赖把直接阵列仅初始化成一个异常慢的版本。但这不排除训练后普通低秩、W4/W8或别的变换用不同函数取得更优交易。

## 6. 完整 A 的迁移范围与未完成项

| A 模块 | 本轮状态 |
|---|---|
| GKPD 多项表示、重排 SVD、固定配置截断 | 已按该方法实现1/2项 |
| GKPD 不重建原核的因子执行 | 当前矩阵向量版已实装；未复现整套 CNN 算子 |
| GKPD 网络恢复训练/随机初始化训练/跨模型评价 | 未运行，不借论文精度 |
| FastKron sliced 顺序、避免实体转置、因素间局部留存 | 固定索引和16元素 local latent 已实现对应权限 |
| FastKron GPU shift caching、共享内存银行优化、GPU自动配置和完整吞吐基线 | 未复现；当前核是小型 flop/mux，不宣称等效 GPU 硬件 |
| 当前 SNN 的原 U/V 舍入与 bias 接口 | 原数组真实输入逐值校验，近似函数显式另标 |
| 同函数 expanded 控制、相同硬件预算、完整事务背压 | 已实跑，并按独立审阅补齐共同整词跳零后重跑 |
| 全网络 AEE、训练恢复、SRAM/ASIC PPA、原 full_chain 连接 | 未做 |

阅读范围继承此前有界 primary 研究并本轮复核链接：GKPD PDF pp.2–4 / 印刷 pp.772–774 Method与Alg.1；FastKron PDF pp.3–6 / §3、4.1–4.2。作者 FastKron [开源仓库](https://github.com/abhijangda/fastkron) 已核可访问，未运行；GKPD 指定作者代码未定位，不暗示无开源。结构化来源见 [source_table.csv](source_table.csv)。

本轮最具体的下一决策是保留这份可运行小核作为 A 基线，暂停把这个免训 K1/K2 端点升为新 X。若后续优先选择该接口做一次恢复训练，需和同参数普通低秩同预算训练、原 R24 和已有低位强控制重新做真实 AEE；当前证据不足以安排整个参数扫描或宣称网络速度收益。

## 重跑与交付

```bash
cd /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/pro_fusion_trials_20260913/kron
/opt/anaconda3/bin/python3.12 reproduce.py
```

[fit_and_fixture.py](fit_and_fixture.py) 生成拟合/真实输入及整数gold；[fit_results.json](fit_results.json) 为局部误差；[rtl_results.json](rtl_results.json) 为TB原始周期；[benefits.csv](benefits.csv) 为可复算收益表；[SUMMARY.json](SUMMARY.json) 为最终机读汇总；[build.log](build.log)、[run.log](run.log) 保留本次实际构建/执行记录。`obj_dir/` 是局部生成产物，未向其他目录写入。未对文件做 hash。
