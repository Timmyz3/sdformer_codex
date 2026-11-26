# R18：实际 dense 成本、公共 k=1 RTL、合法动作边界

2026-10-17。本目录不运行 GPU/训练、不读 GT 或 R18 质量排名、不改原模型。本机 `ic.ismd-nemo` 上完成公共选择器 Verilator/DC；实际描述符分数来自冻结的 R18 CPU 缓存结果。**R17/R18 的执行图仍全 dense，没有减少 FC 行；R16 compact 与 expanded 的数值等价失败仍保留。** 这里不把评分质量与假设的 compact 加速配成一条结果。

## 已实现的控制器

`rtl/k1_f64.sv` 接收固定 row-major 的十二个 IEEE binary64 分数；只在严格大于当前最佳时更新，精确同分和正负零均保留首项，支持有限负数。三个 full-tile 行×四列的获胜序号 `i` 映射成 4×5 网格 bit `i+floor(i/4)`，八个尾块永不被选中。NaN/Inf 置 fault 并禁止可运行 mask；一次只处理一个决策，支持输入暂停和输出 backpressure。该模块只比较已有 score bit，不计算 FP64 分数、不量化、不读取特征。

`tb/tb_k1.cpp` 用宿主 IEEE double 的独立严格比较检查 2,019 个合成事务，包括各赢家、精确 tie、正负零、负数、随机有限 bit pattern、非法数和停顿；另读取 R18 的 **56 个实际分数向量、672 个分数**，全部通过。无停顿时从 start 接受到 decision 需 **13 tick**，含 result 退休为 **14 tick**；每方法十二分数、十一次有效候选比较。七方法的实际 CPU 筛查不是部署时必须同时运行七次评分。`results/verilator/receipt.json` 保留仅合成测试，`actual_receipt.json` 是合成＋真实向量检查，原始向量及来源 SHA 单独保存。

DC V-2023.12-SP3 使用光流 date16 的 TSMC `tcbn28hpcplusbwp7t35p140tt0p9v25c.db`、TT0.9V/25°C；4ns clock、uncertainty/transition各0.05ns、I/O delay0.2ns、load0.01pF、2CPU。实测 cell area **377.201999 库面积单位**，worst setup slack **+2.38412ns**，678 cells（603组合、75时序），0宏/black box，无违反约束。只含比较器、寄存器、计数和 mask 逻辑；不含评分 FP64 单元、SRAM、BN、MLP 或整个执行器。保留 VO-4、check_design LINT-28（27个优化/未用端口）与 LINT-33（共享常量连线）提示，不称零 warning；没有布局、功耗、门级/形式等价结果。不能用此 area/slack 配合下面假定 PE 时间声称完整加速器已综合。

## 同一实际 dense 图的费用

当前有效账本是 `results/SERVICE_ACCOUNT_V2.json`，由 `service_account.py` 产生；5项 CPU 检查覆盖独立描述符算式与作者 ledger 一致、十二 singleton 真尾部覆盖、容量拒绝、dense/compact隔离、池化 repeat 读写。`SERVICE_ACCOUNT_V1.json` 是未冻结的初稿，遗漏了 descriptor repeat 的额外本地读写；已保留而不用于结论。V2 显式增加这些费用，没有更改评分/质量门。

配置为 **2MiB activation SRAM（其中128KiB统计/元数据）、512KiB权重、16个1RW32-bit activation banks、32个权重banks、64B/tick外部接口**。评分以完整 T10、9×9、C64 stripe 流水扫描，每 tile 六 stripe，四份64-bit临时数组＋一份原FP32 stripe＋预留共 **1,997,312B**。评分算术暂假定一个串行 FP64/integer lane，逐操作 load/compute/store，64-bit读写每次占两bank；add/mul/sub latency4、div16、sqrt32，其他列在 JSON。不是实测 FP64 datapath；未验证其归约次序能逐位复现 NumPy 分数。真正 RTL 只保证给定 score word 的选择完全一致。

MLP 仅复用 R16 公共服务模型的 **全 fine、full-domain current-BN** 分支，未调用 weighted compact 或旧54BN串行总数。该模型有限权重 stripe、完整 T、保存 FC 输出跨两次统计屏障并回读，目标 MLP 每 FC 始终 **12,000 行**；整十二 MLP 每 FC 始终 **558,000 行**。参考粗化仍执行 clone、3×3 mean、full-grid repeat，然后同一个 dense MLP，k1 只改变数值，不改变上述 FC 行。十二个可选完整位置都有 81 fine 像素，剩余1,119 coarse像素、131个coarse代表；真尾块没有被当成9×9。

每个部署策略只计自己的一个 score＋selector＋目标 dense MLP＋粗化服务。幅值/RMS/残差/邻差均须读目标前 u；两项 PSN 分数还需产生**原生 u 的完整 PSN1**，账内另加 46,080,000 MAC 及 bias/threshold、读取/保存。缓存 sn1 只是离线证据；不能假定它等于 coarse 输入的 `PSN1(pool(u))`，也不能给这两项方法免费 producer。原完整 u 为18,432,000B，保存在外部，不冒称装进2MiB。

JSON 中的整数 tick 是该阻塞资源假设下的调度值，**不是 FLOPs 直接换周期，也不是原生 BN/FP32 完整数值 RTL**。FP32 PE/BN 延迟、归约次序、FP64评分实现、详细 bank 地址、attention/conv/decoder 日程和全网周期分母仍未闭合。其余11MLP、54外BN也未被假装取消；已知域元素单列为库存。实际 dense 下新增评分及粗化费用没有抵消的 FC 行，不能声称净加速。

## Compact 必须单列成另一算法图

`hypothetical_compact_separate` 只列 V2 的生产服务：目标212个代表→每FC2,120行，整十二MLP每FC548,120行；**两次 BN 仍覆盖23,040,000个完整域元素**，另付 FC输出 expand、BN1 gather、存储/回读和评分费用。不套用 V1 的 weighted-BN 小域服务，不给周期比，不转用 R17/R18 质量。

R16 证明的是该 compact 图不满足旧 expanded 数值等价门，尚未测其自身相对 GT 的质量；允许把现有 V2 另定义为独立近似算法，在新合同中验证全 fine、多输入状态和重放，重采 coarse/联合动作标签及 GT 质量。无需发明第三个 kernel，也不能把此路径称为恢复了旧等价。未来硬件必须对齐这张新图自己的数值/统计服务。单 s2b2 的 CC 最多取消十二 MLP FC MAC 的 **7.361111%**，仍不是整网周期或10%净PE证据。

## 原 Swin 的早决策与最小合法替代

原代码 SHA `8d551eac...7768` 中，`forward` 为 `x+SSA(x)`，再对其加 `MLP(x)`（行824–847）；MLP是 PSN1→FC1→全域BN1→PSN2→FC2→全域BN2（行164–178）。当前 u 在 attention 后，只能选择 MLP。attention 的 q/k/proj 三个 current-BN 统计整个 packed 输入（含 padding），并非逐窗独立；仅把某区域最终 residual 置零，仍需其 linear/FC 输出维持其他区域统计，不能直接算省掉 producer。

根代理的 `../geometry/audit_native_layout.py` 还逐标量核了原 partition/reverse：实际 `[2,100,9,9,384]` 的 PSN time 轴并非原空间块的相邻时间。例如 packed列0的两值来自 `(t,h,w)=(0,0,0)` 与 `(4,18,0)`。独立源码短审确认它照原 q/k reshape、sum、乘法及 proj 重排传播潜在依赖；故意排除 current BN 后，目标层20空间tile的无向结构图仍连通。该结果不是激活Jacobian/最小skip集合证明；但足以否定仅凭 `[2,9,9]` 模块名就宣布可独立跳过整窗。根脚本原 source/GT/GPU均未改变，R17只改MLP，已有质量范围不因此改写。

最小确定合法、无需重写 BN 的**不同问题**是整层 MLP 残差 identity bypass：coarse直接保留 attention 后 x，fine调用完整原 MLP，整T/全空间跳过两FC、两PSN、两BN。更粗的整Swin residual block identity还能取消SSA；其费用和质量须新测。它能作为控制/动作诊断，却失去空间预算粒度，不能冒充区域损伤主故事。局部 zero-fill、active-only BN、冻结BN、重排修正都是新数值图，均不能继承 R17 端点质量。后续主线更值得优先审独立 compact 实际图的有限质量/成本，而非无限修旧等价或把整层 bypass 改名为区域方法。

## 复核与复现

数据代理只读核了 k1 RTL 的 strict tie/±0/负数顺序/真实4×5 mask，以及 service 的费用与边界，未发现阻断；它没有重复跑RTL或给本模型独立完整硬件认证。本目录也只读审了根的 index audit，未复跑神经网络。详见 `PEER_REVIEW.md`。这是内部同行分工复核，不是外部 TVLSI/TCAS-II 评审。

执行 `bash scripts/run_verilator.sh` 或 `bash scripts/run_dc.sh` 会覆盖 build/results；冻结后请复制到新目录再运行。`service_account.py --output <新路径>` 拒绝覆盖输出。所有数字以 `FREEZE_V1.json` 绑定文件为准。
