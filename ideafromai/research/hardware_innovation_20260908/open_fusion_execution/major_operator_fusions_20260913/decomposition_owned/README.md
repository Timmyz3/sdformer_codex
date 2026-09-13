已完成真实多轴分解、整数 pair 基和 H8 共同 pair-ID 三类接口，及同精度强普通控制。root 的完整十帧前向确认：普通无残差 flat SVD R8、激活加权 SVD R8、空间 R16、Tucker R8 全部通过 NB0，必须保留为有效的强 A；不能用局部 L2 淘汰整个分解方向。整数 pair 同样通过质量门槛，但付清公共事件遍历后的有限服务模型仍未胜 dense；当前没有硬件加速 PASS。

| 当前共同父及分解接口 | diverse10 AEE |
|---|---:|
| parent | 1.1597366283 |
| flat SVD R8，无残差 | 1.3479650409 |
| activation SVD R8，无残差 | 1.4030194175 |
| spatial R16，无残差 | 1.2691297866 |
| Tucker R8，无残差 | 1.4170152223 |
| flat + 2:4 R8 | 1.1633597872 |
| spatial + 2:4 R8 | 1.1840618202 |
| 普通 3:4，本组同 tap 四通道分组 | 1.1750728784 |
| signed pair，输出私有 ID | 1.1633027787 |
| unsigned pair，输出私有 ID | 1.1512492510 |
| unsigned pair，H8 共同 ID | 1.1966744037 |
| 普通 3:4，H8 共同 omitted-ID | 1.2144504382 |

全部低于本协议 NB0=1.4546028611。unsigned 私有 pair 的平均 AEE 比 parent 低 0.0084874，不能据此声明显著改进；它的局部 L2 比 signed 差却有更低整网 AEE，因此不再按局部误差选 signed 唯一 winner。全部完整运行及校准帧说明见 [root AEE 汇总](../root_owned/aee_all.md)，没有新增训练，也没有将探索性 diverse10 写成 valid825。

当前真实挂点为 `patch_embed.residual_encoding.resblocks.0.conv2.0`，原生 96×96×3×3。root 的 [当前执行账本](../root_owned/profile.json) 确认 matched dense stage320 实际执行该卷积：名义 dense MAC 63.701 G，占实际采到的 596.546 G 算术 extent 的 10.678%。这是包含 padding 的稠密计算规模，不是 ASIC 周期；历史 patch 34.83%/FFN 26.01% 仅作初始定位。`ParentNetwork` 已将 r1 Conv1 换成 preview-only，本文不再给它虚构完整 fallback 费用。当前导出的 r0 Conv2 W 与历史 capture_train4 的 W 逐值完全一致。

真正的问题是数值域：原输入为静态 θ 乘脉冲位，密集 W 也只需事件驱动的加权加法；一般 SVD/Tucker/空间分解的第一阶段输出会变成连续值，第二阶段仍需乘法。不能把因子后状态继续乘原发放率。PSN/PED 连续成本未机械加进下表。

| 已执行接口 | 真实四帧局部留出结果 | 原算子计算边界 |
|---|---|---|
| 平铺 SVD 与激活二阶矩加权 SVD | 同秩下加权 SVD 更低输出误差 | 第一因子 θg；第二因子连续 |
| 空间 3×1→1×3，M[o,w;c,h] | 与平铺展开正交；空间+2:4 R32 relL2 0.05655 | 2126.14 AAC + 9216 MAC/位置 |
| 普通平铺 SVD+2:4 R24 | relL2 0.05694，与上行接近 | 2600.37 AAC + 2304 MAC/位置 |
| 双通道 Tucker-2 1×1→3×3→1×1 | R64 relL2 0.17931 | 257.34 AAC + 43008 MAC/位置 |
| 普通 3:4 结构零 | relL2 0.07718 | 2594.06 AAC、0 连续 MAC/位置 |
| 原稠密 W | 原捕获归约误差约 8.2–8.5e-5 | 3474.08 AAC、0 连续 MAC/位置 |

前两帧仅用于激活加权 SVD 的二阶矩，后两帧单列局部留出；其余分解只拟合 W。41 个接口×参数配置分别导出 FP32/W8 系数，共同给予逐输出行量化权限，普通 N:M 也有 W8 控制。这不是 41 个新 idea。TT、Kronecker 和 Winograd 只列了候选路线，本轮未执行。这里的 W8 是系数格式，连续状态没有被假装成 A8。41 个实际因子链另在带边界 padding 的完整小图上对重构核独立检查，Float64 误差小于 1e-10。[完整数值及同局部误差比较](results/summary.json)、[当前父首帧输出复核](results/current_capture.json)。

第二条分解是 `selected_pair`。按每个空间 tap 的四个输入通道分组，W 的每个输出组中选两个坐标近似为共同幅值 ±a，另两个保存原值。符号基由同一输入组的所有输出共享，输出只保存 pair-ID。枚举八个符号基，对每个输出找到符号翻转后最接近的两个权重，取其均值；这是该有限组内类的闭式最小二乘解，没有使用验证 GT。

其实际执行为：

```text
4 个源脉冲位 → 源 code
pair-ID + 共享 signs → 选中两位的计数 c ∈ {-2,-1,0,1,2}
两个未选中的坐标 → 原值加权累加
c=0: 跳过；|c|=1: 加/减 θa；|c|=2: 加/减 (θa << 1)
```

θ 静态折入系数；移位示意对实际定点系数成立，CPU 使用同义的乘2常量及加法。原先先算四位总 count 再加 residual 的版本在当前帧需 3507.77 AAC，高于原 2665.35；必须在应用 a **之前**排除两个例外位，才降到 2571.76。两者浮点分解函数一致，差分在计数与例外合并的位置。

| 当前 matched dense 首帧、640 个真实 T/空间样本 | 输出 relative L2 | 全宽加权加法/位置 | 连续 MAC/位置 |
|---|---:|---:|---:|
| 原 W | 约 1e-4 的原生归约差 | 2665.35 | 0 |
| 普通 3:4 | 0.08664 | 约原值的 75% | 0 |
| 普通 flat+2:4 R32 | 0.04838 | 第一因子+残差另计 | 3072 |
| unsigned selected pair | 0.05205 | 2600.84 | 0 |
| signed selected pair | 0.04998 | 2571.76 | 0 |

signed 相对同能力 unsigned 少 1.118% 加权项，而不是把相对原 W 的 3.511% 全部记为新贡献。历史四帧同接口分别少 4.51%、4.72%、5.04%、4.23%；W8 的额外零不得算成该接口独有收益。r0 Conv1 也完成同函数真实首帧检查：relative L2 0.06031，3309.45→3172.64 AAC（−4.13%），仍不是网络 AEE。[计数/位面逐帧结果](count_results/summary.json)、[Conv1 当前帧结果](count_results/current_conv1.json)。

FP32 三项系数加 bias 为 249,216 B；输出私有 unsigned pair-ID 另需 7,776 B，signed 再加 108 B shared signs。新增 H8 接口枚举六 pair，对同一 H8 的八个输出权重误差求和后选共同 pair，各输出 a 仍独立；普通 3:4 同样获得共同 omitted-index 最小二乘控制。metadata 分别降至 972 B 和 648 B，实际 NPZ、四帧数值检查和十帧 AEE 已完成，未增加 rank 扫参。[H8 拟合与数值](count_results/h8_shared_summary.json)。

最终 [有限资源服务表及同值 metadata 预算](PAID_SCHEDULE.md) 明确支付 RF 冲突、公共 slot/time walker、计数/例外选择、全零缓存组检查、累加器初始化和输出服务。全部当前 64 个 P1/T10 样本，dense 为 713544 服务步，unsigned 私有 ID +共享计数为 1797852，H8 pair 为 890128；H8 共同 omitted-ID 的普通 3:4 为 729862，其同值 dense-zero 布局为 617255。H8 pair 比私有 ID 约少一半服务，但仍高于 dense 24.75%。该表仍是端口位流服务模型，未闭合静态物理地址请求、producer gather、实际定点或 RTL，不申领周期/PPA PASS。

可直接接 root 的 GPU 配置见 [GPU_INTERFACE.md](GPU_INTERFACE.md)。GPU 对 count 使用重构核做真实全网数值评价，对一般分解可接实际因子链；CPU `count_basis.execute_pair` 已独立执行整数 pair count 的两位面输出，与同核 Float64 卷积小于 1e-10。AEE 已完成，未训练、未完成 RTL。普通无残差分解已通过质量门槛，后续应直接处理后因子连续 MAC/状态接口，不能因它不是脉冲就退回添加 50% 稀疏残差。

新颖性边界：可借入的完整 A 是低维共享基、低秩加残差、N:M 结构、静态量化及基本位面执行。候选增量句为：“在 θg 卷积的共享二输入整数基中，按消费者选择的例外坐标先从计数义务中删除，使公共幅值只接收未被直接权重消费的有界计数，从而避免‘低秩后因子连续化’和‘先算再抵消’的服务。”这只是待验证假设；需要同样允许输出侧等权/反号合并的普通权重聚类编译器、unsigned 字典及原稠密 W 的同资源对照，不能把 pair 共享本身申领成新概念。

本轮选定的四篇 primary 约束如下，原文已核到所列方法段；没有把综述卡当作全文或完整实现：

| Primary | 已有 A / 本轮借入程度 |
|---|---|
| [TASD，MLSys 2025，§3–4](https://proceedings.mlsys.org/paper_files/paper/2025/file/e2ec2530db26b54d0b3b060c1e4a1bda-Paper-Conference.pdf) | 结构稀疏项分解及共享输入/psum 映射已有。实作 N:M 抽取和例外项，没有迁入 TASDER 层级精度/延迟搜索或 TTC。Grok 原卡误作时间调度的描述不沿用。 |
| [CALDERA，NeurIPS 2024，Algorithms 1–2](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html) | Q+LR、激活加权目标和低精度交替优化已有。本轮实作二阶矩加权 SVD和低秩/稀疏交替投影，没有 LDLQ、Hadamard incoherence、格点量化或完整 CALDERA。 |
| [LQER，ICML 2024，§3](https://arxiv.org/html/2402.02446v3) | 激活诱导缩放、量化误差的低秩重构及低精度硬件已有。本轮只给全部轴共同系数量化，未完整迁入 LQER。 |
| [LegoNet，ICML 2019，§2–3](https://proceedings.mlr.press/v97/yang19c/yang19c.pdf) | 共享低维 filter、二值选择、split-transform-merge 和复用中间特征已有。本轮借分组共享基与消费者选择；第一基受限为脉冲两位的和/差，并增加直接例外项，没有训练 LegoNet 的连续 filters/STE selection。 |

两份 gptpro 原文强调实际共同物理事务、状态生存期及完整链费用。这里遵守同一边界：整数 count 状态和 pair 查询都要收费，不能把逻辑少几项直接当成取消物理读请求。其历史 +0.005 门不继承；本轮 AEE 只按当前同协议 NB0 比较。

对另一组当前 r0 nonzero-fill/request-aware 原型的独立审阅见 [independent_sparse_review.md](independent_sparse_review.md)：核过 Finch、SumMerge 和稀疏卷积组合编译，明确保留 3.06% 同函数族服务计数信号及剩余 X，也指出当前普通 2:4 在 AEE/服务计数上同时占优。评分按新颖性、实际适配、性能证据三项给出，没有接受概率。
