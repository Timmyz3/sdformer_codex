# R34：实际融合特征、单头供数和静态邻预算

**有效交付在 `actual_v2/`；V1 因真实 IOS 生命周期/交接错误判为失败并完整保留。** V2 对 120 个本地冻结完整 T10 输入执行 C++ 融合特征，保存真实 IOS handoff 字节，再执行实际拟合头；通过完整记录的 IOS/RES/DMA 地址审核和同址读取检查。算术延迟与 scalar RF 路由仍是明确的有限资源模型假设，没有新增 RTL/DC，也不声称物理时序、完整网络时延或净硬件收益。

## 已完成的判别

| 首个完整输入，主 row-band 配置 | 特征模型 tick | 单头到 action tick | 新增串行前缀 tick |
|---|---:|---:|---:|
| X128 + folded constant-v | 113,938,624 | 8,599 | **113,947,223** |
| X128 + ordinary direct_raw | 113,938,624 | 8,599 | **113,947,223** |
| X128 + signed_g1 | 113,938,624 | 16,007 | 113,954,631 |
| X160 + signed_g1 | 262,240,560 | 19,952 | **262,260,512** |
| X160 + fit-only calibrated residual_g1 | 262,240,560 | 32,786 | 262,273,346 |

全部 g1/g4、raw/augmented/residual、两个新普通校准头和 folded 头共 17 个单头的实际参数、服务事件与结果见 [HEAD_PREFIX_RESULTS_V1.json](actual_v2/HEAD_PREFIX_RESULTS_V1.json)。单头只支付自己的必要依赖；不把七头研究程序当部署成本。folded 与 ordinary raw 的形状、权重供数和服务完全相同，不能宣称 folded 自带更快微架构。校准 residual 的六个 shrink 乘法也已支付。

在这个冻结的单个 blocking FP 单元配置下，特征远比小头昂贵。X160 的方向特征相对 X128 增加约 148.3M 模型 tick；取消方向特征后，原 X128 统计本身仍需约 113.9M。这个负面证据要求重新审视特征/动作是否值得支付；它只评价当前资源点，不证明所有流式或整数实现都必然昂贵。

## 真实输入、数值和物理交接

- C++ 读取全部 120 个真实 T10×480×640 FP32 `.npy`，逐一核输入 SHA；运行时不读 GT。FP32 特征与 numeric 独立实现最大绝对差 4.76837158203125e-7，**并非逐位相同**，差异已保留。14 个原头、两个普通校准头、folded 头在全部 120 输入上的 fixed-six/allow-Q4 动作均零翻转。不能用零翻转把特征误差写成逐位一致。[执行记录](actual_v2/STREAM_EXECUTION_V1.json)
- 主配置一次借用 RES 存放 3,072,000B 的 120×640×T10 行带，连同 previous/signed-sum 共 3,225,600B，小于既有 4MiB。40 个 307,200B 外存事务覆盖输入恰好一次，共 12,288,000B；每次付 32 startup 和 64B/tick 数据服务。IOS→RES 保守双阶段再付 384,000 tick。
- 唯一邻资源条件用 768,000B 单 cell，连同状态占 921,600B；19,200 个 640B DMA 事务，输入字节相同，特征数值逐位相同，X160 服务为 262,853,680，比主配置多 613,120 tick。没有免费 2D DMA。
- V2 特征真实写在 IOS `[400000,400640)`，不与后续 DMA `[0,307200)` 或梯度 scratch 重叠。C++ 结束时逐项检查实际 IOS readback，输出来自这些字节，并为 120 输入保存完整 512KiB handoff。头用同一 image、同一地址读取；normalization/aux/products/aug 的后续阶段也通过实际 IOS backing array 连接。
- 三份完整地址轨迹记录每个 IOS/RES 范围的开始/结束、字节地址、长度、方向、向量/标量服务；全 DMA 地址另有 CSV。自身审核检查所有记录、端口、边界、DMA 恰好一次覆盖、受保护特征区和 120 handoff payload。[完整轨迹审核](actual_v2/TRACE_HANDOFF_AUDIT_V2.json)

新增 2KiB scalar RF **没有地址级 instrumentation**，其静态活跃数据上界、两读一写和结果路由假设，以及额外 controller 状态单列在 [RESOURCE_SCOPE_V2.md](RESOURCE_SCOPE_V2.md)。不能把 ID2 的零 trace 计数解释成“没有寄存器访问”。算术采用一项阻塞操作：add/sub/mul 3、div/sqrt 16，其余列明标量操作 1 个模型 tick；这些不是 RTL 时序。

## 精确静态邻预算，精度尚未测

以已暴露旧 confirm 的 40 个真实 action 计算平均服务，仅将**新增串行预测前缀 + 四个孤立 conv 服务之和**作为这轮资源预算。它不是网络时延。普通静态方法省下前缀，可多升级完整 tile。新 mask 均先保留 fit 静态 group2 的 IDs302..452，再按最小剩余 tile ID 补充；真实尾保持 Q8。

| 动态候选/策略 | 可用模型预算 | 最大不超预算静态完整 Q8 tiles | 四孤立静态 conv 服务 | 下一 tile 的服务 |
|---|---:|---:|---:|---:|
| folded X128 fixed-six | 1,032,725,470.1 | **360** | 1,032,538,860 | 1,033,083,180 |
| folded X128 allow-Q4 | 1,026,109,634.6 | 348 | 1,026,007,020 | 1,026,551,340 |
| signed X160 fixed-six | 1,181,038,704.6 | **632** | 1,180,593,900 | 1,181,138,220 |
| signed X160 allow-Q4 | 1,174,422,869.1 | 620 | 1,174,062,060 | 1,174,606,380 |

这些边界经过继承的完整有限供数 Simulator 实际重放，未用工作量直接代替服务。最大 tile 数只针对上述冻结的补 tile 顺序，不是所有静态布局的最优下界。新代码只改 mask 来源与 main，Simulator 主体与 R33 冻结模型相同；151-tile 回归精确复现 229,694,508 tick/conv。每次还独立核算 Q4/Q8 行数和 accepted beats。[搜索结果](actual_v2/STATIC_NEIGHBOR_SEARCH_V1.json)、[来源](STATIC_SOURCE_BINDINGS_V1.json)

360-tile 的确定集合为 `302..452` 加 `0..208`，每个热点 conv Q8 rows=322,500、Q4 rows=445,500、accepted beats=183,204,000、服务=258,134,715。明确冻结合同在 [STATIC360_CONTRACT_V2.json](actual_v2/STATIC360_CONTRACT_V2.json)。**这些新增 mask 均无 AEE**，不能从六个原宏组缓存推断，也不能凭预算证明任何精度优势。X128 每身份费用使用固定工作和真实零 mass guard；额外执行了首例及全部四个零 mass 身份的完整 X128 路径。[验证](actual_v2/BASE128_GUARD_VALIDATION_V2.json)

## 因果边界与失败记录

输入起点是完整当前 T10 已在外存；采集/voxel 形成在此起点之前，未声称免费。第一受控层是 `patch_embed.residual_encoding.resblocks.0.conv1.0`，需要共享 stem 的二值 T10×96×240×320 输入。真正开工点为 `max(action_ready, shared_stem_ready)`，stem 服务当前未知，不能将本前缀当完整 raw-voxel→conv 系统时延，也未免费扣除重叠。allow-Q4 身份依旧支付完整前缀。

V1 失败是数值正确但片上持久存储错误；详见 [失败记录](failures/V1_STORAGE_HANDOFF_FAILURE.json)。V2 没有扩大 SRAM、FP 单元数或吞吐来救结果。初版 reducer 的 NumPy 逐位匹配尝试不成立，但真实输出误差与动作稳定性已公开；源编译的误导缩进 warning 已修复。没有新增 GPU、训练、RTL 或 DC。

独立实现者以外复审核过三份完整 trace 共 19,181,120 条记录、120 份实际 IOS handoff 和 17 个头的首读地址及 FP32 score，解除 V1 存储阻断：[独立 V2 审核](../independent_review/HARDWARE_V2_INDEPENDENT_AUDIT_V1.json)。这不扩展 scalar RF、算术时序或共享 stem 的证据范围。

## 复现入口

在 CPU numpy 环境编译 `stream_features_v2.cpp`（`g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra`），顺序运行 `run_stream_v2.py`、`head_prefix_v2.py`；用 `build_static_replay_v1.py` 生成并编译 `static_neighbor_supply.cpp`，运行 `static_neighbor_search_v2.py`；最后运行 `check_base128_guard_v2.py`、`audit_stream_v2.py`。输入和模型都从既有冻结目录只读取得。详细源码、输出与审查 SHA 将记录于冻结清单。
