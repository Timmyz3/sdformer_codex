# Local5 DS-GASR-2C 五 Bank 在线 RTL 与第八轮送审

## 1. 证据边界

本轮只使用当前机器可复现的 RTL、脚本、向量和日志。外部机器或其他 agent 的结果不计入完成度。

评估边界是 Local5 full-resolution `15x15x2=T450` 的 post-G0 稀疏投影子系统，不是完整光流网络、完整 encoder 或芯片 PPA。算法数值合同保持不变：相同 gate、K、权重和 Acc32 累加顺序等价结果。

## 2. 为什么上一版五 Bank 没有赢

第一版五 bank GASR 从 active-index 的 relation-read issue 位置提前发送 geometry。当时 K/gate payload 尚未返回，硬件不知道真实 `descriptor_valid_mask`，只能对所有边界内 Local5 邻居 bank 做 prepare。

真实 profile100 结果：

| 模式 | 总周期 | 相对 direct | 执行期 SRAM 事务 |
|---|---:|---:|---:|
| direct-1RW | 121,501 | 1.000x | 318,974 |
| blind-geometry GASR | 122,360 | 0.993x | 60,510 |

该负结果说明：单 bank 的 `1.609x` 不能外推到在线五 bank；盲预取虽然减少 SRAM 工作，却会阻塞 relation frontier。

## 3. 本土化改进：DS-GASR-2C

DS-GASR 表示 **Descriptor-Synchronized Geometry-Ahead Source Residency**。改进不是增加近似，而是把 geometry 提交延后到完整 descriptor 可见的时刻：

1. relation frontier 先按真实同步 SRAM 延迟返回 K、五方向 gate 和有效 mask；
2. descriptor 只有在 FIFO2 和 GASR backend 同时 ready 时才原子提交；
3. descriptor FIFO 与 backend geometry 采用双 ready 原子提交，不允许部分可见；
4. 当前 source 的 term 流与下一 source 的 bank prepare 继续重叠；
5. source 尾 term 只有在下一 source 所需 bank 均可 activate 时提交。

这利用了 Local5 的固定五邻域和五着色映射，每个有效 role 映射到不同 bank。它不同于复制 dense/sparse 双核，但 descriptor 同步本身仍可能被审稿人视为常规流水解耦。

必须记录一项更正：`scripts/profile_local5_descriptor_geometry_qualification.py` 对 profile100 的 9,427 个 active sources 做了逐 role 重建，边界 geometry 与 descriptor candidate-valid 均为 44,661 个 roles，差异为 0。因此当前数据上不存在 mask pruning，不能把收益解释为“删掉无效 bank”；收益只能归因于 relation-read 与 prepare 顺序重排及原子 descriptor/geometry 提交。

## 4. 新增与修改的 RTL

- `qfit_dual_color_relation_frontier_sync.sv`：提供 relation issue geometry 接口，并保持旧顶层兼容；
- `qfit_source_multicast_term_builder_fifo2.sv`：两项 descriptor 解耦，允许当前 source 发 term 时缓存下一 source；
- `qfit_local5_color_map.sv`：Local5 role 到五颜色 bank/地址映射；
- `qfit_local5_1rw_projection_backend.sv`：公平 direct/GASR 五 bank 共用接口；
- `qfit_local5_1rw_active_projection_tile.sv`：word-skipper、relation frontier、FIFO2 builder 和五 bank backend 在线集成；
- `qfit_local5_1rw_active_projection_assertions.sv`：原子提交、稳定性、source context 和 close 边界断言。

## 5. 公平 profile100 结果

结果目录：`results/local5_qgasr2c_fivebank_postg0_rtl_20260804/`。

一键入口：`sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh`。

| 指标 | direct-1RW | DS-GASR-2C | 变化 |
|---|---:|---:|---:|
| profile100 总周期 | 121,501 | 119,348 | 1.018x |
| term stall | 13,055 | 10,965 | -16.01% |
| SRAM 总事务 | 318,974 | 60,510 | -81.03% |
| descriptors | 9,427 | 9,427 | 相同 |
| terms | 53,085 | 53,085 | 相同 |
| destination updates | 166,080 | 166,080 | 相同 |
| Acc32 | 100/100 PASS | 100/100 PASS | 零失配 |

周期从 `projection_start` 计到五 bank flush 完成，不含结果读回。SRAM 事务只计执行期 backing SRAM 读写，不含每组 900 次结果检查。

逐组 win/equal/loss 为 `17/45/38`，p50 为 `1.000x`，p95 为 `1.125x`，最差为 `0.756x`。固定 DS-GASR 只取得小幅聚合收益，不能写成每个窗口均加速。

## 6. 无泄漏分层模型

只使用前 50 组搜索简单阈值，选出的规则是：

`terms / active_sources >= 4.7241379` 时使用 DS-GASR，否则使用 direct。

阈值冻结后的后 50 组结果：

- 选择 DS-GASR 15 组，其中 3 组仍退化；
- direct 为 80,821 周期，混合模型为 76,823 周期；
- 留出集加速 `1.052x`，留出集 oracle 上界 `1.058x`；
- 全 100 组模型为 `1.050x`。

该项仍是 `[模型]`，因为当前没有共享单 SRAM 的 runtime 双模式 bank，也没有证明该统计量能在不增加一遍扫描的条件下在线获得。不得把 `1.050x` 当 RTL 结果。

## 7. 验证签核

| 项目 | 结果 |
|---|---:|
| direct deterministic profile100 | PASS |
| DS-GASR deterministic profile100 | PASS |
| direct 随机输入/读回空泡 + SVA | PASS |
| DS-GASR 随机输入/读回空泡 + SVA | PASS |
| Verilator lint 两模式 | PASS，无 error |
| Yosys hierarchy/check/stat 两模式 | PASS |

SVA 覆盖：descriptor 在阻塞时稳定、FIFO 与 geometry 原子提交、不允许部分提交、GASR term source 必须匹配 active context、close 只能在 relation/builder/backend 全排空后发生，以及五个 bank 的地址、双槽和单端口合同。

Yosys memory-preserving 统计中，direct/DS-GASR 结构分别为 2,539/4,135 个抽象 cells、21/41 个 `$mem_v2`。该差异只说明 DS-GASR 双槽和控制有面积代价；它不是标准单元面积，也不能与 81.03% 事务下降直接换算成能耗。

## 8. 当前可辩护的架构贡献等级

当前 DS-GASR-2C 已从“离线单 bank 微结构”升级为“真实 word-skipper/frontier/builder 驱动的五 bank 在线数据流”：

- Local5 五着色 bank-local source residency；
- descriptor/geometry 原子提交与 source-boundary activate；
- 相同 1RW 存储合同下 direct/DS-GASR 公平 A/B；
- 真实 T450 profile100 的 bit-exact、周期和 SRAM 事务证据。

但 `1.018x` 固定周期收益偏小，descriptor 同步也偏常规，且尚无 SRAM macro PPA/活动功耗。它可以作为论文中的存储活动与数据流贡献候选，暂不足以单独支撑 DATE 接收。

## 9. 第八轮评审应回答的问题

1. `1.018x cycle + 81.03% SRAM transaction reduction` 是否足以允许继续实现 runtime 双模式；
2. DS-GASR 的 descriptor-synchronized prepare 是否达到可辩护架构新颖性，还是仍属于工程优化；
3. 下一项唯一高优先级应是共享单 SRAM 的 source/window stratifier、bank-local future queue，还是先做 macro PPA；
4. 在无 DC 的当前机器上，哪些证据必须补齐后才值得迁移到 Motion T450；
5. 哪些表述必须降级，避免把模型、Yosys 或事务数写成 PPA/EDP。

## 10. 尚未完成

- 共享同一 backing SRAM 的 runtime direct/DS-GASR 双模式 RTL；
- 可变 SRAM latency 和真正外部消费者反压；
- SRAM macro 绑定后的 OpenROAD/DC/STA/SAIF/功耗；
- Motion T450 对应机制和双线公平比较；
- full encoder 的端到端 FPS 与能耗占比闭环。
