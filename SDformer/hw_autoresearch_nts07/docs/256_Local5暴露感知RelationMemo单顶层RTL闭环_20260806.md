# Local5 暴露感知 Relation Memo 单顶层 RTL 闭环

## 1. 本轮结论

本轮把 `Exposure-Aware Exact Relation Memoization` 从 `[prof]+[模型]` 候选推进到了可综合 RTL 原型：

```text
FCSR 三行 relation transpose
  -> 第一输出 tile：全部 descriptor 走 live 路径
  -> 同拍统计精确 term service，并对 active descriptor 做 speculative write
  -> head 结束：critical 且容量可容纳则原子 commit，否则 rollback
  -> 后续输出 tile：resident head 从 512x112 vault replay
  -> source-major term builder
  -> TCFM5
  -> Acc32
```

在一个可综合 tile engine 中，第一遍 live、resident replay、nonresident miss 和 exact recompute fallback 已经共用同一条投影通路。四个连续 tile 共检查 `7200` 个 Acc32，均与测试台从原始 destination-major K/gate/valid-mask 直接计算的五邻域整数金参考一致 `[rtl]`。

本轮仍不能宣称 Local5 全链路完成。新顶层从 FCSR 输入的定点 gate/K 开始，尚未把 score/Shiftmax5、上游 Q/K/gate 存储及生产级 12-block 调度并入 `[待验证]`。控制器的 `recompute_request/recompute_grant` 已闭合 miss 后的控制协议，但 `grant` 之前的源数据恢复时延仍由外部存储系统负责。

## 2. 修复的真实 RTL 问题

最初 replay 使用“一条在途读 + 一个输出寄存器”。在下游随机反压时，新 SRAM 响应可覆盖尚未消费的旧输出，叶级测试出现：

```text
head 0 replay source mismatch got=14 exp=13
```

修复后的 replay 前端包含：

1. 四项 replay FIFO；
2. 独立 read-tag FIFO，记录每项在途 SRAM 读的 `last`；
3. `FIFO 内数据 + 在途读` 的统一容量预留；
4. 同拍 pop/response/issue 的显式计数更新；
5. blocked 时 payload/last 稳定的 SVA。

修复不改变 admission、数值、descriptor 内容或提交策略，只闭合 ready/valid 流控 `[rtl]`。

## 3. 一个被验证推翻的假设

FCSR 输出是 source-major descriptor，但退休顺序不保证 source ID 严格递增。早期测试把两者混为一谈，因而错误要求 `source_id=0,1,2,...`。

整改后的验证口径是：

1. 每个 T450 source 恰好出现一次；
2. 每个 descriptor 的坐标、K、五方向 gate 和 runtime valid mask 独立重算；
3. Vault replay 必须逐项复现第一遍实际退休的 active-source 顺序；
4. 不要求该顺序等于 raster 顺序。

这项修正很重要：硬件复用的是精确 descriptor 序列，不是测试台假造的排序 `[rtl]`。

## 4. RTL 结构

### 4.1 Vault 物理合同

| 项目 | 当前合同 |
|---|---:|
| 宏深度/宽度 | 512 x 112 bit |
| 容量 | 7 KiB |
| 逻辑 payload | source ID 9 + K 32 + 5xgate9 + valid5 = 91 bit |
| 访问 | 单端口 1RW，相位互斥 |
| 目录 | 每 head 的 base/length/resident，寄存器实现 |
| 提交 | whole-head 原子 commit/rollback |
| replay 解耦 | 四项 FIFO + 在途 tag 预留 |

当前 RTL 用可推断同步 memory 表达宏合同。未绑定 foundry SRAM compiler，因而不具备真实宏面积、时序或功耗证据 `[待验证]`。

### 4.2 暴露感知 admission

当前精确规则为：

```text
service = 15 + sum(popcount(K_source) * unique_nonzero_gate_count_source)
critical = service < 450
resident = critical && !capacity_overflow
```

这不是按稀疏度阈值猜测，而是在第一遍服务过程中统计将由 term builder 实际执行的整数工作量。noncritical head 的 speculative rows 在 head 边界回滚，后续 tile 走 exact recompute fallback `[rtl]`；策略的整帧收益仍是模型结果 `[prof]+[模型]`。

### 4.3 下游共享路径

`qfit_fcsr_relation_memo_projection_top` 只在 descriptor 输入处选择 live 或 replay，后续共享同一套：

```text
descriptor -> gate-equivalence term -> TCFM5 multicast -> Acc32
```

因此 replay 不是旁路的简化算子，也不会跳过 runtime invalid candidate；两条路径使用相同 term 和 accumulator 数值实现 `[rtl]`。

### 4.4 Exact fallback 控制

`qfit_relation_memo_tile_controller` 把 replay miss 转成同一 tile 的精确在线重算：

```text
prefer_replay
  -> replay_issue
  -> hit: replay descriptor -> term -> TCFM5 -> Acc32
  -> miss: fallback_taken -> recompute_request
          -> recompute_grant -> FCSR head_start
          -> live descriptor -> term -> TCFM5 -> Acc32
```

控制器没有为 fallback 建立第二套简化算子。`qfit_local5_relation_memo_tile_engine` 只在 descriptor 边界选择 live/replay，二者之后共用 term builder、TCFM5 和 Acc32。当前 `recompute_grant` 表示上游已经恢复本 tile 的 Q/K/gate 输入；源 SRAM、DMA 和跨 block 仲裁尚不在本顶层范围内 `[rtl]+[待验证]`。

## 5. 验证结果

统一入口：

```bash
sim_new_arch/run_local5_exposure_relation_memo_checks.sh
```

产物目录：`results/local5_relation_memo_rtl_20260806/`。

| 验证项 | 结果 | 证据 |
|---|---:|---|
| DSE 模型单测 | 7/7 PASS | `[模型]` |
| Vault 叶级随机反压 | 2250 live；962 speculative；488 rollback；474 commit/replay；零失配 | `[rtl]` |
| 真实 15x15x2 FCSR→Vault | 900 live；20 resident replay；nonresident miss；零失配 | `[rtl]` |
| 单顶层 descriptor miter | 450 live + 20 replay，逐项零失配 | `[rtl]` |
| 单顶层 Acc32 三方 miter | live/replay/原始输入独立参考共 1800 项零失配 | `[rtl]` |
| 四 tile 混合序列 | resident live/replay + nonresident live/miss/fallback；3 次重算 | `[rtl]` |
| 混合序列 Acc32 miter | 共 7200 项，全部与原始 K/gate/mask 金参考一致 | `[rtl]` |
| Verilator SVA | Vault 10 项合同 PASS | `[rtl]` |
| 控制器 Verilator SVA | 12 项 issue/miss/fallback/close/脉冲合同 PASS | `[rtl]` |
| Verilator lint | 退出码 0；保留既有 scheduler 宽度/未用模式告警 | `[rtl]` |
| Yosys | tile engine hierarchy/check/stat PASS；memory 保持抽象 | `[rtl]` 综合可读 |

Yosys 报告中的 `2347` 个粗粒度 cell 不包含真实 SRAM 映射，不能作为面积，也不能与 DC 标准单元数比较。

## 6. 当前性能证据

profile100 驱动的 DSE 当前给出 `[prof]+[模型]`：

| 策略 | 整帧周期代理 | relation build 减少 | 说明 |
|---|---:|---:|---|
| critical-only，7 KiB | 1.333x | 60.35% | 当前候选 |
| first-fit all，7 KiB | 1.316x | 64.66% | 缓存更多，但挤占关键 head |

这只能说明“暴露感知选择比 first-fit 更符合关键路径”，不能说明芯片达到 1.333x。真实结论还需要多 trace RTL 周期、SRAM 延迟、控制 fallback 开销和同约束 PPA。

## 7. DATE 创新性边界

本轮可辩护的候选不是“增加 7 KiB SRAM”或“加四项 FIFO”，而是：

> 用第一输出 tile 的精确 term service 暴露度决定 relation 表示是否跨输出 tile 驻留，并以 whole-head 事务式提交保证容量失败和 noncritical 路径都无部分可见状态。

它与普通稀疏缓存的差异需要由三项证据共同支撑：

1. admission 使用 exact downstream service，不使用 density proxy；
2. same-capacity `critical-only` 对 `first-fit all` 的反直觉优势；
3. live/replay/fallback 在同一 term/TCFM5/Acc32 数值合同下 bit-exact。

当前第 1、3 项已有 RTL 原型，第 2 项仍是 profile 驱动模型。若缺少真实多样本周期和 PPA，这一机制仍只能称“架构候选”，不能单独支撑 DATE 主贡献。

第一次严格 DATE 子代理审阅给出 `3/5，Reject/Major Revision`。两个直接缺陷是“fallback 只有 miss 检测”和“Acc32 参考可能由 DUT descriptor 自证”。本轮分别以自动 miss→recompute 控制和原始输入独立金参考整改；系统范围、实测性能与 PPA 缺口没有因此消失。

第二次复审仍给出 `3/5，Reject/Major Revision`，但确认：

1. replay miss 到 exact recompute 的 tile-level 控制和数值路径已经局部闭合 `[rtl]`；
2. Acc32 金参考不读取 DUT descriptor，自证循环已基本消除 `[rtl]`；
3. replay descriptor 与首遍 live descriptor 的逐项比较只证明存储/重放保真，不能称为独立算法金参考；
4. `recompute_grant` 仍把源 SRAM/DMA 恢复抽象在顶层之外，因此系统完整度仍为 `2/5`；
5. 架构创新性 `3/5`、局部验证 `4/5`、实验可信度 `2/5`。

复审要求的生产级下一门槛是：在冻结后的 12-block/多 head 调度中接入真实源恢复接口，覆盖 resident hit、nonresident miss、capacity rollback、连续窗口以及 SRAM/DMA 端口冲突。该门槛依赖算法侧新 Local5 rank-1 部署合同，合同释放前只能准备接口和维护回归，不能宣称完成。

## 8. 明确未完成

1. `[待验证]` score/Shiftmax5 到 memo 顶层的定点接口集成；
2. `[待验证]` `recompute_grant` 上游 Q/K/gate SRAM、DMA 时延和端口冲突模型；
3. `[待验证]` 新 Local5 rank-1 部署合同冻结后的 12-block、四 stage、多 head/多输出 tile 调度；
4. `[待验证]` 真实 full-resolution Local5 trace 的多样本 mean/p95/p99、vault occupancy 和 fallback 比例；
5. `[待验证]` SRAM macro 绑定、DC/STA、SAIF/PTPX、面积/功耗/EDP；
6. `[待验证]` random mode switching、reset/window abort、连续多窗口压力回归。

## 9. Motion 并行线

本轮没有停止 Motion。Motion 继续维护 SCS/NMF/DCTF/TESC 回归并等待真实 full-resolution T450 profile。Relation Vault 的具体 relation 表示只属于 Local5；Motion 可借鉴的是“首遍精确服务暴露度驱动驻留”的原则，候选驻留对象应重新选择为 SCS/NMF term 或 gated-K 目录，不能直接复用 Local5 的 7 KiB 数字 `[待验证]`。

## 10. 下一步候选

当前 Local5 的下一门槛不是再增加缓存结构，而是把已闭合的 tile engine 接回真实生产合同：

```text
冻结 theta/Q7/Q1.7/hardware-order/invalid-mask
  -> score/Shiftmax5
  -> relation transpose
  -> memo hit/miss/fallback
  -> source-major term/TCFM5/Acc32
  -> 真实多样本与随机反压
```

固定 12-block 调度必须等算法侧新 rank-1 checkpoint-bound 合同通过后再冻结，避免对旧候选过拟合。Motion 则并行保持回归，并在 T450 profile 到齐后独立选择 SCS/NMF term 或 gated-K 驻留对象及新机制；不把 Local5 的 relation memo 当成 Motion 的既定方案。
