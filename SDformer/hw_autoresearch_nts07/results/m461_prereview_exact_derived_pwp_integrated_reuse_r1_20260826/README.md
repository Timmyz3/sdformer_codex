# M461 H67 ep35 exact derived-PWP / integrated-reuse prereview

## 结论先行

这次只做了预审和资源合同设计；没有读取 M40 payload，没有运行 M453b，也没有产生任何性能、能量、PPA 或系统加速结论。

建议 **有条件继续 M461 的合同与逐 phase event model**，但现在不应启动 RTL。硬结论是：

> 第一个 M461 模型固定为 compact original-order used-center bank，因为它保持 destination 顺序；容量要等 sealed `Nmax`。A) 128 PWP 全缓存因容量故事失败，直接 NO-GO；B) q32 parent cache + 单 child scratch 只是固定容量备选的 event/interface screen，在 destination-tagged 随机 psum 端口闭合前不进 RTL；C) 原行序小 child cache 必须等 ordered selected-ID ledger，当前 unknown。

这个首选模型就是“双 assignment bank + 只物化本 phase `pwp_rows>0` 中心的紧凑 PWP bank + 原 descriptor 顺序重放”。它不改 psum 顺序，但容量依赖 sealed `Nmax`，因此只能等 M453b 后执行。

M461 只能在最终 M453b 的 separate 路径通过预设门槛、且独立打铁 P0=0 后启动；本 prereview 不能授权读 M40、跑 M453b 或写 RTL。

## 为什么旧 M267/M275 只能迁移机制

M267/M275 已证明 PWP 不是独立模型状态，而是 16 个 signed-INT8 weight vector 的确定子集和：

`PWP(child) = PWP(parent) + ΣW(0→1) − ΣW(1→0)`。

可以迁移的是这条精确恒等式、current/next bank 的 valid fence，以及 role-switch 的 fail-closed 纪律。不能迁移的是 PAFT checkpoint 的数值范围、q16 tree flip 数、transition slack、容量或任何速度数字。M453 是 H67 ep35、q32+96 children 的新身份，必须重新做 direct-sum miter 和逐 phase 调度。

M453a train catalog 本身可以给一个不读 M40 的结构界：165,888 条 parent-child edge 的 Hamming 直方图为 `{1:138391, 2:16746, 3:9451, 4:460, 5:679, 6:83, 7:74, 8:1, 9:3}`。每 partition 96 个 child edge 的 flip 总数 min/mean/median/p95/max 是 `98/119.515/119/136/160`；32 parents 从 zero 出发的 Prim MST 是 `38/58.849/58/70/84`。若把全部 128 个中心都生成，仅 8 output-block weight-update 拍平均就是 `1,426.912 cycles/phase`，尚未计 parent read/PWP write/clear/DMA/matcher/port stall。这只是 train-catalog structural bound，不是 M40 周期或性能数字。

独立算法是：只读双封 M453a train catalog，对每个 partition 的 32×3 parent/child mask 算 `popcount(parent XOR child)`；再对 `{zero + 32 parents}` 按 `(distance, child_index, parent_index)` 固定 tie 独立跑 zero-rooted Prim。统计范围是 4×432 partitions，没有打开 M40 或 M453b result。

PWP byte 口径固定为：每 output block 逻辑 144 B signed12，物理信号 padding 按160 B；M453b 的 `pwp_stride=640 B/tile` 就是一个中心的4个 block。两个 tile/8 block 合计 1,280 B/center。因此 q32 parent cache 是 `32×640=20,480 B/tile`，两 tile 是 40,960 B；不得把 per-tile 640 B 和 all-eight-block 1,280 B 混用。

## 对照候选：紧凑 used-center bank，保持原顺序

每个 phase 有 128 个候选中心，但只为 `use_pwp` 实际命中的中心分配 compact slot：

1. phase `i` 从 current assignment/PWP/weight bank 重放时，层级 matcher 把 phase `i+1` 的非零行写入 inactive 48-bit assignment bank，并置 128-bit used-PWP bitmap。
2. 同时装入 next phase 的 12,288 B weights 和 288 B config；不搬 PWP payload。
3. assignment seal 后，按固定 parent group 访问 used center。每组从零构造 parent，再以固定的最小 Hamming transition 顺序访问已用 children；移除 bit 先于增加 bit。
4. 8 个 output block 都用同一个 96-lane signed13 generator。每个 used center/block 采用保守的独立 160 B PWP 写周期，不继承 M275 被评审质疑的 final-update/write 同拍 bypass。
5. center→compact-slot table、assignment、weight、config、PWP 全 valid 后才能切 bank；generator 必须 idle。
6. current replay 仍按原 48-bit descriptor 顺序。只有 `use_pwp` 行查 remap；fallback 完全保持 bit-sparse。

若 `F` 是所有 active parent group 的构造/transition flip 总数、`U` 是 used-PWP center 数、`G` 是 active parent group 数，保守物化周期为：

`C_materialize = 8 × (F + U + G)`。

其中 `U` 计独立 PWP write，`G` 计显式 block clear。准备递推必须使用：

`C_prepare = max(C_match+seal, C_weight/config DMA) + C_materialize`。

冷启动全计；in-sample 只能与上一个 phase 的真实 service window 重叠，超出的每一拍都要暴露。

## 资源与端口下界

主候选的逻辑下界（不是 macro area）为：

| 项目 | 容量/端口 |
|---|---:|
| 两个 full-phase weight bank | 24,576 B；current 96 B read，next 32 B fill 或 96 B generator read |
| 两个 3000×48-bit assignment bank | 36,000 B logical；64-bit macro padding sensitivity 为 48,000 B |
| 两份 q128 config | 576 B |
| 两份 used bitmap | 32 B |
| 两份 128×7-bit compact remap | 224 B |
| 96-lane signed13 generator state | 156 B |
| 两个 compact PWP bank | `2,560 × Nmax` B physical；`Nmax` 必须取 M453b 全 phase 最大 used-PWP center 数 |

固定逻辑小计是 61,564 B，再加 `2,560×Nmax` B PWP。该数字未含 output accumulator、队列/控制、macro rounding/ECC/clock/interconnect，不能写成芯片 SRAM 面积。

关键端口必须同时存在于不同 bank role：

- current PWP：160 B/read；
- current correction weight：96 B/read；
- next compact PWP：160 B/write；
- next materializer weight：96 B/read，且与 next 32 B fill 严格分时；
- inactive assignment 1 write/cycle 与 current assignment 1 read/cycle。

这正是避免 q128 静态 PWP DMA/88,352 B direct-address slot 膨胀的方式：片外只传 weights/config，PWP 由中心掩码精确派生；片上 PWP 只按 sealed `Nmax` 紧凑配置，而不是预留全部 128 个 direct-address slot。

## A/B/C 可执行方案硬比较

### A. 128 PWP 全缓存：NO-GO

一个 phase 的两个 output tile 需要 163,840 B PWP，两个 expanded working slot 是 176,704 B。若真要 current/next phase ping-pong 并行准备，仅 PWP 就是 327,680 B。这个点可以复制 M453b 的 stored-PWP 周期假设，但它把 q128 catalog 直接变成大型 direct-address SRAM，无法支撑 payload-elision 与资源归一化故事，不应做。

### B. q32 parent cache + 单 child scratch：只 GO event/interface screen

每 tile 是 20,480 B parent PWP + 640 B child scratch + 6,144 B weights + 288 B config address space = 27,552 B，距 32 KiB 还有 5,216 B；两 tile 当前 working set 是 55,104 B。它比 A 的容量故事强得多。

它也必须把隐藏成本全部收费：

- 两个 phase 的 3000×64-bit assignment/residual/destination buffer：48,000 B；descriptor 至少包含 row/destination12、original16、global center7、distance5、use1、predecessor12。
- 两个 q32 parent PWP ping-pong：81,920 B；两个 full-phase weight bank：24,576 B。
- 两 tile child scratch：1,280 B；两份 config：576 B；129 条 list 的双 tail table 约 388 B；generator state 156 B。
- 不计 psum macro/control 的已知逻辑下界已是 156,896 B，不是“只加 640 B scratch”。

一个 child 在一个 tile 的保守物化成本是 `4×(1 parent read + H flips + 1 child write)`；两 tile 是 `8×(H+2)`。parent read 160 B/block，每个 flip 读 96 B weights，child write 160 B/block。child derivation 与 correction replay 同抢 current 96 B weight bank，单端口下必须串行，不能偷用重叠。

M433 只证明 tagged `update_delta` adapter，不证明 downstream old_psum SRAM 能按 center-major 的随机 row 顺序做 II=1 RMW。parent ID 0..31 可用 M433 5-bit 字段；child 必须用 local scratch ID，global child ID 留在 assignment controller 中生成 residual。这需要新 protocol shim，不是 M433 原样复用。若 indexed psum 无法在不新增宽端口/大 reorder buffer 的情况下匹配 issue rate，B 立即硬 NO-GO。

### C. q32 parent + 原行序小 child cache：unknown

这个方案可以保持 destination/psum 顺序，但 cache miss 完全依赖 ordered selected-center stream。M453b 目前的 used set、按 ID 的 `count_runs`、per-center count 都不能代替 row-order miss ledger。在 ordered selected-ID ledger 封存前，命中率、物化周期和性能必须保持 unknown。

## matcher → assignment → group replay 是否可行

控制上可行：把 descriptor 扩成 64 bit，增加 12-bit predecessor；为 128 个 PWP center 加一个 fallback list 各存 tail。每行只写一次，之后能按 center 倒序遍历。两份 3000×64-bit bank 是 48,000 B，tail table 约 388 B。

但当前不应选它：

- center-major 顺序会改变 destination update 次序，需要 accumulator 位宽、地址、RAW hazard 和端口的可执行证明；
- materialize next center 与 current correction 同抢 96 B weight bank，除非新增读端口；
- 不新增端口时，center 间 materialization 会暴露；
- 若借重排把可融合行推到 prep 完成之后，M453b 的原顺序周期就失效。

因此推荐的是“used-center **分组物化** + descriptor **原顺序重放**”。只有当 `Nmax` 使 compact bank 的宏 Pareto 失败，才重新打开 true group replay。

## generator 空闲窗口复用 M451 K1 fold

这是 M461 必须测、但现在不得给收益的独立 DSE 轴。

下一 bank 未完成 assignment/weight/config/PWP 准备前，generator 只能处于 `GEN_NEXT`；全部 valid 且最后 PWP write 完成后，才可切到 `FOLD_CURRENT`。每个 fused issue 需要同拍：

- current PWP 160 B；
- current correction weight 96 B；
- 96-lane signed add/sub；
- 一致的 tag/tile/output-block；
- downstream 继续执行 `new_psum=old_psum+update_delta`。

它不会自动得到 M451 的旧机会数字。必须逐 phase 联立 current replay 与 next preparation：prep 前全部 separate；prep 后只有真实调度到的 positive-residual PWP 行，才允许把每个 output block 的首个 correction 和 PWP 合并。融合会缩短 current window，反过来可能暴露 next prep，所以要重新推进 event queue，不能一次性从总周期减去 `positive_residual_rows×8`。

generator reuse 仍要付 current-PWP unpack、current-weight cross-role mux、signed13 mode mux、metadata/仲裁和 256 B 瞬时宽供数布线。M455 的 standalone 面积劣势说明必须做 integrated DC/PT，而不是宣称“复用等于免费”。

## M453b 后必须取得的数字

现有 center ledger 足以提供：每 phase used-PWP center、parent/child、Hamming、`Nmax`、PWP working set、actual-used PWP DRAM bytes、separate issue 和 matcher/service 分量。

但 integrated reuse 还缺有序信息：每个 positive-residual PWP 行/output-block 的 PWP 与首 correction 到达时刻，以及 accumulator 地址顺序。M453b 当前 aggregate center/phase CSV 本身不够。M461 必须二选一：

1. 在 M453b 冻结前增加有序 issue ledger；或
2. 在新的 exact-SHA M461 合同下，从相同 sealed M40 输入确定性重建，禁止任何 catalog/scheduler 参数回调。

不得用 aggregate positive-residual population 估计 fold 命中。

## 风险与 GO/NO-GO

- **P0**：`Nmax` 未知，不能按平均数配 compact bank；必须按 sealed 最大值或显式 spill。
- **P0**：integrated reuse 缺 ordered issue timing；没有就只能给零收益。
- **P1**：48-bit/160 B/96 B 逻辑端口的 macro rounding 可能主导面积与频率。
- **P1**：wide mux 后 generator 未必 II=1、未必保持 3 ns。
- **P1**：true group replay 的 accumulator SRAM 可能比省掉的 PWP bank 更贵。
- **P1**：fold 越多，current window 越短，可能把 next prep stall 暴露出来。

最终建议：

- M461 合同/event simulator：`CONDITIONAL GO`（等 M453b separate gate + 独立评审）；
- compact used-center + original-order replay：唯一首选，等 sealed `Nmax`；
- B q32 parent + child scratch group replay：仅固定容量备选的 interface/event screen；
- integrated generator fold：只作逐 phase、matched-resource DSE；
- true group replay：当前 `NO-GO as primary`；
- RTL：现在不启动；
- cycle speedup / system speedup / energy / PPA / DATE headline：全部 `false/unknown`。

受保护 `docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
