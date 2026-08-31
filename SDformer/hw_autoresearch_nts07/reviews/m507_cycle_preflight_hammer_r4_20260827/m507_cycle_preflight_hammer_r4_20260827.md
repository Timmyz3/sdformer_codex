# M507 r4 APEC-G2 same-resource cycle fast-kill 独立 preflight 评审

日期：2026-08-27  
范围：receipt-blind / production-blind 静态审查。没有 import 或执行 production analyzer，没有打开/解压 trace payload，没有启动 VCS/DC/GPU，也没有修改 production、contract 或 `docs/359`。

## 裁决

**`STATIC_GO__ONE_EXACT_SHA_ONE_SHOT_ONLY__SUPPORTING_PRIOR_ART_NO_RTL`，90/100。**

r4 已经实质关闭 r3 的三个 P0：baseline/common/residual 现在统一为一个 96-lane block-major 累加器数据流；240 KiB 账本按 block padding 重算为 18 KiB overlap scratch 与两个 18 KiB destination slot；final commit 改为 `max(slot,sink)+tail`；输出在 sibling staging 中完整写入、将 `RUN_COMPLETE` 纳入 seal 后才原子 rename。独立审计没有发现会让 candidate 获得免费 compute/SRAM port/state 或改变两臂公平倍率的新 P0。

这只是允许**一次** exact-SHA production fast-kill，不是性能通过。独立小边界显示低事件组的 scratch 固定税很高，production 很可能触发 KILL 门；必须让冻结 validation/train trace 决定，禁止改轴、改 G、重试或在结果出来后调参数。即便全门通过，APEC 仍是 ExSpike 直接先验，只能作 supporting audit，不授权 RTL novelty。

## 1. 身份与 seal 链

- analyzer SHA：`13db92a7...e4968`；contract SHA：`241ae6c8...f3a5a`；contract 内自锁一致。
- contract 的全部 inputs 外层 SHA 匹配；`docs/359` 仍为冻结的 `dedde7ce...`。
- r1/r2/r3 seal 的“seal 文件外层 SHA”和“seal 内容指向的内层 manifest SHA”均与 contract 分栏一致。
- r2、r3 当前 member-by-member 可完整验证。r1 的旧 contract/analyzer 两个 member 已不存在，因此 r1 只能标成 historical provenance；这不改变 r4 当前身份。

## 2. 240 KiB 与 lane/bank 独立复算

| 组件 | 独立复算 bytes |
|---|---:|
| pair bitmap | 192 |
| block-padded overlap scratch | 18,432 |
| destination slot 0 | 18,432 |
| destination slot 1 | 18,432 |
| payload / weight window | 190,272 |
| 合计 | 245,760（240 KiB） |

full-9-tap 有 `768×9/96=72` 个 block。每个 block 是 `96×19=1,824 bit=228 B logical`：scratch 必须占两个 128-B word，即 256 B physical；destination 按 8 bank 条带，每 bank 是 `ceil(12×19/8)=29 B logical`，小于锁定的 32 B/cycle，整个 block 同样是 256 B physical。因此 full scratch 与单 destination 都是 `72×256=18,432 B`，r3 的 packed 16,416 B 幻觉已消失。

两臂得到完全相同的 capacity/port ledger，baseline 只是 clock-gate candidate-only 路径，不回收 scratch 容量。weight 是相同的 8×16 B/cycle，destination 是每 slot 8×32 B/cycle 1R1W，compute 只有一份 96×19-bit lane accumulator。

## 3. 唯一 lane-block dataflow 与周期守恒

r4 的服务组织已经一致：

1. 对每个 output-tap block，首个普通 product 在正常 MAC 拍初始化 96-lane register，后续 active source 在同一 block 上累加；
2. baseline 的两个 source stream、candidate 的 common/left-residual/right-residual 都时间复用同一 accumulator；
3. 每个完成 destination block 仅向普通 destination slot 写一次，没有 zero-fill、per-event SRAM RMW 或第二套免费 psum state；
4. common block 完成后用两个 128-B scratch write cycles drain；两个 destination 各自串行读取，每 block 为两拍 transfer 加一拍 synchronous response tail；
5. destination materialization、scratch 与 common/residual service 不重叠，final slot read 只与同一个 sink write pipeline。

`compute=E×blocks` 与 `weight=E×k×768/128` 独立复算成立。冻结 k∈{4,6,9} 下 compute 对每个非空 phase 都不慢于 weight+1 startup，因此源码把各 stream 汇总后取 `max(compute, weight+startup)` 不会暗中跨 scratch phase 隐藏 weight stall。

full-9-tap final vector 是 16,416 B logical、18,432 B physical destination traffic。slot 为 72 cycles，sink 为 `ceil(16,416/128)=129` cycles，r4 正确收费 `max(72,129)+1=130`，不再漏同步首响应拍。

## 4. 独立小边界算例

这些算例由 review 自己的公式生成，不 import production，也不预测冻结 trace 的最终结果。

| synthetic group | baseline cycles | candidate cycles | baseline/candidate |
|---|---:|---:|---:|
| empty interior | 2 | 3 | 0.666667× |
| one+one, full overlap, interior | 550 | 1,055 | 0.521327× |
| one+one, no overlap, interior | 550 | 551 | 0.998185× |
| 2+3 events, one overlap, interior | 766 | 1,271 | 0.602675× |
| one+one, full overlap, left border | 459 | 988 | 0.464575× |
| one+one, full overlap, top-left | 307 | 660 | 0.465152× |

这些负的小组边界是合理的：candidate 对每个 overlap group 支付 block-padded scratch 的固定税，只有足够多 common events 才可能摊薄。它也说明 production 的 `validation≥1.20×`、`train worst≥1.15×` 与 envelope `≥1.02×` 三个门不能降，不能因 M501 event-work 正结果预写 cycle 正结果。

## 5. validation/train 与发布安全

通过项：

- validation 和 train 都做 per-record、overall、per-sequence M501 event ledger 对账；
- destination materialize/final read/sink 的 logical/physical bytes 与 transaction 在每条 validation+train record 上对称；
- scratch read logical/physical bytes 必须精确为 write 的两倍，并支付每 block 两条读流的同步 tail；
- validation+train 都进入 bank-conflict、destination、scratch 和 queue 最终 gate；
- result 先写唯一 sibling staging，`RUN_COMPLETE` 与其余 payload 一起 seal，self-SHA 与 seal 检查后才 `os.replace()` 到不存在的 final 路径。中断只会留下可区分的 staging，不会毒化 final no-overwrite 目录。

## 6. P1 限制（不阻止本次单次 analytic fast-kill）

1. **mapping/queue 有部分构造性门。** `bank_mapping_mismatch_count` 与 `lane_issue_order_mismatch_count` 初始化为零且没有地址级更新；bank 安全由 contiguous output-tap、`out_channel mod 8`、整除与 byte-width 代数证明。对锁定 cycle model 足够，但不是 functional equivalence 或 RTL 证据。未来若碰 RTL，必须补 address-level miter/VCS。
2. **final width gearbox 隐式。** `8×32 B` destination physical read 与 `8×16 B` sink 的 `max()` 流水需要小型 packing/elastic state；它对两臂完全相同，不给 candidate 免费优势，但没在 SRAM ledger 中具名。M507 因而不能当 PPA/area 证据，未来 RTL 必须把 gearbox 定价并验证。
3. **r1 仅历史 seal。** 外层 seal 与内层 digest 身份正确，但旧 r1 两个 production member 已移走；不得称其当前可完整重放。
4. **发布后仍需 receipt rehash。** production staging 会生成 member hashes并锁 manifest，但内部只复核 manifest digest，没有二次逐 member readback/fsync。exact runner/result hammer 必须对发布目录执行 `sha256sum -c` 后才准入。

## 7. 唯一允许的下一步

创建一个 exact-SHA/no-overwrite runner，钉住 r4 analyzer、contract、`docs/359` 和本 review seal。只有共享主机上 VCS/DC/PT/FM/CPU-DSE 资源门全部为空时，才允许执行一次 production main；发布后立即逐 member 验 seal 并做独立 receipt hammer。

- 任一门失败：永久 `KILL_M501_M507_HARDWARE_LINE`，只保留负 DSE/prior-art audit。
- 全部门通过：只保留为明确引用 ExSpike 的 supporting mechanism；仍不得开发 standalone APEC RTL，不得称 signed-analog novelty、system speedup、PPA 或 DATE headline。

可复现审计器：`audit_m507_r4_preflight_independent.py`。它只读 source/contract/frozen metadata/seals，未触碰 payload 或 production execution。
