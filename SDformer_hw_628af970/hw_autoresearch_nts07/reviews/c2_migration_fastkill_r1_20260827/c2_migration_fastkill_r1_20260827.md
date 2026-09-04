# C2 公开机制迁移 fast-kill 审阅 r1

日期：2026-08-27  
范围：只读审计；未修改 RTL，未运行 VCS/DC/PT/PTPX，未修改 `docs/359`。  
裁定：`TICKET_SUPPORT_ONLY__STRICT_METADATA_GATE_MISSED__NO_SECOND_CANDIDATE`。

## 结论

C2 已经不是“再找一种稀疏”的问题，而是把同一 K8 signed-source 服务做成更好的
物理 Pareto。M519 在相同八 bank 峰值带宽下，K8 相对 K1x8 的 directed 周期优势只有
`1.0122x--1.0392x`；因此新迁移不得声称新的多倍加速，必须降低控制传输、宽状态、面积
或切换能量，同时保持八 bank、端口、L4 响应、Acc24 和周期边界不变。

只保留一个协议候选：**central typed context + tag-elided leaf ticket**。它受 ELSA
bundled AER 的公共身份摊销和 FireFly-T bank dispatch 启发，但 H67 的差分是对八个独立、
可背压、乱序返回的 1R1W weight bank 使用 fail-closed typed ticket。

严格地址审计后，它**没有通过直接 GO 门**。`output_block3 + slice3 + source_channel12`
共 18 bit 是生成/寻址 128-bit weight 的必需字段，不能从 leaf request 删除。合法候选只能
删除 leaf 不使用的 `tag24`：request metadata `93 -> 69 bit`，response metadata
`75 -> 51 bit`，接口 metadata movement 仅降 `28.57%`。把 leaf entry 和中央 slot table
的写读也纳入，每个平均 occupancy transaction 的上界仅降 `27.53%`，低于 `30%` 门。
因此本轮状态是 **support-only**；除非将来 matched logic+FF area 实测降低 `>=15%`，否则
不写 RTL、不列贡献。为尽快收口，不再扩展第二候选。

其余公开机制不再开支线：FireFly-T multi-NZ/OOO bank issue 已由 M216/M218 覆盖；
OpenEye stream constructor/variable-length FIFO 已由 M214/M523 覆盖；LoAS temporal
packing/inner join 与当前 bundler/bank join 重复且二值 dual-sparse 假设不适配 H67；
ELSA row-stationary RMW 已被 M484 的 same-K8 resident baseline `1.0000x` 判死；SNE 只作为
event-proportional resident-state 强基线。

## 当前公平基线

| 证据 | 已验证事实 | 对本审阅的约束 |
|---|---|---|
| M519 r2 VCS | K8/K1x8：B1 `1.0392x`、B2 `1.0153x`、B4 `1.0267x`、B8 `1.0122x`；bit-exact；四次协议攻击；request/result/raw stall=`375/45/1165` | 同带宽上没有可包装的多倍 cycle 增益；新点主打物理效率 |
| M218 full trace recompute | 120 records；`73,380,812` K8 ordered groups；`2,477,402,364` active bank reads | 六个 16-lane slice 后为 `440,284,872` bundle requests，平均 `5.6268` active bank/request |
| M517 density fast-kill | `97.32%` tiles 低于 25% density；dense routing 无收益，最终 KILL | 不重开 dense/sparse stratifier |
| M490/M218 RTL 静态审计 | M490 保存八 slot 的完整 bank weight；M218 又保存一份 response skid；M490 还把中央已有 identity 复制给每个 bank | 剩余空间是协议 metadata 和 response state，而不是新 matcher |

当前 M490 leaf request 每个 active bank 携带
`epoch16 + slot3 + generation32 + tag24 + block3 + slice3 + channel12 = 93 bit`；
leaf response 回显 `epoch16 + slot3 + generation32 + tag24 = 75 bit`。M490 中央 slot 已经
持有 epoch/generation/tag，而 M218 scoreboard 也持有 epoch/generation/tag/block/slice/mask。
唯一可安全删除的 leaf 身份是 `tag24`；地址 18 bit 和 fail-closed ticket 51 bit 均必须保留。

## 唯一候选：tag-elided typed ticket leaf-bank protocol

### 原工作与合法迁移边界

| 项 | 内容 |
|---|---|
| 原工作 | [ELSA, ISCA 2026](https://arxiv.org/abs/2605.20802) 的 bundled AER 公共头摊销；[FireFly-T](https://arxiv.org/abs/2505.12771) 的 bank-aware dispatch/worker |
| 原机制 | 把同一 row/group 的公共身份集中编码，并把有效工作分派到独立 bank/worker |
| H67 换了什么 | 对象由 binary spike packet 换为 K8 signed INT8 weight bundle；协议必须支持八个独立 bank 的 ready/valid、乱序 response、slot reuse、soft flush 和 Acc24 context |
| 可写 claim | 只有过 matched area 门后，才可写 “A fail-closed typed-ticket protocol removes replicated tag identity from independently backpressured FC2 weight-bank leaves.” |
| 不能写 | BAER/dispatch 是我方发明；由 metadata bit 减少推导 cycle speedup；与 K1 的 `4.9--6.3x` 相乘 |

### 字段与存储严格账本

| 边界 | 当前 | 合法候选 | 说明 |
|---|---:|---:|---|
| leaf request：transaction identity | epoch16+slot3+generation32+tag24=`75` | epoch16+slot3+generation32=`51` | epoch/generation/slot 均保留，防 slot reuse/flush 后 stale alias |
| leaf request：weight address | block3+slice3+channel12=`18` | 同为 `18` | **全部必需，不删除**；TB 的 `weight_value` 明确由 bank/lane/channel/block/slice 决定 |
| leaf request 总 metadata | `93` | `69` | 只删除 tag24 |
| leaf response 总 metadata | epoch16+slot3+generation32+tag24=`75` | epoch16+slot3+generation32=`51` | 只删除 tag24；weight payload 128 bit 不变 |
| M490 central slot/table | valid1+epoch16+generation32+tag24+expected8+arrived8=`89` | `65` | 不新增 bit，删除 tag24；最终 tag 由 M218 既有 scoreboard 按合法 slot/ticket 恢复 |
| scalar leaf live entry（不计不变 due/pending） | epoch16+generation32+tag24+block3+slice3+channel12=`90` | `66` | slot 是 entry index；地址字段全部保留 |
| M490 response identity comparator | epoch16+generation32+tag24=`72` equality bits/active response | `48` | 每个 bank response lane 少 24-bit tag compare；expected/arrived 检查不变 |
| M218 scoreboard | 既有 typed context | 不变 | tag/block/slice/mask 的 owner；无新增存储 |

这里的“51 bit ticket”只指 `{epoch,generation,slot}`；request 是
`51-bit ticket + 18-bit address = 69 bit`，不再把 channel 单独混入 ticket 口径。
**不能只发 slot，也不能删除 block/slice/channel。**

每个有 `n` 个 active banks 的 bundle transaction：

- interface metadata movement：当前 `n*(93+75)=168n`，候选 `n*(69+51)=120n`，
  仅降 `28.5714%`；
- 再计 leaf metadata entry 的一次写和一次 response 读，以及 M490 central slot 的一次写/读：
  当前 `168n + 2*90n + 2*89 = 348n+178`，候选
  `120n + 2*66n + 2*65 = 252n+130`；
- 冻结 trace 的平均 `n=5.6268169` 时为 `2136.13 -> 1547.96 bit/transaction`，
  降 `27.5346%`。`n=1..8` 均只有 `27.38%--27.55%`。

这已经是对候选有利的上界：未把不变的 M218 scoreboard、mask updates、valid/due state
计入分母；计入只会令比例更低。也不采用“公共 header 广播只算一次”的乐观口径，因为
八个物理 leaf sink 的 fanout/capacitance 尚无布局和时钟门控证据。

### 一日 fast-kill 门

本轮静态门已经失败，状态降为 support-only。若未来仍申请实现，必须同时满足：

- 固定同一 M218/M519 request/response multiset、八 bank、端口、L4、O8/FIFO4、Acc24；
- 冻结 120-record H67 FC2 trace，并重放 M519 r2 的 request/result/raw stalls；
- slot reuse、response reorder、soft flush、stale/duplicate/wrong-ticket 攻击均 fail-closed；
- bank request、bank response、Acc24 update 和 done tuple 均 0 mismatch；
- total metadata bit-movement/transaction 降低 `>=30%`，**或**同边界 matched
  logic+FF area 降低 `>=15%`；
- K8 周期改变不超过 `1%`，bank read/traffic 不增加。

bit-movement 已知只有 `27.53%`，所以只有 matched logic+FF area `>=15%` 才能升格；在没有
该证据前不得称 GO、不得列论文贡献。M490 已有 final-beat cut-through，ticket 预期不改善
周期，不能用 latency 弥补面积门。

## 重复性/NO-GO 矩阵

| 公开机制 | 本地已有覆盖 | 裁定 |
|---|---|---|
| FireFly-T multi-NZ grouped-carry + OOO bank worker | M216 descriptor、M218 group FIFO/scoreboard/八 bank coissue；M519 等带宽比较 | 不单列贡献，不再开发 |
| LoAS temporal spike packing + prefix-sum inner join | 当前 group FIFO 已把一组 source 发六个 slice，M216 已做 bitmap/bank join；LoAS binary/dual-sparse 假设不适配 analog signed ATLIF | NO-GO |
| ELSA row-coherent Gustavson/state RMW | M484 same-K8 resident baseline `1.0000x` | NO-GO |
| OpenEye stream constructor/varlen FIFO | M214/M523 bundler/runner | NO-GO |
| SNE event-proportional execution/resident state | 定义公平强基线 | 只引用，不作我方 novelty |
| FEATHER generic reorder/reduction | 会引入新 reducer/多写口，且本轮要求不扩第二候选 | 不开发 |

## DATE 写法与停止条件

该候选若未来过 matched-area 门，也只能写进现有 **C2 typed signed-source service**，
不能成为 C4：

- C2 主句仍是 K8 相对低面积 K1 的 bank utilization，以及相对等带宽 K1x8 的共享
  descriptor/scoreboard/queue/Acc24 物理 Pareto；
- 过门后只能写 “typed ticket removes replicated tag identity”；
- fast-kill bit proxy、有损数字、公开论文倍率均不得进性能主表。

当前 transport 门已不过，因此默认停止。只有已有收口排期允许、且能直接做 matched
logic+FF area 审计时才可重开；面积降低 `<15%` 立即永久 NO-GO。它失败不影响 C2 按
M519 三点 Pareto 收口；不得因此复活 dense/sparse routing、FC1 factor wrapper、row bundle
或第四条 Conv matcher。

## 身份与局限

本审阅是静态、只读 fast-kill，数值仅来自已封存 JSON/log/RTL 和公开一手论文/官方代码。
没有生成新的 cycle、area、power、energy 或系统 speedup。OpenEye 本地镜像固定为
`fe2f5ed169d4e4bf1d2e960601588e7c5d71c4e3`，SNE 镜像固定为
`92449df7a49f485f331dc785522b82acd33759ae`。`docs/359` 复核 SHA-256 为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
