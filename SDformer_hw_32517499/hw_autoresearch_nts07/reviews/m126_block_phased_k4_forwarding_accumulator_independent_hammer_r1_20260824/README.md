# M126 block-phased K4 forwarding accumulator independent hammer

## Verdict

**92/100，directed functional 与 reset-boundary repair 通过；P0=0、P1=0、P2=4。**

M126 已经把 M125 canonical K4/tail folding 与 M123 same-address forwarding accumulator 真正接通，而且独立证据支持 source → fold update → accumulator accept → exact lane write → full commit 的守恒。更重要的是，M125 的 reset-high phantom handshake 和 M123 的 reset-edge physical write 两个既有反例，在 M126 外部边界上都被非空重放并闭合。

Exact-SHA production commercial VCS 已独立重编重跑，完全复现 24,802 source contributions → 7,326 fold updates → 7,326 lane writes → 3,072 commit vectors / 294,912 commit-lane checks。隔离 adversarial TB 另外用独立 lowest-4、weight、pending-write 和 full-commit oracle 验证 65 sources → 18 folds → 18 accumulator accepts → 18 exact writes，包含 7 个 consecutive same-address forwarding pairs、K4 与 K1/K2/K3 tails、block transition、commit stalls，以及 `+512/-512`。

Reset 攻击不是 vacuous：独立 TB 先确认 M123 internal pending write 和未门控外部 write 在 reset 前确实为 1，再拉高 reset；内部 pending write 保持到 reset edge，但 M126 外部 `lane_mem_wr_en` 立即为 0，物理 memory model 没有增加写计数。同时在 reset 高电平并发拉高 start/fill/row/end valid，9 个 reset cycles 中外部 handshake/enable violation 为 0。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| K4/tail source-fold conservation | 24/24 | Independent lowest-4 oracle；65/65 source；18 folds；K1/K2/K3 tails；±512。 |
| Forwarding、write 与 commit exactness | 25/25 | 18/18 accumulator accepts/writes；1728 write-lane checks；7 forwarding pairs；3072 vectors / 294912 commit lanes。 |
| Reset counterexample closure | 20/20 | Pending internal write 已 sensitized；reset-high external violation=0；reset-edge physical write=0；SVA cover=1。 |
| Identity/overflow protocol | 11/16 | Fail-closed、无 silent corruption，但 row 384 和 overflowing update 都是 accepted-before-fault，且无 retry。 |
| Physical evidence与论文口径 | 12/15 | 两个 3.x 数字边界正确；仍无 foundry macros、macro-inclusive PPA/energy/Fmax。 |
| **Total** | **92/100** | **可作为强 functional integration milestone；不能升级成 physical/system performance admission。** |

## Exact-SHA production rerun

冻结的 M125、M123 core/adapter、M126 wrapper、SVA、TB、filelist 和 contract SHA 均匹配。VCS compile/sim rc=0，PASS line 与 sealed run 完全一致：

- 正向 3072 rows / 3072 row_done；
- 24,802 source contributions；
- 7,326 fold updates / 7,326 physical lane writes；
- 5,115 full-K4 / 2,211 tails；
- 4,262 consecutive same-row update pairs；
- 3,072 commits / 294,912 exact lane checks；
- 401 commit stall cycles；
- 1 个 pending-update reset attack，physical write 被抑制。

Production SVA cover 精确复现：

```text
cp_four_consecutive_same_row_folds = 160
cp_full_k4_to_write               = 5115
cp_tail_to_write                  = 2211
cp_commit_stall_release           = 384
cp_reset_with_prior_update        = 1
```

## Independent adversarial coverage

| Scenario | Result |
|---|---|
| Canonical lowest-4 / source conservation | PASS，65 sources 无丢失、无重复 |
| M125 fold accept = M123 accumulator accept | PASS，18/18 |
| Exact write address/data | PASS，18 writes / 1728 lane checks |
| Four consecutive updates to one address | PASS，macro read 被抑制，forwarding sum 精确 |
| K4、K1/K2/K3 tails | PASS，14 full-K4 + 4 tails |
| Same row 的跨 transaction 再累加 | PASS，两组 replay |
| Block 0 → block 1 cache transition | PASS，旧 cache valid 被清除，accumulator 两个 block 均保留 |
| Mixed polarity、`+512/-512` | PASS，fold signed11 → accumulator signed19 精确 |
| Commit backpressure | PASS，1088 stall cycles，commit data/identity 稳定 |
| Full 8×384×96 commit | PASS，3072 vectors / 294912 lanes |
| Reset-high start/fill/row/end 并发攻击 | PASS，9 cycles、0 external handshake/enable violation |
| Pending internal write 后 reset | PASS，internal path 已 sensitized，external physical writes=0 |
| Row 384 | Fail-closed；1 row accept、0 fold、0 write、sticky accumulator fault |
| Signed19 `+262144` overflow | Fail-closed；512 fold accepts、511 writes、last valid=261632、无 wrapped write |

## P0

**0 个。** 合法 directed scope 内没有 source loss、update divergence、write loss/duplication、write address/data error、forwarding error、commit error或 reset leakage。

## P1

**0 个。** M125/M123 既有两个 reset P1 在 M126 integration boundary 上均已被独立、非空地关闭。

## P2

### P2-1 — Out-of-range row 在 wrapper ingress 被接受

独立 VCS 对 `row_offset=384` 得到：

```text
identity_row_accepts=1
identity_fold_accepts=0
identity_writes=0
accumulator_protocol_error=1
```

M125 接口允许 9-bit row，而 M123 才检查 `<384`。因此请求已对上游宣告 accept，之后才 sticky fault。没有写坏数据，但更干净的集成方式是在 M126 `row_ready/row_accept` 前检查 `row_offset < 384`。

### P2-2 — Overflow update 是 accepted-before-fault，且无 retry

使用只有 lane0 为 `+512` 的 K4 row，重复 512 次：

```text
overflow_fold_accepts=512
overflow_writes=511
last_valid_lane0=261632
attempted_next_value=262144
overflow_fail_closed=true
overflow_retry=false
```

RTL 正确禁止 signed19 wrap，没有 silent corruption；但 overflowing update 已经被 fold/accumulator 接受，随后不写并进入 sticky fault。因此“每个 accepted update exact-once write”只适用于已证明不 overflow 的合法 workload。要支持一般输入，需要 pre-accept headroom check、credit 或明确的全窗 reset/retry protocol。

### P2-3 — Weight/accumulator memory 仍是 behavioral model

M126 暴露 96 个 logical `3072×19` lane memory，并复用 M125 的 logical 1536-byte single-block cache；但没有绑定 foundry SRAM/register-file macro。四个 768-bit logical weight vector read、96-lane fold tree、lane macro ports、Fmax、area 和 energy 均未物理闭合。

### P2-4 — `3.385476x` 与 `3.1725369x` 不是同一种性能数字

`3.385476385476x` 只是 production synthetic directed traffic 的压缩比：

```text
24,802 source contributions / 7,326 accepted fold updates
= 3.3854763854763856 source/update
```

它不包含 cycle time、stall、memory access、commit、频率、面积或能耗，不能叫 latency/throughput/speedup。

`3.172536900846x` 则来自 M122 heldout cycle simulator：

```text
fixed8 service-island baseline / K4 candidate
= 1,114,863,448 / 351,410,711
= 3.1725369008459166x

K1 candidate / K4 candidate
= 439,708,199 / 351,410,711
= 1.2512657845537327x
```

它不是 M126 VCS 测量，也不是同面积、同频率、macro-inclusive 或 full-system speedup。论文最稳妥的硬件性能数字仍是 `3.1725x projected vs fixed8 service island`，并同时披露 `1.2513x incremental vs K1`；`3.3855x` 应只放 traffic reduction/packing efficiency。

## Safe claim

> Exact-SHA commercial VCS and an independent end-to-end scoreboard validate the directed reset-free M126 path from canonical K4/tail source folding through same-address signed19 forwarding, exact lane writes, and all 3072×96 commit values. A sensitized pending internal write and simultaneous reset-high requests produce zero external handshake or physical-write leakage, closing the prior M125/M123 reset counterexamples at the M126 boundary. The 3.385476× value is directed traffic compression, while 3.172537× is a separate M122 fixed8 service-island cycle projection (1.251266× incremental over K1); neither is a physical or system speedup.

## Artifacts

- `m126_block_phased_k4_forwarding_accumulator_independent_audit.json`：machine-readable counters、score、P0/P1/P2 与 claim boundary。
- `audit_m126_independent.py`：fail-closed SHA/log/cover/ratio 审计。
- `sealed_vcs_rerun/`：production exact-SHA commercial VCS rebuild/rerun。
- `independent_vcs/`：隔离 source→fold→write→commit 与 reset replay。
- `boundary_vcs/`：row identity 和 signed19 overflow characterization；刻意不启用只适用于正向 non-overflow 的 conservation SVA。
- `manifest.sha256`：review source 与文本证据清单；VCS binary/database 可由 runner 重建，未纳入清单。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
