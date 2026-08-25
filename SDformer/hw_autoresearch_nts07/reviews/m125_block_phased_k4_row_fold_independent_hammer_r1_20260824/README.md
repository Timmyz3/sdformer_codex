# M125 block-phased K4 row-fold independent hammer

## Verdict

**87/100，reset-free standalone 功能条件通过；P0=0、P1=1、P2=3。**

M125 的主要功能创新经受住了独立打铁：它不是只在 production TB 的固定 mask 上输出一个漂亮计数。exact-SHA production VCS 已独立重编重跑；隔离 adversarial TB 用自建 lowest-4 oracle 和逐 lane 整数 miter 复核了 canonical select-and-clear、source 不丢不重、stall 稳定、block/cache identity、连续同 row、混合极性以及 signed11 的 `+512/-512` 两端。

在 reset-free、单 resident block、behavioral weight-cache 的合同内，没有找到 accepted update 的 source loss、duplication、错误选择、错误 delta 或 stall 漂移。独立路径覆盖 9 rows、12 accepted updates、40 selected sources、1152 lane checks、28 stall cycles、8 个 full-K4 update 和 4 个 K1/K2/K3 tail update；三类 cache/block/fill 攻击均 fail-closed。

但 standalone admission 不能无条件通过。独立 VCS 在 `rst_core=1` 时分别观察到 fill accept、row accept、update valid 和 update accept；同步 reset 分支随后清空状态，因此外部看见的握手可以被内部丢弃。这是可复现的 reset-edge counterexample，必须把当前结论限定为 **reset-free functional admission**。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| Canonical K4、source conservation 与算术 | 32/32 | 40/40 source 精确一次消费；12/12 update；1152 lane miter；`+512/-512`；K1/K2/K3 tails。 |
| Block/cache identity 与 fail-closed | 17/18 | Block transition 将旧 `ffff` valid 收缩为新 block 的 `0001`；missing source、wrong block、wrong beat 三类攻击 sticky fault。 |
| 独立 VCS/SVA 证据质量 | 20/20 | exact-SHA sealed rerun；隔离 TB；production 与独立 cover 均 non-vacuous；无 assertion/fatal。 |
| Reset 与接口协议 | 5/15 | Reset 后状态能清空，但 reset 高电平仍可见四类 handshake；输入 valid 不是普通 decoupled ready/valid。 |
| 物理实现与论文口径 | 13/15 | 合同已正确禁止 physical/system/headline；仍无 realizable 4-read cache、M123 集成和 PPA。 |
| **Total** | **87/100** | **功能 datapath 成立；reset 和物理 cache 闭合前不得升级为无条件/物理 admission。** |

## Exact-SHA production rerun

冻结输入 SHA 全部匹配合同：RTL `cc343bd5...`、SVA `35f637d8...`、production TB `ad90e409...`、filelist `ee2d94cd...`、contract `0e351208...`。商业 VCS 重新编译和仿真均 rc=0，PASS line 与封存一致：

- 51 fill accepts，其中 48 是正向完整 cache fill；
- 66 rows / 66 row_done；
- 155 accepted updates / 528 selected sources / 14880 lane checks；
- 105 full-K4、50 tails、64 个 consecutive same-row update pairs、47 stall cycles；
- `cp_full_k4=105`、`cp_tail_k1=14`、`cp_two_fold_same_row=64`、`cp_update_stall_release=42`、`cp_empty_row=1`、`cp_fault=1`。

## Independent adversarial coverage

| Scenario | Result |
|---|---|
| All-16、稀疏非连续 mask、最低位与最高位 | PASS，独立 lowest-4 oracle |
| Source conservation | PASS，40 selected source 无 loss/duplication |
| K4 与 K1/K2/K3 tails | PASS，8 full-K4 + 4 tails |
| 连续同 block/同 row 的两次 row transaction | PASS |
| Update stall / release | PASS，28 stalled cycles，block/row/mask/delta 全稳定 |
| Mixed polarity、negated `-128` | PASS |
| Generic signed11 `+512` 与 `-512` | PASS，均精确 sign-extend 到 signed19 |
| Block transition cache invalidation | PASS，旧 block 16 vectors 不可继续引用 |
| Missing cache source / wrong block / wrong fill beat | PASS fail-closed，fault sticky，reset 可恢复 |
| Reset 高电平 fill request | Finding：`weight_fill_accept=1`，但事务被 reset 丢弃 |
| Reset 高电平 valid row | Finding：`row_accept=1`，但事务被 reset 丢弃 |
| Reset 打断 stalled update | Finding：`update_valid=1` 且 `update_accept=1`，但 row state 被 reset 清空 |

## P0

**0 个。** 在明确限定的 reset-free standalone scope 内，没有找到破坏 canonical lowest-4、source conservation、数值精确性、stall stability 或 block/cache identity 的反例。

## P1

### P1-1 — Reset 不隔离 public handshake

RTL 的 `weight_fill_ready`、`row_ready`、`update_valid` 和三个 accept 都未显式由 `!rst_core` 门控。独立 TB 在同步 reset edge 得到：

```text
reset_fill_phantom=1
reset_row_phantom=1
reset_update_visible=1
reset_update_phantom=1
reset_quiescence=false
```

内部 architectural state 在 reset 后确实清零，但这不能补偿外部已经观察到的 accept。修复要求：

- Reset 高电平时强制 ready/valid/accept 全部为 0；
- 加 reset-quiescence SVA，并且不要全部 `disable iff (rst_core)` 后完全失去 reset 可观察性；
- 明确定义 reset 前 pending row/update 是 abort 还是 drain；
- 重跑 fill beat1/beat2、stalled update 和 row_done 同边界攻击。

## P2

### P2-1 — `1536 B` 与 `3072 bit/update` 只是 logical architecture

两个数字的算术都正确：

```text
16 sources × 96 lanes × 8 bit / 8 = 1536 B
4 sources × 96 lanes × 8 bit = 3072 logical read bits/update
```

`1536 B` 能称为 total cache，是因为 M125 实现了严格的一次只驻留一个 output block，并在 block transition 清空旧 valid。它仍是 `logic weight_cache_q` behavioral register array 加组合选择，不是 foundry SRAM，也没有证明 3072-bit 物理读带宽能以目标频率、面积和能耗实现。

### P2-2 — 尚未接入 M123 accumulator

M125 只输出 signed19 delta。连续同 row 的 row-fold output 已验证，但没有经过 M123 forwarding accumulator 的 exact-write/commit miter。合同的 `m123_accumulator_integrated=false` 必须保持。

### P2-3 — Request valid 是 fail-closed，不是普通 backpressure

当 row/fill semantic prerequisites 不满足时，只要 producer 拉高 `*_valid`，RTL 就进入 sticky `protocol_error`；producer 不能像标准 ready/valid 那样提前保持 valid 等待 ready。这个选择可以作为强协议使用，但必须写入 wrapper/接口合同，否则组合时容易把正常 backpressure 误判为攻击。

## Performance claim audit

M125 RTL VCS **没有测得 3.1725x**。该数字来自 M122 heldout cycle simulator：

```text
fixed8 service-island baseline = 1,114,863,448 cycles
K1 candidate                  =   439,708,199 cycles
K4 candidate                  =   351,410,711 cycles
baseline / K4                 = 3.1725369008459166x
K1 / K4                       = 1.2512657845537327x
```

因此 `3.1725x` 是相对 inherited fixed8 service-island denominator 的 cycle projection；K4 相对 K1 的增量只有 `1.2513x`。它不是同面积、同频率、同 memory-port 成本的 physical speedup，更不是 full-network 或 system speedup。M125 VCS 只验证每个被接受 update 的功能；没有测 Fmax、PPA、能耗、FPS 或 end-to-end scheduling。

## Safe claim

> Exact-SHA commercial VCS and an independent adversarial scoreboard validate reset-free M125 canonical lowest-four select-and-clear, source conservation, block-phased cache identity, stall stability, mixed polarity, and exact signed11 fold arithmetic including +512 and -512. The 1536-byte cache and 3072-bit reads are logical architecture quantities. The 3.1725x result remains an M122 fixed8 service-island cycle projection (1.2513x incremental over K1), not an M125 RTL-measured, physical, full-network, or system speedup.

## Artifacts

- `m125_block_phased_k4_row_fold_independent_audit.json`：machine-readable score、计数、claim boundary 与 P0/P1/P2。
- `audit_m125_independent.py`：fail-closed hash/log/cover/ratio 审计。
- `sealed_vcs_rerun/`：冻结 production source 的独立商业 VCS rebuild/rerun。
- `independent_vcs/`：隔离 adversarial TB 的商业 VCS 证据。
- `tb_m125_independent_hammer.sv`：独立 oracle、scoreboard、identity fault 与 reset counterexample。
- `manifest.sha256`：review source 和文本证据的完整封存清单；VCS binary/database 可由 runner 重建，未纳入清单。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
