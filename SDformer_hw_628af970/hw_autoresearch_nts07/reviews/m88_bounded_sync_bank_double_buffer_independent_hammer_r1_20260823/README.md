# M88 bounded sync-bank double-buffer 独立打铁评审

## 结论

M88 的核心数字可以独立重现，结论为：

- `P0=0 / P1=5 / P2=3`；
- **GO**：valid825-internal、五样本、隔离模块、同 32B/cycle 带宽模型的周期估算；
- **NO-GO**：RTL 可执行模块倍速、等面积/等功耗倍速、物理 SRAM PPA/能耗、精度、
  全网/系统倍速及 DATE headline。

评分：硬件创新 68、性能优势 73、证据质量 91、里程碑完整度 92、DATE 完整度 58，
本里程碑综合 84/100。

## 独立重算数字

评审脚本没有 import M88 或生产 M78。它使用上一轮已封存、直接从 M41/M72/M40
重建的独立 M78 结果，并直接解码 `/tmp/m85_inputs` 下 exact-SHA M83 records/offsets。

| 指标 | 独立结果 |
|---|---:|
| M83 phases / entries | 1,728 / 221,184 |
| signed8/9/10/11/escape | 52,248 / 128,893 / 37,144 / 2,898 / 1 |
| M86 bank issues（含 escape control） | 835,383 |
| numeric payload / header / padding | 23,776,068 / 82,944 / 24,988 B |
| canonical records | 23,884,000 B |
| prepare range | 786–847 cycles |
| aggregate candidate | 790,706,475 cycles |
| aggregate bit-sparse | 1,114,402,488 cycles |
| aggregate speedup | **1.409375695323603x** |
| midstream phase-refill stalls | **0** |
| listed storage | 116,525 B = **113.793945 KiB** |

逐样本也全部 exact match：

| sample | candidate | bit-sparse | speedup |
|---:|---:|---:|---:|
| 5 | 154,847,259 | 221,547,408 | 1.430748012x |
| 6 | 159,669,145 | 224,991,456 | 1.409110420x |
| 7 | 159,116,417 | 221,167,536 | 1.389973079x |
| 8 | 159,411,903 | 222,126,528 | 1.393412435x |
| 9 | 157,661,751 | 224,569,560 | 1.424375656x |

每个 sample 相对 frozen M78 仅增加 3,458 cycles：
`2 sync-fill × 1728 phases + (838 new phase-0 prepare - 836 old prepare)`。

## 双缓冲无装载停顿证明

上一轮独立 M78 全 8,640 phases 的最小
`candidate_compute - max(matcher, packer, old_next_DMA)` 为 12,637 cycles。
直接解码 M83 后，新 preparation 相对旧 M78 preparation 每 phase 只增加 2 或 3 cycles。
因此保守的新余量仍为 `12,637 - 3 = 12,634 > 0`。

两槽归纳关系是：phase `i+2` 的槽在 compute `i` 结束时释放；由于 compute `i+1`
始终比 prepare `i+2` 至少长 12,634 cycles，下一 phase 必在 compute `i+1` 结束前 ready。
因此只暴露 phase-0 startup，candidate 为 838 cycles、baseline 为 384 cycles；中途 refill
stall 确为 0。

这里的 **0 只表示 phase-refill stall 为 0**，不表示 FIFO、output backpressure、accumulator、
correction 或 escape fallback 都无 stall。

## 双算/漏算审计

核心账本没有发现双算或漏算：

- record 每 phase 在 shared DRAM 只收一次，已包含 48B header、numeric payload 和 zero padding；
- weight phase 在 DRAM preparation 收一次，event 执行时的三周期 weight service 是片上读/数据通路，
  不是再次传同一笔 DRAM；
- 74B metadata 中的 48B header 已在 record，另 26B pattern base 由 128-entry parser 生成，
  因而不应再收一笔 DRAM；
- 460 row writes 是独立 writer 资源，通过 `max(DRAM,row-write,parser)+commit` 收费，
  不是漏掉。该重叠在时间上可行，但尚未由真实 DMA/parser/ping-pong RTL 实现。

五样本 candidate shared-DRAM 总量为 225,588,320 B，bit-sparse 为 106,168,320 B；
两者都执行相同 phase weight reload。比较是同 workload、同 bandwidth 公平，不是等面积公平。

## 主要缺口

### P1

1. M86 仍是单 bank image；testbench 离线提供 256-bit rows 和 74B metadata。M88 的双缓冲
   DMA/parser/writer 是周期模型，不是 RTL。
2. 3/4/4/5 service 假定 always-ready；M86 stress phases 不准入这些 II，唯一 escape 仍为
   zero placeholder。
3. baseline 同带宽但非等面积：candidate 额外拥有 113.8KiB subtotal 和 matcher/packer/frontend。
4. 113.8KiB 只含两组 PWP+metadata、两组 weight、response FIFO、pattern table 和 offset table；
   不含 activation/descriptor queue、accumulator、fallback/correction routing、parser、ECC、macro padding。
5. catalog 与 heldout 都是 valid825 internal，无 train-only catalog、accuracy 或全网 composition。

### P2

1. pattern/offset table 的 cold preload、matcher read ports、仲裁和能耗未收费；
2. 每 catalog pass 的 794,880 row writes 中有 51,149 个 zero-tail write（6.88%），已收费但可优化；
3. M88 analyzer 对 M86 RUN_COMPLETE/input manifest 记录 SHA，但没有 expected-SHA pre-gate。

机器可读结果为 `m88_independent_hammer_review.json`，复跑入口为
`audit_m88_independent.py`。未修改 M88 分析器、结果或任何生产证据。
