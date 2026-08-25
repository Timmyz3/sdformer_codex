# M127 pipelined K4 row-fold independent hammer

## Verdict

**91/100，功能流水线条件通过；P0=0、P1=1、P2=4。**

M127 的核心功能成立。冻结 production 输入以 exact SHA 重新通过 Synopsys VCS V-2023.12-SP1；独立差分 TB 又在 5,129 个合法 accepted cycles 上逐周期对比 M125，并对 168 个 accepted update 做 16,128 次 lane 数值检查。canonical groups、source conservation、K1/K2/K3/K4 tails、`-512/+512`、63-cycle 长反压、pipeline stall stability、block/cache identity、sticky protocol fault 和 reset isolation 均通过。

独立评审找到一个必须收窄的吞吐口径：**II=1 只在同一 active row 内的连续预解码 groups 成立，不是任意 row transaction 的 II=1。** 一个 16-source row 的四组 update 连续四拍接受；但四个连续 single-group K4 rows 的 accepted update 间隔全部是 2 cycles。原因是 `row_ready` 要求 `row_active_q` 和 `pipe_valid_q` 同时为 0，最后一组被接受的同拍没有 next-row look-ahead。

## Evidence summary

### Exact-SHA production VCS replay

- Compile/simulation rc：0/0；商业工具版本：VCS V-2023.12-SP1 Full64。
- 原 production PASS line 精确重现。
- 507 accepted-cycle checks、176 updates、16,896 lane checks。
- SVA cover：four-group II1 1、full-K4 126、K1 tail 18、stall-release 35、empty row 2、reset quiescence 2。
- 所有 production RTL、SVA、TB、filelist、contract、runner、M125 review manifest SHA 均与冻结身份一致。

### Independent adversarial VCS

| Check | Result |
|---|---:|
| Legal M127/M125 cycle-exact comparisons | 5,129 |
| Rows / accepted updates / selected sources | 81 / 168 / 572 |
| Canonical checks / lane numeric checks | 168 / 16,128 |
| Backpressure cycles / longest burst | 4,642 / 63 |
| Final-group K1/K2/K3/K4 coverage | 18 / 14 / 18 / 30 |
| Same-row consecutive update pairs | 9 |
| Four-group same-row II1 sequences | 1 |
| Single-group cross-row update interval | 2 / 2 cycles min/max |
| `-512/+512` | 1 / 1 |
| Cache transition / cache fault / block fault / fill fault | 1 / 1 / 1 / 1 |
| Reset isolation / stalled-pipeline reset flush | 2 / 1 |

There were no accepted-cycle control, valid-payload, numeric, assertion, stall-stability, conservation or reset-isolation mismatches in the final race-free test. An earlier exploratory mismatch was traced to the review TB changing `update_ready` in the active region of a `posedge`; the final test drives ready only on `negedge` and is the sole admitted independent result.

## Throughput and latency boundary

- Same-row four-group update II: **1 cycle**.
- Consecutive single-group row update II: **2 cycles**.
- No-stall row accept to first accepted update: **1 cycle** in both M125 and M127.
- M127 additional first-group cycles versus M125: **0**.

Therefore the contract field `no_backpressure_update_initiation_interval=1` is too broad unless read as an intra-row group metric. Required P1 repair is either to rename it `intra_row_group_update_ii=1` and publish cross-row II=2, or implement next-row look-ahead on the last group and reverify under stalls.

## 1,920-bit audit

The arithmetic is correct for pair-sum payload:

```text
2 pair arrays x 96 lanes x signed10 = 1,920 bits
```

It is not total elastic-stage storage. `valid + block + row + selected_mask + last` adds 30 bits, so the stage has at least 1,950 bits before additional predecoded group and row state. Paper wording must retain “1,920-bit pair-sum payload,” not “1,920-bit total pipeline.”

## Findings

### P0

None. No functional or data-integrity counterexample was found in the admitted legal-input scope.

### P1

1. Unqualified II=1 overstates row-level service rate. Same-row groups are II1; consecutive single-group K4 rows are deterministically II2.

### P2

1. 1,920 bits excludes 30 bits of stage metadata and all predecoded group/row state.
2. “First group 0 cycles” is only zero additional cycles versus M125; absolute latency is one cycle.
3. VCS does not admit frequency, area, power or physical speedup; matched DC/STA is still required.
4. The 1,536-byte cache remains behavioral multi-read logic, not a foundry SRAM macro.

## Paper-safe claim

> Exact-SHA commercial VCS and an independent M125 differential scoreboard verify M127 legal accepted-cycle behavior, canonical K1-K4 grouping, exact signed arithmetic including -512/+512, long elastic stalls, block/cache identity and reset isolation. M127 sustains II=1 for four consecutive groups within one active 16-source row; consecutive single-group K4 rows remain II=2. The 1,920-bit quantity is pair-sum payload, while no-stall first-group latency is one cycle with zero additional cycles versus M125. Frequency, physical speedup, macro PPA and system speedup remain unadmitted.

## Reproduce and audit

The VCS runner is write-once and refuses to overwrite the two evidence directories. Run it in a clean checkout/review directory:

```bash
reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/run_vcs_m127_independent_hammer.sh
python3 reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/audit_m127_independent.py
sha256sum -c reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/input_manifest.sha256
(cd reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824 && sha256sum -c manifest.sha256)
```

`manifest.sha256` covers every top-level review artifact and all durable files at depth two, including both rebuilt VCS binaries; rebuild databases under `csrc/`, `simv.daidir/` and `simv.vdb/` are reproducible intermediates and intentionally excluded. Production source files and `docs/359_DATE终局冻结_20260813.md` were not modified; the latter remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
