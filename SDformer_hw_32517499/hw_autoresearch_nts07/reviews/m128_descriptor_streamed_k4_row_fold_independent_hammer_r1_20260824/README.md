# M128 descriptor-streamed K4 row-fold independent hammer

## Verdict

**88/100，descriptor-local 功能与跨 row II=1 条件通过；P0=0、P1=2、P2=4。**

冻结 production 输入以 exact SHA 重新通过 Synopsys VCS V-2023.12-SP1；独立 TB 又完成 140 个 descriptor accept、139 个 update、13,344 次 lane 数值检查。K1/K2/K3/K4、`-512/+512`、97-cycle 长反压、同拍 retire/replace、128 个连续 row、八类非法 descriptor/fill 攻击和 stalled-pipeline reset 均通过。128 个连续独立 row 形成 127 个相邻 descriptor II1 和 127 个相邻 update II1；没有数值、stall-stability、assertion 或 reset-isolation mismatch。

但 M128 只关闭了 **descriptor-local arithmetic/transport**，没有关闭完整 row-fold。独立正向反例证明 RTL 会接受 unsorted source IDs、holey `source_valid`，也会在同一 row 的两个 descriptor 中重复接受同一个 source。排序、left-pack 和跨 descriptor 的 exactly-once source conservation 均属于尚未实现、尚未计成本的外部 predecode producer。

第二个 P1 是 `row_done` 所有权。在跨 row II1 下，独立测试观察到 134 次 `row_done` 与下一 row 的有效 update 重叠；`row_done` 是前一 accepted `update_last` 的延迟脉冲，却没有 block/row tag。消费者不能用并发的 `update_row` 解释这个 pulse。

## Evidence summary

### Exact-SHA production VCS replay

- Compile/simulation rc：0/0；商业工具版本：VCS V-2023.12-SP1 Full64。
- 原 production PASS line 精确重现：384 descriptors、384 updates、36,864 lane checks、98 stall cycles、63 个 cross-row II1 intervals。
- SVA cover：K4 144、tail 240、reset 6、cross-row replace 60、stall-release 78。
- production RTL、SVA、TB、filelist、contract、runner、M127 correction/review manifest 与 docs/359 SHA 均冻结一致。

### Independent adversarial VCS

| Check | Result |
|---|---:|
| Accepted descriptors / updates / reset-aborted | 140 / 139 / 1 |
| Selected sources / numeric lane checks | 543 / 13,344 |
| K1 / K2 / K3 / K4 | 3 / 3 / 2 / 132 |
| Cross-row descriptor II1 intervals | 127 |
| Cross-row update II1 intervals | 127 |
| Output/input stall cycles / maximum burst | 97 / 97 / 97 |
| Same-cycle retire/replace after long stall | 1 |
| `-512/+512` | 9 / 9 |
| row_done checks / overlap-next-row | 138 / 134 |
| Duplicate/dirty-source/dirty-negate/mask/cache/block/empty/collision attacks | 1 each |
| Reset while output and next input are stalled | 1 |
| Accepted unsorted / holey-valid / cross-descriptor duplicate probes | 1 / 1 / 2 |

## Ready/error dependency audit

No internal combinational ready/error loop was found. `quarantine` and `protocol_error` depend on current requests/cache state, while `group_ready` consumes `quarantine`, semantic validity, fill state and output readiness; neither descriptor/request audit consumes `group_ready` or `group_accept`.

The interface still has an integration hazard: `group_ready` is payload-semantic even with `group_valid=0`. An independent probe made ready high with a legal inactive payload and low with an invalid-mask inactive payload. Therefore the producer must present defined/stable payload and must not derive valid or payload from ready; otherwise use a registered/skid boundary and reverify the composed graph.

## 53-bit claim boundary

The payload arithmetic is exact:

```text
block3 + row9 + source_valid4 + source_ids(4x4) + negate4
       + selected_mask16 + last1 = 53 bits
```

It excludes ready/valid, framing/queue metadata, producer state and the cost of transporting descriptors. It is not evidence of bandwidth reduction because no complete baseline-equivalent format/traffic ledger exists.

## Findings

### P0

None. No descriptor-local numeric, protocol, stall, or reset counterexample was found in the admitted scope.

### P1

1. Full canonical row-fold correctness depends on a missing external producer. M128 locally accepts unsorted IDs, holey valid slots and same-row cross-descriptor source duplication. Implement/seal sorted and left-packed descriptor generation with per-row exactly-once source conservation, or retain a trusted-input descriptor-local claim.
2. Untagged `row_done` has ambiguous ownership at cross-row II1. Define completion by the accepted `update_last` handshake, or add delayed block/row tags and reverify consumers under stalls.

### P2

1. `group_ready` depends on inactive descriptor payload; document the no-ready-dependence/stable-payload rule or add a registered/skid boundary.
2. 53 bits is descriptor payload only, not complete interface or bandwidth cost.
3. External predecode latency, buffering, area, power and traffic are unimplemented/unmodeled.
4. There is no matched DC/frequency, foundry SRAM macro, macro PPA, physical speedup or system speedup evidence; the 1,536-byte cache is behavioral.

## Paper-safe claim

> Exact-SHA Synopsys VCS and an independent adversarial test verify M128 descriptor-local signed K1-K4 arithmetic over 13,344 independent lane checks, fail-closed malformed descriptors, a 97-cycle ready-valid stall with same-cycle retire/replace, reset isolation, and 127 consecutive cross-row descriptor/update II1 intervals. The descriptor payload is 53 bits. Canonical descriptor production, valid left-packing, cross-descriptor per-row source conservation, descriptor traffic/cost, unambiguous tagged row completion, matched DC frequency, macro PPA, physical speedup and system speedup remain unadmitted.

## Reproduce and audit

The VCS runner is write-once and refuses to overwrite its two evidence directories. Run it in a clean review directory:

```bash
reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/run_vcs_m128_independent_hammer.sh
python3 reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/audit_m128_independent.py
sha256sum -c reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824/input_manifest.sha256
(cd reviews/m128_descriptor_streamed_k4_row_fold_independent_hammer_r1_20260824 && sha256sum -c manifest.sha256)
```

`manifest.sha256` covers every top-level review artifact and all durable files at depth two, including both rebuilt VCS binaries; `csrc/`, `simv.daidir/` and `simv.vdb/` are reproducible intermediates and intentionally excluded. Production files and `docs/359_DATE终局冻结_20260813.md` were not modified; docs/359 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
