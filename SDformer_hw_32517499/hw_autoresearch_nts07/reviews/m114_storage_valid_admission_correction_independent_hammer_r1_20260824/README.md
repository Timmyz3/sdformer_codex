# M114 storage-valid/admission correction：独立打铁复审

日期：2026-08-24

结论：**94/100，P0=0 / P1=3 / P2=5。11 个 W 的 valid-bit 修正、byte ceiling、M109 schedule 冻结和 M110/M111/M112 standalone admission 映射全部正确；producer 的六项 SHA 清单可验证，但没有完整列出 analyzer 的全部输入。**

## 独立复算

统一公式：

`combined_bits = 2×128×W×2 + 314 + W×8×96×24 + W×8`

最后一项是 M111/M112 实际实现的 `8W` 个 lazy-valid bits；由于正好是每 row 1 byte，所有窗口相对 M109-r2 都增加 `W` bytes。

| W | valid bits | valid bytes | M109-r2 | M114 corrected |
|---:|---:|---:|---:|---:|
| 43 | 344 | 43 | 101,864 B | 101,907 B |
| 64 | 512 | 64 | 151,592 B | **151,656 B** |
| 96 | 768 | 96 | 227,368 B | 227,464 B |
| 128 | 1,024 | 128 | 303,144 B | 303,272 B |
| 192 | 1,536 | 192 | 454,696 B | 454,888 B |
| 256 | 2,048 | 256 | 606,248 B | 606,504 B |
| 294 | 2,352 | 294 | 696,232 B | 696,526 B |
| 384 | 3,072 | 384 | 909,352 B | **909,736 B** |
| 512 | 4,096 | 512 | 1,212,456 B | 1,212,968 B |
| 1,024 | 8,192 | 1,024 | 2,424,872 B | 2,425,896 B |
| 3,000 | 24,000 | 3,000 | 7,104,040 B | 7,107,040 B |

这仍是 logical lower bound，不是 foundry macro capacity/area；controller/grace、ECC、macro rounding/periphery、interconnect 和 weight SRAM 仍未计入。

## M109 schedule 冻结

- 11 个 `windows_per_phase` 全部逐字段相等。
- `exact_work` 共 33 个字段全部 deep-equal。
- `dual_timeline_recurrence` 共 240 个字段全部 deep-equal。
- 11 个 candidate、baseline、serialized ratio 原样未变。
- 另用 `baseline/candidate` 独立除法复核，最大误差仅 `2.26054e-16`。
- W384 仍是 `439,708,199 / 1,114,863,448 / 2.53546204172554×`。

因此 M114 只改 storage/admission metadata，没有改 work 或 schedule。

## Admission 核验

三个 sealed commercial-VCS run 的 input/output manifest 均重新验证：M110、M111、M112 各 `8 input + 4 output`，compile/sim RC 均为 0。

- W64：仅保留既有 controller geometry VCS。
- W384：准入 standalone controller、standalone signed24 accumulator、standalone lane-sliced adapter directed VCS。
- 其他 W：没有错误扩张 accumulator/adapter admission。
- integrated controller+accumulator、actual heldout replay、foundry macro、macro-inclusive PPA、physical/system/headline 全为 false。

没有发现当前 admission overreach。

## Exact/strict 攻击

- 当前 M109/M114 result、M114 contract、M111 review 均通过 duplicate-key/nonfinite strict JSON。
- duplicate JSON key、NaN、Infinity、duplicate manifest path、malformed hash、`..` traversal 均被独立解析器拒绝。
- M114 `SHA256SUMS.txt` 六项全部验证，但缺少 analyzer 实际读取的两项：M109-r2 result 与 M111 independent review。
- 两项 SHA 都存在于已固定的 M114 result identity 中，所以当前结果没有失去身份；但 `SHA256SUMS.txt` 自身不是完整 input closure，这是 P1。

## Claim boundary

`2.53546204172554×` 只能称为 **same-clock precompacted service-island software projection**。合同已明确禁止称为 RTL-measured、physical、equal-area、full-network、system 或 headline speedup，也禁止从三个 standalone run 推导 integration。

机器审计见 `m114_independent_audit.json`，评分与 findings 见 `m114_storage_valid_admission_correction_independent_hammer_review.json`。本评审只写本目录，未修改 analyzer/result/contract 或 `docs/359`；后者 SHA 仍为 `dedde7ce...`。
