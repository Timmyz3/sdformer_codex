# M599｜M597/M593 parent-scratch energy r2 source static hammer

## 裁决

**PASS，96/100；P0/P1/P2 = 0/0/1。** M595 的 traffic 配对、严格 frozen identity 与 future-result provenance 三项缺陷已经在 r2 source 中闭合。允许 root **另立** exact runner/attempt/release-candidate 链；本评审不授权直接正式运行 analyzer，也不授权把 `38.2283079189%` 或 `1.2622562287 mJ` 写成 admitted/paper 结果。

本评审没有运行正式 analyzer，没有创建 canonical result/attempt/runner/launch，没有调用 EDA、GPU 或远程服务，也没有修改被审 source、contract、handoff 或 `docs/359`。仅执行了 Python 3.6 原生编译、内建无业务输入 self-test、独立 Decimal 复算、冻结输入双封校验与 `/tmp` 中的 publish fault test。

## M595 三项关闭

### P0｜traffic 与周期现在同源配对

- all-write 完整取自 sealed M504：`456,016,645 cycles`、`16,490,761 macro reads`、`1,714,628 RAW forwards`、`27,305,568 writes`。
- `reads + forwards = 18,205,389 parent edges`；forward 在 future row 中显式标记为不收 macro-read 能量。
- dead-only 取自 sealed M528/M528 hammer：`435,293,339 cycles`、`16,490,761 reads`、`1,714,628 forwards`、`9,947,701 writes`、`17,357,867 elisions`。
- `writes + elisions = 27,305,568 active rows`，M504/M528 的 all-write cycle 交叉值一致；analyzer 只选择唯一 `m505_dead_write_only_1rw` row，combined-PVRF row 不可能进入计算。
- 两条 row 均显式执行 `8 banks × 144 B`。独立复算得到 read bytes `18,997,356,672`；all-write/dead-only write bytes 分别为 `31,456,014,336` 和 `11,459,751,552`。

### P1｜调用者不再选择业务输入或 digest

CLI 只有 `--source-contract`、`--output-dir` 与 `--self-test`；没有 M504/M528/macro-map path 或 expected SHA 参数。analyzer 内建完整 path/SHA/manifest/outer map，在解析业务 JSON 前验证 exact r2 contract path/SHA、顶层 key set、完整 frozen-input map，以及每个 sealed directory 的 member/manifest/outer seal。analyzer 嵌入 contract SHA `90399b6c...`，双封 author handoff 反向绑定 analyzer SHA `6896c8a4...`，没有 contract/analyzer 的自指 SHA 环。

### P2 lineage｜future rows 已保留防错账本

future row 保留 cycle source、traffic source、read/forward/write、parent/active count、8-bank/144-B multiplier、access/byte count与 executed conservation；结果边界明确为 **per-frozen-sampled-inference**、九宏 parent-scratch datasheet component model，不是 camera frame、C1、full-network、system 或 silicon energy。

## 独立数值复算（仍仅是 review diagnostic）

slow 0.9 V 下，九宏 1152-bit access：read `94.57074 pJ`、write `90.65763 pJ`，leakage `0.54009423 mW`。独立 Decimal 结果：

| 项 | M504 all-write | M528 dead-only |
|---|---:|---:|
| dynamic mJ / frozen sampled inference | 3.228001241293584 | 1.969102774033416 |
| leakage mJ / frozen sampled inference | 0.073887587624537505 | 0.070529826225400191 |
| modeled component total mJ / frozen sampled inference | 3.301888828918121505 | 2.039632600258816191 |

诊断差值为 `1.262256228659305314 mJ`，组件能量降幅为 `38.2283079189219449%`，cycle ablation 为 `1.0476076800247109×`。这些值只证明 source 算式与 M595 修正诊断一致，**尚非正式 M597 result，也不是论文能量数据**。

## 唯一 P2｜publish 端路径/失败残留应由 exact runner 加固

analyzer 的 `--output-dir` 未冻结到唯一 repo canonical 坐标，也未拒绝 symlink parent；`Path.exists()` 不能区分 dangling symlink，且 `os.rename()` 不是显式 `RENAME_NOREPLACE`。`/tmp` fault test 证明 dangling target 会 fail-close 而不发布 canonical，但会遗留完整 staging directory。可信 exact runner 必须：

1. 内建唯一 canonical result path，并验证其所有 parent 均为 repo 内普通目录、无 symlink；
2. 用 `lexists`/`lstat` 拒绝 dangling symlink 与任何预存目录项；
3. 用 no-replace publish 或等价原子锁，捕获异常时将 staging 移入有封印的 quarantine/attempt；
4. rename 后直接重算 canonical member/manifest/outer identity，再允许 success receipt。

这是 publish/runner hardening，不改变业务输入 identity 或能量算式，因此记 P2，不阻止 authoring 一个单独、受静态评审约束的 exact runner。

## 授权边界

- `exact_runner_authoring_allowed = true`
- `formal_analyzer_execution_allowed = false`
- `canonical_result_allowed = false`
- `paper_energy_data_allowed = false`

下一步只能由 root 另立 exact runner/attempt/release candidate，关闭上述 P2 并再做独立 runner static hammer。之后才可进行一次正式 analyzer run；正式结果还必须另做 fresh result hammer。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
