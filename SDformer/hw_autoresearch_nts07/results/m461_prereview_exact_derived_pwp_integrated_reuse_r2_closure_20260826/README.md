# M461 prereview R2 closure

## 结论

R2 只关闭独立 M461r2 95 分评审留下的 `P1=1/P2=2`；R1 不改。没有读取 M40，没有读取或运行 M453b，没有运行 RTL，也没有修改 `docs/359`。

唯一首选仍是 `compact used-center bank + original-order replay`，等待最终 sealed `Nmax`。A 全 q128 cache 继续 NO-GO；B q32 parent + child scratch 只作固定容量备选接口筛选；C 缺 ordered selected-ID ledger 时保持 unknown；group replay 不是主线。全部性能、能量、PPA 与 headline 字段仍为 `false`。

## P1：48-bit descriptor 与生命周期冻结

Primary assignment bank 固定为每 bank `3000×48 bit`，两 bank 逻辑 payload 为 36,000 B。未来 M461 descriptor 的 LSB0 布局为：

| bits | 字段 | 合法约束 |
|---|---|---|
| 11:0 | `destination_row_id` | 0..2999；compact replay 中严格递增 |
| 27:12 | `original16` | 必须非零 |
| 34:28 | `global_center_id7` | M453a flat ID 0..127 |
| 39:35 | `hamming_distance5` | 必须等于 `popcount(original XOR center)` |
| 40 | `use_pwp` | 必须严格等于 `1+distance < popcount(original)` |
| 41 | `descriptor_valid` | 合法 descriptor 必须为 1 |
| 47:42 | `reserved6` | 必须全零 |

`use_pwp=0` 完全走 exact bit-sparse：不得查 remap/PWP，不产生 minus mask；correction source 是 `original16`。`use_pwp=1` 才允许查 compact slot，并生成 `plus=original&~center`、`minus=center&~original`。

结束 sentinel 固定为全零 `48'h0`，但它是 controller 在 `read_ptr==sealed_active_count` 时合成的，不占 SRAM word，也绝不送到 backend。因此全 3000 行合法：地址 0..2999 都能装 descriptor，pointer=3000 时结束。空 phase 的 count=0，完全不发 descriptor/PWP read。inactive SRAM 的旧数据不必清零；bank epoch/seal 未 valid 时不可达。

中心 remap 的 valid 由 sealed 128-bit `used_pwp_center_bitmap` 给出。slot 固定为该 center 在升序 set-bit 中的 rank，必须满足：

- slot 范围 `0..Nused-1`、连续且一一映射；
- `slot_to_center[slot]==center`；
- 每 slot 的 8 个 output-block PWP valid 全部置位后才允许 bank-ready；
- fallback 完全旁路 lookup；
- bitmap 无效、slot 越界、inverse mismatch、PWP block 未 valid、tag/epoch/bank 错误或任一 X/Z，都在同拍禁止 accept/request/output 并设置 sticky error，不能当 miss、fallback 或 slot0。

生命周期固定为：

1. invalidate NEXT：清 epoch/seal/count/bitmap/remap-ready/PWP-ready，payload 不必清。
2. capture：按 source row 0..2999 接收；零行不写，非零行经 matcher exact check 后写 compact `active_count` 地址；只有 write ack 才递增。
3. seal：row2999 后等 matcher drain 和全部 write ack，原子封 `active_count/bitmap/tag/epoch`。
4. materialize：从 sealed bitmap 建 deterministic remap，完成每个 slot×8 block 的独立 PWP write 后才置 valid。
5. role switch：assignment/remap/PWP/weight/config 全 valid、generator idle、无 pending write，且旧 current replay 和 downstream update 已 drain，才可原子切换。
6. replay：只读 `0..count-1`，验证 bank/epoch/tag/address、descriptor 数值与 destination 严格递增；count 位置合成 sentinel。完成还要求所有 descriptor/remap/PWP/weight response、FIFO 和 adapter output 都 drain。

任一 early read/write/seal/switch/reload、same-bank R/W、非法 descriptor/remap、fallback PWP read、sentinel 泄漏、stale response 或 X/Z 都原子 fail-closed：同拍抑制所有 accept/output，sticky `protocol_error` 只可 reset 清除。

## 48-bit logical 与 64-bit macro sensitivity

- 48-bit logical：两 bank 36,000 B。
- 64-bit-row macro sensitivity：两 bank 48,000 B，多 12,000 B，即 `+33.33%`。
- 64-bit 点的 upper16 仅为 padding，写入强制零、读出检查零，不增加语义容量。

36,000 B 不能写成物理宏；target macro 未定前必须两列并报。跨 row packing 是另一个未准入 DSE。

## P2：p95 口径

all128×8 weight-update-only 的 1728 个值同时保留两种明确命名的统计：

- 标准 nearest-rank：`rank=ceil(0.95×1728)`，zero-based index 1641，值为 **1584**；这是后续表的 canonical p95。
- 旧 floor 定义：`index=floor(0.95×(1728−1))=1640`，值为 **1576**；只能称 legacy floor-index statistic，绝不能冒充标准 p95。

二者都是 train-catalog structural reference，不是 M40 runtime 或性能数字。

## P2：B config footprint

B 的 `27,552 B/tile` 是兼容地址视图：`288 config + 6144 weight + 20480 parent PWP + 640 child scratch`；两 tile address footprint 为 55,104 B。这不自动代表两个 288 B config 物理实例。

- shared physical：每 phase role 一份 288 B，两 role 共 576 B；B 已知 lower bound 156,896 B。
- per-tile replicated physical：两 tile×两 role 共 1,152 B；B lower bound 157,472 B。

未来资源表必须明确选哪一种，不能一边描述 duplicated config ports，一边引用 shared 576 B 下界。

## 下一门

R2 必须先做独立 delta hammer：验证 R1/95 分 review 的双封、重算 descriptor/storage/p95/config 算术，攻击所有 fail-closed 条款并复查 `docs/359`。通过前仍不能读 M40、运行 M453b、写 RTL或使用任何性能数字。

受保护文件 SHA256 仍应为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
