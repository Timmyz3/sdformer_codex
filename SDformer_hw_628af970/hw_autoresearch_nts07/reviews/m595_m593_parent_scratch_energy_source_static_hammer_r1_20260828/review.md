# M595｜M593 parent-scratch generated-macro energy source static hammer

## 裁决

**FAIL，64/100；P0/P1/P2 = 1/1/1。** 不允许创建 exact runner、launch admission、canonical result 或论文能量数字。下一步只能另立不可变修订身份，修正 M504 traffic 语义与 frozen-SHA 绑定后，再交新的独立静态打铁。

本评审没有运行正式 analyzer，没有创建 M593 result/attempt/launch，没有调用 EDA、GPU 或远程服务，也没有修改被审 source/contract 或 `docs/359`。

## P0｜40.5634% 的分子分母配错了

M593 在 analyzer 第 166、175–179 行把：

- `m504_all_write_1rw_cycles = 456,016,645`
- `m473_fused_concurrent_1r1w_ceiling` 的 read/write traffic

拼成同一 baseline。写流量相同没有问题；读流量不同。

M504 的单口 all-write 调度已执行同址 RAW forwarding。冻结 M505 结果明确给出每个 output block：

- M504 macro read：`16,490,761`
- RAW forward：`1,714,628`
- 二者之和：`18,205,389` parent edge
- M504 macro write：`27,305,568`

M593 使用的 M473 read row 把全部 `18,205,389` edge 都算成宏读，得到 `20,972,608,128 B`；实际与 M504 周期配套的宏读应是：

`16,490,761 × 8 banks × 144 B = 18,997,356,672 B`。

修正后独立 Decimal 复算如下：

| 项 | M504 all-write | M505 dead-write-only |
|---|---:|---:|
| cycles / S10 | 456,016,645 | 435,293,339 |
| read bytes / S10 | 18,997,356,672 | 18,997,356,672 |
| write bytes / S10 | 31,456,014,336 | 11,459,751,552 |
| dynamic mJ/frame | 3.2280012413 | 1.9691027740 |
| leakage mJ/frame | 0.0738875876 | 0.0705298262 |
| component total mJ/frame | 3.3018888289 | 2.0396326003 |

所以正确的静态诊断值是：

- cycle speedup：`1.0476076800×`
- parent-scratch generated-macro component energy saved：`1.2622562287 mJ/frame`
- component energy reduction：`38.2283079189%`

当前 source contract 的 `40.5634216565%` 和 `1.3919791397 mJ/frame` 分别高了 `2.3351137375` 个百分点和 `0.1297229110 mJ/frame`。单位公式本身是对的，错的是 all-write read traffic 与 M504 cycle 的语义配对。

修法应优先选一条：

1. 新版 M528 结果显式输出唯一的 `m504_all_write_1rw` traffic row；或
2. M593 新增并冻结 sealed M505 result，从 `m504_macro_reads/writes` 生成 traffic。

同时必须执行 `reads + forwards = parent_edges`、`writes = active_rows`、M504/dead-only read equality 和 `8 × 144 B` 守恒，不能再借用 M473 row 名义替代。

## P1｜三条 SHA 仍由调用者自己指定

`docs/359` 的 SHA 是 analyzer 常量；M528 result、M528 hammer、macro map 的 expected SHA 却来自 CLI。调用者可提交一个已修改文件及其新 SHA，仍通过第 122–131 行的检查。analyzer 没有读取 M593 source contract，也没把 CLI SHA 与 contract 中的三条 frozen digest 比较；当前又没有 sealed exact runner 来封死参数。

新版本应在 analyzer 内硬编码全部 frozen digest，或者冻结 source-contract 本身的 path/SHA，并在解析任何业务输入前验证 CLI path/SHA 与 contract 条目完全一致。未来 runner 只能是 defense in depth，不能成为唯一身份绑定。

## P2｜结果缺少 traffic provenance

输出把 row 重命名为 `all_write_1rw_parent_scratch`，却不保留原 traffic source、per-output-block read/forward/write、bank multiplier 与守恒谓词。建议把这些字段写进 bounded result；这会直接防止本次 M473/M504 错配再次发生。

## 已通过检查

- Python 3.6.8 原生 `compile()` 通过。
- clean-room JSON fault injection 正确拒绝 duplicate key、NaN、Infinity 和非 object 顶层。
- 当前四个输入文件 SHA 与 source contract 一致；`docs/359` SHA 保持 `dedde7ce...`。
- M528 result hammer 的 exact-CPU admission 与 “logical bytes != energy” 边界存在。
- 生成宏 cell/shape、13/13 状态、slow 0.9 V、面积/周期/access/current/leakage 均匹配；九宏面积 `78,825.2454 µm²` 与 M528 一致。
- `uA/MHz × V = pJ/access`、动态能量、leakage/frame 的单位换算正确。
- future result 的 claim boundary 已排除 integrated PPA、logic/other SRAM/DRAM、C1 total、full-network/system、silicon 与 DATE headline。
- combined-PVRF traffic 未被误选；磁盘上不存在 M593 canonical result 或 attempt。

注意：上述 `38.2283%` 只是本评审用于证明 P0 的独立诊断重算，**不是 admitted result，不得写进论文表**。
