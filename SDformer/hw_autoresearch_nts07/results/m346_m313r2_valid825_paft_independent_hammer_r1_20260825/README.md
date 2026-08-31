# M346：M313r2 valid825 PAFT 独立打铁复核

结论：**96/100，P0/P1/P2=`0/0/2`。M313r2 的证据链可信，但结果是确定的 `NO_GO_MODIFIED_FORWARD_ACCURACY_GATE`。按冻结的 M309/M313 政策，enable023 的正距离 PAFT 主线永久关闭；禁止再用 valid825 搜索其他 operator subset 或距离阈值。**

这不是说所有 PAFT、tau0 exact 或 q128 都失败。关闭范围仅是当前 `tau=1`、enabled operators `[0,2,3]` 的 positive-distance near-match residual-elision。M312r2 tau0 exact baseline 和独立 M338/M340 q128 train-only 线不受此结论撤销，但 M346 也不替它们准入周期、能耗、PPA、系统加速或论文 headline。

## 独立结论

M312r2 baseline AEE 为 `1.4691506710196987`，M313r2 candidate AEE 为 `1.498478640643033`，完整精度差值为 `+0.0293279696233343`。冻结门槛是绝对增加不超过 `+0.02`，所以 candidate 比最大允许 AEE `1.4891506710196987` 仍高 `0.0093279696233343`，使用了允许退化预算的约 `146.64%`。

直接对两个 `per_frame.csv` 的 825 个 AEE 十进制字段求等权均值，得到：

- baseline：`1.4691506688129696969...`；
- candidate：`1.4984786414984242424...`；
- 独立差值：`+0.0293279726854545454...`；
- 相对 `+0.02` 门仍失败 `0.0093279726854545454...`。

CSV 只保留十位小数，导致均值与 profile 的完整精度值相差 `1e-9` 量级；这个差距比失败余量小约六个数量级，不可能翻转结论。

## 825 帧与身份

- baseline/candidate 均为 825 行、825 个唯一 `(sequence,file)` key；
- 两者顺序逐行相同，并与 SHA `7f3dc280...` 的官方 valid list 完全一致；
- 两个 receipt 的 `ordered_population` 均逐项等于各自 CSV；
- `valid_pixels` 和 `gt_flow_mag` 逐行一致；
- checkpoint/config SHA 分别为 `cf4833b2...` 和 `070d0dfe...`；两次 checkpoint load 都是 missing/unexpected/overlay mismatch 全零；
- M309、M312r2、M313r2 的 34 项合同输入全部重新哈希，mismatch=`0`；launcher、wrapper、evaluator、基础模块和选择模块 SHA 均与冻结合同/回执一致；
- M312 baseline 与 M313 candidate 的四文件 manifest 和二级 seal 均在本机逐项 replay 通过。

## tau1 实际命中账本

| Operator | Enabled | Exact hit | Positive distance | Total snapped |
|---|---:|---:|---:|---:|
| resblocks.0.conv1.0 | yes | 78,687,829 | 87,101,876 | 165,789,705 |
| resblocks.0.conv2.0 | no | 0 | 0 | 0 |
| resblocks.1.conv1.0 | yes | 89,209,895 | 103,992,277 | 193,202,172 |
| resblocks.1.conv2.0 | yes | 63,007,466 | 54,536,554 | 117,544,020 |
| **Aggregate** | `[0,2,3]` | **230,905,190** | **245,630,707** | **476,535,897** |

逐 operator 与 aggregate 均满足 `total=exact+positive`；正距离命中非零且占 snapped work 的 `51.5451%`。operator 1 的 calls、partition、exact、positive 和 total 全零，证明实际执行策略确为 `[0,2,3]`。

## 关闭边界

1. **positive-distance PAFT：永久关闭。** M309 在 valid825 前已冻结唯一候选并明确禁止失败后追 subset/threshold；M313r2 又重复了这一 red line。继续在这 825 帧上挑组合或阈值将构成 validation-set search。
2. **tau0 exact：精度边界不受影响。** M312r2 是独立封存的 exact production forward；M313r2 的失败来自 tau1 正距离替换。exact-only 路线仍可作为无损机制研究，但不能从本复核自动继承任何硬件性能结论。
3. **q128：不受撤销。** M338/M340 是独立的 train-only nested catalog/vector-work 线，不是 M309 候选 subset 或 threshold 变体；M313r2 无权撤销它，也不能替它提供 runtime/cycle/system admission。

## 两个 P2

- 回执保留生成服务器 `/root/private_data/...` 绝对路径。本机搬迁到 `/home/zhumd/...` 后仍能完整复核字节 SHA 和 seal，但不能原样重演绝对路径 gate。以后应同时封存 repo-relative identity 与 producing-host resolved path。
- per-frame CSV AEE 仅十位小数，无法 bit-exact 重建 profile 的完整精度 aggregate。以后应输出 round-trip 精度或精确 aggregate numerator/count。

本复核只使用只读哈希、JSON/CSV 解析和十进制重算；未运行 GPU、RTL、VCS 或新思，未修改 M309/M312/M313、冻结合同和 `docs/359`。
