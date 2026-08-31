# M185 fixed-bank K8 accumulator 独立打铁评审 r1

结论：**87/100，`PASS_AS_FIXED_BANK_K8_INTERFACE_AND_TIMING_POINT_REJECT_AREA_WIN_KEEP_K4_STANDALONE_DENSITY`，P0/P1/P2 = 3/6/5。**

M185 的 RTL、sealed VCS/SVA、独立 fresh-seed 数值 miter 和同约束 28 nm 3 ns DC 均通过。
它解决了 M184 与 M183 不能直连的接口问题，并将关键路径从 `1.77 ns` 缩短到 `1.66 ns`；
但删掉 bank ID、prefix 约束和 28 组 pairwise comparison **没有带来 mapped area 收益**：
M185 为 `27,129.815772 um2`，反而比 M183 的 `27,031.031773 um2` 大 `0.365447%`。
因此不能把 M185 写成面积优化，也不能用它取代 M169 K4 的 standalone 吞吐密度点。

## 独立验证

- sealed VCS 输入、输出和 runner SHA 全部通过；主回归 481/481 issue/result、13/13 cover、
  0 assertion failure。
- review-only fresh VCS 使用 seed `185925`，不复用主 TB 的 stimulus/scoreboard：遍历全部
  `255/255` 个非空 bank mask，另加 256 个随机向量、`-128/127`、signed24 正负精确边界、
  正负 overflow、随机 backpressure、II=1 replace，并将 numeric 与 protocol fail-close 分离测试。
  结果 515/515 accept/retire、509 次 same-cycle replace、92 个 stall、13/13 原 SVA cover 非零，
  compile/sim/assert 均 0 error/failure signature。
- 数值范围独立核对：8 路 signed INT8 和为 `[-1024,1016]`，signed11 足够；与 signed24
  accumulator 相加的完整范围为 `[-8389632,8389623]`，signed25 足够；
  `extended_sum[24] != extended_sum[23]` 是正确的 signed24 overflow 判据。
- sealed DC 的 input/evidence/runner manifest 全部校验，五类 constraint clean，
  check-design/check-timing 均通过，0 macro、0 multiplier。

## 同约束 DC 裁决

| 项 | M170 K1 | M169 K4 | M183 generic K8 | M185 fixed-bank K8 |
|---|---:|---:|---:|---:|
| Cell area (um2) | 11,940.011991 | 18,522.881882 | 27,031.031773 | 27,129.815772 |
| Cells | 14,275 | 20,498 | 28,365 | 28,266 |
| Sequential cells | 2,341 | 2,343 | 2,344 | 2,344 |
| Logic levels | 9 | 38 | 42 | 40 |
| Critical path (ns) | 0.91 | 1.67 | 1.77 | 1.66 |
| Setup / hold slack (ns) | +1.6146 / +0.0221 | +0.8670 / +0.0224 | +0.7691 / +0.0224 | +0.8771 / +0.0233 |

M185 相对 M183 少 24 个 port bit、99 个 cell 和 2 层逻辑，关键路径缩短 `6.214689%`，
但 combinational/total cell area 增加 `98.783999 um2`。直接 RTL 差分给出了合理解释：M183
的 legal prefix 保证 slot0 必有事件，所以第一路不需 valid gate；M185 支持任意 mask，必须在
96 个输出 lane 上为 bank0 重新加入数据门控。另一方面，所谓被删除的“external packing
crossbar”从未包含在 M183 standalone DC 中，所以它的系统级节省不能由这组面积差证明。

## 性能与密度裁决

- M182 bounded K8 的 `97,607,807` analytic schedule cycles 相对 M179 K4 的
  `127,581,198` 为 `1.307079853x`；这是 exact-payload schedule ratio，不是 RTL/physical
  speedup。
- M185 是 M169 面积的 `1.464664945x`，所以 standalone K8/K4 条件性
  schedule-throughput/logic-area 只有 `0.892408777x`。M185 比 `24,210.885723 um2`
  break-even 高 `2,918.930049 um2`（`12.056271%`）。M169 K4 继续保留 standalone 密度点。
- 仅把两个 standalone block 相加：M180+M169 为 `32,940.809935 um2`，M184+M185 为
  `37,156.643801 um2`，K8 组合面积增加 `12.798209%`，但周期减少 `23.493580%`，得到
  `1.158777128x` 条件性 schedule-throughput/logic-area。这是目前最有价值的正结果，但它
  **不是 flat composition、不是 physical、complete-FC2、system 或 headline 指标**。

因此架构定位应是：M185 取代 M183 成为 M184 的 mask-native K8 算术接口与更短关键路径点；
M169 K4 仍是 standalone arithmetic density 点；M184+M185 只作为等待 flat composition
验证的候选组合点。

## 关键缺口

P0：

1. 把 M184、八 bank weight response/stall、M185、Acc24 context owner/writeback、BN2 与 residual
   commit 做成可执行组合；完成前不得称 complete FC2。
2. 对 PAFT 发布 checkpoint 做 12 个 `sn2` 的专用 threshold census 和 valid825。任何非精确 1
   的值都要求可验证的 folded-weight/量化桥或 multiplier path，冻结 ep35 的 threshold-one
   事实不能自动迁移到 PAFT。
3. 用冻结 representative/all-120 payload materialized group stream 回放组合 RTL，并固定周期
   区间；`1.307079853x`、`4.344533568x` 与 `1.158777128x` 均不得升级为物理、系统或 headline。

P1：

1. 明确承认 fixed-bank specialization 没有 M183 mapped area win；收益仅是接口匹配和 timing。
2. M185 standalone 未跨过 K4 density break-even，K4 不能被删除；若需要中间 Pareto 点，预提交
   K5/K6/K7 的 schedule+DC sweep。
3. 计入 6,144-bit/cycle weight payload、八 bank SRAM 地址/响应、read stall 与布线；当前 0 macro。
4. 跑 RTL-to-netlist Formality，并在组合后补 interface-level equivalence/conservation。
5. 补 macro-aware timing、route/CTS、SAIF/PTPX；当前 ideal clock、ZeroWireload 下还有 clock 和
   一个 2,343-load 高扇出网的 TIM-134 近似。
6. 修正 lineage 命名：127,581,198 与 97,607,807 来自 M179/M182 scheduler，不是 M169/M185
   arithmetic RTL 自己测得的 wall cycles。

P2：

1. 把 review-only 的 255 masks、极值与分离 fault 测试迁入 canonical exact-SHA regression。
2. 给 SVA 增加 `source_count==popcount(mask)`、result mask identity 与 accepted activity-mask 的
   关系型断言；当前主要由 TB scoreboard 检查。
3. 清理或逐行 waiver DC 的一处 VER-318 signed-to-unsigned truncation warning。
4. 下一版 DC runner 应对 break-even 双向接受再生成 verdict；当前 runner 硬编码
   `area > 24210.885723`，会拒绝意外出现的更优结果，不能称为中性的预提交判据。
5. 如需要统一 machine admission，新增 keyed-to-VCS/DC RUN_COMPLETE 的非破坏性 overlay，
   不改 pre-run contract。

机器可读裁决见 `m185_independent_hammer_review_r1.json`。本评审只写当前 review 目录；
`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
