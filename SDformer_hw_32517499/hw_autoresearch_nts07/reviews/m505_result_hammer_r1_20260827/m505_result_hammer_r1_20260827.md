# M505 liveness-aware 1RW parent scratch 全量结果独立打铁（r1）

日期：2026-08-27  
角色：独立 result hammer；不修改生产文件  
裁决：**98/100，PASS_PERMANENT_KILL_M505_PVRF_CONV_RTL；仅允许无 RTL 的宏能量敏感性分析**

## 1. 结论先行

M505 的结果包、冻结输入、40 行 CSV、三档消融和五道合取门可信。五门独立重算为：

`cycle=FAIL / retained-speed=PASS / exact-DP-area=PASS / proxy-DP-area=PASS / scratch-access=PASS`。

因此生产结果中的 `NO_GO_M505_RTL` 正确。M505 完整四层 Conv 周期为
`435,293,339`，相对 M473 理想 1R1W 的 `389,974,420` 仍有
`11.6210%` 端口税，超过 `<=5%` 上限 `25,820,198` cycles。这个失败不是
边缘误差，不能由表格舍入、流水边界或独立重算误差翻转。

`dead-write-only` 与 `combined_pvrf` 的 issue cycles、full-pipeline cycles、
holds 和 stalls 在 40/40 个 sample/operator 切片上全部相同。组合 PVRF 只比
dead-only 再少 `244,346` 次写：占 M504 原访问的 `0.5579%`，占 dead-only
访问的 `0.9242%`；**周期收益严格为 0**。所以 M505/PVRF 不能开发 Conv RTL，
也不得继续提出同族性能变体。允许的唯一后续是把 `40.1911%` scratch access
reduction 放进带 metadata、读写能量差异和 leakage 的宏能量敏感性模型；这不
授权 RTL，也不授权能量、系统倍速或 DATE headline claim。

## 2. 身份、封存与执行边界

| 对象 | 独立 SHA-256 | 结果 |
|---|---|---|
| analyzer | `9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced` | 匹配合同、结果身份与 runner |
| contract | `3c1e769fbb9f99e3b3bf50ee7d4658d62ae70aedcc736d5b5d59708f9b0bd5a5` | 匹配 runner 与结果身份 |
| runner | `80c11b9886290f8e64731c5654673b593f383cbb1d00cc10d7d43a9d23790c7e` | exact-SHA、3 workers、拒绝覆盖 |
| result JSON | `b8a29f2fafc0e7d051d66ed206cd5c25efb866d4a1ab02082aa71bad4b14eb61` | manifest 匹配 |
| 40-row CSV | `c776b6121fe2d23acbdb247a099ffd6ce4b33acbb4a8df642e3638cb4fe7cfbc` | manifest 匹配 |
| result manifest | `b1622a7190046101d905cc27baf7c09220f71c8182898ec65c5452f6649eba64` | outer seal 匹配 |
| M504 result | `a0d2234a3a660df42bb87be04d42085c6c19025e55bdc35a1d61b9c48a54634b` | 冻结锚点匹配 |
| M473 result | `a415f8474f3a351d123670c2d3691a6414f620e3d60848a9c51242802a6956e5` | 冻结锚点匹配 |
| docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | 未变 |

结果目录内 `sha256sum -c SHA256SUMS` 与 outer-seal 检查均通过；结果声明的
所有冻结输入重算为 0 mismatch。preflight 的 JSON/Markdown/manifest/seal
也通过。本 result hammer 额外记录 runner SHA，关闭 preflight 指定的封存义务。

本审阅没有运行 VCS、DC、Formality、PT/PTPX 或 GPU，没有修改生产文件或
`docs/359`。

## 3. 40 行 CSV 与人口独立重算

- CSV 恰有 40 个唯一 `(sample, operator)`：完整覆盖 sample 0--9、operator
  0--3；0 重复、0 缺行、0 负计数。
- 冻结人口为 `10 samples x 4 operators x 432 partitions x 47 row chunks`
  = `812,160 tasks`；原始行数为 `51,840,000`。
- 40 行合计与结果 JSON 的 active row、parent edge、三档 read/write、三档
  issue cycle、dead/combined hold/stall、refcount 和 forwarding 字段全部
  0 mismatch。
- 逐行 40/40 均满足：
  - `active = refcount_zero + refcount_one + refcount_two_or_more`；
  - `unique_live_parent = refcount_one + refcount_two_or_more`；
  - `dead_write + dead_elision = active`；
  - `combined_write + dead_elision + single_forward_elision = active`；
  - 与冻结 M504 CSV 联接后的 cycle、edge、macro-read、forward 0 mismatch；
  - `mode_issue - ideal_issue = mode_stall`。
- refcount 全量为 zero `17,357,867`、one `5,525,030`、two-or-more
  `4,422,671`，和为 active rows `27,305,568`；精确最大 refcount 为 `56`。

CSV 的 `*_pipeline_slice_cycles_no_commit` 是每个 operator 独立启动流水的
切片值，不能直接相加冒充 sample 内四 operator 连续流水。直接相加会比全量
sample-major 模型多 `2,612` cycles（M504）和 `2,604` cycles（dead/combined）。
这是 operator 边界 overlap 的报告粒度效应，不是全量结果错误；但论文表必须
使用结果 JSON 的 sample-major full-pipeline 值，并明确切片列不可加。

## 4. 三档消融总表

以下 issue-window 项均为一输出块计数；full-pipeline 项包含 8 个 resident banks
与固定 `960,000` commit cycles，仍只覆盖冻结四层 Conv，不是 full network。

| 指标 | M504 baseline | dead-write-only | combined PVRF |
|---|---:|---:|---:|
| parent macro reads | 16,490,761 | 16,490,761 | 16,490,761 |
| parent macro writes | 27,305,568 | 9,947,701 | 9,703,355 |
| parent macro accesses | 43,796,329 | 26,438,462 | 26,194,116 |
| forwarded parent edges | 1,714,628 | 1,714,628 | 1,714,628 |
| suppressed writes | 0 | 17,357,867 | 17,602,213 |
| deadline holds | 8,411,485 | 2,141,342 | 2,141,342 |
| total stalls | 8,411,485 | 5,777,543 | 5,777,543 |
| issue-window cycles | 53,850,734 | 51,216,792 | 51,216,792 |
| full-pipeline cycles | 456,016,645 | 435,293,339 | 435,293,339 |
| overhead vs M473 ideal | 16.9350% | 11.6210% | 11.6210% |

dead-write-only 回收 M504 的 `20,723,306` full-pipeline cycles，即 M504 原
`66,042,225` 新增周期的 `31.38%`；要过 5% 门还需再回收 `25,820,198`
cycles。它相对 M504 只有 `1.0476x`，不足以挽救单端口性能线。

## 5. 为什么 dead-only 与 combined 周期完全相同

这是当前合同下的真实结果，不是 CSV 漏计：

1. single-use store elision 只在唯一 consumer 是下一 active row、且该 parent
   已在 producer final beat 同拍 forward 时成立；
2. dead-only 在这一拍已允许 `store + internal forward` 并行，forward 不占
   parent macro read port；
3. combined 只去掉这次 store。单 lookahead descriptor 不能在同拍再服务
   下一条独立 read，所以空出的 1RW 端口不会减少一个 cycle；
4. 该 parent 只有一次 use，forward 后没有未来 macro read，`written=false`
   不会产生后续 stall。

独立枚举长度 1--5、mask 0--7 的全部 `37,448` 个小任务：combined 相对
dead-only 的 cycle 严格下降 `0` 例、回归 `0` 例；访问下降 `18,455` 例，且
恰与发生 single-use forwarded-store elision 的 `18,455` 例一致。冻结全量
40 行也全部满足 combined/dead-only cycle、hold、stall 相等。

因此 `244,346` 额外写抑制是纯访问/潜在动态能量项，不能被描述成 performance
optimization；为它增加 128-bit/task 的 2-bit refcount metadata 与控制逻辑在
没有 matched RTL/DC 和宏能量闭环前甚至不能声称净能量为正。

## 6. 五道硬门独立重算

| 门 | 阈值 | 重算值 | 裁决 |
|---|---:|---:|---|
| full-pipeline cycle overhead vs M473 | `<=5%` | `11.620998%` | **FAIL** |
| retained speed vs M468 same-budget zero | `>=1.50x` | `1.746753x` | PASS |
| 1RW area reduction vs exact DP fallback | `>=80%` | `83.336266%` | PASS |
| 1RW area reduction vs overdepth DP proxy | `>=70%` | `72.376005%` | PASS |
| scratch access reduction vs M504 | `>=10%` | `40.191069%` | PASS |

五门是合取关系。主周期门失败即 `rtl_nomination=false`；不能用四个 PASS 投票
覆盖一个 FAIL。宏面积仍只是 9 个已有 generated 128x128b 1RW 宏与 foundry
QRT DP fallback/proxy 的映射比较，preferred DP PPA 与 integrated macro PPA
继续为 OPEN。

## 7. Claim boundary 与永久裁决

- 可用：冻结 H67 ep35 四层 Conv 上的 exact CPU cycle/access fast-kill；三档
  消融；refcount 分布；生成 1RW 宏映射面积敏感性。
- 不可用：RTL/VCS/DC/Formality/PTPX、integrated macro PPA、full-network 或
  system speedup、能量节省、DATE headline。
- `1.746753x` 仅是 M505 四层 Conv cycle model 相对 M468 same-budget zero 的
  局部保留倍率，而且 performance admission 明确为 false。
- `40.191069%` 仅是 parent-scratch macro-access count reduction，不是 SRAM
  总流量、DRAM 流量、功耗或 energy/frame reduction。

最终裁决为：**在冻结 H67 DATE 主线上永久 KILL M505/PVRF Conv RTL，并停止
同族性能变体。** 若写作需要，可做一次不改 RTL 的宏能量敏感性：分别使用生成
宏的 read/write 能量，计入 128-bit/task metadata 生成/存取、控制翻转与 leakage，
并把结论标成 `[model]`。无论该敏感性结果如何，都不重新开放 M505 RTL。

