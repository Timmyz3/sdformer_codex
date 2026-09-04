# M504 r3 全量结果独立打铁

结论：**98/100，PASS_NO_GO_M504_RTL；仅 GO 一次 M505-PVRF 冻结离线审计**。

M504 r3 的执行、封存和算术账本可信，四个合同门独立重算为 `FAIL / PASS / PASS / PASS`。因此 `NO_GO_M504_RTL` 是正确裁决，不能因为仍保留 `1.6674x` 的局部周期倍率而偷换成 RTL 准入。M505 只获准做一次不改 RTL 的精确机会审计；若不能把同一四层 Conv 周期模型的单端口税降到 `<=5%`，立即停止这条线。

## 1. 身份与封存

| 对象 | 独立 SHA-256 | 结果 |
|---|---|---|
| r3 analyzer | `9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e` | 匹配合同与结果身份 |
| r3 contract | `64f1ac425520816af5250647d251c14a34e28a715723c98a50b4234b01bd9a5d` | 匹配 runner 与结果身份 |
| r3 runner | `15a0a4b6c45de15f133f76070090c086cefe6587a883ad1b28f69753ee87a0f9` | exact-SHA、拒绝覆盖 |
| result JSON | `a0d2234a3a660df42bb87be04d42085c6c19025e55bdc35a1d61b9c48a54634b` | 内层 manifest 匹配 |
| summary CSV | `265884d3af040a9d985066a471b71f6c0607c5f871c876726ddab98a4f94915e` | 内层 manifest 匹配 |
| result manifest | `f682a43c35847fa1fd2d9234bff9f225943ed582db7c65bb3590fb634b51212c` | 外层 seal 匹配 |
| docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | 未变 |

结果目录内执行 `sha256sum -c SHA256SUMS` 和 `sha256sum -c SHA256SUMS.seal.sha256` 均通过。结果 JSON 声明的六个冻结输入逐一重算为 0 mismatch。r3 preflight 的 JSON/Markdown/manifest/seal 也全部通过。

## 2. 人口、任务顺序与 M473 锚点

- 冻结人口：H67 ep35、10 samples、4 Conv operators、432 partitions、每 phase 3000 rows、`row_tile=64`。
- 每 phase 为 `ceil(3000/64)=47` 个 row chunk；任务数独立重算为 `10*4*47*432 = 812,160`，与 M504 和 M473 一致。
- analyzer 的数组布局和 phase 解码保持 M473 的 `sample -> operator -> row-chunk -> partition`，没有退回已否决的 partition/chunk 交换。
- 总 row 数为 `10*4*432*3000 = 51,840,000`；active rows 为 `27,305,568`，parent edges 为 `18,205,389`。
- M473 锚点三路一致：`product_cycles_without_commit 389,014,420 + commit 960,000 = 389,974,420`；M504 重建值、M473 SHA 冻结 selected point 和合同 `required_anchor` 均为 `389,974,420`。

CSV 有且仅有 40 个唯一 `(sample, operator)` 行，覆盖 sample 0--9 与 operator 0--3，无重复、无负数。CSV 的 ideal/work/deadline/stall/hold/edge/read/forward 九项合计与 JSON aggregate 0 mismatch；逐行满足：

- `ideal + work_stall = work_cycles`
- `ideal + deadline_stall = deadline_cycles`
- `deadline_hold = deadline_stall`
- `macro_read + forward = parent_edges`

## 3. 四门独立重算

| 门 | 阈值 | 重算值 | 裁决 |
|---|---:|---:|---|
| 单端口周期税 | `<=5%` | `(456,016,645 / 389,974,420)-1 = 16.9350%` | **FAIL** |
| 相对 M468 same-budget zero 保留倍率 | `>=1.50x` | `760,350,133 / 456,016,645 = 1.66737x` | PASS |
| 相对 exact-capacity DP fallback 面积下降 | `>=80%` | `1-78,825.2454/473,034.72 = 83.3363%` | PASS |
| 相对 DP overdepth proxy 面积下降 | `>=70%` | `1-78,825.2454/285,350.64 = 72.3760%` | PASS |

四门是合取关系，一门失败即不准 RTL。宏面积只是 9 个现有生成 128x128b 1RW 宏与 QRT DP fallback/proxy 的比较；preferred DP 和 integrated macro PPA 仍为 OPEN，不能写成最终芯片面积。

## 4. 16.935% 端口税由什么组成

一输出块的 ideal issue 为 `45,439,249` cycles，deadline schedule 为 `53,850,734` cycles，恰好多出 `8,411,485` cycles（issue-window 税 `18.5115%`）。这不是队列空等或算术变化：

- `macro_reads = 16,490,761`
- `concurrent_issue_reads = 8,079,276`
- `macro_reads - concurrent_issue_reads = 8,411,485`
- `deadline_holds = deadline_stalls = 8,411,485`
- `empty_queue_parent_stalls = 0`

所以每一个额外拍都对应一笔无法与算术 issue 重叠、必须占用 1RW 端口的 parent read。`1,714,628 / 18,205,389 = 9.418%` 的 parent edges 已被同拍 forwarding 消掉；剩余 macro reads 中 `51.007%` 仍要靠 hold 发出。

八 bank 的原始新增 issue 为 `67,291,880` cycles；pipeline 覆盖了 `1,249,655`，最终仍新增 `66,042,225` cycles，即全量四层 Conv 模型上的 `16.9350%`。deadline policy 相对 work-conserving 的 `469,975,780` cycles 已省 `13,959,135` cycles，但不足以过门。

该负结果跨样本稳定：十个样本的一输出块 issue-window 税在 `17.691%--19.539%`，不是单一 outlier。按 operator 聚合如下：

| operator | ideal issue | deadline issue | issue-window 税 | parent edges | holds |
|---:|---:|---:|---:|---:|---:|
| 0 | 16,155,942 | 18,857,753 | 16.723% | 6,181,396 | 2,701,811 |
| 1 | 5,822,215 | 7,201,917 | 23.697% | 2,534,999 | 1,379,702 |
| 2 | 17,509,795 | 20,296,284 | 15.914% | 6,704,842 | 2,786,489 |
| 3 | 5,951,297 | 7,494,780 | 25.935% | 2,784,152 | 1,543,483 |

这张表只描述一输出块的 issue window，不是 full-network 或 system speedup。

## 5. M505-PVRF：只准一次离线审计

裁决：**GO_FOR_ONE_FROZEN_OFFLINE_AUDIT_ONLY**。理由不是 M504 已通过，而是失败机制已经被定位为“final write 与未来 parent read 争用同一 1RW 端口”。对 dead-parent write elision 与 single-use immediate forward 做一次精确反事实审计，能够直接判断是否有足够的可释放端口拍；在此之前禁止 RTL。

当前 M504 有 `27,305,568` 次 active-row writes、`16,490,761` 次 reads，总 scratch macro accesses 为 `43,796,329`。另有未封存的代表性 sample0/operator0 观察提示 dead writes 可能较多，但该观察**不得进入结论或论文数字**；M505 必须在冻结全量 10x4 人口中重算。

### M505 最小建模义务

1. 完全继承 M504 的输入 SHA、M473 坐标、任务顺序、1-cycle sync response、两项 FIFO+pending 容量、producer validity、no consume credit 和 `389,974,420` anchor。
2. 每个 64-row task 在 matching 完成后精确生成 parent refcount；至少报告 `0/1/2+` 及完整最大值。硬件义务按 saturating `0/1/2+` 元数据计入，不能把 refcount 当免费 oracle。
3. dead-parent 仅定义为 refcount=0；其 final row 可抑制 scratch write，但算术 issue、psum commit 和 row 完成语义不变。
4. single-use immediate forward 仅在 refcount=1、唯一 consumer 恰为下一 active row、FIFO 顺序与容量合法时抑制 store。多用 parent 仍必须写入。
5. 单周期仍最多一个 macro read 或一个 macro write。若声称 forwarding 同拍再读第二条 edge，必须显式增加第二个 lookahead descriptor、保持响应次序并把该硬件计入后续 matched DC；否则离线模型只能消费一个 parent edge request/拍。
6. 分三档独立报告：M504 baseline、dead-write-only、dead-write+single-use-immediate-forward；禁止只报组合最优点。
7. 对每个 sample/operator 以及 aggregate 报：dead writes、unique referenced parents、refcount histogram、single-use/immediate 数、suppressed writes、macro R/W/total accesses、holds/stalls、issue-window cycles、完整 pipeline cycles。
8. 用状态 ID/断言证明 0 dropped/reordered/duplicated parent edge、0 read-before-producer、0 simultaneous macro R/W，且 arithmetic issue、active row、parent edge 和 commit 数与 M504 完全一致。
9. 输出 exact-SHA contract/analyzer/runner/result/CSV/seal，并再做一次独立 result hammer；审计本身仍标记 `rtl=false, system_speedup=false, date_headline=false`。

### M505 硬门

- **主门**：完整四层 Conv pipeline `cycles <= 409,473,141`，即相对 `389,974,420` 的端口税 `<=5%`。当前需从 `456,016,645` 至少回收 `46,543,504` cycles，约为当前新增周期的 `70.48%`；任何 one-block 代理都不能替代完整 pipeline 重算。
- 保留 M504 其余三门：相对 M468 zero `>=1.50x`、两项宏面积下降 `>=80% / >=70%`。
- 支撑门：相对 M504 的 `43,796,329` 次 macro accesses 至少下降 `10%`，否则 liveness 元数据不具物理化价值。
- **只有以上五门全过且独立 result hammer 通过，才允许讨论 RTL**。若主周期门失败，即使写访问下降很多，也只能作为 energy opportunity；在没有宏能量和 PTPX 前不得开发该 RTL。

## 6. Claim boundary 与最终裁决

- M504 r3 是可信的 exact CPU cycle fast-kill，不是 RTL/VCS/DC/Formality/PTPX、integrated PPA、full network、system speedup 或 DATE headline。
- `1.66737x` 是同一四层 Conv、相对 M468 same-budget zero 的局部模型倍率；不能与其他模块倍率相乘。
- M504 不再开发 RTL。只允许 M505-PVRF 一次离线审计，按上述门 fail-closed；失败即封线。
- 本审阅没有运行 VCS/DC/PT/PTPX/GPU，没有修改生产文件或 docs/359。

