# M598｜M594/M579 r3 fresh source-static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_SOURCE_HAMMER__NO_FORMAL_CPU_GPU_EDA_REMOTE_RUN`  
裁决：**PASS_SOURCE_STATIC，98/100，P0/P1/P2 = 0/0/1。**

## 一、裁决

M594 r3 已关闭 M592 的 `2 P1 + 1 P2`：同一份 execution-contract bytes 与 runner bytes 在 validator、
production result binding、terminal 和 pre-publish 均由 launch SHA 约束；execution input 不再是“所有已声明项”，
而是冻结的精确 15-key path/SHA 集；EXIT/INT/TERM/HUP trap 在第一个 canonical mutation 前安装，失败 attempt
与 staging 进入同一个 `RENAME_NOREPLACE` quarantine，并生成 failure receipt、member manifest 与 outer seal。

独立执行 exact runner `--preflight-only` 通过：Python 3.10.16、NumPy 2.0.1、spawn child、M43/M504/M505
导入和八行 recurrence 均正确，得到 ideal issue=6、liveness=8；前后正式 result/attempt 均不存在。
另以临时 v3 execution contract 执行 full validator，只重哈输入而不运行正式 record：15/15 required input 和
80/80 packed payload 通过，`formal_trace_records_processed=0`。少 key、多 key、历史 path/SHA 漂移、r3
analyzer/runner path/SHA 漂移、错误 start SHA、错误 runner SHA，以及“语义相同但 bytes 不同”的合同均 fail-closed。

因此本轮允许 root **另建** `launch_now=false` execution candidate 并再走 candidate/release review；本 source
hammer 不授权 execution release，也不授权 80-record CPU production。PAFT 仍无 accuracy/performance Pareto，
本轮没有生成任何性能结果。

## 二、评分

| 维度 | 得分 | 满分 | 结论 |
|---|---:|---:|---|
| immutable source/dependency identity 与 r3-only wrapper | 20 | 20 | r1/r2/M43/M504/M505 与 r3 tuple 全部 exact-SHA |
| runtime/spawn/task order/15 inputs/80 payload preflight | 25 | 25 | 实际执行通过，零正式 record |
| contract/runner start-to-terminal bytes identity | 20 | 20 | entry/exit/result/terminal/pre-publish 全闭合 |
| attempt/quarantine/result 原子状态机 | 18 | 20 | trap、双封、NOREPLACE 闭合；保留一项 dangling-symlink P2 |
| M255、容量与 claim boundary | 15 | 15 | 三口径、64 帧退化、九行 213,376 B、全 false 边界正确 |
| **总分** | **98** | **100** | **PASS_SOURCE_STATIC** |

## 三、M592 findings closure

### M592-P1-1｜开始/终端 execution-contract identity：CLOSED

- runner 在 validation 前记录 live `CONTRACT_SHA_START` 与 `RUNNER_SHA_START`。
- analyzer 在 validator entry/exit、r3 result binding、terminal entry/exit 均要求当前 contract/runner SHA 等于
  start；result 与 terminal receipt 均携带相同 start identity。
- runner 在 validation 后、production 后和 final rename 前再次直接重哈 contract/runner，同时重验 Python 与
  analyzer frozen SHA。
- 临时 fault matrix 证明：旧 launch SHA 配语义等价但重排/压缩过 bytes 的合同立即报
  `execution contract changed`；注入 validator 内中途换 bytes 也在 validator exit 被拒绝。

### M592-P1-2｜trap-before-mutation 与失败双封 quarantine：CLOSED

- `trap cleanup EXIT` 和 INT/TERM/HUP trap 均位于 canonical attempt `mkdir` 之前；validation、坐标冻结和
  collision precheck 都是只读操作。
- marker 写失败、production/terminal/seal/pre-publish/publish/attempt-seal/consume 任一非零退出，cleanup
  按 `STAGE` 收口；attempt 与 staging 用 `RENAME_NOREPLACE` 搬进同一 unique container。
- failure receipt 含 exit、stage、signal、contract/runner/analyzer start/current identities 与 final-result-exists；
  quarantine 对全部成员生成 `SHA256SUMS` 和 `SHA256SUMS.seal.sha256`，final quarantine 亦 no-replace。
- success 路径先 terminal rehash 和 result 双封，再 pre-publish identity check、result no-replace；成功 attempt
  单独生成 completion、member/outer seal，最后 no-replace 消费。
- 独立调用 runner 所用的 Linux `renameat2(RENAME_NOREPLACE)` 对已有 target 返回 `EEXIST`，source/target
  内容均未覆盖。

### M592-P2-1｜精确 required input set 与 terminal runner identity：CLOSED

- analyzer 冻结 13 个历史依赖和 2 个 r3 runtime source，共 15 个精确 key；`set(inputs)` 必须完全相等。
- 13 个历史项逐项比较冻结 path/SHA；r3 analyzer/runner 逐项比较冻结 path 与 live SHA，并与 top-level SHA
  相等。
- full validator 实际重哈 15 inputs 与 80 payload；terminal 复用该 validator，并直接重哈 runner，不依赖
  optional input key。

## 四、P2 finding

### M598-P2-01｜canonical absence predicate 没有显式区分 dangling symlink

runner 的初始 result/attempt/consumed/staging 检查使用 shell `-e`，failure receipt 的 result observation 使用
`Path.exists()`。dangling symlink 会被这些 predicate 当作 absent。后续 `RENAME_NOREPLACE` 仍会因目录项已存在
而拒绝发布，所以该缺口不会覆盖目标、不会产生 false PASS，也不阻塞本轮 source-static admission；但它可能让
正式运行在最终发布才发现碰撞并浪费一次 attempt。建议 execution-candidate overlay 或下一版 runner 对所有
canonical 坐标增加 `-L`/`lexists` 与 no-symlink 检查，并把该状态写入 failure receipt。

## 五、冻结计算、accuracy、容量与 claim boundary

- r3 的 `worker_init`、`spawn_probe`、`analyze_record` 都只委托 exact-SHA r2；r2 再绑定 exact-SHA r1、M43、
  M504、M505。r1 每 task 只调用一次 `simulate_liveness_task(tile, False)`。
- r1 的七个 partition-major cost array 仍由 r2 执行 `(432,47).T.reshape(-1)`，执行顺序为
  `[sample, operator, row-chunk, partition]`；anchor `[0,47,94,141]`、20,304 tasks/operator、末 chunk 56 rows。
- DMA=160、tail=2、commit=96,000/sample、8 output blocks 未改变；r3 没有新的支持算术路径。
- M255 三口径同时保留：valid825 control/PAFT AEE=`1.4776177362723637/1.4691506688129696`，单 seed
  +0.5730215096601543%；硬件十帧 `1.0035290166/1.00114762892`，5 win/5 loss；完整 64 帧
  `1.351884619446875/1.36565899929375`，PAFT **退化 1.0189020311889285%**。无共同 evaluator runtime SHA、
  无多 seed，`accuracy_performance_pareto=false`。
- M528 容量账精确九行、合计 213,376 B，240 KiB 预算 245,760 B，余量 32,384 B；只证明 capacity fits，
  macro integration/PPA/energy 仍 open。
- arithmetic-work、local-cycle、PAFT/control activity increment 不相乘；numeric Conv/Acc24、RTL、VCS、
  Synopsys PPA、integrated macro、energy、decoder complete、system speedup、headline 均为 false。

## 六、执行边界与下一步

本 review 没有运行正式 analyzer/runner、80-record CPU、GPU、EDA 或远程，没有创建 execution contract、
result、attempt、launch candidate 或 release。`docs/359` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

唯一允许的下一步：root 另建 `launch_now=false` execution candidate，显式绑定本 review 和 M594 exact source
SHA；candidate/release 必须独立评审，P2 dangling-symlink hardening 应在正式 launch 前关闭或形成显式 overlay。

