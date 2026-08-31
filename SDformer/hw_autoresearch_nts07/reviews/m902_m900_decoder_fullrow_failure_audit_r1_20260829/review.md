# M902 M900 decoder full-row failure audit

## 裁决

**M900 的 100x host-runtime 门已真实失败，但 return 143 后的 `FileNotFoundError` 是 runner 进程控制竞态，不是 decoder 算法、冻结身份或内存耗尽。** 当前证据不准入 full-row exactness、512 MiB full-row state gate、decoder cycles、speedup 或任何论文数字。

- **NO-GO**：不得恢复或重用已消费的 M900 attempt；不得用新时限把 M900 的 100x 门改判 PASS；不得把 partial heartbeat 当封存结果。
- **CONDITIONAL GO**：若项目仍需要取得一行完整 exact 诊断，可新建独立 R2 namespace、fresh release 和 fresh final hammer，以保守的可运行 timeout 完成一次非生产测量。R2 的科学结论只能是测得的 host runtime 与 exact aggregate；所有硬件、系统和论文 claim 继续为 false。

## 第一性原理根因

M900 在后台调用 shell function `m900_driver_env`，所以 `$!` 是 function 的后台 subshell，而不是实际 Python worker。monitor 在第三个 over-runtime 样本后只向该 wrapper 发 `SIGTERM`：

1. wrapper 被终止，`wait wrapper` 返回 **143 = 128 + SIGTERM(15)**；
2. `/usr/bin/env ... python3.10` 后代仍在运行；
3. EXIT trap 看到失败，将 private stage 原子移动到 `.partial_artifact`；
4. orphan Python 仍按旧 stage pathname 更新 heartbeat，父目录已不存在，于是封存的 stderr 出现 `FileNotFoundError`。

有界复现得到 `broken_wrapper_rc=143 orphan_after_wait=yes`；把 `env ... python` 直接放到后台后，`wait` 返回时 worker 已不存在。M900 snapshot 中所谓 `child_rss_kib=2152` 也只是 wrapper RSS；真正 Python 的 heartbeat `ru_maxrss` 已到 **1,896,264 KiB**。因此同一进程树错误同时污染了 terminate/reap 和 child-RSS 观察，但没有改变冻结事务身份。

## 9.3208 秒为什么不能作为 R2 的运行安全时限

`9.320783571 s = 932.0783571209759 / 100` 是待验证的 **100x 性能假设阈值**，不是由 M896 bounded measurement 推导出的安全 timeout。M899 已明确写着 `full_row_runtime_100x_gate_passed=false`。M900 到 11.413 s 时仍处于 `BUILD_COMPRESSED_RUN_IR`，只观察到 1,572,864 / 9,582,057 compressed transactions；因此 100x 假设已经失败，重跑不能改变这个事实。

若目的改为完成 full-row exact/scalability 诊断，R2 应把 acceptance threshold 与 operational timeout 分离：

```text
expanded_ratio   = 38,672,612 / 100,000 = 386.72612
compressed_ratio = 9,582,057 / 24,852   = 385.5648237566
bounded_time     = max(3.51 s independent hammer, 3.44 s author) = 3.51 s
scaled_time      = bounded_time * max(ratios) = 1,357.4086812 s
safety_timeout   = ceil(2.0 * scaled_time) = 2,715 s
```

2.0 倍只是一项保守的 host-execution guard，不是加速数字。1 秒 monitor 连续三个 over-time 样本后终止，故墙钟最迟约为 threshold + 3 个采样周期。完成后仍须单独比较实际 elapsed 与 932.078 s；`elapsed <= 9.3208 s` 的 100x 条件已经由 M900 失败证据封死，不得在 R2 复活。

## 身份、算法与内存判断

- M896 source、M900 driver/runner/release、M899、M901、consumed attempt 和 failure receipt 的 SHA/双层 seal 均复核通过；`docs/359` 仍是 `dedde7ce...`。没有身份漂移。
- M899 的 real-100K exact miter 仍成立。M900 没走到 `SCHEDULE_RUN_GTLS`，所以没有观察到 full-row mismatch；也同样没有证明 full-row exactness。结论是 **algorithm mismatch not observed, full-row algorithm not verified**。
- 11 个 snapshot 的 `over_resource=0` 且 `over_counted_state=0`；MemAvailable 仍约 395 GiB、commit headroom 约 109 GiB，没有 OOM/资源门失败。
- counted scheduler state 始终为 `NA`，因为 schedule 尚未开始。M899 的 470.096 MiB 只是 100K 线性投影，M900 没有实测确认 512 MiB full-row gate。
- `.partial_artifact/runtime_heartbeat.json` 本身未被 failure receipt 双封，仅作为与已封 traceback/snapshot 一致的 forensic 辅证；不能单独形成 authority。

## R2 最小修复合同

1. **新 attempt 身份**：使用全新 `...runtime_gate_r2...` result、attempt、stage、quarantine prefix；显式把 M900 consumed attempt 和 failure SHA 作为只读前序证据。不得删除、恢复、覆盖或别名复用 M900 namespace。
2. **目标重命名**：R2 是 `full-first-row exact/scalability diagnostic`，不是 `100x runtime gate retry`。`production/full_population/decoder_complete/cycles_or_speedup_citable/system_speedup/energy/paper_ppa_ready/paper_citable=false`。
3. **时限分离**：`scientific_100x_threshold=9.320783571 s` 仅记录为 M900 已失败假设；`operational_safety_timeout=2715 s` 按上述 100K population scaling 推导，三次采样 grace 单列。
4. **无 wrapper worker**：正式 worker 必须直接启动为可追踪 PID，或由 `setsid --wait` 建立独立 process group。禁止把 shell function 的 `$!` 当 Python PID。
5. **终止与回收顺序**：超门后先向整个私有 process group 发 TERM，有限 grace 后向仍存活的 group 发 KILL；reap supervisor/worker，并确认 group 不存在；然后才允许 stage rename、failure sealing 和日志删除。EXIT/signal trap 也必须先执行同一 drain routine。
6. **正常完成顺序**：worker exit 0 且已完全 reap → 检查 heartbeat/diagnostic → 复制 snapshot → seal private stage → no-replace publish。任何一步失败都走 drain-before-quarantine。
7. **资源观察**：RSS 若保留，只能读取直接 Python PID或聚合 process group，继续 diagnostic-only；system MemAvailable/commit/disk gate保持独立。counted-state 只认 worker 在 schedule 后写出的字段。
8. **fresh hammer**：R2 source/release 必须先做无 full-row 的 bounded race/TERM/KILL/normal-exit attacks，再由不同 reviewer 封 final launch；最后仍只准一次 R2 attempt。

## 最终 GO / NO-GO

**NO-GO for M900 100x retry；CONDITIONAL GO for one fresh R2 exact/scalability diagnostic。** R2 只有在上述进程回收、独立 namespace、2715 s safety timeout、claim 全 false 与 fresh hammer 全部落地后才可执行。否则维持 decoder full-row FAIL_CLOSED。
