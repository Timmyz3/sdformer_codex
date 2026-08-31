# M528 r2 单口同账本重算独立静态锤审

## 裁决

**98/100，P0=0、P1=0、P2=3；允许 root 签发一个全新的、双封存的 r2 static admission。** 本审阅本身不是 launch admission，不授权直接启动。全程只读：没有运行 production analyzer/runner，没有 CPU production、EDA、GPU 或 RTL 动作，也没有修改 author 包或 `docs/359`。

冻结身份：r2 runner `36152576c07f8da496af99b2632a11ebfe04be2a00bc913e55b6f73ae866d386`，r2 execution contract `fc0c3aee93d4055f0f1feda8268009d82d957c4b4d0adf5111ad8464122a95e2`，未变 analyzer `c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a`，author handoff outer-seal file `7c9fbbc8d5b13a6d27c2a9b5ef072c0cbb3e313f144013558d09b30d35bf8f29`。所有 SOURCE 条目、内封和外封均通过。

## r1 P0 已彻底修复

r1 的确定性失败来自三份根日志被 `cp` 进 `result` 后仍留在 work root，导致 canonical 已创建后 `rmdir` 必败。r2 的正常路径现在是：

1. analyzer 成功并先验证 analyzer 自身双封；
2. `production_stdout.log`、`production_stderr.log`、`resource_preflight.log` 三者全部用 `mv` 移入 `result`；
3. 加入 static admission、输入身份与 `RUN_COMPLETE.txt` 后重新生成并验证双封；
4. 用 `find` fail-close，确认 work root 除 `result` 外无任何残留；
5. `mv -T result canonical` 原子提交；
6. 紧接着、没有任何可失败命令插入地置 `m528_canonical_committed=1`；
7. 此时 work root 必为空，`rmdir`、`m528_complete=1` 和最终 PASS 均可达。

EXIT trap 只有在 `m528_complete!=1`、`m528_canonical_committed!=1` 且 work 仍存在时才会打失败标记并 quarantine。canonical 一旦提交，trap 不再写入、移动或标记它。因此 r2 同时修复了 success-path 不可达问题，并避免 post-commit 失败破坏已经提交的 canonical。

## 资源门与旧 admission 隔离

48 GiB 的精确门为 `50,331,648 KiB`，是已封存 6 GiB 保守 commit 峰值上界的 8 倍。r2 只把原 64 GiB 门改成此值；以下门均原样保留：`MemAvailable >= 134,217,728 KiB`、`SwapFree >= 33,554,432 KiB`、三次 launch 前快照、`failcnt/under_oom/oom_kill=0`、`workers=3`、`chunksize=2`，以及当前 UID 的 Synopsys/VCS/simv 精确进程冲突拒绝。

18:02:19 的只读快照为 commit headroom `55,472,148 KiB`、MemAvailable `412,072,660 KiB`、SwapFree `57,278,716 KiB`，OOM 三项全零，列出的 EDA/sim 进程全零；它通过 r2 动态资源门，但只是一张非授权快照，真正启动仍必须由 runner 连续采三次并全部通过。

封存 r1 admission 无法复用：它的 schema/status、runner SHA、execution-contract SHA、author outer-seal、canonical/attempt 身份和 commit 门均与 r2 不同。r2 runner 同时检查 schema `m528_single_port_same_ledger_static_admission_v2` 和 status `AUTHORIZED_ONE_M528_R2_CPU_PRODUCTION_RUN`，所以 r1 admission 在进入资源门或 Python 前即失败。

## 语义与结果身份

Analyzer SHA、governing contract、冻结输入、sample-major 与 operator-isolated 粒度、统计聚合、cycle/traffic/capacity/conservation 门、CPU 决策门、输出 schema/文件名和 claim boundary 都未改变。把 revision 身份、cleanup 描述和资源审计字段正规化后，r1/r2 execution contract 的计算语义 diff 为空。

审阅时 r2 canonical、attempt sentinel、work 和 failure quarantine 候选均不存在。`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## P2 与下一步硬门

- 6 GiB 是保守静态上界，不是实测 high-water；论文或回执不得改写成实测值。
- 未变 analyzer 仍通过精确 sealed objects 与 cycle anchors 间接锁定 row64/B8/128 B/cycle/CAM64；result hammer 必须再次显式核对这些坐标。
- traffic CSV 没有冗余列出同坐标 bit 行；其 weight/source/DMA/commit 等同性由 sealed M473 对象提供。若进 paper-facing energy ledger，应在未来 revision 显式补行，而不是改本次冻结 analyzer。

Root 现在可以签发**恰好一个**新的 r2 admission；它必须双封并钉死上述 runner、analyzer、r2 execution contract、governing contract、author outer seal、`docs/359` 和五项 runtime 值。生产运行成功后仍只是 raw one-sequence/four-Conv CPU 结果，必须另做独立 result hammer；当前没有 RTL/PPA/energy/full-network/system speedup/DATE headline 准入。
