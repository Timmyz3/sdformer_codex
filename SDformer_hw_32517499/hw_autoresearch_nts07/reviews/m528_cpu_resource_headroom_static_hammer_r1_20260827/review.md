# M528 CPU 重算资源门静态打铁

日期：2026-08-27  
裁决：**98/100，P0=0、P1=0；M528 专用 `64 GiB` commit-headroom 门过度保守，建议在一个全新 revision 中降为 `48 GiB`。本审阅不授权修改现有封存文件，也不授权 production run。**

## 1. 边界与身份

本审阅只读检查 M528 analyzer、runner、两级冻结 analyzer、execution contract、static admission、Python 3.10 `ProcessPoolExecutor` 实现及 `/proc`/cgroup 资源状态。未运行 M528 production analyzer/runner，未运行 EDA/GPU，未创建 canonical result 或 attempt sentinel，未修改任何既有源码、合同、runner 或 `docs/359`。

冻结身份：

- M528 analyzer：`c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a`
- M528 runner：`a31d891ab83a8c87fa98f31cabbc7a81174362ef9b4f469fe0a3220b80711531`
- M505 analyzer：`9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced`
- M504 analyzer：`9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e`
- execution contract：`910c804a9a9df13395ab4f6b2ef5988ea0dee56ab7e52a21f887fa8fe0d73a34`
- static admission：`96832f878b6be79dbc342aeb1758ed7deaca09d618f283e72df13ed0bc08f8d7`
- `docs/359`：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

审阅结束时，canonical result 与 attempt sentinel 均不存在。

## 2. 静态内存账本

### 2.1 主进程确定性数组

冻结 shape 是 `10 samples × 4 operators × 47 chunks × 432 partitions = 812,160` 元素。`FIELD_NAMES` 恰为 24 个，每项为 `int32`：

`812,160 × 24 × 4 = 77,967,360 B = 74.355469 MiB`。

后处理的 `astype(int64)`、sample/operator flatten 和 pipeline 临时数组逐项/逐样本产生，不会复制完整 24 组账本。给这些临时对象、JSON/CSV 行及主进程 allocator high-water 合计预留 `0.375 GiB`，已经远高于确定性 payload。

### 2.2 worker 不是整包 load

冻结行账本是 `466,560,000 B = 444.946289 MiB`。M504 `worker_init` 只执行 `os.open`；每个 `worker_phase` 用 `os.pread` 读取：

`3,000 rows × 9 B = 27,000 B/phase`。

三个 worker 不会各自载入 466.56 MB。顺序读最多把整文件形成约 `444.95 MiB` 可回收 page cache；这影响物理内存，不计入匿名 `Committed_AS` 峰值。每 worker 同时只处理一个 phase，47 个 64-row tile 顺序复用小数组和 Python list。

### 2.3 futures 与最大乱序返回

Python 3.10 `Executor.map` 会预提交 `17,280 / 2 = 8,640` 个 chunk futures；process call queue 上限为 `max_workers + EXTRA_QUEUED_CALLS = 4`。结果必须按输入次序消费，因此最坏情况下可有大量后继结果已完成而前项仍未完成。

每 phase 的原始返回 payload 为 `24 × 47 × 4 = 4,512 B`；即使把全部 `17,280` phase 同时滞留，原始 NumPy payload 仍为 `74.355469 MiB`。本审阅按 4 倍计入 ndarray/dict/pickle/future 容器，再额外计入 executor bookkeeping，合并进下述 `0.75 GiB` 主进程/IPC 档。

### 2.4 保守 commit 峰值上界

| 项 | 保守上界 |
|---|---:|
| 主数组、全量乱序返回、后处理与 IPC/futures | `0.75 GiB` |
| 4 个 Python/NumPy 进程的 interpreter、heap 与 allocator high-water | `1.00 GiB` |
| NumPy OpenBLAS `MAX_THREADS=64`，按 4 进程 × 64 线程 × 8 MiB 栈全部收费 | `2.00 GiB` |
| spawn/queue/shared-library private writable、碎片与未建模裕度 | `0.75 GiB` |
| 额外 fail-closed 余量 | `1.50 GiB` |
| **M528 commit 峰值上界** | **`6.00 GiB`** |

代码只使用 bitwise、popcount lookup、lexsort、sum/max 等 NumPy kernel，没有 BLAS 矩阵调用；`2 GiB` OpenBLAS 栈项是故意收费的极端上界。故 `6 GiB` 不是期望 RSS，而是禁止实跑测量条件下的 fail-closed 静态上界。

物理内存再加入至多 `444.95 MiB` 行账本 page cache后，增量仍小于 `6.5 GiB`。

## 3. 门限裁决

- 现门：`64 GiB = 67,108,864 KiB`，相对 `6 GiB` 静态上界为 `10.67×`。
- 建议门：`48 GiB = 50,331,648 KiB`，相对 `6 GiB` 静态上界为 **`8.00×`**。
- 不建议低于 `48 GiB`：共享机没有 user.slice 硬内存上限，其他用户的 `Committed_AS` 在只读审阅期间发生了数 GiB 波动；保留额外共享机波动余量比追求更低启动门更重要。
- `MemAvailable ≥128 GiB` 相对 `<6.5 GiB` 物理增量约有 `19.7×` 裕度。
- `SwapFree ≥32 GiB` 相对 `6 GiB` commit 上界有 `5.33×`，继续保留。

因此，**只针对冻结的三 worker、chunksize=2 的 M528 CPU workload，64 GiB 是过度门控；48 GiB 足够且仍满足至少 8× commit 安全裕度。**

## 4. 只读资源快照

2026-08-27 17:49:30+08:00 连续三次：

| snapshot | commit headroom KiB | MemAvailable KiB | SwapFree KiB | failcnt / under_oom / oom_kill |
|---:|---:|---:|---:|---:|
| 1 | 60,239,800 | 415,422,888 | 57,279,228 | `0 / 0 / 0` |
| 2 | 60,239,800 | 415,421,872 | 57,279,228 | `0 / 0 / 0` |
| 3 | 60,267,284 | 415,457,536 | 57,279,228 | `0 / 0 / 0` |

三点会通过建议的 48 GiB 门、不会通过封存的 64 GiB 门。当前 UID 的 `dc_shell/dc_shell-t/fm_shell/pt_shell/vcs/vcs1/vlogan/simv` 均为空。这只是资源观察，**不是 launch admission**。

## 5. 新 revision 的强制条件

现有 runner、execution contract 和 static admission 均由 SHA 封存，禁止直接改。若 root 接受本裁决，必须另起 revision 并重新独立静态评审；新 revision 只允许变更 commit-headroom 常量为 `50,331,648 KiB`，并保持：

1. `workers=3`、`chunksize=2`，禁止 override；
2. launch 前连续三次资源快照；
3. `MemAvailable ≥134,217,728 KiB`；
4. `SwapFree ≥33,554,432 KiB`；
5. `failcnt=under_oom=oom_kill=0`；
6. 当前 UID Synopsys/VCS/simv collision 全拒绝；
7. canonical/attempt fail-closed、exact-SHA、一次 attempt 和失败 quarantine 规则不变；
8. 新 runner/static admission 未双封通过前不得启动 production analyzer。

## 6. 评分与限制

- 正确性：`39/40`
- 资源模型完整性：`29/30`
- fail-closed 边界：`20/20`
- 可复核性：`10/10`
- 总分：`98/100`

P2：没有通过实跑采集 RSS/VMS high-water；这是任务明确禁止 production run 的结果。审阅用 `6 GiB` 上界和 `48 GiB` 门保留 8×，不得把该静态估算改写成实测内存数字。

