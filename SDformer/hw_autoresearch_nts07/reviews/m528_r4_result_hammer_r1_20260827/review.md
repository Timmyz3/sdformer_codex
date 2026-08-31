# M528 r4 同账本 raw result 独立打铁（r1）

日期：2026-08-27  
角色：全新独立 result hammer；只读生产结果与冻结证据链  
裁决：**99/100，P0/P1/P2 = 0/0/1；准入 exact CPU 同账本候选，只授权 root 创建一个 bounded dead-write-only 1RW RTL author admission**

## 1. 结论

M528 r4 已把 M505 最像“错杀”的点重新放回正确分母。冻结 `row64 / B8 /
128 B/cycle / CAM64` 坐标上，dead-write-only 1RW 为 `435,293,339`
cycles：

- 相对可落地 M468 strong-zero `760,350,133` cycles：`1.7467534301x`；
- 相对同坐标 bit `757,946,784` cycles：`1.7412322131x`；
- `213,376 B <= 245,760 B`，宏舍入容量余量 `32,384 B`；
- 相对冻结 dead-only 锚点 0 cycle 回归；
- 身份、人口、算术、parent edge、completion 和 commit 守恒全部通过。

因此旧 M505 hammer 用未物理化 concurrent-1R1W ceiling 的 `<=5%` 距离作合取
拒绝门，确实会把可落地单口点错杀。M473 的 `389,974,420` cycles 仍应报告为
诊断 ceiling；它不再是拒绝单口实现的分母。

这不是直接 RTL 准入。当前只允许 root 新建一个独立、双封的 RTL-author
admission，冻结唯一结构：`dead-write-only 1RW parent scratch + 现有 same-address
RAW forward`。本审阅不授权直接写/跑 RTL，不授权第二次 CPU、VCS、DC、GPU 或
任何论文 headline。

## 2. 身份与一次性执行链

结果 JSON SHA 为
`778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1`；
结果 outer-seal 文件 SHA 为
`02abbf7f9209d9a41d803c9942bfb43550be0d40945e3c094c1e457bda0db053`。
目录内所有 member seal 与 outer seal 均通过。

production attempt 的状态为
`CONSUMED_AT_FIRST_R4_CPU_PRODUCTION_LAUNCH_AFTER_PREFLIGHT`，一次性 sentinel 双封
通过，未发现第二次 attempt。生产 stderr 为 0 byte；pre-attempt spawn/schema
token 恰出现一次且 stderr 为 0。三次资源快照的最小值为：commit headroom
`59,675,876 KiB`、MemAvailable `415,205,904 KiB`、SwapFree
`57,265,404 KiB`，cgroup failcnt/under_oom/oom_kill 全 0；runner 在 attempt
之前执行了同 UID Synopsys/VCS/simv collision gate。

static review、preflight admission、preflight admission hammer、preflight receipt、
receipt hammer、production admission 与 production admission hammer 的双封、精确
PASS status/verdict、P0=P1=0 和身份绑定全部重新核过。所有受审 JSON 也通过重复
key 拒绝式解析。

`docs/359` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 3. 人口、CSV 与周期重算

- row ledger SHA 匹配，独立 `wc -l` 为 `51,840,000`。
- 人口为 `10 samples x 4 Conv x 432 partitions x 47 row chunks = 812,160`
  tasks；每 phase 的原始行仍为 3000。
- sample-major CSV 恰 10 行，六个周期列求和精确回到
  `760350133 / 757946784 / 389974420 / 456016645 / 435293339 /
  435293339`。
- operator-isolated CSV 恰 40 个唯一 `(sample, operator)`，与封存 M505 CSV 的
  M504/dead-only/combined 三列逐片交叉核验 40/40、0 mismatch。
- 每行 ratio 与 cycles 重新相除、JSON 中 mean/geomean/min/max/population-CV
  重新计算均通过。

sample-major 10 样本并非由单个异常点撑起来：相对 M468 的逐样本范围为
`1.7138300680--1.7807850800x`，population CV `1.1097%`；相对 same-bit 为
`1.7089390583--1.7744949870x`，CV `1.0883%`。10/10 样本均超过两个
`1.50x` 局部门。

operator-isolated 40 片只用于异质性：相对 M468/same-bit 的最小值仍分别为
`1.5336512127x / 1.5223833656x`，但每片都会独立启动/排空且不含 commit，**禁止
求和冒充 sample-major runtime**。

## 4. 守恒复核

- arithmetic：`42,806,256 residual + 2,632,993 exact-parent = 45,439,249`
  issues/output-block；8 blocks 为 `363,513,992`。
- parent edge：dead-only 与 combined 均为
  `16,490,761 reads + 1,714,628 forwards = 18,205,389`。
- completion：dead-only 为
  `9,947,701 writes + 17,357,867 dead elisions = 27,305,568 active rows`；
  combined 为 `9,703,355 + 17,357,867 + 244,346 = 27,305,568`。
- row-tile commits 为 `1,880`；committed accumulator vectors 与固定 commit cycles
  均为 `960,000`。

所有等式与 result JSON、冻结 M505 CSV 和 governing contract 一致。

## 5. 容量与 traffic 消融

M505 宏舍入账本为 `213,376 B`：包含 1RW parent scratch 的 9 个
`128x128-bit` 宏（物理 `18,432 B`，低 64 行承载逻辑 `9,216 B`）、descriptor、
mask、liveness、response/scheduler reserve、resident psum 及 ping-pong 义务。
相对 `240 KiB` 尚余 `32,384 B`。9 宏 slow-view 面积为
`9 x 8,758.3606 = 78,825.2454 um²`，单独报告，**没有换算成免费 SRAM 容量**。

parent logical on-chip movement 的独立消融为：

| 项 | M473 ceiling | dead-only 1RW | 变化 |
|---|---:|---:|---:|
| parent read bytes | 20,972,608,128 | 18,997,356,672 | -9.42% |
| parent write bytes | 31,456,014,336 | 11,459,751,552 | **-63.57%** |
| parent total bytes | 52,428,622,464 | 30,457,108,224 | **-41.91%** |

combined PVRF 再把 parent total 降到 `30,175,621,632 B`，但周期仍与 dead-only
完全相同，所以不得实现 combined PVRF。以上全是 logical access bytes，**不是
SRAM/DRAM energy，也不是 measured power**。

## 6. 后续唯一合法链

1. root 创建一个 bounded RTL-author admission，不得直接开 RTL；
2. admission-only 独立评审通过后，只实现 dead-write-only 1RW 一个结构；
3. Synopsys VCS/SVA 覆盖 read-XOR-write、1-cycle response、queue full、RAW
   forward、dead-store、backpressure、reset、wrong identity、overflow 和 completion；
4. trace recurrence 需在 `435,293,339` 的 1% 内，且相对两个局部基线均保留
   `>=1.50x`；
5. 集成 9 个真实宏视图后做 DC/STA/Formality、matched area-throughput、
   mapped-gate SAIF/PTPX 与宏/DRAM energy；logic-only 或 zero-macro 不能过门；
6. decoder-inclusive full-network Amdahl 与多 sequence 完成前，不能写系统倍速或
   DATE headline。

唯一 P2 是 SRAM mapping 仍为 `PARTIAL_FAIL_CLOSED`，生成宏 slow-view 面积可用于
本次容量/预估账，但 `.db`/macro integration 尚不在当前 repo 闭合。这不阻塞一份
RTL 规格的 author admission，却严格阻塞物理 PPA、energy 与论文 headline。
