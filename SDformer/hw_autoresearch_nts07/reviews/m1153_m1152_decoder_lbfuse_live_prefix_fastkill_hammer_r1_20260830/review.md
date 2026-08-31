# M1153｜M1152 decoder LB-FUSE 冻结前缀独立打铁

结论：**PASS 独立复核；KILL LB-FUSE 性能机制，不开 RTL，保留现有 A1-OSG 调度。**

## 独立重算

本评审只读 live JSONL 的冻结前缀，没有读取第 22 行之后作为证据，也没有停止或修改 producer。前缀严格为 21 行、67,751 B，SHA256 为 `584b47f2b74dc877dac22084283ea9f028387c2d1eb86e045dac573ad11d98c0`。逐行校验 canonical JSON、call/cycle/transaction 连续性、依赖投影、配置与 claim boundary 后，独立求和如下：

| 项 | 独立值 |
|---|---:|
| calls | 21（5 个完整 sample + 1 个 D0 call） |
| diagnostic cycles | 628,231,055 |
| psum read | 45,226,030,752 B |
| psum write | 45,226,030,752 B |
| psum RMW（read + write，各计一次） | 90,452,061,504 B |
| dense output commits | 5,568,000 |

资源从冻结 runner contract 独立取得：96 lane、Acc24、240 KiB 总片上 SRAM、其中 psum 221,184 B、六 bank 1RW、48 B/bank row、读 2 cycle、写 1 cycle。M722 的公平 A1 权威同时证明其片外 psum spill 为 0。

因此，把相同 source-order psum 状态改放到三行 buffer 只改变地址和 lifetime；每个 exact contributor 仍需同一次 on-chip read-compute-write，并被同一 1RW 端口串行化。公平候选的可执行下界仍是 628,231,055 cycle，故 baseline/candidate speedup 上界为 **1.000000×**；片上 psum byte reduction 与片外 spill reduction均为 **0%**。

## 容量复核

| 层 | 3 行 Acc24 × 96 | 240 KiB | 221,184 B psum partition | 3 行 Acc24 × 48 |
|---|---:|:---:|:---:|---:|
| D0 | 34,560 B | fit | fit | 17,280 B |
| D1 | 69,120 B | fit | fit | 34,560 B |
| D2 | 138,240 B | fit | fit | 69,120 B |
| D3 | 276,480 B | no | no | 138,240 B |

D3 的 48-channel split 虽可装下，但必须做两次 source pass；若 descriptor 不额外保留，冻结前缀会新增 **1,332,936,320 B** input-descriptor read，因此不能继续声称同一 96-lane 吞吐。D3 Acc16 的 184,320 B 只说明容量能装下；本 21-row 前缀没有 numeric admission，必须保持 `false`。

## 攻击与论文边界

独立 checker 共通过 367 项检查，并拒绝六类受控攻击：弱基线、把 on-chip RMW 误标成 off-chip、RMW 重复计数、把 partial 当 full、无证明启用 Acc16、授权修改 live producer。

该结果仅可作为 model-labeled、H67_ep35、partial-prefix 的负消融。不得升级为完整 decoder、最终 checkpoint、性能 headline 或系统倍速，也不得据此启动 RTL/VCS/DC/PTPX。若未来要减少 RMW，必须真正改变 destination packing/执行顺序，并与 A1-OSG/PIDP 做同资源对比；不能再拿 unstriped spilling A1 当弱基线。
