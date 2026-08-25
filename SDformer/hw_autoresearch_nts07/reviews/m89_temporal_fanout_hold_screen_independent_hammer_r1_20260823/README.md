# M89 temporal fanout / hold screen 独立打铁评审

## 结论

M89 的十份选定日志、冻结 M43/M53 基线及 receipt 算术已经独立复算通过：

- `P0=0 / P1=6 / P2=4`，综合 **81/100**；
- **GO**：日志级筛选数字、K4 policy 排名、K5–K8 排名、结构宽度和容量算术、
  明确标注为 composition 的两种比值及 PAFT 缺口；
- **NO-GO**：trace-level 独立事件重放、K6 等面积最优、RTL/VCS 倍速、宏/PPA/能效、
  PAFT 已达 2x、全网/系统及 DATE headline。

评分分项为：硬件创新 62、性能优势 69、证据质量 84、筛选里程碑完整度 91、
DATE 完整度 47，综合 81。

## 独立复算

评审脚本不 import 或执行 M89 builder/probe；它直接解析 receipt 选定的十份 raw log，
验证每份正好有 `1..40` 条互异 record marker，并从 M43/M53 冻结结果重建全部算术。
它能证明日志内部账本和 receipt 一致，但**不能**证明日志确由所列源程序运行产生，因为日志本身
不带生成器 SHA、命令、Git commit、环境或完整输出身份。

K4 policy 的 integrated-cycle 排名为：

| 排名 | policy | source cycles | hold cycles | integrated cycles |
|---:|---|---:|---:|---:|
| 1 | K4_NOHOLD | 72,089,712 | 0 | **78,803,200** |
| 2 | K4_ONLY_THREE | 71,581,896 | 636,544 | 79,006,656 |
| 3 | K4_ONLY_TWO | 71,334,128 | 999,656 | 79,032,056 |
| 4 | K4_ONLY_ONE | 70,791,776 | 1,976,456 | 79,097,976 |
| 5 | K4_TWO_OR_THREE | 70,585,576 | 1,957,240 | 79,327,960 |
| 6 | K4_UP_TO_TWO | 69,786,880 | 3,324,856 | 79,506,024 |

这说明 hold 会降低 source work，却以更大的 hold/调度代价恶化 integrated cycles；
当前六个策略中 `NOHOLD` 确是正确筛选赢家。

K5–K8 排名为：

| 排名 | fanout | source cycles | overhead cycles | integrated cycles | p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | K8 | 69,609,240 | 6,728,112 | **76,337,352** | 7,814,384 |
| 2 | K7 | 69,744,144 | 6,723,240 | 76,467,384 | 7,827,736 |
| 3 | K6 | 69,964,176 | 6,713,144 | 76,677,320 | 7,843,680 |
| 4 | K5 | 70,391,112 | 6,701,792 | 77,092,904 | 7,874,872 |

K6 相对 K8 多 339,968 cycles，即 K8 比 K6 快 **0.44534948%**；K6 的三项
K-scaled 结构宽度都是 K8 的 75%，即少 25%。这使 K6 成为合理的下一 RTL 候选，
但不是已证明的面积性能最优。纯周期冠军仍是 K8，且 K6 阈值是在观察 DSE 结果后选出。

边际收益快速递减：K4 nohold→K5、K5→K6、K6→K7、K7→K8 分别只减少
1,710,296、415,584、209,936、130,032 cycles；而每一步结构宽度增加一份 K4 的 25%。

## 容量与宽度

位宽公式重算为 response=`21K+8` bit，按 64-bit 对齐、16 entries。K4/K5 每 entry
16B，combined=176,688B，headroom=17,040B；K6/K7/K8 每 entry 24B，
combined=176,816B，headroom=16,912B。

相对 16KiB headroom gate，K4/K5 只多 656B，K6–K8 只多 **528B**。
算术正确，但固定 176,432B subtotal 和 193,728B allowance 没有宏、ECC、padding、
端口、队列、控制和 K-scaled datapath state 的物理证明，因此只能叫 structural ledger。

结构宽度独立重算如下：

| K | accumulator paths | signed bank terms | atomic payload bits |
|---:|---:|---:|---:|
| 4 | 384 | 3,072 | 7,296 |
| 5 | 480 | 3,840 | 9,120 |
| 6 | 576 | 4,608 | 10,944 |
| 7 | 672 | 5,376 | 12,768 |
| 8 | 768 | 6,144 | 14,592 |

这些是线性结构宽度，不是 DC area、Fmax、功耗或 SRAM port 代价。

## 两种比值与 PAFT 2x 缺口

| candidate | local source-only / candidate integrated | equal-candidate-overhead composition | 达 2x 还需减 source cycles | 占 candidate source |
|---|---:|---:|---:|---:|
| K6 | 1.845198554x | **1.932749136x** | 2,578,308 | 3.685183% |
| K8 | 1.853416136x | **1.941552701x** | 2,230,856 | 3.204827% |

公式为：候选 source=`S`、候选 overhead=`O`、冻结 local source=`L`，假设 PAFT 只把
候选 source 减少 `r`，令 `(L+O)/(S-r+O) >= 2`，可得
`r=ceil((2S+O-L)/2)`。

这里有两个强限制：第一种比值拿 source-only baseline 对 integrated candidate，并非同口径执行；
第二种把候选 overhead 人工赋给 baseline，也只是 composition。PAFT 缺口还假定 `O` 和 `L`
完全不变。因此不能写成已实现 1.93x/1.94x，更不能写成已达 2x。

## P1

1. raw logs 不绑定生成器/transformed source SHA、命令、commit、环境或完整结果；只能重算
   log 算术，不能独立重放 simulator event decision。
2. 两个性能比值都不是执行过的、完整收费的同资源 baseline。
3. K6 knee 是事后基于 `<0.5%` 和结构宽度挑选，没有预冻结 objective。
4. K4→K8 结构宽度翻倍而对 K4 nohold 只快 3.129%；无 K5–K8 RTL/VCS/DC/PT/宏收费。
5. capacity 是 hardcoded subtotal，K6–K8 距 gate 仅 528B，没有 macro/ports/ECC 证明。
6. 只覆盖 H67 ep35 四个昂贵 Conv3x3、十个 valid825-internal window；无准确率、
   sequence-disjoint、PAFT checkpoint 或全网 composition。

## P2

1. 六份 K4 policy log 无 per-sample ledger，不能独立检查逐窗回归和 p95。
2. K5–K8 marker 省略 `fusion_hold`，builder 用缺省零补入 receipt。
3. 旧 nohold log 与选定 r2 log 同目录共存，虽有 SHA 可区分，但没有 supersession receipt。
4. 十个窗口的 nearest-rank p95 等于最大值，尾部统计能力很弱。

机器结果为 `m89_temporal_fanout_hold_screen_independent_hammer_review.json`；独立复跑入口为
`audit_m89_independent.py`，输出为 `m89_independent_recompute.json`。未修改任何 M89 producer、
receipt 或 remote log。
