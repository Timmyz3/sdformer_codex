# M262 FC1 descriptor lifecycle author milestone

M262 把 M230 留在模型外的 FC1 tile 生命周期做成了一个真实可执行的 Synopsys VCS wrapper。当前几何是 `8 lanes × 8 contexts × Acc19`，不是 96-lane 完整实现。

## 已经执行的闭环

- 空 tile 在 header/done 同一握手原子 bypass，不访问 factor、weight 或 Acc 存储。
- 非空 tile 显式执行八个 Acc19 初始化写、tagged factor 取数、tagged INT8 weight 取数、tagged Acc read-modify-write、八拍 commit 和 done。
- factor、weight 和 Acc response 都检查 `tag/epoch/descriptor` 及端口专属身份；stale response 进入 sticky abort。
- Acc19 溢出在 write handshake 前隔离，commit 不可见。
- factor、weight、Acc read、Acc write、commit 和 abort 六类 backpressure 都有非零 VCS cover。
- 每个无 stall descriptor 的实测周期严格为 `6 + 3×popcount(context_mask)`。

Directed VCS 结果为：5 tiles、1 empty bypass、18 retired descriptors、14 个 clean cycle equality、32 commit beats、4 个负向攻击，numeric/transaction/assertion mismatch 均为 0。

## 冻结 trace 的同端口比较

M230 的 raw 100-record population 被零 mismatch 映射到同一个小宽生命周期。每个冻结 96-lane output block 串行执行 12 个 8-lane slice；没有把 8-lane RTL 冒充 96-lane RTL。

| 模式 | 8-lane serialized lifecycle cycles | 相对 dense | 相对 bit-sparse |
|---|---:|---:|---:|
| dense | 798,024,960,000 | 1.000000× | — |
| bit-sparse | 110,840,148,144 | 7.199783× | 1.000000× |
| context-factorized | 66,282,442,128 | 12.039764× | 1.672240× |

Context-factorized 相对 bit-sparse 的 weight request reduction 是 `2.580060×`；100 records 的逐记录 cycle speedup 最小/均值/最大为 `1.477213× / 1.657647× / 1.798352×`。

`12.039764×` 只表示 dense 连零 factor 也收费时的同端口模块生命周期比较。它不是完整 FC1、FFN 或系统加速。`1.672240×` 是更有用的增量指标，但仍是固定 latency、无 stall 的 trace mapping，不是 full-trace VCS 或物理 SRAM 结果。

## 尚未完成

- 96 output lanes 的实现与 VCS。
- 数十亿 descriptor 的 full-trace RTL replay。
- SRAM macro latency、冲突、容量和地址级执行。
- DC、STA、SAIF/PTPX；遵守 M237 的工具所有权，本里程碑没有启动新 DC。
- 完整 FC1/FFN/system/headline admission。

本目录是作者里程碑，必须由不同 agent 独立打铁后才可晋级。`docs/359` 未修改。
