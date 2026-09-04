# Local5 密封 RTL Release 失败审计与 v5 准入

## 1. 裁决

本轮完成了 Local5 numeric RTL 回归的共享构建物闭包，但不改变 formal 状态：

```text
sealed numeric RTL release v5 = ACCEPT_FOR_SHARD_REGRESSION
single H3 numeric canary       = PASS
sample2 12-window shard        = PASS
formal G0                      = DENY
```

独立 DATE/RTL 风格复审对 v5 给出 **4.2/5，Conditional Accept**。该裁决只允许
复用 v5 扩跑 numeric shard，不代表 Local5 架构性能、full encoder 或 ASIC PPA 已经
闭环。

## 2. 为什么必须重建三次

| 版本 | 结果 | 发现的问题 | 证据处理 |
|---|---|---|---|
| v3 | Reject，2.5/5 | consumer 在密封源树生成 `__pycache__`，release 自污染 | 保留失败产物；sample2 H3 不计正式证据 |
| v4 | canary 失败 | miter 可执行依赖闭包漏掉 archive replay 的三项传递依赖 | DUT 的 H3 `PASS_WINDOW` 不计完整 canary |
| v5 | Conditional Accept，4.2/5 | 关闭上述 P0/P1，真实 H3 消费后仍可验封 | 允许扩跑 sample2 12 窗口 |

这里没有原地修改已发布目录。每次修复都生成新的 release manifest，使失败版本继续
可审计。

## 3. v5 密封合同

发布目录：

`results/local5_erep_numeric_rtl_release_v5_20260811`

manifest SHA256：

`c620cf6a33f1c9bbdb1c7d85ba0fa485580f8f578287850d08b7c6ee52939bf9`

v5 绑定：

1. 44 个 RTL、SVA、TB、adapter、金参考和 runner 源文件的相对路径与 SHA；
2. source tree 与 tar member 的精确集合，拒绝重复、额外文件和非普通文件；
3. H3/H6/H12/H24 四个 Verilator executable 及 SHA；
4. 每个 H-class 的精确 compile argv 和固定 compile CWD；
5. bash、C/C++、make、Python、NumPy、Verilator 等主要工具路径、版本和 binary SHA；
6. 真实运行时的 executable、输入、权重、坐标、seed 和实际输出路径；
7. `PYTHONDONTWRITEBYTECODE=1`、内存 `compile()` 语法检查和只读 source tree；
8. consumer 完成后再次从独立 CWD 验封。

v4 暴露的三项传递依赖已纳入 v5：

- `local5_erep_ledger_replay_v4.py`
- `local5_erep_capacity_baselines_v4.py`
- `local5_erep_command_schedule_v4.py`

## 4. 真实 H3 canary

结果目录：

`results/local5_erep_numeric_sample2_h3_canary_v5_release_20260811`

| 指标 | 结果 | 证据等级 |
|---|---:|---|
| sample/stage/block/window | 2/0/0/249 | `[prof]` |
| head | 3 | `[prof]` |
| DUT cycle | 691,588 | `[rtl]` 验证环境延迟，不是性能主数字 |
| Acc32 标量 | 43,200 | `[rtl]+[软件整数金参考]` |
| mismatch | 0 | `[rtl]` |
| max abs error | 0 | `[rtl]` |
| canary 后 release verify | PASS | `[rtl-build-provenance]` |
| `__pycache__` / `.pyc` | 0 | `[rtl-build-provenance]` |

数值边界是 `pre-bias/pre-BN/pre-requant/pre-residual Acc32`。收据
`canary_receipt_sha256.txt` 对 miter NPZ、miter report、消费后验封日志和
`canary_complete.json` 的 SHA 检查全部通过。

## 5. 能证明与不能证明

该 canary 能证明：固定真实窗口、真实 checkpoint 权重和固定随机 service seed 下，
H3 DUT 独立导出的 43,200 个 Acc32 与软件整数金参考逐值一致，并且 consumer 没有修改
密封 release。

该 canary不能证明：

- H6/H12/H24 真实运行正确；
- sample2 其余 11 个窗口或 100-sample formal archive；
- 多 seed、长期 stall、恢复路径和覆盖率闭合；
- bias/BN/requant/residual、网络输出或 full encoder；
- Local5 架构吞吐、能效、面积、ASIC PPA 或部署 FPS；
- formal G0。

## 6. 独立复审剩余问题

### P1

1. H6/H12/H24 尚无 v5 真实 runtime canary；
2. 只有一个 service seed；
3. `flock + staging + atomic rename` 尚缺并发创建、中断和残留 staging 动态测试；
4. Python 依赖检查仍是已知文件集合，不是通用 import graph closure。

### P2

1. 工具闭包不包含动态库、链接器依赖及完整 Python 环境镜像；
2. receipt 使用本机绝对路径，跨机器搬迁能力有限；
3. SHA 链是本地一致性证明，不是外部签名或可信时间戳。

这些问题不阻塞同机 sample shard 回归，但正式长期归档前必须保留边界。

## 7. 下一步准入

sample2 已同时满足以下条件，并记为第三个正式 `[rtl]` numeric shard：

1. 12/12 block/window 完成；
2. H3/H6/H12/H24 均有真实运行；
3. Acc32 标量总数严格为 1,987,200；
4. mismatch 和 max abs error 均为 0；
5. 聚合 NPZ 落盘后由独立 parser 重放一致；
6. 回归后 v5 release 再验封通过；
7. 继续标注 `formal G0 = DENY`。

正式规模更新为 3/100。该升级只关闭 numeric shard，不关闭 phase ledger 或 formal G0。
