# Local5 H24 紧凑 Phase 遥测恢复闭环与 Post-Rename 封存

> 日期：2026-08-13  
> 主包：`results/local5_h24_phase_summary_pilot_v2_20260812/`  
> 外层封存：`results/local5_h24_phase_summary_pilot_v2_postrename_sidecar_v4_final_20260813/`  
> 证据等级：`[rtl]+[软件整数金参考]+[流式独立oracle]`  
> Formal G0：**DENY**

## 1. 结论

Local5 最大 head 配置 H24 的单窗口紧凑 phase-summary 已闭环。主 RTL 不因后验证中断
而第三次重跑；恢复器复用 SHA 绑定的已完成 RTL 输出与 numeric verifier，并重新执行
全部 phase/resource/oracle/tamper 后验证。

| 项 | H24 实测 |
|---|---:|
| RTL cycle | 31,718,450 |
| reference trace | 47,941,735 行，2,999,951,581 bytes |
| semantic phase interval | 2,929 |
| aligned accepted event | 2,043,648 |
| cross-Acc command | 16,588,800 |
| cross read/write | 8,294,400 / 8,294,400 |
| TCFM5 term/update | 398,376 / 1,212,096 |
| TCFM5 mask mismatch | 0 |
| Acc32 | 345,600，mismatch=0 |
| fail-closed tamper | 7/7 |

这些 cycle 和后验证 wall time 只属于验证环境，不能作为架构性能、吞吐或 PPA。

## 2. 闭式计数与流式比较

H24 闭式合同全部与 RTL 摘要一致：

```text
phase              = 1 + 2H + 5H^2       = 2,929
relation req/rsp   = 450H^2               = 259,200 / 259,200
weight req/rsp     = 1,024H^2             = 589,824 / 589,824
final / Acc32      = 14,400H              = 345,600
cross read/write   = 14,400H^2            = 8,294,400 / 8,294,400
```

五类 accepted event 对 3.0 GB 冻结 reference trace 完整流式读取，count、双 64-bit 有序
digest 与首尾 anchor 一致。reference trace SHA256 为：

```text
096d4e0c6f6154cb80433d088a6355af941046749ed55f6c33da591e8ae56e9c
```

Cross-Acc 由冻结 C oracle 独立重建 16,588,800 条命令，得到与 RTL 相同的两个 digest：

```text
b62d67328ef9c0d9
579ef539bac6bf11
```

冻结 verifier 单测 19/19、C/Python oracle 单测 10/10 通过。

## 3. 中断、负结果与恢复

第一次完整运行的 RTL 和 numeric verification 均通过，但 runner 将预期 trace rows
错误写成少 1 行，后验证 fail-closed；现场保留为 `.failed.478589`。修正后第二次 RTL
再次通过，外部会话在后续 3.0 GB trace 多遍扫描时中断，留下
`.staging.523847`。

恢复过程中另有两个正确保留的失败现场：

1. NumPy 2.1.1 与冻结 1.26.4 环境不一致；
2. 直接从 snapshot 目录执行测试时仓库相对布局不成立。

最终恢复使用系统 Python 3.12.3，验证 live/snapshot 测试 SHA 相等，在仓库布局中执行
测试，并重新运行 phase、五类资源、C oracle、TCFM5 和 7 类 tamper。复用已完成 RTL
输出在这里是可接受的 post-verification recovery，因为输入、输出、日志和 verifier
均有 SHA 绑定；它不等价于从 compile/simulation 开始的第三次独立重跑。

## 4. 首轮独立 DATE 审阅

主包首轮审阅为：

```text
4/5，Conditional GO
P0：无
P1：顶层 complete SHA 未由外层 receipt 固定；rename 后 4 份 JSON 保留失效 staging 路径
```

审阅重新核验 116 项 `complete.internal_bindings`、36 项 evidence manifest、13 组
live/snapshot source、外部 release/reference/table/vector SHA，均无失配。问题属于最终
封存层，不是 RTL 功能错误。

## 5. Post-Rename 外层封存

在不修改不可变主包的前提下新增 sidecar：

1. 重算主包全部 117 个 regular file，合计 156,252,132 bytes；
2. 显式绑定主包 `complete.json` SHA256：
   `f886770e...007d5`；
3. 将 4 份 JSON 中 6 个历史 staging path occurrence 映射到最终主包路径；
4. 从最终主包的 C 源重新编译 oracle；
5. sidecar rename 后，再从最终 `sidecar/build` 路径实际运行 oracle；
6. live test 与主包 snapshot SHA 相等后，在正确仓库布局重新执行 19+10 项测试；
7. 最后写 sidecar manifest 与 complete，避免把 pre-rename 路径当作最终可重放路径。

第一次 sidecar 因直接从 snapshot 布局执行测试而 fail-closed；第二次两阶段版因 rename
后仍引用临时 source 路径而 fail-closed；v3 又因漏扫无尾斜杠的
`payload_audit.root` 被独立复审拒绝。失败现场均保留，v4 才是最终正向证据。

## 6. 存储口径

| 口径 | 大小 |
|---|---:|
| compact evidence payload（排除 `build/source`） | 3,639,504 bytes |
| 完整主包 117 文件 | 156,252,132 bytes |
| 冻结上限 | 536,870,912 bytes |

`3.64 MB` 只能写成 compact payload，不能写成完整包大小。即使将 `build/source` 纳入，
完整主包仍低于 512 MiB，因此排除项没有把超限结果变成 PASS。

## 7. 最终独立复审

v4 sidecar 修复根路径漏扫后，独立 DATE 证据审阅结果为：

```text
5/5，GO
P0：无
P1：无
P2：无包内缺陷
```

复审独立确认 117/117 主包文件 SHA、4 records/6 occurrences、5 项 sidecar 单测和最终
路径 C oracle。发布时使用以下包外终端信任锚：

```text
sidecar_complete.json SHA256
13447a611b939b3a95b1caba13f9386f9996d4b8f1f1a2d9ff2e299df83e8043
```

`5/5` 只评价该单窗口验证包的方法学完整性，不是整篇 DATE 论文评分。

## 8. 证据边界

本包允许证明：

- 指定 `sample2/stage3/block0/window1/H24` direct 配置的 Acc32 数值一致；
- H24 phase interval、五类 accepted event、Cross-Acc 与 TCFM5 摘要一致；
- 指定中断现场可以通过 SHA 绑定的 post-verification recovery 收口。

本包不能证明：

- formal G0、形式证明或完整 1,200-window phase archive；
- relation-memo/vector-result 配置或其他 sample/window 的普遍性；
- full encoder、latency、FPS、能耗、DC/STA/SAIF 或 ASIC PPA；
- telemetry、observer、恢复器或 sidecar 是 DATE 架构创新。

## 9. 下一步

H24 证明紧凑摘要在最大 H 下可实现，但仍是单窗口。下一步按冻结 30-anchor 计划扩展
H3/H6/H12/H24 与 sequence/density 边际覆盖，并为全量 phase formal 选择“有限真实 RTL
anchor + 参数化合同/形式性质”路线，不能将单窗口外推为 462,600 phase 已闭环。
