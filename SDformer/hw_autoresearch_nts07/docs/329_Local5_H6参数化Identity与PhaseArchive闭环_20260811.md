# Local5 H6 参数化 Identity 与 Phase Archive 闭环

## 1. 本轮结论

本轮只关闭 Local5 从 H3 扩到 H6 时的验证链参数化缺口，不增加候选架构，也不生成
formal admission。

```text
身份：sample2 / stage1 / block0 / window26 / H6
no-hold 基线：PASS [rtl]
hold2 候选：PASS [rtl]
Acc32：86,400 / 86,400，三方逐元素零失配
phase-template 独立展开：3,190,783 / 3,190,783
九类 archive 篡改：9 / 9 被拒绝
formal G0：DENY
```

主要结果：

- `results/local5_identity_service_tables_sample2_h6_v4_20260811`
- `results/local5_h6_identity_phase_canary_v2_20260811`
- `results/local5_h6_phase_template_tamper_regression_v2_20260811`

## 2. 参数化范围

旧 verifier 把 H3 的 `sample2/stage0/block0/window249`、`HEADS=3`、4,050 个
relation 和 43,200 个 final 写死。本轮改为从经独立验证的 identity-table manifest
读取身份，并由 `H` 推导全部事务数：

| 项 | 公式 | H6 实测 |
|---|---:|---:|
| relation unique | `H x 450` | 2,700 |
| relation runtime | `H x H x 450` | 16,200 |
| weight runtime | `H x H x 32 x 32` | 36,864 |
| final/Acc32 | `H x 450 x 32` | 86,400 |

PASS 行中的 stage、block、window、token、result 和 final 数量也由同一身份独立重算，
不再接受 H3 固定正则。旧 H3 密封结果用新 verifier 回放仍通过。

## 3. 两份独立 RTL 构建

本轮没有重新编译后再用同一二进制冒充两种协议条件，而是直接绑定两个已有密封
release 的 H6 executable：

| 角色 | release manifest SHA256 | weight hold |
|---|---|---:|
| no-hold 基线 v8 | `9e6fe6426d1da24afecf3ca7605e7e0e91e6fc5a85fa664e1bc0f7c02044a08d` | 0 |
| hold2 候选 v10 | `87c4cf52c26ee7fe05335de0aaadc2e706ef3f23309cec2860e6c3175bfd2359` | 2 |

两边 release 在运行前后均重新验封；complete 直接绑定 release manifest、H6
executable、compile argv、真实输入、INT8 权重、软件 expected 和验证源码。完整包共
42 个直接 SHA 绑定，复核为零错误，结果目录全部只读。

## 4. 逐事件协议证据

| 指标 | no-hold 基线 | hold2 候选 |
|---|---:|---:|
| trace 行 | 3,117,055 | 3,190,783 |
| relation available/accept 对 | 16,200 | 16,200 |
| weight available/accept 对 | 36,864 | 36,864 |
| final request/accept 对 | 86,400 | 86,400 |
| `weight_response_stall` | 0 | 73,728 |
| 每个 weight response 的 stall | 0 | 恰为连续 2 cycle |

候选的 36,864 个 weight response 全部在服务侧出现 `valid=1,ready=0`，每个 payload
从 available 到 accept 保持不变。覆盖边界是 **producer + hold-adapter 的服务侧接口**，
不是 DUT 自己主动拉低 ready 的随机反压。

基线验证周期为 2,230,231，候选为 2,303,959，差值恰为 73,728。该关系仅用于证明
hold 注入没有漏记；两者都是验证环境时延，不能作为硬件性能收益或损失。

## 5. 数值和状态全序

两份 RTL Acc32 memh 的字节摘要相同：

```text
315129e4e0c81e81e98bf708f07c6d4dd7296cb33f861dc880a9564f58a0bd15
```

软件整数 expected 的 86,400 个坐标与两份 RTL 逐元素 mismatch 为 0。phase verifier
去掉合法服务等待的绝对 cycle 后，还比较了握手、边界和内部状态
的统一全序账本：

```text
core-all count  = 3,117,053
core-all SHA256 = b8f37982eac3e95ac264b65bd23056424729c7c5263da367598b9b109b1e2f76
```

候选与基线完全相同。该 cycle-free ledger 保留 event、身份、delay、index、origin 和
payload，只移除绝对 cycle 与 protocol telemetry。

## 6. H6 模板扩展结果

H6 没有引入新的模板类，仍是 prefix、head-seed、inter-head-gap、head-accumulate、
tile-tail、tile-transition 和 suffix 七类。实例数从 H3 的 22 增加到 H6 的 79，模板
行数仍为 206,078：

| 指标 | H3 | H6 |
|---|---:|---:|
| 展开行 | 862,507 | 3,190,783 |
| 模板行 | 206,078 | 206,078 |
| 结构实例 | 22 | 79 |
| event/origin 骨架复用因子 | 4.185x | 15.483x |
| 原 CSV / 完整 NPZ 文件缩减 | 2.123x | 2.264x |

`15.483x` 只表示 event/origin 骨架复用；`2.264x` 只表示含 typed patch 的验证 archive
文件大小。二者都不是片上 SRAM、吞吐、能耗或 ASIC PPA。

## 7. 负测试与回归

对 template event、patch cycle、payload code、instance class、patch offset、instance
tile、instance head、越界 dictionary code 和 patch identity 共九类定向篡改，独立
expander 均返回 `PASS_REJECTED`。篡改包会重算 archive SHA，因此拒绝原因来自内容
语义不等价，不是只靠文件摘要。

此外，identity-table 与 release 基础设施的 27 项单元测试全部通过；H3 密封 trace 和
phase archive 均用参数化 verifier 兼容回放通过。

## 8. 证据边界

- `[rtl]`：一个真实 H6 窗口的 no-hold/hold2 Verilator 回放与 86,400 个 Acc32；
- `[rtl-build-provenance]`：两个密封 release 的 H6 executable、compile argv 和工具链；
- `[软件确定性服务合同]`：identity table 由 generator 和不导入项目 oracle 的独立
  verifier 双向闭合；
- `[待验证]`：H12、H24、旧 `WindowCommandWork` 同构、剩余 97 个 sample、正式
  1,200-window archive、admission receipt；
- formal G0、full encoder、DC/STA/SAIF 与 ASIC PPA 均未完成。

## 9. 独立 DATE 复审

独立审稿代理现场重算后的评分为 `4/5 Conditional Accept`，仅接受本包作为 H6 单窗
验证扩规模证据：

- P0：0；42 个 SHA、事务数、held-valid、phase 展开、core-all 和九类篡改均通过；
- P1：该窗口 13,500 个 K word 和 86,400 个 Acc32 全为零，只覆盖控制、排序、接口
  和零值传播，不能扩大为非零 MAC、正负权重累加或溢出边界证据；
- P2：容量字段仍需防 manifest-only 篡改，结果包需提高源码可携带性并绑定 software
  expected receipt；
- held-valid 仍只覆盖服务侧 producer/hold-adapter，不是 DUT 自身输入反压。

该 P1 成立。后续已筛选到非退化真实窗口
`sample1/stage1/block1/window71/H6`：1,202/13,500 个 K word 非零，58,238/86,400
个 expected Acc32 非零，其中正值 30,663、负值 27,575，范围 `[-39527,49042]`。
在该窗口完成同链复验前，不把本包称为充分的 H6 数值覆盖。

## 10. 当前裁决

H6 证明 H3 的 phase-template 不是只对三头特例成立，而且服务协议在 head 数翻倍后
仍保持精确；但首个 H6 窗口数值退化为全零。它提升的是 Local5 formal 验证完整度，
不是新的 DATE 架构贡献。先关闭非零 H6 P1，再考虑 H12 或正式 G0。
