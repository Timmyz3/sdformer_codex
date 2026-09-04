# Local5 H12 全 Head 全分支 Identity 与 Phase 闭环

## 1. 本轮目标

H6 非退化包已经关闭 projection 全零 P1，但仍有两个覆盖边界：Q 全零、部分 input
head 的 K 全零。本轮不新增 RTL 机制，只选择一个更强的真实 H12 窗口，验证：

1. 12 个 input head 都有非零 Q 和 K；
2. 12 个 head 都实际命中 `q&k` score 分支；
3. 12 个 output tile 都产生正负非零 Acc32；
4. identity-service、hold2、phase-template 和 Acc32 miter 在 H12 规模仍成立。

formal G0 继续为 `DENY`。

## 2. 窗口选择

从已完成软件 expected 的 18 个 H12 窗口中，冻结：

```text
sample1 / stage2 / block2 / window21 / H12
```

该窗口不是 K 最密的窗口，但覆盖更均衡：

| 指标 | 值 |
|---|---:|
| 非零 Q word | 1,090 / 5,400 |
| 非零 K word | 11,636 / 27,000 |
| 有 K 的 input head | 12 / 12 |
| `q&k` bit | 511 |
| 命中 `q&k` 的 input head | 12 / 12 |
| 非零 Acc32 | 172,732 / 172,800 |
| 同时有正负 Acc32 的 output tile | 12 / 12 |

例如 K 最密的 s0/b5/w13 有 24,844 个非零 K，但 `q&k` 只有 18 bit，且只覆盖
4 个 input head；因此未选择它。

## 3. 结果包

- identity table：`results/local5_identity_service_tables_sample1_h12b2_v4_20260811`
- RTL/phase：`results/local5_h12_nonzero_identity_phase_canary_v2_20260811`
- 数值/分支：`results/local5_h12_fullbranch_numeric_diversity_audit_v4_20260812`
- 九类负测试：`results/local5_h12_phase_template_tamper_regression_v3_20260812`

主包 18 个 external 和 24 个 internal SHA 全部匹配；数值审计另有 10 项绑定。两个
RTL release、H12 executable、compile argv、table、真实输入、权重、expected、receipt
和验证源码均被直接绑定。

## 4. H12 事务规模

| 项 | 公式 | H12 实测 |
|---|---:|---:|
| head job | `H x H` | 144 |
| relation unique | `H x 450` | 5,400 |
| relation runtime | `H x H x 450` | 64,800 |
| weight | `H x H x 32 x 32` | 147,456 |
| final/Acc32 | `H x 450 x 32` | 172,800 |

identity table 由 generator 写出，再由不导入项目 oracle 的 verifier 独立重算；manifest
SHA 为 `8f3b9573062a7c1b26b5ce3eda506c3fa5bf414f0545c7d079a21db6e4aeee8f`。

## 5. 数值和 score 分支

数值 auditor v4 严格检查 plane/y/x/head 坐标全集、Q/K/weight hex 宽度、NPZ schema、
expected receipt 语义，并把两个 expected 生成器固定到规范化绝对路径、可信 SHA 常量
和 baseline release 的 source binding 三重合同。

| 数值域 | 总数 | 非零 | 正 | 负 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| INT8 weight | 147,456 | 145,475 | 73,026 | 72,449 | -127 | 127 |
| Acc32 | 172,800 | 172,732 | 87,014 | 85,718 | -81,429 | 87,545 |

Q/K 是 bit-vector，不使用有符号大小解释。对 valid candidate 的 817,920 个 bit：

| score 输入关系 | bit 数 |
|---|---:|
| `q & k` | 511 |
| `q xor k` | 43,713 |
| `~q & ~k` | 773,696 |

三类之和精确等于 817,920。每个 input head 的 `q&k` 分别为
`[153,136,67,44,12,57,26,2,6,2,2,4]`，全部非零。

no-hold RTL、hold2 RTL 和软件 expected 在 172,800 个固定坐标上逐元素一致，
mismatch 与 max absolute error 都为 0；两份 RTL memh SHA 同为：

```text
c9b06235124f2262d1d043b95dbbabf1e7bf9d7dfdaef667d7eb9619a98cb35f
```

这覆盖真实非零 Q/K 分支和有符号 projection 累加，但不等于溢出边界穷尽或第二套
clean-room 算术 oracle。

## 6. 服务协议

| 指标 | no-hold | hold2 |
|---|---:|---:|
| trace 行 | 11,949,751 | 12,244,663 |
| relation pair | 64,800 | 64,800 |
| weight pair | 147,456 | 147,456 |
| final pair | 172,800 | 172,800 |
| held-valid pair | 0 | 147,456 |
| `valid=1,ready=0` cycle | 0 | 294,912 |

每个候选 weight response 恰连续 stall 两周期，accept 在 available+2，payload 与身份
保持不变。基线与候选验证周期分别为 8,153,286 和 8,448,198，差值恰为 294,912。
这些周期只证明 hold 注入与 trace 记账完整，不能用作架构性能。

## 7. Phase Archive 扩规模

| 指标 | H3 | H6 | H12 |
|---|---:|---:|---:|
| instance | 22 | 79 | 301 |
| template row | 206,078 | 206,078 | 206,078 |
| expanded row | 862,507 | 3,190,783 | 12,244,663 |
| event/origin 骨架复用 | 4.185x | 15.483x | 59.418x |
| 完整 NPZ 文件缩减 | 2.123x | 2.264x | 2.353x |

H12 仍只需七类结构模板。完整 archive 为 318,939,163 字节，包含 typed patch；独立
verifier 精确展开 12,244,663 行并重建 candidate trace SHA。候选与基线的 cycle-free
core-all 都是 11,949,749 条，SHA 为：

```text
de85104de0ac7a7e4309771e559e9862fc9a8e319fc0b564bec63a0976ae9c68
```

九类 template/patch 篡改全部 `PASS_REJECTED`。

## 8. 工具扩展性负结果

H12 暴露出当前验证归档工具的工程瓶颈：generator 会把 750 MB candidate CSV 全部
载入 Python 对象，并在现场观察到约 10.9 GB RSS；phase archive 生成与逐行复验显著
慢于 RTL 本身。该观察没有密封的 `/usr/bin/time` receipt，只能标为 `[现场观察]`。

这不影响 H12 正确性结论，但说明 H24 前应先把 generator/expander 改成流式或分片，
否则继续按当前实现扩展会浪费内存和 wall-time。它是验证基础设施问题，不是候选硬件
面积或吞吐问题。

## 9. 证据边界

- `[rtl]`：一个真实 H12 窗口的 no-hold/hold2 回放、协议 trace 和 172,800 个 Acc32；
- `[独立软件数值审计]`：Q/K 分支、INT8/Acc32 分布、receipt 与三方 miter；
- `[rtl-trace-derived]`：H12 phase-template archive、cycle-free ledger 和九类负测试；
- `[现场观察]`：当前 Python generator 的约 10.9 GB RSS，仅用于指导下一步工具改造；
- `[待验证]`：更多 sample/window、DUT 自身输入反压、溢出边界、旧
  `WindowCommandWork` 同构、正式 1,200-window archive、clean-room rebuild；
- `[rtl]`：H24 单个真实窗口的 identity/phase/Acc32、hold2 和解析 state 全序已在
  `docs/333` 关闭；
- formal G0、full encoder、DC/STA/SAIF、ASIC PPA 和架构性能均未完成。

## 10. 独立 DATE 复审

独立审稿评分为 `4/5 Conditional Accept`，仅接受 H12 单窗口本地 RTL 扩规模证据；
P0 为 0。审稿代理独立重算 42/42 主绑定、H12 事务、Q/K 三分支、172,800 点 miter、
147,456 对 held-valid、core-all 全序和 phase 规模，均与本文一致。

复审提出两个 P1：

1. 旧 auditor v3 只按 basename 约束 expected source，不能称强白名单；
2. 当前非流式 phase generator 已在 H12 暴露扩展性问题，H24 前必须改造并密封
   wall/RSS receipt。

第 1 项已由 auditor v4 关闭；numeric complete 已增加明确 schema、identity/formal G0，
tamper complete 已增加明确 schema/formal G0。第 2 项已由两遍流式 Phase Array Store、
周期性 `MADV_DONTNEED`、H12 全量等价和独立复审关闭，详见 `docs/332`。H24 后续
真实 RTL trace、Acc32、hold2、解析 state oracle 和 source-only Phase Store 已完成，
详见 `docs/333`。

## 11. 当前裁决

本地正向、负向和数值分支审计均通过。H12 关闭了 H6 留下的 Q 全零与部分 input head
不活跃边界，但本包仍属于验证扩规模证据，不是 DATE 架构创新。流式/分片 phase 工具
P1 已在 `docs/332` 关闭；真实 H24 trace 已在 `docs/333` 完成。下一步回到正式
1,200-window ledger/admission，不能用单窗口闭环代替 formal G0。
