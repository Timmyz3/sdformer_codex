# Local5 H6 非退化数值 Identity 与 Phase 闭环

## 1. 为什么重跑

`docs/329` 的首轮独立 DATE 复审为 `4/5 Conditional Accept`，唯一 P1 是所选 H6
窗口的 K 和 Acc32 全为零。该结果足以验证控制、排序和服务协议，但不能支撑非零
projection 乘加、正负权重或有符号累加。

本轮只关闭这个 P1，不扩新架构，不进入 H12，也不生成 formal admission。

## 2. 真实窗口选择

在已经完成软件 expected 和向量导出的六个 H6 窗口中，只读统计如下：

| sample/block/window | 非零 K word | 非零 Acc32 | 正/负 Acc32 | 范围 |
|---|---:|---:|---:|---:|
| s0/b0/w54 | 0 | 0 | 0 / 0 | 0 |
| s0/b1/w44 | 41 | 7,206 | 3,594 / 3,612 | `[-5408,3808]` |
| s1/b0/w45 | 0 | 0 | 0 / 0 | 0 |
| **s1/b1/w71** | **1,202** | **58,238** | **30,663 / 27,575** | **`[-39527,49042]`** |
| s2/b0/w26 | 0 | 0 | 0 / 0 | 0 |
| s2/b1/w14 | 112 | 18,286 | 9,429 / 8,857 | `[-8544,4704]` |

因此冻结 `sample1/stage1/block1/window71/H6`，不是在结果出来后从 RTL mismatch 中
挑选，而是按软件 expected 的非退化覆盖最大化选择。

## 3. 结果包

- identity table：`results/local5_identity_service_tables_sample1_h6b1_v4_20260811`
- RTL/phase：`results/local5_h6_nonzero_identity_phase_canary_v2_20260811`
- 数值多样性：`results/local5_h6_nonzero_numeric_diversity_audit_v2_20260811`
- 九类负测试：`results/local5_h6_nonzero_phase_template_tamper_regression_v2_20260811`

formal G0 在所有包中均为 `DENY`。

## 4. 非退化数值审计

独立 auditor 不读取 RTL 自报统计，而是重新解析 combined inputs、INT8 权重、两份
RTL Acc32 memh 和软件 expected NPZ：

| 数值域 | 总数 | 非零 | 正 | 负 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| K word | 13,500 | 1,202 | 不适用 | 不适用 | 不适用 | 不适用 |
| INT8 weight | 36,864 | 36,286 | 18,229 | 18,057 | -125 | 124 |
| Acc32 | 86,400 | 58,238 | 30,663 | 27,575 | -39,527 | 49,042 |

no-hold RTL、hold2 RTL 和软件 expected 在全部 86,400 个固定坐标上逐元素一致，
mismatch 和 max absolute error 均为 0。auditor 另外绑定了此前主 complete 未直接绑定的
`software_expected_receipt.json`，逐项验证其 identity、task-plan SHA、expected SHA、
shape、scalar count 和两份生成器源码 SHA，并把自身源码复制进结果包。

限制：2,700 个 Q word 仍全部为零。因此本轮证明的是非零 K 驱动下 projection 的
正负 INT8 乘加与 Acc32 累加，不证明非零 Q score 路径或 Acc32 溢出边界穷尽。
非零 K 只分布在 input head 3/4/5，数量分别为 67/418/717；input head 0/1/2 仍为
零。另一方面，六个 output tile 各有约 9,700 个非零 Acc32，且每个 tile 都同时包含
正值和负值。

## 5. 协议与 phase 结果

| 指标 | no-hold | hold2 |
|---|---:|---:|
| relation runtime | 16,200 | 16,200 |
| weight response | 36,864 | 36,864 |
| final/Acc32 | 86,400 | 86,400 |
| trace 行 | 3,117,055 | 3,190,783 |
| 服务侧 held-valid 对 | 0 | 36,864 |
| `valid=1,ready=0` cycle | 0 | 73,728 |

每个 hold2 response 都连续 stall 两周期，accept 发生在 available+2；payload 始终
稳定。cycle-free core-all ledger 在两边均为 3,117,053 条，SHA 为：

```text
d74425634e373abc0d2000706b526257ecd5eba0528dade0ff244c4641147659
```

phase archive 仍由七类模板、79 个实例和 206,078 个模板行精确展开 3,190,783 行；
九类 template/patch 篡改全部 fail-closed。模板复用和 archive 文件缩减只属于验证
归档指标，不是硬件性能。

## 6. P1 关闭口径

首轮 P1 的准确关闭范围是：

```text
非零 K                 PASS
正负 INT8 weight       PASS
非零且正负 Acc32       PASS
两份 RTL/软件三方等价  PASS
非零 Q                 未覆盖
溢出边界穷尽           未覆盖
```

因此可以把 H6 证据从“零传播控制 canary”提升为“非退化 projection 数值 + 协议 +
phase archive canary”，但不能写成完整 Local5 数值空间证明。

## 7. 证据边界

- `[rtl]`：一个真实非退化 H6 窗口、两份密封 Verilator executable、86,400 个 Acc32；
- `[独立软件数值审计]`：K/weight/Acc32 分布和三方逐元素 miter；
- `[待验证]`：非零 Q 窗口、DUT 自身输入 backpressure、H12/H24、更多窗口、clean-room
  rebuild、manifest-only tamper、正式 1,200-window archive；
- 验证周期、trace 行数和 archive 压缩率都不是 throughput、energy 或 ASIC PPA；
- formal G0、full encoder、DC/STA/SAIF 均未完成。

## 8. 第二轮独立 DATE 复审

第二轮评分为 `4/5 Conditional Accept`，P0 为 0，首轮“全零 projection 数值”P1
正式关闭。审稿代理独立确认：42/42 主绑定、10/10 数值审计附加绑定、86,400 点
三方 miter、36,864 对 held-valid、3,190,783 行展开和九类 tamper 均通过。

复审发现本文初稿误沿用了首轮全零窗口的 core-all SHA；本版已改为非退化窗口真实
SHA `d7442563...47659`。auditor v2 已进一步严格检查字段宽度、完整坐标集合、NPZ
schema version 和 receipt 语义，并修复同号数组 min/max 会被 0 污染的问题。保留 P2
为 Q 全零、非零 K 仅分布于 input head 3/4/5，以及当前 auditor 是独立 parser/miter
而不是第二套独立算术 oracle。

## 9. 当前裁决

本轮本地正向、负向和数值多样性审计全部通过，独立复审已关闭首轮 P1。该包可作为
“非退化 H6 projection 数值 + identity/hold + phase archive”验证证据；formal G0、
性能、PPA 和架构创新主张仍为 `DENY`。
