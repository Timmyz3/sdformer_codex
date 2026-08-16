# Adaptive CSR配置合同与Selector SVA整改

> 2026-07-18后续更新：本文的fail-fast是格式元数据完成前的临时保护，现已由`docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md`取代。当前`Adaptive + ENABLE_RESIDENCY=1`是合法配置，但驻留资格严格限于IPD32W；FADC24和RAW41不会查询或填充descriptor cache。本文其余selector SVA仍有效，首字属性同时保留为无元数据legacy fallback的验证。

## 一、来源

独立第三轮DATE复审见`docs/106_DATE审稿人第三轮_AdaptiveCSR后评估_20260718.md`，评分为`2.8/5`、Weak Reject。复审确认统一双格式RTL已成立，同时指出两个直接问题：

1. `Adaptive CSR + 旧descriptor residency`存在格式合同冲突；
2. 原专属SVA主要检查外部valid/ready，没有直接证明selector不变式。

## 二、非法配置fail-fast

现有warm replay offset采用IPD32W的`2 + ceil(term_count/2)`布局，不能解释FADC24的24-bit descriptor以及list/bitmap destination。因此在format-aware residency完成前，以下组合被定义为非法：

```text
CSR_FORMAT_FADC24 == 2 && ENABLE_RESIDENCY != 0
```

`gatestack_single_context_execution_top`新增可综合的admission fail-closed：非法组合将`protocol_error`置高，并强制阻断`group_ready`与scheduler的`group_valid`。专用TB显式实例化该组合并确认不能接收group，日志为：

```text
PASS: invalid Adaptive plus residency configuration blocked
```

这避免默认参数误用导致cold tile正常、warm tile错误offset的潜在静默数值错误，同时不在ASIC RTL中引入仿真专用`initial`。它不是residency修复；最终方案仍需选择“FADC format-aware residency”或“仅IPD cacheable、FADC强制non-cacheable”。

## 三、Selector不变式SVA

新增：

- `verif_hitflow/gatestack_adaptive_csr_selector_assertions.sv`；
- `verif_hitflow/bind_gatestack_adaptive_csr_selector_assertions.sv`。

覆盖属性：

| 属性 | 目的 |
|---|---|
| start-ready仅在IDLE | 禁止重入 |
| word0 magic决定child | 证明header-steered分派 |
| START/RUN期间selection稳定 | 禁止session中途换格式 |
| child start严格one-hot且匹配selection | 未选decoder不启动 |
| child word-valid严格one-hot且匹配selection | 未选decoder不接收payload |
| 首字缓存期间上游ready关闭 | 避免首字重复或后续字越过 |
| child反压时首字data/index/last稳定 | 保持valid/ready合同 |
| child接受首字后pending清除 | 首字只重放一次 |
| done只在RUN发生 | 关闭异常相序 |

原格式无关外部SVA和IPD/FADC各自专属SVA继续保留。

## 四、回归结果

`sim_hitflow/run_gatestack_adaptive_csr_fulltop.sh`重新执行：

- 四stage真实trace全部双工具通过；
- `11 IPD + 12 FADC + 1 RAW`同context通过；
- `11 IPD + 13 FADC`同context通过；
- 六个Verilator构建均为0 warning/error；
- selector、外部接口和两个child专属SVA均未触发；
- accumulator mismatch、done error、protocol error和abort均为零；
- 非法`Adaptive + residency`组合按预期被拒绝。

## 五、仍缺

- unknown magic/version、word0 index错误、word0即last和截断payload错误矩阵；
- 中途reset、连续多group和多随机seed；
- format选择计数器尚未作为顶层硬件计数输出；当前格式数量来自向量manifest；
- commit时解析format metadata、避免每次replay约2周期peek开销；
- format-aware residency与warm replay。

因此当前准确名称仍是`header-steered heterogeneous CSR replay`，不是硬件在线fanout-aware编码决策器。
