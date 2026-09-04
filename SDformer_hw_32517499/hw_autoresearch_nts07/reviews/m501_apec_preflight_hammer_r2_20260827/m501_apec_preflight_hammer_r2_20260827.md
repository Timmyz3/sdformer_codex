# M501 ExSpike-APEC 预执行独立打铁评审（r2 delta）

日期：2026-08-27  
范围：只读复审 r1 后的 contract/analyzer/exact-SHA runner；未运行 analyzer，未启动 VCS/DC/PT/GPU，未修改生产文件或 `docs/359`。

## 裁决

**`GO_FOR_EXACT_OPPORTUNITY_AUDIT_ONLY`，94/100。r1 三个 P0 已关闭。**

这不是 APEC RTL 准入。r2 只允许运行冻结的机会审计；即使 opportunity gate 通过，下一步也只能进入 identical-resource cycle fast-kill，仍需独立定价 scratch macro、端口/延迟、group buffer/comparator、weight readiness、shifted destination commit，以及证明相对 ExSpike 的 H67-native novelty delta。

## r1 → r2 P0 关闭表

| r1 P0 | r2 证据 | 裁决 |
|---|---|---|
| 24 bit 被误称为 published width | selected width 改成冻结 H67 signed19 proxy；明确 ExSpike 论文只给符号公式、固定官方 RTL 为 16 bit；合同列出 16/19/24/32 位 DSE | **CLOSED** |
| 精确 cohort/geometry fail-closed 缺失 | 精确 schema/status、40/128 records、sample ID、1/18 sequences、四 operator set、sample×operator 笛卡尔积、`[10,1,768,15,20]` input/output、Co/Ci/groups/K/stride/padding/dilation 全部成为硬断言 | **CLOSED** |
| signed-analog novelty 边界过宽 | 逐 record 验证 zero + one finite-positive codeword、negative count 0，并在 result/README 显式声明 exact overlap 在本 trace 等于 support intersection、general signed-analog novelty 未激活 | **CLOSED** |

## 1. 身份与 runner

当前静态 SHA 与 runner 固定值一致：

```text
contract  bbb7bce5015ab3a3a5772b86d594853da353380df8dcd85a295e480d422eb2d6
analyzer  5bdfa6f6fa81510d11751d6867748515763d3d4b31927b8cfe03e03ee597b7e7
M40       e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3
M73       3fb3468066fe1f7d61f5e39398cb2f8655643080f03e5b1deb58ef2911db17e2
docs/359  dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
```

runner 继续要求 output directory 不存在，并在完成后封存 contract/analyzer/manifests/docs359/result/README/RUN_COMPLETE。没有发现 SHA 漏钉或旧版本引用。

## 2. 精确 fail-closed 复核

r2 新增检查形成了完整的三层链：

1. **cohort identity**：精确 schema/status/record 数；sample IDs 必须为完整 `0..N-1`；sequence 数、operator set、sample×operator Cartesian product 精确匹配。
2. **record geometry**：input/output shape 均为 `[10,1,768,15,20]`；`Ci=Co=768`、groups=1、K=3×3、stride/padding/dilation=1。
3. **payload identity/numeric**：compressed SHA/bytes、decoded bytes/SHA、float32 elements/bytes、finite、nonzero count、negative count、完整两码本和 decoded nonzero codeword count。

端序也已从宿主隐含 `np.uint32` 收紧为 `<f4` decode + `<u4` bit view，满足 LE payload 的 bit-exact 条件。

静态对照冻结 manifests 的实际结构为：

| Cohort | Records | Sample IDs/keys | Sequences | Operators | Cartesian pairs | Shape/geometry variants | Negative max |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation S10 | 40 | 10/10 | 1 | 4 | 40 | 1/1 | 0 |
| train calibration S32 | 128 | 32/32 | 18 | 4 | 128 | 1/1 | 0 |

四个 operator 在两 cohort 中分别只有一个固定正 codeword，且 codebook nonzero counts 与 record nonzero counts 汇总一致。当前固定输入会通过新增身份门；这不是运行 analyzer 的结果，而是 manifest 静态核对。

## 3. 数学、边界和输出口径

r1 已确认的 horizontal/vertical G2/G4/G8 reshape 与 tail 保留逻辑未被改变。r2 新增三条守恒断言：

```text
candidate + redundant = baseline
redundant = (g-1) * exact_overlap
exact_overlap <= grouped_baseline / g
```

均与 APEC event accounting 一致。

决策输出现在明确分开：

- `status` 仅表示 exact opportunity audit 成功；
- `event_gate_pass` 与 `sensitivity_gate_pass` 分列；
- `opportunity_gate_pass` 是二者合取；
- `next_action` 只能是 `ALLOW_SAME_RESOURCE_CYCLE_FASTKILL_ONLY` 或 `KILL_ADJACENT_OVERLAP_LINE`；
- `new_rtl_admitted=false` 无条件保持。

这关闭了“脚本 PASS 被误读成硬件晋级”的风险。19-bit selected scratch 为：

```text
768 * 3 * 3 * 19 = 131,328 bit = 16,416 byte = 16.03125 KiB
```

result 同时标记 `costs_unpriced=true`，README 也带 width source 和非系统边界，口径严谨。

train-only 18-sequence horizontal-G2 现在输出 sequence-level min/median/max，并明确 `heldout=false`；validation headline 仍只来自预声明 S10 horizontal-G2，没有发生 train 挑点或 heldout 冒充。

## 4. 非阻塞改进项

以下均不阻止当前固定 SHA 的 opportunity audit：

1. **sample key 一致性**：当前代码以 sample ID 断言样本数，并以 `(sample_id,operator)` 检查笛卡尔积；可再断言每个 sample ID 只映射一个 sample key，且 unique sample keys 等于 expectation。当前 manifests 已满足。
2. **operator-constant 措辞**：逐 record 两码本足以证明本 record 内 exact overlap=support intersection；但 `all_records_zero_plus_one_operator_constant_positive_amplitude` 的布尔值目前由 record count 推出，没有在代码中显式断言同 operator 跨样本 codeword 恒定。当前 manifests 确实恒定。可增加 `operator -> nonzero_bits` 单值映射断言，或把字段改成 `per_record_single_positive_amplitude`。
3. **码本 count 完整性**：当前检查两 entry 之和、decoded nonzero count和 record nonzero count；可再逐项检查 `entry[1].count == nonzero_count` 与 `entry[0].count == elements-nonzero_count`，使证据更直接。
4. **width DSE 展示**：contract 已列 16/19/24/32，但 result 只显式计算 selected 19-bit 点。建议顺手输出四档容量 `13.5/16.03125/20.25/27 KiB`，并交叉检查 contract 中 `for_768x3x3` 数值，方便后续 macro DSE。
5. README 可补 train min/median/max；当前 JSON 已有，不影响证据完整性。

## 5. 最终准入边界

允许：运行当前重钉 SHA 的 M501 exact opportunity audit，并根据预声明 validation-G2 两门决定是否做后续同资源 cycle fast-kill。

禁止：

- 把 event reduction 或 Amdahl sensitivity 写成 throughput/system speedup；
- 把 train 18-sequence 写成 heldout；
- 把 positive two-codeword trace 写成一般 signed-analog 支持；
- 把 16.03125 KiB 容量代理写成完整 SRAM 面积/能耗；
- 从 opportunity gate 直接启动或准入 APEC RTL；
- 把 APEC 机制本身写成项目创新。

**最终判断：r2 已达到“可安全运行机会审计”的标准；是否继续这条线必须由结果中的预声明 gate 决定，且最高只进入同资源周期模型。**
