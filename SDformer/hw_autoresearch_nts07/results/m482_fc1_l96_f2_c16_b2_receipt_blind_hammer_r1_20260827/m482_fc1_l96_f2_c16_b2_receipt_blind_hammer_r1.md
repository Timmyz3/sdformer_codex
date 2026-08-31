# M482 receipt-blind 独立打铁评审

日期：2026-08-27（Asia/Shanghai）

## Overall Assessment: Share with caveats

评分：**4.0 / 5.0**。

M482 的负向门判定可信：`L96_F2_C16_B2` 在冻结 H67 ep35 100-record workload 上的 RTL-handshake 等价递推只有 **1.359896673×**，低于合同的 `1.50×`，因此必须维持 `NO_GO_L96_F2_C16_B2_AS_PERFORMANCE_POINT`，不得进入 DC，也不得升级成 FC1、FFN 或系统性能结果。

本评审先隔离官方 result/receipt，只读取合同、RTL、TB/SVA、原始 VCS log 和冻结 workload，逐文件校验并重新解码 100 个 bitpack；完成独立重算以后，才核对官方 result、receipt、`SHA256SUMS` 和 seal。四种周期视图、envelope 敏感性、F2/B4 与 F4/B4 敏感性均与官方结果精确一致。

## 数据与身份质量

- 冻结人口：H67 motion、ep35、10 个 sample × 10 个 binary FC1 module，共 100 条唯一记录。
- 100 个 payload 均重新计算 SHA256，packed size、shape 与 active-element count 全部一致。
- 重新解码所得 255 个非零 context-mask bin 与 M481 冻结直方图逐 bin 相等；逐 record 的 group、nonempty、weight-read、context-update 汇总也完全一致。
- 总 group stream：`4,320,000`；nonempty：`4,171,068`；empty：`148,932`；nonempty rate：`96.5525%`。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，本评审未修改该文件。

## 独立周期重算

| 口径 | Baseline cycles | Candidate cycles | 比率 | 可否作为 RTL 门值 |
|---|---:|---:|---:|---|
| Serial analytical | 9,292,511,340 | 4,603,346,064 | 2.018642790× | 否 |
| Factor+weight parallel analytical | 6,260,940,084 | 3,428,345,892 | 1.826227656× | 否 |
| Full-overlap analytical | 3,229,368,828 | 2,253,345,720 | 1.433143969× | 否 |
| RTL-handshake exact recurrence | 1,229,325,596 | 903,984,560 | **1.359896673×** | **是，P0 NO-GO** |

从原始 trace 独立得到：

- Expanded onehot baseline descriptors / issue rounds：`1,010,523,752`。
- Context-factorized descriptors：`391,666,724`。
- B2 bank-aware issue rounds：`685,182,716`。
- Chunk-16 directory cycles：`55,832,328`。
- RTL recurrence：nonempty tile 为 `issue_rounds + chunk_count + 39`，empty tile 为 2 cycles。

## 四项重点攻击

### 1. Factor / sparse 阶段是否误重叠

裁定：**在 fixed-2-cycle、in-order 合同内通过**。

RTL 的 factor request、weight request 和 accumulator issue 是独立 ready/valid 路径；VCS cover 分别观测到 factor+weight overlap `1622` 次、weight+update overlap `1874` 次、triple overlap `1593` 次。每个非零 descriptor 至少需要一个 update issue round，而 factor、weight 各为单请求端口、每周期吞吐至多一个；因此稳态 update 不会比前两级消耗更少，固定延迟的 fill/drain 可被常数项吸收。TB 还对无 stall group 强制检查 `cycles = rounds + chunks + 39`。

这不等于任意存储系统均成立。实际 SRAM variable latency、out-of-order response、跨 tile contention 尚未验证；当前结论只覆盖合同冻结的 fixed-2-cycle、in-order 模型。

### 2. Fixed two-cycle / in-order 是否被 synthetic mask 偏置

裁定：**延迟合同没有被 mask 偏置，但 synthetic 局部比率有明显偏置**。

TB 对每个 accepted factor/weight request 直接检查 response acceptance 相差恰为 2 cycles，共 `3828` 次 latency check。Factor response 还必须匹配严格递增的 `factor_response_q`，因此其合法域明确为 in-order。当前没有 variable-latency 或 legal reorder 覆盖，receipt 对此如实写为 `legal_reorder_supported=false`。

所谓 all-255 是“mask 1..255 各一个 source”，不是 255 个 `0xff`，也不是 100-record trace：

| 指标 | all-255 synthetic | 冻结 100-record workload |
|---|---:|---:|
| 平均 popcount / nonzero source | 4.015686 | 2.580060 |
| 平均 B2 rounds / descriptor | 2.556863 | 1.749402 |
| RTL recurrence ratio | 1.526167× | 1.359897× |

Synthetic ratio 相对真实门值高约 **12.23%**。它只可证明 mask 类别、数值、协议和递推覆盖，绝不能作为 trace 性能数字。官方门判定使用冻结 workload 的 `1.3599×`，因此最终 NO-GO 没有被该偏置污染。

### 3. F4/B4 throughput/mm² 是否只是线性代理

裁定：**是，只是 pre-audit 线性代理，不是物理结论**。

- F2/B4 同递推敏感性：`1.438200567×`，仍低于 `1.50×`。
- F4/B4 同递推敏感性：`1.696926427×`，但未实现 RTL。
- F2→F4 candidate throughput gain：`1.247834825×`。
- Lane adders：192→384；Acc banks：2→4；端口：2R2W→4R4W。
- 用 dominant parallel structure 线性翻倍作面积代理时，relative throughput/area proxy 为 `0.623917412×`。

该代理足以支持保守的 `NO_GO_AUTOMATIC_F4_RTL_OR_DC` 项目决策，却不能证明实际 throughput/mm² 一定下降。没有 SRAM macro、DC 面积或布局后布线数据，就不得把 `0.6239×` 放进论文的物理效率表。M483（SHA `eb60ea57...`）也明确禁止在 compact point 失败后用 F4/C64 解析敏感性挽救 headline。

### 4. `1.045×` 是否被误写为系统实测

裁定：**官方 result/receipt 标注正确**。

独立重算得到 `1.044983135×`，其公式是把 `1.359896673×` 线性投影到 `620,302,905` cycle envelope 中的 `100,895,624` eligible binary-FC1 cycles，并保持 `17,474,490` stage3 fallback 不变。这只是 ideal scope-corrected envelope sensitivity，不是端到端仿真，更不是 measured system speedup。

官方 receipt 明确写有 `ideal_scope_corrected_envelope_sensitivity_not_speedup`、`measured_performance=false`、`system_speedup=false`，没有发现偷换口径。但后续表格必须继续保持这一标签，不能简写成“system speedup”。

## 额外发现：物理资源向量尚未冻结

这是不影响负向 cycle gate、但会阻止任何 PPA 继承的中风险问题：

- M481 compact DSE 预算为 128×22 bit=`2816 bit` descriptor buffer。
- M482 factor FIFO 实际至少保存 descriptor 12 bit、chunk 5 bit、offset 4 bit、context mask 8 bit、sign mask 8 bit，即 37 bit/entry、至少 `4736 bit`，尚未计控制位。
- M481 的 maximum chunk directory 为 24 bit；M482 实现的是 24×8 bit descriptor-count directory=`192 bit`。

Baseline 和 candidate 使用同一个 M482 RTL，所以 `1.3599×` 的 same-RTL 周期比较仍公平，且 NO-GO 不会被推翻；但 M482 不得直接继承 M481 的 compact physical resource vector。若未来重新打开物理实现，必须重新统计所有 FIFO、directory、credit metadata 和真实 SRAM macro。

## 官方回执与封存核对

- 独立重算完成后，四种周期视图与官方 JSON 全部精确相等。
- Envelope、F2/B4、F4/B4 和 F4 线性代理均一致。
- 官方 `SHA256SUMS` 共核验 61 个文件，0 mismatch。
- `SHA256SUMS` SHA：`0c8b382e0e1719e168886fbf91783917276d866182b2b119f7c88b9f6ff0bc54`。
- Seal SHA：`7f67c7110a19997d0c244f12ebc640e44b705fb351d0cfe214323d382198e2c4`，正确绑定当前 manifest。
- 官方 result SHA：`d125962b1f293e3408b5bffb83a32300428fc9fbed8bf3049e467b0a954f0683`。
- 官方 receipt SHA：`a3c0f579bf3da9d45a25dc872a699eb66912206e706ff73a92067db608f3cda5`。

## 最终裁定与红线

M482 可作为一份可靠的**负向性能门证据**和 full-width directed RTL 验证证据分享；不能作为被准入的性能点。保持以下结论：

1. `L96_F2_C16_B2`：P0 NO-GO，禁止 DC。
2. `1.526×`：synthetic directed local ratio，不是 trace 结果。
3. `1.3599×`：冻结 trace 的 RTL-handshake 等价递推，不是 literal full-trace VCS，也不是完整 FC1/FFN/system speedup。
4. `1.045×`：ideal envelope sensitivity，不是 measured system speedup。
5. F4/B4 与 `0.6239×`：未实现 cycle sensitivity 与线性 area proxy，不是物理 throughput/mm²。
6. M483 的 F4/C64 headline-rescue 禁令继续有效，不得将 NO-GO 升格。

可复现的独立计算见同目录 `audit_m482_receipt_blind.py` 与 JSON；审计脚本不调用 M482 官方 analyzer，也不以 receipt 数字作为计算输入。
