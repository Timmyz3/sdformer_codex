# TCAS-II C1+C2/TSBG 四页工作稿独立模拟审稿（r2）

- 审阅日期：2026-09-04 18:02 CST（Asia/Shanghai）
- 审阅对象：`paper/tcasii/main.tex`、`references.bib`、当前四页 PDF、claim linter，以及其引用的 C1/C2/TSBG sealed evidence
- 审阅模式：只读；未修改论文、RTL、结果或 `docs/359`，未运行 EDA、VCS/simv、GPU 或许可证查询
- Git HEAD：`fff3ce8f5c4b83dfc4464c3526f6a417b0ef3006`
- 稿源 SHA256：`5ad370e2193d43ed079fc2e4ecb82b21cbd638711ab45c37a70570ba043addf2`
- 参考文献 SHA256：`096cbe6ab803605e84056580b4a799054d1bab02a36684ecd4ee149af51d945d`
- PDF SHA256：`82d11b4794fb6b4331c0a7f68ece1d7e788177247447a69f7418832731eaa091`

## 总裁决

**双主线选择正确，数字主体真实，但当前文件不能直接投 TCAS-II。** 如果把格式硬退回单独计算，当前上传件是 **Return without review**：PDF 只有 4 页，且最后一栏同时出现 Conclusion 与 References，违反官方“正文 4.5 页 + references 0.5 页、总计 5 页、最后一栏 only references”的硬要求。官方原文见 [TCAS-II Guidelines for Authors](https://ieee-cas.org/publication/TCAS-II/guidelines-author)。

忽略该机械格式、只评科学内容，我给 **3.55/5，Borderline / Weak Accept，约 45--60% 外审接收倾向**。它已经是一篇可辨认的 component Express Brief：C1 有单 1RW、有限 lifetime 的物理化对象，C2/TSBG 有同端口/cache 的 1,920-workload VCS 周期和等带宽面积效率。它尚不是稳 Accept，主要因为：

1. C1 正文把九宏的真实对象写错；
2. TSBG 人口的 directed-weight、自然符号和跨 attempt lineage 未按 admission 披露；
3. K8 的 `[DC/PT]` 标签超过实际 DC-only 证据；
4. 对 Prosperity 和 FireFly-T 的直接 prior 承认仍偏轻，容易被判“把已有 subset residual / broadcast 改名”；
5. C2/TSBG 的 hold 与功耗仍是空的。

完成下列 P0、把新增内容排成严格 5 页后，我预计 **4.0--4.2/5，可信 Weak Accept / Accept，约 65--80%**。如果同时获得同身份 matched hold-clean 与 logic+weight-store energy，才可能接近 **4.25/5**。概率只是审稿倾向估计，不是保证。

## 一、主数字与证据边界核验

五个核心独立评审目录的 `SHA256SUMS` 与 outer seal 均由本次只读复验通过：

- `m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901`
- `m1789_m1782_c1_expected_macro_leaf_blackbox_energy_result_hammer_r1_20260902`
- `m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829`
- `m2030_m2029_m2018_c2_tsbg_b4_divfree_matched_dc_result_hammer_r1_20260902`
- `m2057_m2053_ep34_tsbg_full40_missing3_vcs_result_hammer_r1_20260903`

### C1

| 稿件数字 | 独立重算 | 裁决 |
|---|---:|---|
| `648,741,051 / 382,848,700` | `1.694510262x` | 正确；只准称四层 bottleneck Conv、十个 `zurich_city_09_a` sample、51.84M source-row 的 same-ledger `[cycle model]` |
| 时间下降 | `1 - 382,848,700/648,741,051 = 40.985899%` | 正确，四舍五入 `40.99%` |
| same-coordinate bit | `648,741,051/646,619,098 = 1.003282x` | 正确；该 baseline 与 strong-zero 几乎相同 |
| concurrent-access ceiling | `648,741,051/341,057,992 = 1.902x` | 数值正确，只是未物理化 ceiling |
| single-port tax | `(382,848,700-341,057,992)/341,057,992 = 12.2533%` | 与正文 `12.25%` 一致，但最好明说分母是 ceiling cycles；若用 C1 cycles 作分母则是 10.9157% |
| 九宏 mapped island | `166,514.31208 um2` | 正确；仅是九个 parent-product SRAM leaf 加逻辑，不是完整 214,912-B ledger |
| PT setup/hold | `+27.871/+1.827 ps` | 正确；3 ns、prelayout、ideal clock、ZeroWireload、无 parasitics |
| Formality | 16,549 passing compare points | 正确；mapped-to-mapped，不是 RTL-to-gate |
| directed power window | 29.0763016 mW；253 cycles x 3 ns -> 22.0689129 nJ | 正确；mixed-corner、含九个 SRAM Liberty、无 SPEF，不是 frame energy |

### C2/K8

| 稿件数字 | 独立重算 | 裁决 |
|---|---:|---|
| K8 vs K1x8 cycles | `1913/1945`，即 `1.016727653x` | 正确；五个 frozen directed component workloads，不是 trace-weighted 或系统结果 |
| logic area | `131,086.241/585,479.154 um2` | 正确；logic-only、0 macro |
| logic area reduction | `77.610434%` | 正确 |
| throughput/logic-area | `(1945/1913)*(585479.154/131086.241) = 4.541078x` | 正确；必须与只有 `1.0167x` 的周期改善同列 |

这里存在一个确定性标签错误：Table III 的 K8 行使用 `\dctag`，而命令定义为 `[DC/PT]`。M903 只准入 **DC setup/area**，没有 PT、hold signoff、power 或 energy。该行必须改成 `[VCS][DC]`，不能借 C1 的 PT 标签。

### TSBG

| 稿件数字 | 独立重算 | 裁决 |
|---|---:|---|
| VCS post-load cycles | `12,522,876 -> 5,124,365` | 正确 |
| ratio-of-sums speedup | `2.443790792x` | 正确 |
| time reduction | `59.0799669%` | 正确 |
| scalar bank requests | `8,774,304 -> 3,136,608` | 正确 |
| request reduction | `64.2523441%` | 正确 |
| ordinary/TSBG logic | `249,710.451846/249,739.809848 um2` | 正确；matched logic-only schedule-mode ablation |
| area overhead | `0.011756817%` | 正确 |
| sequence rows | 四序列逐行及 aggregate 与 M2057 完全相加 | 正确；新 Table II 有价值，证明不是 best-sequence cherry-pick |

但稿件只写了“real-activity component workloads”，仍少三项 M2057 要求的必要边界：

1. 活动/符号 descriptor 来自 ep34，但 functional weight 是 **deterministic directed INT8 verification weight**，不是捕获的模型权重；
2. 该人口的自然非零 descriptor 全是 `+1`；负号由 directed protocol cover 注入，不能写成真实负极性分布；
3. 1,920 条可引用人口是 **1,917 个 failed-parent attempt 的有效同一 simv 日志 + 3 个 successor 同一 simv 日志**。M2053 仍是 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。若不用一个新的干净 single-image result 替代，正文或表注必须披露这一 lineage。

现稿已经正确写出：383-cycle preload 被排除、286 个 empty workload 保留、7 个非空 workload 略慢、只覆盖 12 个 FC1 和 4 个 G48-supported FC2、不是 full-FC/network。这些边界应保留。

## 二、一个会被电路审稿人抓住的 C1 对象错误

正文约 L121--L124 写成：

> Nine 128x128 1RW SRAM macros hold parent masks and psums.

这与 mapped RTL 不一致。`m528_dead_write_only_1rw_product_capture_island_r2.sv` 中九宏 `u_parent_scratch` 的数据口是 `1152 bit = 96 lanes x signed12` 的 `row_final_packed_w`，保存的是可供 residual reconstruction 的 **parent product row**。`mask_q`/directory 在逻辑状态中，外部 `issue_psum_prior` 与 `psum_write_*` 是另一个边界。M1591 的完整 ledger 另含 metadata/reserve、psum 和 weight，共 214,912 B；九宏只对应其中 18,432 B parent scratch。

因此当前句子不是小措辞，而是会让 reviewer 误以为九宏物理化了 mask+psum storage。合法改法应明确：九宏保存 96-lane parent product rows；mask/directory 为 control state；完整 psum/weight ledger 仅有 105-macro、0.988049 mm2 `[macro area model]`，尚未集成/定时。

同时，“place-equivalent mapping”在没有 placement/CTS/route 的稿里容易造成误解。证据只支持 `mapped synthesis and prelayout PrimeTime`。

## 三、novelty 边界：当前仍有 desk-reject 风险

### C1 对 Prosperity

Prosperity 不只是“隔离 product sparsity 与 bit sparsity”。它已经以运行时 prefix/subset row、residual activation 和 dependency-aware scheduling复用内积，并公开 cycle-accurate simulator。因此，C1 不能把下列对象当新意：

- exact subset parent；
- parent + residual reconstruction；
- product sparsity 本身；
- runtime relation discovery 的一般概念。

C1 可守住的 circuits claim 是：**Prosperity-style opportunity 在有限 214,912-B ledger 和一个真实 1RW parent-product store 上如何变成可执行电路**，具体是 grant-time liveness recheck、single-port deadline arbiter、reserved response queue、forwarding、dead-write suppression 与 atomic commit。当前 Related Work 只说 Prosperity 是“mechanism-level analogue”，承认得不够具体；审稿人容易认为有意淡化最直接 prior。

### C2/TSBG 对 FireFly-T、ELSA、SpikeX 与 WS-LOS

FireFly-T 的直接 prior 不只是“multi-spike execution in a different operator”。其公开文本明确描述单一 memory bank 向 spatial-temporal 维广播 wider vectorized data，以规避多 bank/crossbar conflict。ELSA 也明确使用 bundled AER 与 mini-batch spiking Gustavson product 降 memory access。SpikeX 与 WS-LOS 提供 weight reuse / hierarchical-memory 背景。

TSBG 因此不能声称发明 broadcast、group-major schedule、bundle 或 weight reuse。它可守住的 specialization 是：

- 广播的对象是一个 weight row，而不是 signed product；
- 四个 token 的 sign、destination、tag、terminal 和 Acc24 ownership 私有；
- reuse hit 在 bank request 之前决议，真正抑制 read/address/response，而非乘法器后门控；
- same ports、same LRU4 capacity、same preload 的 ordinary/TSBG A/B；
- stale/out-of-order response 的 typed ownership 检查。

当前文字已承认“not the invention of broadcast”，方向正确，但对 FireFly-T 的直接 memory-broadcast prior 写得过轻。最稳的做法不是增加第三条 idea，而是加一张 **prior/object/resource/what-remains-private** 的四行对照，明确上述对象差。

### Phi

Phi 的主要对象是 offline pattern selection、precomputed pattern-weight product 与 sparse residual/PAFT；它不是 C1 的动态 single-1RW parent lifetime，也不是 TSBG 的 private signed contexts。当前把 Phi 放在 pattern/DSE 背景中基本准确。不要用 Phi 的 `3.45x` 或 Prosperity 的 `7.4x` 来衬托本文 component ratio；跨论文不同网络/范围不能直接排名。

## 四、标题、摘要、主图与表格

### 标题

当前标题准确，但 “Exact Product Capture” 与 Prosperity 的 product sparsity 太接近，第一眼没有把新的约束说出来。更能避免 desk-reject 的标题应把 `single-1RW` 或 `finite-lifetime` 提前，例如：

> Finite-Lifetime Single-Port Product Capture and Context-Safe Weight Broadcast for Spiking Optical Flow

不建议出现 `full-network accelerator`、`processor` 或 `system`。

### 摘要

173 words、6 keywords，形式通过；所有主数字都能重算。科学表达仍有三个问题：

1. 第一结果是 model-only C1，可能让 circuits editor 在首屏认为“性能来自模拟器”。建议把 direct VCS TSBG + equal-bandwidth physical efficiency 放在第一个结果簇，C1 明确写成 cycle-model + mapped island 的第二簇；
2. `fixed real-activity workloads` 没说明 deterministic directed weights 和部分 FC2 scope；至少正文同页必须补齐；
3. `0.0118% matched logic-area overhead`、`2.4438x VCS`来自两个相容但独立的 evidence axes，语法不要让读者理解为“同一 mapped netlist 实测 2.4438x”。

### 主图

Figure 1 左侧仍保留 `C3 coverage only / exact fixed-T=10`。这与正文宣称“两条贡献”冲突，也把已经降级的第三条弱机制重新放回 reviewer 第一眼。应从图中删去 C3，仅画 C1 与 C2/TSBG；腾出的空间用于画普通模式和 TSBG 模式的 weight-read/Acc24 ownership 时序。

### 表格

新加入的四序列表有价值；当前 Table III 仍混合 model、VCS、DC/PT 与 energy，读者必须逐行解码。严格五页版本应保留两张表：

1. C1 ladder（明确 model-only 与 ceiling）；
2. 唯一主表，把 K8 与 TSBG 合成一个 C2 block，并增加 `workload scope / execution evidence / physical evidence / open` 四列。

若页面紧张，四序列数字可做 compact min--max + aggregate，并把空间给 TSBG physical/energy；但不能只留 aggregate、删掉跨序列稳定性。

## 五、硬格式与版面核验

- 当前 PDF：Letter，4 pages，17:59:02 CST 生成；字体全部嵌入，视觉上无明显裁切或 overfull。
- 当前 page 4 right column 上部是 Conclusion，下部才是 References。
- 官方要求是 4.5 页 content + 0.5 页 references，总计 5 页，最后一栏仅 references。当前文件确定性不合规；claim linter 只检查 abstract/字符串，不检查页数和 last-column purity，因此 linter PASS 不能覆盖该失败。
- author block 仍是 `Authors to be inserted` 与操作说明；TCAS-II single-blind 首投必须填真实作者、affiliation、e-mail、ORCID/funding（适用时）。

## 六、P0：首投前必须修复

1. **硬格式**：形成正好 5 页；正文必须在 page 5 left column 结束，page 5 right column only references；用最终 PDF 人工/机器双检，不靠当前 linter。
2. **C1 物理对象**：把“九宏存 parent masks and psums”改为九宏存 96-lane parent product rows，并拆清 18,432-B mapped scratch 与 214,912-B/105-macro full-ledger model。
3. **TSBG 人口边界**：补 deterministic directed INT8 weights、自然 descriptor 全 `+1`、directed negative protocol coverage；披露 1917+3 same-simv lineage，或用一个新干净结果替代。
4. **K8 证据标签**：Table III K8 从 `[DC/PT]` 改为 `[DC]`；不得暗示 hold、PTPX 或 macro-inclusive PPA。
5. **直接 prior**：具体承认 Prosperity 的 subset-prefix/residual/dependency mechanism 与 FireFly-T 的 single-bank spatial-temporal broadcast；把 novelty 限定为 single-1RW lifetime/atomic circuit 和 typed-signed private-context read suppression。
6. **C2/TSBG 物理闭环**：首选用最终相同 ordinary/TSBG identity 闭合 hold 和 matched logic+weight-store energy。若投稿时仍为 `-16.4 ps` 和 power-open，必须从摘要删去任何暗示 timing-closed 的 C2 语法，并接受稿件只能处于边缘 Accept。
7. **投稿元数据**：替换 placeholder authors、操作说明、日期，填合规 affiliation/e-mail/ORCID/funding；封最终 manuscript 与 cover letter。

## 七、P1：显著提升接收率但不新增第三机制

1. **Continuation-safe FC2**：只有在新 960-workload VCS 独立封存后才加入；它是 C2 coverage，不是新 contribution。报告 G>48 chunk 之间 Acc24/terminal ownership 与合并后 2,880-workload ratio-of-sums。
2. **TSBG timing/energy 图**：普通 LRU4 与 B4 TSBG 并排，画清 `hit -> suppress read enable/address/response -> four private signed products/Acc24 commits`；这是最该用来填篇幅的 circuits 内容。
3. **C1 evidence ladder 图**：opportunity/strong-zero -> 1RW cycle model -> real-mask VCS calibration -> nine-macro PT/FM/energy，禁止把各轴拼成同一 RTL speedup。
4. **C2 scope 分解**：说明 K8 base top（131,086 um2）与包含 G48/B4 scheduler 的 matched ordinary/TSBG top（约 249,7xx um2）不是同一个面积 scope，防止 reviewer 误以为面积跳变无法解释。
5. **一张直接 prior 对照表**：只比较 reuse object、decision time、private state、capacity/port 和 evidence type，不做跨网络 raw speedup 排名。

## 八、第五页应该补什么

当前正文大约在 page 4 right column 中段结束，references 也不足以自然推到 page 5。合法补足应全部服务现有两条贡献：

| 优先级 | 内容 | 预计占用 | 作用 |
|---|---|---:|---|
| 1 | TSBG ordinary-vs-broadcast cycle/read-enable 时序图 | 0.45--0.60 column | 直接证明判定发生在被省 SRAM read 之前，最符合 TCAS-II |
| 2 | matched hold/power + common 288-KiB SRAM read-energy 分列表 | 0.35--0.55 column | 把 `-64.25% requests` 落成 circuits energy；不把共同 SRAM 面积算作节省 |
| 3 | C1 model/VCS/physical evidence ladder | 0.35--0.45 column | 防止 1.6945x 被误读为 RTL/mapped speedup |
| 4 | Prosperity/FireFly-T/ELSA/Phi/object-difference compact table | 0.30--0.45 column | 修 novelty desk risk，不比较跨网络倍率 |
| 5 | sealed continuation-safe FC2 coverage 行 | 0.15--0.25 column | 只补范围，不新增机制 |

推荐组合是 1+2+3+4，并让正文在 page 5 left column 结束。若 matched power 尚未通过，不得用 request count 冒充 energy 来填版；宁可扩大 timing/state diagram、基线定义与 prior 差异，也不要加入 C3、S2、RQTB、decoder 或有损剪枝。

## 九、模拟审稿评分

| 维度 | 当前 /5 | 完成 P0 后 /5 | 评语 |
|---|---:|---:|---|
| Novelty | 3.25 | 3.55--3.7 | 两个 specialization 有真实对象差，但 Prosperity 与 FireFly-T/ELSA prior 很近；必须更具体承认 prior |
| Technical soundness | 3.9 | 4.5 | 数字与 seals 很强；C1 SRAM 对象、TSBG人口披露和 K8 标签是当前硬错误 |
| Circuit relevance | 4.1 | 4.5 | single-port、SRAM read suppression、Acc ownership 很契合；matched power/hold 决定上限 |
| Implementation | 3.65 | 4.2--4.5 | C1 商业流强；C2/TSBG hold/power/macro open |
| Evaluation | 3.8 | 4.25 | 四序列/1,920 workload 很好；FC2 不完整、自然负号未出现、weights 为 directed |
| Presentation/compliance | 2.4 | 4.25 | 当前 4 页且 last column 非 refs-only，会被硬退；主图还残留 C3 |
| **Overall scientific** | **3.55** | **4.0--4.2** | 当前 Borderline；修完是可信 Accept 路径 |
| **Upload readiness** | **0/1** | **1/1** | 当前硬格式不合规，不能上传 |

## 十、最终审稿意见

建议 **Major Revision before submission**。稿件不需要第三条弱机制，也不需要为了 TCAS-II 临时做 FPGA。最稳的版本就是两条 circuits contribution：

1. **C1：** Prosperity-style exact product opportunity 在有限容量、single-1RW parent-product lifetime 下的可执行化；
2. **C2/TSBG：** 等带宽 typed-K8 加 context-safe weight-row broadcast，复用 delivery、绝不复用 signed product/Acc24 ownership，并在 SRAM read enable 之前抑制访问。

四序列 TSBG 表是正确增量。下一步应把篇幅用于 timing/state ownership、matched energy/hold、direct-prior 对照和 evidence ladder，而不是再添加 C3、attention、decoder 或有损机制。完成这些后，摘要和表格才能同时满足“数字醒目”与“对象/分母诚实”，显著降低 TCAS-II desk-reject 风险。

## 审阅操作与边界

- `docs/359` SHA256 复验为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- claim linter 当前 PASS：abstract 173 words，6 keywords；但不覆盖页数、末栏、物理对象或 evidence-tag 语义。
- 所有数字均从现有 sealed evidence 重算；没有把正在运行或准备中的 continuation、power、decoder 结果计入正证据。
- 未修改被审论文、RTL、实验、合同或 predecessor。
- 未运行商业工具、simulator、GPU、license query 或网络实验；网络只用于读取官方投稿要求与公开论文/项目页面。
