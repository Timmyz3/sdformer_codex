# M501 ExSpike-APEC 预执行独立打铁评审（r1）

日期：2026-08-27  
评审范围：只读检查合同、分析器、exact-SHA runner、两份冻结 manifest 与 ExSpike 固定提交；未运行 M501 analyzer，未启动 VCS/DC/PT/GPU，未修改生产文件或 `docs/359`。

## 结论

**裁决：`HOLD_BEFORE_EXECUTION`，78/100。**

M501 的核心计数公式与 `T,B,C,H,W` 空间重排是正确的；horizontal-G2 预声明点、train/validation 身份隔离、event-work/Amdahl/RTL 边界也基本守住。当前不应直接运行 exact runner，因为有两个必须先修的 P0：

1. 合同把 24-bit scratch 写成“published overlap-partial-sum storage width”，但 ExSpike 论文只给符号化公式，固定提交的官方 RTL 实际使用 16-bit `overlap_cal_res`。24 bit 可以保留为保守的项目代理，不能写成 ExSpike 已发表位宽。
2. 代码只检查“五维 shape”，没有兑现合同中“shape mismatch fail-closed”的 H67 精确几何/覆盖要求；应冻结并断言两个 cohort 的 schema、status、record/sample/sequence/operator 数、精确算子集合、`[10,1,768,15,20]`、`Co=768`、`K=3x3` 及关键 Conv 几何。

另有一个必须收窄的创新口径：当前 168 个 record 都是“零 + 每个 operator 唯一的正幅值”两码本，负值计数为零。因此本次 bit-exact overlap 在该 trace 上等价于 support intersection；它不能证明一般 signed-analog APEC，也不能形成相对 ExSpike 的 novelty delta。现有 `new_rtl_admitted=false` 是正确结论。

## 评分

| 维度 | 分数 | 结论 |
|---|---:|---|
| APEC 语义映射 | 17/20 | 交集、non-overlap、`(g-1)|O_G|` 均正确；signed-analog 差异化尚未被真实数据激活 |
| 张量与边界计数 | 20/20 | horizontal/vertical G2/G4/G8 reshape 正确，边界保守保留，不跨行伪造相邻 |
| fail-closed / identity | 14/20 | SHA/zlib/raw SHA/dtype/元素/非零计数很强；精确 cohort 几何和覆盖断言缺失 |
| claim boundary | 17/20 | S10/S32、Amdahl、非系统、无 RTL 边界正确；状态和 README 的 gate 决策还不够显式 |
| 成本/可执行性 | 10/20 | 20.25 KiB 算术正确且未冒充完整成本；位宽来源错误，端口/延迟/比较/commit 仍未定价 |
| **总分** | **78/100** | **修完 P0 后才允许跑 opportunity audit；无论结果多高仍不准入新 RTL** |

## 1. ExSpike APEC 语义映射

固定提交 `51accc76936588705255487d101fcc80092b98ce` 的三项外部身份已独立复核：

- `rtl/sparse_processing.v` SHA256 = `3b1001cc520386d136808594374a93a5f489de00e8496a101bba65de95519444`；
- `rtl/weight_acc.v` SHA256 = `a855476563553af8826dea477a6b77d5c4bdb60b1fe280d7a16709cf43f7632c`；
- 官方 PDF SHA256 = `93d92f41f816cee482da731987e70204f6cdb23cfb2e3726db97642c530bb3d5`。

论文定义 `O_G = intersection(S_i)`，消除事件数为 `(g-1)|O_G|`；官方 RTL 对 group 内 spike vectors 做 AND 得到 overlap，再以 XOR 得到各 member 的 non-overlap。M501 的：

```text
all_active AND exact_same_float32_bits
redundant = (g - 1) * intersection_population
candidate = baseline - redundant
```

是对二值 APEC 的保守 exact-value 泛化。不同非零幅值、不同符号、仅 support 相同都不会被合并，数值安全方向正确。

但是，当前两份冻结 manifest 的四个 operator 各自都只有两个 float32 codeword：`0x00000000` 与一个 operator-constant 正幅值；所有 record 的 `negative_count=0`。因此当前 exact-value 条件不会比 support-only 条件更严格。本轮可以证明“这批 H67 bottleneck 输入上 APEC 支持交集可精确复用”，不能证明“支持任意 signed analog 值的差异化硬件”。合同第 30 行的 future novelty 描述必须收窄或明确写成尚未激活的条件路径。

外部对照还支持 M501 的谨慎边界：ExSpike 报告 G2 event reduction 明显高于实际 throughput gain，并明确指出 weight/buffer cycles 会限制转化。因此 event ratio 不能改名为 throughput。

## 2. `T,B,C,H,W` reshape 与边界公平性

冻结记录精确形状均为 `[10,1,768,15,20]`。

### Horizontal

`values[..., :full_extent]` 保留 `T,B,C,H,W` 顺序，再 reshape 为：

```text
T,B,C,H,(W/G),G
```

最后一维取 AND/equality，确实是在同一个 `t,b,c,h` 内对连续 x 位置分组，不会混 channel、timestep、batch 或 row。

### Vertical

先把 H 从 axis 3 移到末维，得到：

```text
T,B,C,W,H
```

再 reshape 为 `T,B,C,W,(H/G),G`，最后一维是连续 y 位置。此实现正确，不会把 W 当成 H。

### 边界

对于本 workload：

| Axis | Extent | G2 | G4 | G8 |
|---|---:|---:|---:|---:|
| horizontal | W=20 | grouped 20 / tail 0 | grouped 20 / tail 0 | grouped 16 / tail 4 |
| vertical | H=15 | grouped 14 / tail 1 | grouped 12 / tail 3 | grouped 8 / tail 7 |

tail 没有从 baseline 中删除；算法只减去完整 group 的 redundant events，所以未分组边界按原事件保留。horizontal 不会跨 row 把 `(x=19,y)` 与 `(x=0,y+1)` 当相邻。horizontal-G2 又与 ExSpike 默认 G2 和官方 raster 方向一致，是合理的预声明点；vertical/G4/G8 只能留作 DSE，不应 post-hoc 替换 headline 点。

## 3. SHA / zlib / bit-exact fail-closed

已通过静态检查确认 runner 固定：

- contract SHA；
- analyzer SHA；
- M40 validation manifest SHA；
- M73 train-calibration manifest SHA；
- `docs/359` SHA；
- output directory 必须不存在。

analyzer 还逐 record 检查 compressed payload SHA/bytes、zlib 解码长度、decoded SHA、float32 dtype、元素数、nonzero count。`np.frombuffer(..., dtype="<f4")` 与 manifest 的 LE payload 一致；`view(uint32)` 后按位相等能区分所有非零 float32 codeword。输出 result/README/RUN_COMPLETE 最后进入 SHA seal，且 analyzer 在完成所有输入分析后才创建输出目录，失败原子性较好。

必须补的 fail-closed 项：

1. 对 M40/M73 分别断言精确 schema 与完整 status，不接受任意 `PASS_*`。
2. 断言 M40 为 40 records/10 samples/1 sequence/4 operators，M73 为 128/32/18/4，并断言 `(sample,operator)` 唯一且为完整笛卡尔积。
3. 断言精确 operator set、`T=10,B=1,C=768,H=15,W=20`、`in_channels=out_channels=768`、`groups=1`、`K=3x3`、stride/dilation/padding 均为冻结值。
4. 拒绝 NaN/Inf，并将 bit view 明写为 `view("<u4")` 或增加 little-endian host 断言，避免当前 `view(np.uint32)` 的宿主端序隐含假设。
5. 校验 `elements == prod(shape)`、`input_content_bytes == elements*4`、`output_shape` 与预期一致。

这些缺口目前不会让已固定 manifest 被静默替换，因为 manifest SHA 已钉死；但它们使“合同字段变化后仍 fail-closed”的可维护性和外部复跑可信度不够，属于执行前 P0。

## 4. S32 train 与 S10 validation 边界

当前身份事实是：

- M40：validation S10，10 个窗口，全部来自 `zurich_city_09_a`，不是 heldout multi-sequence；
- M73：train-only S32，32 个窗口，覆盖 18 个 DSEC train sequences，只能作 robustness/calibration 支撑；
- 两者都使用 H67 ep35 四个 bottleneck Conv inputs。

合同的 `train_calibration_multi_sequence=true`、`heldout_multi_sequence=false`、`full_network=false` 是准确的。预声明决策只使用 validation horizontal-G2，避免使用 train DSE 挑最好点。

建议 result 同时输出 train 18-sequence 的 min/median/max 或至少最弱序列，但不得把它升级成 heldout generalization。S10 也不能表述为十条序列。

## 5. event reduction、Amdahl 与 scratch 税

### event reduction

公式正确：`baseline/candidate` 是 event-work reduction ratio，不是 cycle speedup。`exact_overlap_events` 是压缩后只执行一次的 group intersection population；`redundant_events` 才是消除的原事件数。论文和 README 应保持这两个术语不混用。

### Amdahl sensitivity

冻结占比为：

```text
79,630,957 / 620,302,905 = 12.8374309322%
```

`1 / (1-p+p/s)` 公式正确。它是把 validation event-work ratio 理想地施加到全部四层 Conv envelope 的敏感性，不含 grouping/weight/SRAM/commit 周期，现有 `system_speedup=false` 与 “ideal envelope sensitivity only” 标签正确。

两个门几乎重合：event ratio `1.2x` 对应 ideal sensitivity `1.0218635x`；`1.02x` sensitivity 对应 event ratio约 `1.1802747x`。这不是错误，但应说明第二个门不是独立强证据。

### 20.25 KiB

算术正确：

```text
768 * 3 * 3 * 24 = 165,888 bit = 20,736 byte = 20.25 KiB
```

来源表述不正确：论文只发表 `M_ov ≈ Co*k^2*w_acc`，固定提交官方 RTL 中 `overlap_cal_res` 是 16 bit；项目当前 bottleneck Conv 主路径常用 19-bit accumulator。建议二选一：

- 把 24 bit 明确标成“保守 H67 proxy”，并同时列 19/24/32-bit DSE；或
- 对齐冻结 H67 19-bit 数值合同，报告 16.03125 KiB，再单列 guard/port/ECC/rounding 的物理税。

无论选择哪条，scratch 只能写成容量项，不能写成完整面积/能量税。group input buffering、comparison、1R1W/2R1W 端口、macro latency、shifted destination commit、边界 mask、与 M473/M498 parent/PWP 的冲突仍必须另计。当前 `new_rtl_admitted=false` 正确。

## 6. 代码错误与性能风险

### P0

1. 修正 24-bit “published width” 的错误来源陈述。
2. 增加精确 cohort/geometry/coverage fail-closed 断言。
3. 把“signed-analog novelty”改成条件性未来主张；当前结果必须显式记录 `two_codeword_positive_only=true` 或等价 codebook 事实。

### P1

1. `status` 目前无论 gate 真假都写 `PASS_EXACT_OPPORTUNITY_AUDIT_NO_RTL_ADMISSION`。应增加单一 `opportunity_gate_pass = event_gate && sensitivity_gate` 与 `next_action = KILL / ALLOW_SAME_RESOURCE_CYCLE_FASTKILL`；README 也要显示两个 gate，避免把“脚本成功”误读为“机会晋级”。
2. 顶层 result/README 应显式输出 selected point 的 scratch 位数/字节/KiB、位宽来源标签和 `costs_unpriced=true`，不要只把 scratch 藏在 detailed rows。
3. 增加守恒断言：`candidate + redundant == baseline`、`redundant == (g-1)*overlap`、`overlap <= floor(grouped_baseline/g)`。
4. train robustness 不参与 headline，但可输出 18-sequence min/median/max，防止 aggregate 掩盖场景崩塌。

### P2 / 性能

当前实现对 168 records × 6 DSE points × 2.304M values 重复做约 23.22 亿元素级 active/equality 检查。峰值内存是逐 record、可控，主要风险是运行时间而非 OOM。可以一次预计算 bit/support 后复用，但不要为了提速改成 support-only；当前 exact bit condition 必须保留。

## 7. 执行准入清单

修正后可以运行 M501 opportunity audit，当且仅当：

1. exact-SHA runner 更新并重新钉住 contract/analyzer SHA；
2. 上述两个 P0（成本来源、精确 cohort/geometry）关闭；
3. result 明确当前 trace 是 positive two-codeword，而非一般 signed analog；
4. 无论 gate 是否通过，都保持 `new_rtl_admitted=false`；
5. 只有 opportunity 两门通过，才允许做下一步 identical-resource cycle fast-kill；下一步还必须定价 scratch macro 端口/延迟、group buffer/comparator、weight readiness 和 shifted destination commit；
6. 没有 H67-native novelty delta 之前，不开发 APEC RTL，不把 ExSpike 的机制写成项目贡献。

最终判断：**数学机会审计值得跑，但当前版本应先修合同和 fail-closed；它最多决定“是否值得做同资源周期模型”，不能决定“是否开发新 RTL”。**
