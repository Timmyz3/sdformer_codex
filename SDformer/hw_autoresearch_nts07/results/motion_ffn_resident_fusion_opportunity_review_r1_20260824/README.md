# Motion/H67 FFN expand-contract resident fusion 独立机会审计 r1

## 结论

FFN 是值得做 standalone resident handoff 的方向，但当前只能把它定义为**数据驻留与调度模块**，不能声称已有加速：当前 620,302,905-cycle RQTB 包络没有任何物理 memory cycle，resident fusion 在这个分母里可证明的加速是 **1.0000×**。

H67 有 12 组严格有序的 `fc1 -> sn2 ATLIF -> fc2`，profile100 中合计 159,784,111 model cycles/frame，占 RQTB 包络 25.7590%。十样本 ordered trace 中，12 组相邻关系全部 10/10 出现。模块范围和热点可信，但中间地址、SRAM/DRAM 读写、数值桥与 macro 都未闭合。

本审计纠正一个容易夸大的表述：350,208,000 B 的 `fc1` output 和 350,208,000 B 的 `fc2` input 是 materialize-all 代理中的两个端点，二者之间还有 ATLIF，且前者是稠密连续值、后者是二值事件，**不是同一张量的一次 write + read**。700,416,000 B 只能称为两个 INT8 endpoint proxy 的和，不能称为已证明可消除流量。

独立评分：**6.2/10，P0=0、P1=4、P2=4**。结论等级为 `PASS_OPPORTUNITY_DECOMPOSED_RESIDENT_SPEEDUP_NOT_ADMITTED`。

## other 算子的精确拆分

| 类别 | 模块数 | 周期模型/帧 | 占 other | materialize-all INT8 I+O 代理 | 唯一 INT8 权重 |
|---|---:|---:|---:|---:|---:|
| FFN expand `fc1` | 12 | 118,370,114 | 59.41% | 437,760,000 B | 8,626,176 B |
| FFN contract `fc2` | 12 | 41,413,997 | 20.79% | 437,760,000 B | 8,626,176 B |
| downsample Linear | 3 | 21,012,750 | 10.55% | 48,384,000 B | 1,548,288 B |
| patch projection Conv1x1 | 1 | 18,432,000 | 9.25% | 92,160,000 B | 9,216 B |
| other 合计 | 28 | 199,228,861 | 100% | 1,016,064,000 B | 18,809,856 B |

FFN 占 other 周期的 80.20%。24 个 FFN module 的逐模块形状、活动、周期、字节和 M63 eligibility 已写入 `ffn_module_ledger.csv`；12 个 pair 的 join 与 buffer 下界见 `ffn_pair_ledger.csv`。

## 四个 stage

| stage | pair | 形状 `C -> 4C -> C` | H×W | fc1 cycles | fc2 cycles | pair cycles | fc1 output INT8 endpoint | fc2 input packed-1b endpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 96→384→96 | 120×160 | 22,727,084 | 4,723,595 | 27,450,679 | 147,456,000 B | 18,432,000 B |
| 1 | 2 | 192→768→192 | 60×80 | 9,655,207 | 6,659,649 | 16,314,856 | 73,728,000 B | 9,216,000 B |
| 2 | 6 | 384→1536→384 | 30×40 | 68,513,333 | 21,232,455 | 89,745,788 | 110,592,000 B | 13,824,000 B |
| 3 | 2 | 768→3072→768 | 15×20 | 17,474,490 | 8,798,298 | 26,272,788 | 18,432,000 B | 2,304,000 B |

stage 2 是最大热点，占 FFN 周期 56.17%。所有 pair 都是 T10，且 expanded channel 恒为输入的 4 倍。

## 四本互斥机会账

### 1. Resident movement

- `fc1` output INT8 endpoint proxy：350,208,000 B/frame。
- `fc2` input INT8 endpoint proxy：350,208,000 B/frame。
- `fc2` input 按已知二值合同打包：43,776,000 B/frame。
- 可证明 external bytes saved：**0**，因为没有地址与 residency trace。

用于敏感性分析时，可把前 ATLIF 的 INT8 endpoint 与后 ATLIF 的 packed-1b endpoint 分开相加为 393,984,000 B，但仍不能称为真实 R/W。若额外假设这些字节完全串行、完全离片、没有 overlap 和本地访问代价，256-bit/cycle 接口对应 12,312,000 cycles，条件比值仅 1.01985×；它不是实测 speedup。

### 2. Weight buffer reuse

`fc1` 和 `fc2` 的权重各 8,626,176 B，总计 17,252,352 B。它们形状互为 expand/contract，但名称和参数对象不同，没有 checkpoint 证据证明数值 tied/transposed，因此 weight-value reuse 为 false。

可复用的是**容量**：一个 256×96×INT8 tile buffer 为 24,576 B，可串行供 `fc1` 和 `fc2` 使用。M63 对 eligible module 已经假设每个完整矩阵每帧 cold-load 一次，resident bridge 不能再把这笔 weight DMA 当新增 savings。

### 3. Activation sparsity

- `fc1` profile100 输入活动率：13.3827%。
- `fc2` profile100 输入活动率：4.12935%，即 sn2 后约 95.87% 为零。

这些零值已进入 activity-weighted cycle model；resident handoff 只保留稀疏事件，不会再次节省相同 product term。M63 覆盖 10/12 `fc1` 和 12/12 `fc2`，其 FFN-only opportunity model 中 spatial K1/K4 为 1.5045×、temporal K1/K4 为 1.7042×，但 K4 保持 product update 数不变，只减少 source/front-end service，不能与 residency 相加，也不是 RTL/system speedup。两层 stage3 `fc1` 因输入非二值没有进入 M63。

### 4. Structural sparsity

当前没有 checkpoint-bound weight-zero、N:M、block mask、paired-channel pruning 或 rank census。结构稀疏收益一律为零，不能从激活率推导。

算法侧若要反哺硬件，建议训练共享 pair mask：只有在删除某个 `fc1` output channel 的同时删除对应 `fc2` input column，且 mask 对齐 16/32-channel group 时，才允许硬件跳过完整 weight group。另可正则化 sn2 在 T10×spatial tile 上的全零 channel group，但必须与现有 scalar activation sparsity分开统计，并以 valid825 AEE 与 event-rate guardrail 验收。

## 最小 standalone 合同

建议模块名：`qfit_ffn_t10_resident_pair_bridge_l96`。

它是 movement/controller island：上游 `fc1` 算术、精确 ATLIF 服务和现有 M66-class `fc2` source engine 通过 ready/valid 连接；模块自身不发明第二套 MAC，也不把 M31 rank3 当作当前 H67 的透明 ATLIF。

### 接口与几何

- 固定 T10、96 lanes、256-bit source tile、8 weight banks。
- 支持 `C={96,192,384,768}`、`Cexp=4C`，上游 `fc1` 算术外置，因此 resident handoff 可覆盖全部 12 pair。
- 48-bit context tag 必须包含 sequence、module、spatial、expanded-tile 和 timestep 身份。
- `fc1_acc` 输入为 96×signed19；ATLIF reply 为 96-bit binary event；`fc2` 最终 global accumulator 默认 signed24。
- 权重为 signed INT8。3072 个最坏 `-128/127` 项的和为 `[-393216,390144]`，signed20 可覆盖不含 bias 的 weight sum；bias/requant 范围尚未冻结，因此 prototype 使用 signed24 并要求数值证明。

### 最小 buffer 下界

| 项 | 大小 |
|---|---:|
| 一组 96-lane、T10、signed19 的 fc1 temporal tile | 2,850 B |
| 最大 Cexp=3072 的单 spatial token、T10 sn2 binary replay | 3,840 B |
| 一组 96-lane、T10、signed24 的 fc2 output accumulator | 2,880 B |
| 一个共享 256×96×INT8 weight tile | 24,576 B |
| 串行最小合计 | **34,146 B** |
| temporal/acc/weight 双缓冲合计 | **64,452 B** |

这些是 bit-tight 下界，不含 macro rounding、tag、FIFO、ECC 和 bank padding。

### VCS 边界

必须覆盖四个 stage、12 个 pair、T10 完整性、expanded-channel tail、output-tile replay、任意 backpressure、buffer full、同拍 pop/push、reset/sequence/module switch 和 stale-tag/malformed-timestep 攻击；使用真实 ordered tile payload 做 fused/unfused bit、顺序、multiplicity 和最终 Acc24 miter。

VCS 还必须证明只存在一个 shared weight tile 和一个 M66-class source engine。若 ATLIF 仍是外部 oracle，VCS 只能 admission movement；不能 admission 训练后数值、物理流量、cycle、energy 或 speedup。

### DC 边界

- Phase A：TSMC28、3 ns、ideal clock、ZeroWireload 的 controller/datapath logic-only DC；SRAM 抽象、macro_count=0，只能报告逻辑面积和时序。
- Phase B：用至少 34,146 B 串行或 64,452 B 双缓冲的真实 SRAM macro，加入真实端口、weight bandwidth、PT/SAIF/PTPX 和仲裁后，才允许谈 PPA。
- 必须审计一个 source engine，而非隐式复制两套 96-lane MAC。

## Amdahl 边界

| 假设 | 上限 |
|---|---:|
| 只做 residency，使用当前 compute-only 包络 | **1.0000× admitted** |
| `fc1` 算术完全免费，与 residency 无关 | 1.2358× |
| `fc2` 算术完全免费，与 residency 无关 | 1.0715× |
| 全部 FFN 算术完全免费，与 residency 无关 | 1.3470× |

后面三个只用于说明热点规模，不是设计预测。任何 M63 K4、结构剪枝或 residency 数字都不得相乘或直接相加。

## 缺少的数据与最短闭环

1. 生成带 object/residency/address/cycle 的 `fc1 output -> sn2 input/output -> fc2 input` baseline trace。
2. 导出全 12 pair、profile100 与 heldout 的 ordered `fc1 Acc` 和 `sn2 bit` tile payload。
3. 冻结 FFN INT8 weight/bias、Acc19→ATLIF 与 final requantization，证明 signed24 全链数值。
4. 选择 34–65 KB 对应的 SRAM macro、端口和 banking，跑 macro-aware DC/PT/PTPX。
5. 算法侧再训练 paired-channel group mask，并跑 valid825；此前结构稀疏为零收益。

## 评分

| 维度 | 分数/10 |
|---|---:|
| 数据身份 | 9.5 |
| 算子与 pair 覆盖 | 9.5 |
| movement 证据 | 4.0 |
| 数值就绪度 | 3.0 |
| standalone 合同 | 7.5 |
| 声明卫生 | 10.0 |
| 综合 | **6.2** |

未修改 production 或 `docs/359`，其 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复跑

```bash
python3 results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/build_review.py
sha256sum -c results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/source_manifest.sha256
sha256sum -c results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/manifest.sha256
```
