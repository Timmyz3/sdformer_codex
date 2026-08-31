# M689 decoder ConvTranspose2d → 官方 Prosperity `run_fc` 适配设计审阅

## 1. 裁决

**96/100，`CONDITIONAL_GO_DESIGN_ONLY`，P0/P1/P2 = 0/0/2。**

可以实现一个严格标注为 *external official-artifact decoder support-work replay* 的 Prosperity 适配器，但必须等 M686-r6 产生非覆盖、顶层双封的 canonical payload，并经新的 result hammer 明确准入。当前 canonical M686 目录不存在；因此本审阅不授权运行适配器，也不产生周期或倍率。

M672-r3 已经足以作为精确数学映射核：它把 K3/S2/P1/OP1 ConvTranspose2d 拆成 4 个互斥的目的奇偶 phase，不物化插零图。M618 只能作为“如何导入未修改官方 API、抓取计数器和分层聚合”的参考，不能直接复用其 FC1 形状和 N-tile 扩展逻辑。

## 2. 精确 workload 映射

### 2.1 源 payload 与行序

M686 每条记录的逻辑输入是 `[T,1,Cin,H,W]` FP32 激活的 little-bit-first、C-order mask，`T=10, B=1`。D0/D2/D3 只允许 `EXACT_BINARY_BITPACK`；D1 只允许已准入的 `EXACT_SCALED_BINARY_BITPACK` mask，否则走公共 FP32 fallback。

M672 对每个 sample/module 产生 4 个 `[T,S,Kphase]` 矩阵，`S=H*W`。送入官方 `FC` 时设置：

- `time_steps=T`；
- `sequence_length=S`；
- `input_dim=Kphase`；
- `output_dim=Cout`；
- `activation_tensor.sparse_map` 保持 `[T,S,Kphase]`。

官方 `run_fc` 内部固定执行 `[T,S,K] -> [S,T,K] -> [M,K]`，所以有效 M 行序是 `destination-site-major, then timestep`，`M=T*S`。product 和 bit 必须使用同一行序；不允许为了改善 product matching 单独重排某一模式。

每个 mode 都必须新建 `FC` 对象并重新绑定 activation，因为官方 `run_fc` 会就地改写 `sparse_map` 的 shape/order。operator 名不得以 `_fc` 结尾，否则官方代码会误认为 Conv2d/img2col 并改变 DRAM 计费。建议名称为 `h67_decoder_d{module}_phase{bank}_polyphase`。

### 2.2 phase/tap/K 序

phase bank 顺序必须固定为 `3,2,1,0`，bank 编码为 `(dst_y_lsb<<1)|dst_x_lsb`：

| bank | destination parity `(y,x)` | tap 顺序 | tap 数 |
|---:|---|---|---:|
| 3 | `(1,1)` | `(0,0),(0,2),(2,0),(2,2)` | 4 |
| 2 | `(1,0)` | `(0,1),(2,1)` | 2 |
| 1 | `(0,1)` | `(1,0),(1,2)` | 2 |
| 0 | `(0,0)` | `(1,1)` | 1 |

K 必须是 **tap-major, then source-channel**：

`k = tap_index * Cin + cin`.

越界 source 是结构零，在两个官方 mode 的矩形支持矩阵中位置完全一致。必须另报 `valid_tap_slots`、`structural_padding_zero_entries` 和 `active_tap_events`；不得把越界零写成数据稀疏收益。四个 phase 的 destination 互斥，合起来覆盖 `[T,Cout,2H,2W]`，无跨 phase psum 归并。

### 2.3 权重布局

M686 原始权重是 PyTorch ConvTranspose2d 布局 `[Cin,Cout,Ky,Kx]`、FP32 little-endian C-order。每个 phase 的概念 FC 权重矩阵为：

`Wphase[tap_index*Cin + cin, cout] = W[cin,cout,ky,kx]`,

即 `[Kphase,Cout]`。适配器必须对入参重排后的字节 SHA、shape 与 tap 序做身份审计，并用小形状整数权重 miter 复核 M672 reconstruction。

但官方 Prosperity `run_fc` **不读取权重数值**，只根据 `[K,N]` 和 `weight_tensor.nbits=8` 计算权重流量/支持操作。所以此处的权重映射是 workload identity 和数学完整性证据，不是权重值相似或量化的官方周期证据。

D1 只有在 `exact_zero_or_runtime_scalar_theta_s10=true` **且** `folded_weight_deployment_admitted=true` 时，才允许用 mask 和 `d1.weight.folded_theta.f32le` 进入 exact decoder aggregate。只有 scaled-binary 表示但 folded miter 非 bit-exact 时，可做明确标签的 opportunity-only 统计，不得进 exact 表。

## 3. 冻结形状与 tile 账本

官方配置保持 M618 的 `Mtile=256, Ktile=16, Ntile=128, mem_if_width=1024 bit/cycle, weight=8 bit, activation=1 bit`。不造活动 padding；官方 `cur_tile_size_*` 对尾 tile 按真实尺寸收费，同时另报物理 padding。

| module | `Cin/Cout/H/W` | `M=T*H*W` (`tiles,pad`) | bank 3 `K` (`tiles,pad`) | bank 2/1 `K` (`tiles,pad`) | bank 0 `K` (`tiles,pad`) | `N` (`tiles,pad`) |
|---|---|---|---|---|---|---|
| D0 | 1536/384/15/20 | 3000 (12,72) | 6144 (384,0) | 3072 (192,0) | 1536 (96,0) | 384 (3,0) |
| D1 | 770/192/30/40 | 12000 (47,32) | 3080 (193,8) | 1540 (97,12) | 770 (49,14) | 192 (2,64) |
| D2 | 386/96/60/80 | 48000 (188,128) | 1544 (97,8) | 772 (49,12) | 386 (25,14) | 96 (1,32) |
| D3 | 194/96/120/160 | 192000 (750,0) | 776 (49,8) | 388 (25,12) | 194 (13,14) | 96 (1,32) |

M618 的“先跑 N=128，再乘 `N/128`”只对它的 FC1 整 N-tile 群成立。Decoder 的 D1/D2/D3 有 partial N，多数 phase 也有 partial K；因此每个 sample/module/phase/mode 必须直接调用完整真实 N，禁止用 M618 `expand_n_tiles`。D0 可以额外做 N128×3 与 direct-N384 计数器 miter，但正式结果仍用 direct full-N。

## 4. 官方 support-work 聚合与公平分母

运行粒度是 `10 samples × 4 decoder modules × 4 phases × 2 official modes = 320` 个独立 direct-full-N `run_fc`。每个 phase 一次只物化该 phase，避免四个大矩阵同时占内存；CPU worker 不超过 3，不使用 CUDA。

只允许下列 product-vs-bit 官方倍率：

`speedup_ratio_of_sums = sum(bit.total_cycles over identical admitted calls) / sum(product.total_cycles over the same calls)`.

两个 mode 必须同时锁定：官方 commit/source SHA、Accelerator 全部参数、sample/module/phase 集合、M/K/N、行序、结构零 mask、`spike_stored_in_buffer=false`、`weight_stored_in_buffer=false`和 direct-full-N 调用。不允许 product 模式拿一个更小的子集。

结果分层必须有 phase、sample/module、module、sample 和 overall；overall 除 ratio-of-sums 外另报 per-support-call geomean/min/max/arithmetic mean。四个 phase 周期求和只表示 **support-tile aggregation**，不是 monolithic ConvTranspose2d latency，不假定 phase 并行、流水 overlap、全 decoder 调度或全网完成。

必须同时报 `compute_cycles, raw_issue_cycles, raw_preprocess_cycles, preprocess_stall_cycles, memory_stall_cycles, num_ops, DRAM bits, global-buffer bits, support_nnz`。官方结果只标记 `external_official_artifact=true, ours=false, same_resource=false, full_decoder_latency=false, system_speedup=false`，不得与 K8/Ours 本地周期相除或相乘。

## 5. D1 fallback 处理

输出必须并列两个 population：

1. `official_binary_support_subset`：总是可以包含已准入的 D0/D2/D3；D1 只在上述 scaled-mask + folded-weight bit-exact 门成立时进入。必须明写 module 集合，不得称 decoder-complete。
2. `exact_decoder_complete`：只有 D0--D3 全部有可执行 exact 路径时才产生官方 product-vs-bit 数值。若 D1 是 `COMMON_FP32_DENSE_FALLBACK`，该聚合的 official 周期/倍率为 `null`，原因为官方 spike `run_fc` 只有 1-bit activation memory 合同，不能冒充 FP32 typed-source 引擎。

未来本地同资源表可在 D1 fallback 时引入一个单独验证的 common-FP32 路径，但必须对 B0/B1/K1/K1x8/K8/Ours 每行收取同一个 D1 compute/memory/completion 成本。禁止将 D1 排除后仍标记 decoder-complete，禁止将 fallback 费用设为零，禁止阈值化/四舍五入 D1。

## 6. B0/B1/K1/K1x8/K8/Ours 同资源表接口

这张表属于本地统一 simulator，**不是**官方 Prosperity 表。官方 adapter 只为每个 decoder support workload 提供已封存的形状/稀疏/计数器对照，不直接填下表。

公共资源头必须锁定 `28 nm, 3.0 ns, source_group_lanes=96, peak_source_service=8/cycle, Acc24, macro-rounded SRAM<=245760 B, DRAM=192 B/cycle`，以及完全相同的 D0--D3 payload、phase/K 序、legal-tap mask、权重精度、完成语义和 D1 policy。`iso-lane` 不自动等于 `iso-area`，必须同时有 matched area 和 throughput/mm2。

| row | 执行合同 | 公平用途 |
|---|---|---|
| B0 | 所有 T10 上的每个**合法** source/tap 都执行，包括数值 0；越界 tap 不执行；8-source 公共峰值 | strongest dense same-peak denominator |
| B1 | K 序中固定 8-source group；整组全零才跳过，非空组收取 8 个 source；8 个复制 K1 状态/控制均收费 | project-defined PTB-like structured baseline，不得写 official PTB |
| K1 | 只发出精确 nonzero source/tap，每拍 1 source | low-service diagnostic，不是主要公平分母 |
| K1x8 | 与 K8 相同的 exact nonzero 多重集，8 个独立 K1 端点，复制 scoreboard/state/control/weight ports 全部收费 | K8 的 strongest equal-service denominator |
| K8 | 与 K1x8 相同的 exact nonzero 多重集，共享 typed-K8 scoreboard/Acc24/completion state，外部 8-bank 峰值相同 | 只与 K1x8 讨论同服务周期、面积、能量 |
| Ours | K8 + source-centric phase/tap bundle + 合法边界抑制 + exact atomic completion；D1 fold 只在 bit-exact miter 准入时开启 | decoder-local candidate；仍要对 K1x8 和 K8 直接重跑 |

六行都使用同一数学 polyphase workload。B0--K8 的矩形/扫描开销与 Ours 的 source-centric 生成开销必须显式计入；不允许候选方免费拿 descriptor，而 baseline 还在付矩形物化。Ours 若只节省 frontend/descriptor 而 arithmetic 多重集不变，必须如实报告。

每行 schema 至少包含：

- `configuration_id, workload_manifest_sha256, resource_manifest_sha256, exactness, D1_route, operator_population`;
- `compute/source_scan/preprocess/bank/weight_refill/accumulator/completion/memory_stall/total_cycles`;
- `issued_sources, legal_tap_terms, retired_destinations, descriptor_bits`;
- 每个 SRAM array 的 `read/write bytes` 和 DRAM `read/write bytes`;
- `macro_rounded_capacity, bank/port, clock, lane/service, precision, Acc bits`;
- `area_mm2, throughput_per_mm2, logic/SRAM/DRAM energy` 及为 null 时的 blocking reason;
- 以 B0、B1、K1x8 为固定分母的 ratio-of-summed-cycles，以及 sample/module 的 geomean/min/max；
- `external_official_artifact=false` 与 `decoder_component_only=true`。

这个 adapter 不得直接填 `Ours_C1_C2_C3_exact` 全网行；它只返回 decoder sub-ledger，后续必须在同一统一 simulator 中与 Conv/FC/BN/ATLIF/attention 直接合并重跑。

## 7. 必须绑定的 SHA 与封存门

### 已冻结输入

- M672-r3 mapper: `989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254`;
- M672-r3 test: `d0dd9e5ff7236f0f0436467be28e3158051d3d14c87d00ec5851f5a1d19b20e4`;
- 它实际 import 的 M670-r2: `875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b`;
- M677 review JSON: `46594795672e12ae6fa7ad56d0b4b77c50cf6daf7c5cc646888974f1bf6d76bd`，outer-seal file: `1b348283f4d9f7fafbeca1f1dc9d29d1300b27ececbb72562c3a842a38846ade`;
- M686-r6 capture contract: `cd17f141c2e7dc26b6b9093251ebe98b793e3e3436c7ea1f598dc2b4e1959b04`;
- M686-r6 producer: `1bcff2257e95983ddc77485a41cc4727e082c9297e7312ad534abbb28cf2c630`;
- M511 module/spec contract: `e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e`;
- M618 reference runner: `b6cc50463b7fea36c0ce7403824f28a3af2c2b0b2d9254b949615f9cf505287b`，contract: `b3fd297d2eb24ff70851641d26517fd99e6c882c64017cfdbcc3e6fb2780d928`，M619 review: `3aabe34979f267a5344c1e904a7b0634ec685c63e8de370b2736426860967d80`;
- official Prosperity repository commit `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，必须 clean；`simulator.py/accelerator.py/networks.py/utils.py` SHA 分别为 `eed85a3d.../0e7da67d.../96e217d7.../9d74f729...`;
- docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，只读。

M677 的 GO 只授权 M660 payload integration，不自动授权 official replay。新 adapter 在运行前必须另有冻结 contract 和 fresh static hammer，P0=0/P1=0 且显式 GO。

### M686 canonical 结果到达后的强制门

1. 绑定 canonical 目录、`manifest.json`、`RUN_COMPLETE.txt`、`SHA256SUMS` 和 `SHA256SUMS.seal.sha256` 的文件 SHA，验证顶层双封与文件集恰好一致。
2. 验证 `calls/` 和 `weights/` 的 nested seals，40-cell sample/module lattice，30 条 D0/D2/D3 binary payload，D1 route-specific 文件数，每个 payload 的 path/size/SHA/popcount/tail/shape。
3. 绑定 M686 后继 result-hammer review JSON/内层 manifest/outer seal，要求 P0=0/P1=0 和明确 `payload_admitted=true`。
4. 适配器输出先写新 staging，成功后写 `RUN_COMPLETE`、生成双封、身份复查后 atomic rename；禁止 overwrite。任何异常产生双封 failure receipt，不得留 canonical 半成品。
5. 结果经新的 receipt-blind hammer 复算全部 320 调用群、计数器聚合、D1 边界和双封后，才可标记 external opportunity admitted；始终不是 ours/system/headline。

## 8. 不能直接复用 M618 的地方

1. M618 的 FC1 只有原通道 K；decoder K 是 phase-specific `tap*Cin` 并含逐行结构零。
2. M618 的输入是 `[T,B,H,W,C]`；M686 是 `[T,1,C,H,W]`，必须由 M672 做坐标 gather，不能只 reshape。
3. M618 的 K/N 是整 16/128 tile；decoder 有 partial K/N，不能复用 `expand_n_tiles`。
4. M618 每条是一个 FC；decoder 每条要分成 4 个 phase support operators，并且四 phase 求和仍不是 monolithic latency。
5. M618 的全部输入 exact binary；decoder D1 是 route-dependent，必须有 scaled/folded 准入门和公共 FP32 fallback。
6. M618 不需权重值布局证明；decoder 需要 `[Cin,Cout,Ky,Kx] -> [tap*Cin,Cout]` 身份 miter，即使官方周期路径不读数值。
7. M618 可用相同 `FC1` 名字；decoder adapter 必须避免 `_fc` 后缀的隐式 Conv/img2col DRAM 分支。
8. M618 是开发结果后另审阅；本 adapter 必须从起点就加入非覆盖 staging/atomic publish/双封 failure 边界。

## 9. 最小实现文件与测试清单

最小实现文件：

1. `contracts/m690_h67_ep35_decoder_official_prosperity_iso_workload_contract_r1_20260828.json`；
2. `scripts/run_m690_h67_ep35_decoder_official_prosperity_iso_workload.py`；
3. `system_simulator/tests/test_m690_decoder_official_prosperity_adapter.py`；
4. `system_handoff/scripts/run_m690_h67_ep35_decoder_official_prosperity_one_shot.sh`；
5. fresh static-hammer request/review；正式 result 目录及 receipt-blind result hammer。

最小测试集：

- strict JSON duplicate/nonfinite/path traversal/symlink/extra-file/双封攻击；
- canonical M686 schema/status/40-cell route lattice、D1 三种结果分支、payload SHA/size/popcount/tail 攻击；
- 4 phase 的 tap/order/parity/K 及非方形、边界、尾 M/K/N 攻击；
- 整数/随机小形状 `phase matmul + scatter == torch conv_transpose2d`，权重重排 SHA 复算；
- direct N384 与 N128×3 在 D0 两 mode 的全计数器 miter；D1/D2/D3 强制拒绝 N expansion；
- 每个 mode 新 `FC` 对象，名称不触发 `_fc` Conv 分支，官方 API/source SHA 不变；
- 1/2/3 workers 的输出记录集、次序归一化和聚合哈希一致；
- phase→record→module/sample→overall 整数计数器守恒，ratio-of-sums 独立复算；
- D1 fallback 时 exact-decoder 为 null 且有 blocking reason，不能被设为零或从 population 静默删除；
- B0/B1/K1/K1x8/K8/Ours 表 schema 的公共 workload/resource SHA、同精度、同 legal-tap 集、同 D1 route 强制门；
- 不可变 claim 断言：`ours/full_decoder_latency/system_speedup/energy/ppa/headline=false`；
- docs/359 SHA 在执行前后不变。

## 10. P2 留项

1. 官方 `run_fc` 对已物化的矩形支持进行计费，会把逐行越界 tap 零也包含在 activation-buffer 矩形尺寸中。这对 product/bit 的相对分母是对称的，但不等于 Ours source-centric 硬件流量；论文必须保留此差异。
2. D1 fallback 需要一个独立 typed-FP32 周期/存储模型才能填 decoder-complete 同资源表。在该路径准入前，适配器只能给出 binary support subset 的外部 opportunity。

## 11. Claim boundary

本目录是只读设计审阅。未运行 GPU/EDA/official simulator，未修改 M618/M672/M686/docs359，未产生 payload、周期、倍率、RTL、能量、PPA 或 DATE headline。
