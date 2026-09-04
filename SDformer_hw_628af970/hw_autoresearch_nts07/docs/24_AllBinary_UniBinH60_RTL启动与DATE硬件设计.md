# All-Binary UniBin-H60 RTL 启动与 DATE 硬件设计

**日期**：2026-06-20  
**主线**：All-Binary ATLIF + all12 NTS/H60  
**目标**：判断是否能开始 RTL，提炼可写进 DATE 的硬件设计 idea，并完成主要模块级 RTL/仿真起点。

## 1. 是否足够开始 RTL

结论：**足够开始主要模块级 RTL，不建议直接进入完整顶层控制器和 P&R 级面积功耗结论。**

已经具备的证据：

| 证据 | 状态 | 说明 |
|---|---|---|
| 主 checkpoint | 已完成 | all-binary NTS/H60 ft ep2 |
| valid825 精度 | 已完成 | AEE `1.4891`，AAE `9.7785` |
| spikes / energy | 已完成 | `23.8206G`，`21045.91 uJ` |
| full encoder H60 覆盖 | 已完成 | 40 样本 `480=40×12` 次 H60 调用 |
| ATLIF 格式 | 已完成 | 105 个 wrapper 全 binary，0 ternary |
| P0 Q/K 活性 | 已完成 | Q/K activity 极低，支持 event-gated popcount |
| INT8 score/gate | 已完成 | AEE `1.4916`，几乎不掉点 |
| skip buffer 口径 | 已完成 | 1-bit packed 后 S0/S1/S2 skip 每样本约 `1.45 MB` |

还不适合直接做完整顶层的原因：

1. 还没有逐层 cycle-accurate schedule；
2. 还没有完整 SRAM bank / NoC / decoder replay 控制；
3. ATLIF membrane 内部位宽还未做最终 sweep；
4. 105 vs 93 ATLIF forward coverage 仍需解释未记录模块。

因此 RTL 路线应分两层：

```text
现在做：主要模块级 RTL + testbench
后面做：descriptor controller / SRAM banking / full top schedule
```

## 2. 可借鉴的硬件设计 idea

### 2.1 FireFly-T：binary attention engine + SRAM data manipulation

FireFly-T 提出 dual-engine overlay：稀疏引擎处理 activation sparsity，binary engine 处理 spiking attention；其 binary engine 使用 AND-PopCount，并利用 SRAM byte-write 能力处理 spiking attention 所需的数据流变换，减少专用 transpose buffer。

对 UniBin-H60 的迁移：

| FireFly-T idea | UniBin-H60 迁移 |
|---|---|
| binary attention engine | Binary H60 Consensus Attention Engine |
| AND-PopCount | Q/K overlap popcount、active count、mismatch count |
| byte-write SRAM data manipulation | 1-bit packed event SRAM + token/head/window 布局重排 |
| overlay orchestrator | layer descriptor controller，复用 12 个 H60 block |

可写进 DATE 的点：

> 与通用 spiking-transformer binary attention engine 不同，UniBin-H60 面向 SDformer optical-flow encoder 的 fixed 12-block H60 pattern，score 不直接是 QK 点积，而是 TX/SC-compatible overlap consensus score。因此我们可以用更小的 binary popcount consensus engine 和固定 descriptor schedule 替代通用 overlay 控制。

参考：FireFly-T arXiv 页面说明其 dual-engine overlay、binary engine、AND-PopCount 与 SRAM byte-write 数据流思想：<https://arxiv.org/html/2505.12771v1>

### 2.2 Bishop：Token-Time Bundle 作为调度粒度

Bishop 提出 Token-Time Bundle，把多个 token 和 time point 的 spike workload 打包，作为硬件处理单元；并用 stratifier 把高密度/低密度 bundle 分给不同 core，以利用 spatiotemporal sparsity 和 weight reuse。

对 UniBin-H60 的迁移：

| Bishop idea | UniBin-H60 迁移 |
|---|---|
| TTB as work unit | TTB1/TTB2 作为 H60 score engine 发射粒度 |
| stratifier | `ttb_skip_unit` 判断 empty/active bundle |
| dense/sparse route | 第一版只做 skip / issue，不做复杂双核 |
| bundle-level sparsity | all-binary P0 中 TTB2 empty 在 S1/S2/S3 约 `73.8%/63.0%/64.5%` |

可写进 DATE 的点：

> 我们不照搬 Bishop 的异构 dense/sparse core，而是把 TTB 简化为 H60 前端 work-issue gate：空 TTB 直接跳过 popcount score 和 gated-K，非空 TTB 进入 binary consensus engine。这样控制简单，更适合 DATE 版模块级硬件设计。

参考：Bishop arXiv 页面描述 TTB、stratifier 和 heterogeneous cores：<https://arxiv.org/html/2505.12281v1>

### 2.3 BESTformer：binary event-driven transformer 的软件动机

BESTformer 说明 transformer-based SNN 可以通过 binary event-driven 表示降低存储和计算；同时指出纯二值化存在信息表达能力下降风险，需要配套训练/恢复策略。

对 UniBin-H60 的迁移：

| BESTformer observation | UniBin-H60 结果 |
|---|---|
| binary event 可降低存储/计算 | all-binary 1-bit event SRAM/NoC |
| binary 可能掉精度 | 我们通过 short fine-tune 恢复到 AEE `1.4891` |
| attention binarization 是关键 | H60 score/gate INT8 valid825 稳定 |

可写进 DATE 的点：

> all-binary 不是硬件端强行简化，而是软件训练已验证的结构选择：在 DSEC optical flow valid825 上几乎保持 NB0 精度，并将 spike/energy 降低约 46%/44%。

参考：BESTformer arXiv 页面描述 1-bit 表示降低存储和计算、以及需要缓解二值信息损失：<https://arxiv.org/html/2501.05904v1>

### 2.4 Xpikeformer / SSA：用轻量逻辑替代重乘法

Xpikeformer 的 SSA engine 基于 binary query/key/value，用逻辑 AND 和加法替代真实乘法，强调 spiking attention 的软硬协同。

对 UniBin-H60 的迁移：

| SSA idea | UniBin-H60 迁移 |
|---|---|
| binary Q/K/V | binary Q/K event，gated-K output |
| AND + addition | overlap popcount + integer score fusion |
| attention-specific tile | Binary H60 attention tile |

可写进 DATE 的点：

> UniBin-H60 也避免 dense MAC-style attention，但不同于 stochastic attention；我们保留软件已验证的 H60 Shiftmax gate，并证明 score/gate 可 INT8 部署。

参考：Xpikeformer arXiv 页面描述 binary attention 中逻辑 AND 和 addition 替代真实乘法：<https://arxiv.org/html/2408.08794v1>

### 2.5 Reconfigurable timestep computing：时间维并行和可重构 neuron

硬件高效 Spiking Transformer accelerator 工作强调多 timestep 是 SNN 延迟来源，并用 parallel tick-batching / timestep-reconfigurable neuron 缓解延迟和 membrane memory。

对 UniBin-H60 的迁移：

| idea | UniBin-H60 迁移 |
|---|---|
| time-step reconfigurable neuron | shared binary ATLIF lane 按 descriptor 复用 |
| tick batching | TTB2 聚合 token/time work item |
| membrane memory pressure | 第一版用 INT16 membrane，后续 profiling 决定位宽 |

可写进 DATE 的点：

> 我们暂不做全 T 并行，而采用 TTB2 作为面积/控制折中：既利用 all-binary 的高空 bundle 比例，又避免完整 tick-parallel top controller 复杂化。

参考：Hardware Efficient Accelerator for Spiking Transformer with Reconfigurable Parallel Time Step Computing：<https://arxiv.org/html/2503.19643v1>

## 3. DATE 论文可讲的硬件贡献

建议最终贡献点写成：

1. **Unified Binary Eventization**  
   105 个 ATLIF site 全部输出 1-bit event，消除 ternary rail 和 mixed-format 控制。

2. **Binary H60 Consensus Attention Engine**  
   12 个 encoder H60 block 共享同一 score-gate-output 数据流，score 由 overlap / active / mismatch popcount 派生。

3. **TTB2 Work-Issue Gating**  
   以 Token-Time Bundle 作为调度单元，空 TTB 跳过 score engine 和 gated-K。all-binary P0 中 S1/S2/S3 的 TTB2 empty ratio 约 `73.8%/63.0%/64.5%`。

4. **INT8 Deployable Shiftmax Gate**  
   score/gate INT8 + `mu=1/16` valid825 几乎无损，AEE `1.4916` vs float `1.4891`。

5. **1-bit Packed Event SRAM and Skip Replay**  
   S0/S1/S2 pre-downsample skip 每样本从 FP16 `23.22 MB` 降到 1-bit packed `1.45 MB`。

## 4. 本次新增 RTL

新增目录：

```text
/root/private_data/work/SDformer/hw_autoresearch_nts07/rtl_allbinary
/root/private_data/work/SDformer/hw_autoresearch_nts07/tb_allbinary
/root/private_data/work/SDformer/hw_autoresearch_nts07/sim_allbinary
```

模块：

| 文件 | 模块 | 作用 |
|---|---|---|
| `unibin_h60_pkg.vh` | constants | all-binary H60 参数，`HEAD_DIM=32`，`MAX_TOKENS=162`，`mu=1/16` |
| `binary_atlif_unit.v` | `binary_atlif_unit` | binary ATLIF comparator，输出 1-bit event |
| `binary_popcount_consensus.v` | `binary_popcount_consensus` | Q/K overlap、active、mismatch、TX/SC-compatible fused score |
| `ttb_skip_unit.v` | `ttb_skip_unit` | TTB empty 检测和 active_count |
| `shiftmax_int8_unit.v` | `shiftmax_int8_unit` | INT8 gate 的 Shiftmax 近似 scaffold |
| `gated_k_unit.v` | `gated_k_unit` | K event 与 INT8 gate 调制 |
| `unibin_h60_token_core.v` | `unibin_h60_token_core` | token-level score + gated-K 组合 |
| `tb_unibin_h60_modules.v` | testbench | 模块级烟测 |

这不是完整加速器顶层，而是 DATE 主图里最关键的可综合模块起点。

## 5. 仿真环境和结果

环境：

```text
Ubuntu
iverilog 12.0
```

运行命令：

```bash
cd /root/private_data/work/SDformer/hw_autoresearch_nts07
./sim_allbinary/run_iverilog.sh
```

结果：

```text
PASS: UniBin-H60 module smoke tests passed
```

仿真覆盖：

1. ATLIF threshold comparator；
2. binary Q/K popcount 统计；
3. overlap / mismatch / fused score；
4. TTB empty / non-empty 检测；
5. Shiftmax INT8 gate 非零输出；
6. gated-K 输出。

## 6. 最小量化状态

已完成：

| 项 | 状态 |
|---|---|
| event activation | all-binary，天然 1-bit |
| H60 score | INT8 deploy valid825 已验证 |
| H60 gate | INT8 deploy valid825 已验证 |
| `mu` | `1/16` power-of-two valid825 已验证 |
| Q/K score accumulator | 静态范围可覆盖，RTL 使用 16-bit internal |

仍需后补：

| 项 | 是否阻塞当前 RTL | 说明 |
|---|---|---|
| ATLIF membrane 位宽 sweep | 不阻塞 | 当前 RTL 用 INT16 comparator，后续可压到 INT12/INT10 |
| threshold 定点格式 | 不阻塞 | all-binary official ATLIF 当前 threshold 以 1.0 为主 |
| Conv/Linear weight quant | 不阻塞 DATE 主线 | 可作为 future work 或 appendix |
| full-chip SRAM banking | 不阻塞模块 RTL | 顶层控制阶段再做 |

## 7. 下一步

建议下一步按这个顺序：

1. 把 `binary_popcount_consensus` 的 score 公式和 PyTorch H60 中 TX/SC 的归一化口径进一步对齐；
2. 增加 per-stage descriptor test：S0/S1/S2/S3 token/head 参数不同，但复用同一模块；
3. 增加 TTB2 issue test：空 bundle 不触发 score/gated-K；
4. 写 all-binary 版 module interface spec，替代旧 `05_module_interface_spec.md` 的 binary/ternary mixed 字段；
5. 再考虑简化 top wrapper，不做完整复杂 controller，只展示 descriptor-driven module reuse。

当前判断：

```text
all-binary UniBin-H60 已经可以进入主要模块级 RTL。
DATE 论文第一版不需要完整顶层控制器，重点应放在 binary event datapath、
popcount consensus H60、TTB2 work gating、INT8 Shiftmax gate、1-bit skip SRAM。
```
