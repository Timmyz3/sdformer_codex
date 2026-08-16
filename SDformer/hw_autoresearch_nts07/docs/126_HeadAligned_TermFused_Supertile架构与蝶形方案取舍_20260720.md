# Head-Aligned Term-Fused Supertile 架构与蝶形方案取舍

## 1. 架构决策

真实 S0-S3 完整 projection 表明，在同步 bias SRAM 合同下 C0 的 Builder 只占总周期 6.68%，C1 即使把 Builder 加速 1.408x，系统也只提高 1.020x。因此下一轮主架构不再扩 Builder workspace，而是直接优化占 93.3% 的 replay/decode、term×output-tile delivery、AccTile 和 bias/final。

本轮提出并实测 **HATF-Supertile（Head-Aligned Term-Fused Supertile，按 head 对齐的 term 融合输出超块）**：

```text
一个 typed-slot head payload
        |
        v
   单次 replay/decode
        |
        v
 final-gate/lane term -----+----> 32-lane weight/Acc bank 0
                           +----> 32-lane weight/Acc bank 1
                           +----> 32-lane weight/Acc bank 2
                                      |
                                      v
                               96-lane supertile
```

它不是增加第二个稀疏核，也不改变 H67 数值。核心是将原来三个独立的 32-lane output tile 合并为一个 96-lane supertile，使相同 typed-slot payload、descriptor 和 term 只 replay/decode 一次，再并行送往三个 32-lane 权重/Acc bank。权重总位数不变，只改变 banking、读带宽和驻留粒度。

当前 RTL 的 `OUT_TILE` 参数已经能表达该 exact 数据流并完成逐元素验证；后续物理实现仍需把 96 lane 显式拆成 `3 x 32` bank，而不是实现一个不可布线的单体超宽 SRAM。

## 2. 为什么是 96 lane

H67 四个 stage 的投影维度为：

| Stage | Head | 通道维度 | 32-lane tile | 96-lane supertile |
|---:|---:|---:|---:|---:|
| S0 | 3 | 96 | 3 | 1 |
| S1 | 6 | 192 | 6 | 2 |
| S2 | 12 | 384 | 12 | 4 |
| S3 | 24 | 768 | 24 | 8 |

96 lane 等于三个 32-lane head dimension，四个 stage 都无 padding。64 lane 在 S0 需要填充 32 lane；128 lane 在 S0/S1/S2 存在不同程度的尾部浪费。96 lane 还允许沿用 32-lane 权重 bank、乘积向量和 AccTile 子块，控制只需在 supertile 级共享 replay/term。

## 3. 真实 RTL 周期消融

范围为 `sample0/B0/window0` 的 S0-S3 共 45 个 head，C0+BPB、adaptive CSR、residency enabled。四种宽度均使用同一 final-gate/K、INT8 权重、bias 和 expected acc32，逐元素失配为 0。

| OUT_TILE | S0 | S1 | S2 | S3 | 四 stage 总周期 | 相对 32 lane |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4375 | 4652 | 28370 | 173906 | 211303 | 1.000x |
| 64 | 3270 | 2860 | 15830 | 91586 | 113546 | 1.861x |
| 96 | 2165 | 2262 | 11650 | 64146 | 80223 | **2.634x** |
| 128 | 2165 | 2262 | 9557 | 50426 | 64410 | 3.281x |

机制证据以 S3 为例：

| 指标 | 32 lane | 96 lane | 变化 |
|---|---:|---:|---:|
| projection head issue | 576 | 192 | 3.00x 减少 |
| projection term | 12888 | 4296 | 3.00x 减少 |
| bias commit | 3888 | 1296 | 3.00x 减少 |
| slot replay | 415 | 143 | 2.90x 减少 |
| projection 周期 | 164985 | 55225 | 2.99x 减少 |

这说明收益来自跨 output tile 的 exact term 复用，而不是删除 token、近似剪枝或改变 Shiftmax/gate/K。

## 4. 开放目标库逻辑面积联合消融

使用同一 Nangate45 typical Liberty、Yosys `dfflibmap+ABC` 对 `gatestack_multihead_decoder_projection_top` 做逻辑映射：

| OUT_TILE | logic area | 面积比 | RTL加速 | 面积归一吞吐 | 未映射 `$mem_v2` |
|---:|---:|---:|---:|---:|---:|
| 32 | 90159.902 | 1.000x | 1.000x | 1.000x | 3 |
| 64 | 146983.886 | 1.630x | 1.861x | 1.142x | 3 |
| 96 | 203975.716 | 2.262x | 2.634x | **1.164x** | 3 |
| 128 | 261017.820 | 2.895x | 3.281x | 1.133x | 3 |

96 lane 在当前代理中具有最高面积归一吞吐，超过 32 lane 约 16.4%；128 lane 虽然绝对周期最低，但边际面积效率下降。因此冻结：

- **默认研究候选**：C0+BPB+HATF96；
- **面积保底**：C0+BPB+32 lane；
- **高吞吐上限**：HATF128，仅作消融；
- **C1**：降级为局部 Builder 流水对照，不与 HATF96 叠加，除非后续绝对吞吐仍不达标。

这些面积数字是 `[开放目标库逻辑映射代理]`，三个 `$mem_v2` 均未计面积，也没有 SDC、STA、SAIF、布线和 SRAM macro。它们只够筛选 96 lane 进入 DC，不是论文最终 PPA。

### 4.1 权重流量守恒与带宽代价

HATF96 不减少稠密 projection 权重的总比特读取量。相对 32 lane，逻辑 weight request 数约减少 3 倍，但每次请求从 256 bit 增为 768 bit；在无 padding 的四个 stage 中，总 weight payload bit 近似守恒。它减少的是重复 replay、decoder、term 控制和请求事务数，并用三个 32-lane bank 的并行带宽换取周期下降。

因此当前不能声称“权重 SRAM 能耗降低 3 倍”。论文必须分别报告：

- logical weight request 数；
- physical 32-lane bank access 数；
- weight payload 总 bit；
- replay/decoder/term 活动；
- 三 bank 峰值带宽与利用率；
- SRAM macro 动态能耗和逻辑动态能耗。

只有 mapped SAIF 与 SRAM 能耗模型证明总能量下降后，才能把 HATF96 写成能效贡献；在此之前只能报告 RTL 周期和开放逻辑面积效率。

真实 RTL 日志分账为：

| OUT_TILE | logical weight req | physical 32-lane bank access | weight payload bit | weight padding | bias payload bit | bias padding |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 15030 | 15030 | 3847680 | 0.00% | 7464960 | 0.00% |
| 64 | 7546 | 15092 | 3863552 | 0.41% | 7630848 | 2.22% |
| 96 | 5010 | 15030 | 3847680 | 0.00% | 7464960 | 0.00% |
| 128 | 3773 | 15092 | 3863552 | 0.41% | 7962624 | 6.67% |

HATF96 恰好做到逻辑请求减少 3 倍、物理 bank access 与 payload bit 不增；64/128 lane 均因 stage 尾部产生额外流量。完整结果见 results/hatf_supertile_traffic_20260720/report.md。

## 5. 与蝶形网络工作的关系

### 5.1 复旦 ISSCC 2023 天溪

复旦团队的 ISSCC 2023 工作使用 **in-memory butterfly zero skipper** 支持非结构化剪枝 Transformer，并结合 CIM local-attention reusable engine。该工作的可迁移思想是：先在存储侧识别可跳过项，再用受控网络只路由有效工作；不能直接搬用其零跳过网络，因为我们的权重统计不支持。

来源：[复旦大学官方介绍](https://fics.fudan.edu.cn/70/b1/c22203a487601/page.htm)、[DBLP论文条目](https://dblp.org/rec/conf/isscc/LiuLZWZJTCLL23)。

### 5.2 MICRO 2022 FABNet

FABNet 用统一 butterfly sparsity 近似 attention 与 FFN，并由可重构 butterfly engine 执行。它属于算法结构化稀疏与硬件联合设计，必须训练网络后才能获得收益；不能作为当前 exact H67 权重的透明替换。

来源：[论文与摘要](https://arxiv.org/abs/2209.09570)、[开源 artifact](https://zenodo.org/record/7010800)。

### 5.3 当前 H67 权重证据

| Stage | INT8零值率 | 2:4可跳过块 | 4:8可跳过块 |
|---:|---:|---:|---:|
| S0 | 1.194% | 0.130% | 0.000% |
| S1 | 1.576% | 0.206% | 0.000% |
| S2 | 1.294% | 0.103% | 0.000% |
| S3 | 1.570% | 0.161% | 0.000% |

当前 projection 权重接近稠密，因此：

1. exact 主线不采用 weight-zero skipping 或 butterfly 稀疏网络；
2. 天溪的“存储侧筛选再路由”被修改为“typed-slot 侧 term 合并后跨三个 head bank 广播”；
3. FABNet 只保留为未来结构化投影算法分支，必须 full30+valid825 重训后与 exact HATF 分表比较；
4. 论文创新不能写成“提出 butterfly network”，而应写成由 H67 final-gate term 与 96-channel stage geometry 共同驱动的 exact head-aligned term reuse。

## 6. 可辩护的架构创新表述

> 提出 HATF-Supertile，将 all-binary 事件注意力产生的 typed final-gate/lane term 在三个 32-lane head bank 间进行一次解码、多 bank 消费；该数据流保持 H67 数值精确、避免跨 output tile 重复 replay/decode，并利用四 stage 均为 96 整数倍的通道几何选择 supertile 粒度。

与普通 SIMD 加宽的区别必须由以下消融支撑：

- 32/64/96/128 lane 同一真实 trace 的周期、面积、功耗；
- 96 lane 的无 padding 与 128 lane 尾部浪费；
- replay、term、bias 和 weight-bank 访问分别分账；
- 96-lane 单体实现与 `3 x 32` banked implementation 的 STA/拥塞/能耗对照；
- 相对 Direct RAW、C0+BPB32 和 C1+BPB32 的同约束 EDP。

## 7. 下一步 RTL 与物理门槛

1. 将 HATF96 从参数化宽总线细化为三个 32-lane weight bank、product lane group 和 AccTile bank；
2. term decoder 只实例化一套，广播前放 elastic register，三个 bank 各有独立 ready，禁止最慢 bank 丢 term；
3. bias 改成带 tag/token 的同步 SRAM req/rsp，数据宽度为 `3 x 32 x ACC_W`；
4. final 接口可按 32-lane bank 分三拍或三路并行，必须把 IO 周期和 SRAM 端口计入；
5. 在同一 500 MHz SDC 下完成 32/96 lane 的 SRAM macro、STA、SAIF 与 mapped LEC；
6. 晋级条件：相对 C0+BPB32，projection EDP 改善至少 15%、WNS/TNS 通过、总面积和功耗不超预算80%。

### 7.1 三 Bank 权重端口已实现

gatestack_hatf96_weight_coalescer 已将一个逻辑 96-lane 请求拆为三个独立 32-lane bank 请求，支持错峰/同拍握手、返回反压、身份校验和三份返回原子拼接。双模拟器、动态 SVA 与 Yosys check 通过，96 lane 逐 lane 0 失配。

这项结果只关闭了权重端口的叶模块可实现性；尚未关闭完整 projection 集成、SRAM macro、STA 和功耗。详见 docs/127_HATF96三Bank权重接口RTL闭环_20260720.md。

## 8. 复现入口

```bash
bash sim_hitflow/run_gatestack_projection_supertile_sweep.sh
bash dc_handoff/scripts/run_gatestack_supertile_nangate45_mapping.sh
python3 scripts/analyze_projection_weight_sparsity.py \
  --vector-root tb_hitflow/vectors \
  --output-dir results/projection_weight_sparsity_20260720
python3 scripts/analyze_hatf_supertile_traffic.py
```

结果目录：

- `results/gatestack_supertile_mapping_20260720/`；
- `results/projection_weight_sparsity_20260720/`。
- `results/hatf_supertile_traffic_20260720/`。

## 9. 后续复审更新

- BSF 已完成 exact RTL 与同约束开放映射，但 flop-based 面积归一吞吐只有 `0.963x`，因此条件降级；详见 `docs/130_BSF精确Bias驻留终结器RTL闭环_20260720.md`。
- `3xIndependent32/HATF96-Central/DCTF96-Distributed` 的等并行度资源与评估合同已冻结；详见 `docs/131_等并行度96Lane公平基线与DCTF晋级合同_20260720.md`。
- 在真实三路独立 wrapper 与 DCTF bank-local 后端完成前，HATF96 仍是候选参数，不是已成立的主架构创新。
