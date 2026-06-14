# 统一 ATLIF-PSN 异构算子与数据流（软件双模收敛版）

**软件锚点**：`ATLIFTernaryPSN(output_mode="ternary"|"binary")`  
**硬件模块**：`rtl/atlif_unified_encode_unit.v`  
**模式开关**：`ternary_en`（per-layer，来自 descriptor / APB LUT）

---

## 1. 为什么改叙事：两种神经元，一个算子

软件侧正在收敛为 **只保留两种 ATLIF PSN 推理模式**，不再混用独立 PSN + ATLIF + PSN 三条路径：

| 模式 | 软件类 | 输出 | 典型挂载 |
|------|--------|------|----------|
| 三值 | `output_mode="ternary"` | {-thre, 0, +thre} | S2 Q/K（H60 输入） |
| 二值 | `output_mode="binary"` | {0, +thre} | Patch / FFN / 其余 SN |

二者共享同一套 ATLIF 阈值比较与前向累加语义，仅 surrogate 出口不同。硬件用 **单一比较器树 + `ternary_en` 掩码负半轴** 即可覆盖，比「三套神经元、三套编码器」更适合 DATE 的 co-design 故事。

**DATE 表述建议**：

> We deploy one inference-frozen ATLIF-PSN operator across the network; software only configures per-layer ternary or binary emission, while the accelerator exposes a single `ternary_en` mode bit in the layer descriptor.

---

## 2. 统一算子微架构

```text
activation (FP16)
    │
    ├─ compare ≥ pos_thresh ──► pos_fire
    └─ compare ≤ neg_thresh ──► neg_fire_raw
              │
              ▼
         neg_fire = ternary_en & neg_fire_raw
              │
    spike_out[1:0] = pos ? POS : neg ? NEG : SILENT
    binary_out     = pos_fire          // Sparse MAC 直接吃 1bit
```

| `ternary_en` | `spike_out` | `binary_out` | 下游 |
|--------------|-------------|--------------|------|
| 1 | 2-bit 三值 | 忽略 | H60 TX/SC、三值 DMA 打包 |
| 0 | 恒为 POS/SILENT | 1-bit | Sparse MAC AND-popcount |

`ternary_encode_unit.v` 保留为 **ternary_en=1 的薄封装**，仿真脚本无需改动。

---

## 3. 数据流变化评估：**主干不变**

与 `docs/02_end_to_end_dataflow.md` 对照，**事件→体素→Swin→H60@S2→Decoder→光流** 拓扑不变。变化集中在 SN 出口与元数据：

| 环节 | 原方案 | 双模 ATLIF 方案 | 是否变数据流 |
|------|--------|-----------------|--------------|
| Event Scatter | 相同 | 相同 | 否 |
| Patch Conv+SN | PSN 或二值 ATLIF | **二值 ATLIF**（`ternary_en=0`） | 仅编码器实例统一 |
| 全线 Attn | Legacy+H60 混用 | **统一 H60（11bc+）** | 单注意力引擎 |
| 全线 Q/K | 三值 ATLIF | **三值 ATLIF**（`ternary_en=1`） | 否 |
| FFN / 其余 | 二值 ATLIF | 不变 | 否 |
| H60 Engine | 仅 S2/S3 | **S0–S3 全线** | 周期↑、Legacy 面积↓ |
| Sparse MAC | 吃 1-bit | 二值模式仍 1-bit；三值层在 MAC 前 **零扩展**（NEG 不参与 AND） | 否（旁路 1bit） |
| Decoder | Dense MAC | 不变 | 否 |

**带宽**：Q/K 仍 2bit packed；FFN/其余仍 1bit。统一算子不增加通路宽度，只减少 RTL 重复。

---

## 4. 控制面：per-layer `neuron_mode`

在 128-bit work descriptor 增加 1 bit（或复用保留位）：

| 字段 | 位 | 值 |
|------|-----|-----|
| `neuron_mode` | 1 | 0=binary ATLIF，1=ternary ATLIF |

控制器在 DMA 出口 latch `ternary_en = neuron_mode`，同一物理编码阵列服务全线 SN 层。  
APB 侧可增加 `NEURON_MODE_LUT[layer_id]` 供调试；推理主路径以 descriptor 为准。

**软件导出**（checkpoint → `hw_metadata.json`）：

```json
{
  "layer.2.0.attn.q_sn": {"neuron_mode": "ternary", "pos_thresh": 1.0, "neg_thresh": -5.0},
  "layer.0.0.ffn.sn1":   {"neuron_mode": "binary",  "pos_thresh": 0.8}
}
```

---

## 5. 与文献的对接

| 文献 | 统一算子如何引用 |
|------|------------------|
| FireFly-T | 同样强调 **单一路径上的脉冲编码 + popcount**；我们扩展到 signed ternary Q/K |
| Bishop TTB | descriptor 里 `neuron_mode` 与 `window_enable` 同包，不增加调度维度 |
| SDformerFlow | 任务与拓扑不变；神经元从「PSN 全家桶」收敛为 **ATLIF 双模** |

---

## 6. RTL / 验证清单

- [x] `atlif_unified_encode_unit.v`
- [x] `ternary_encode_unit.v` → 封装 unified
- [ ] `nts07_controller.v`：descriptor 解析 `neuron_mode`
- [ ] Golden：`ATLIFTernaryPSN` PyTorch vs RTL 逐通道对比（ternary/binary 各 1k 向量）
- [ ] 更新 `run_nts07_sim.sh` 加入 unified 模块

---

## 7. 结论（给论文一句话）

**数据流几乎不变**；硬件故事从「三种神经元、四种编码」收敛为 **「一种 ATLIF-PSN 异构算子 + 三值模式开关」**，与软件双模实验对齐，且有利于面积（共享比较器树，见 Segment-1 autoresearch）。