# T45b 报告：把活动率换成我们自己的 ep34 权重，并修正两处口径错误

日期：2026-09-17　　state：**完成**（数字可直接进论文，边界见 §5）

---

## 1. 为什么不需要重跑

T45 的 dense MAC 列是硬的（真前向 hook），但活动加权 SOP 列借的是
`neuron_experiments/_profiles/sops_20260511_120258/layer_firing_rates.csv`
—— **另一个 checkpoint（tokenmix_pool）、288×384 crop、40 样本**。

本步发现：与我们部署配置同批封存的
`hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/spike_profile.json`
里已经有**我们自己的 ep34 权重**跑出来的逐层发放率：

| 项 | 值 |
|---|---|
| 层数 | 93（逐个点名可用） |
| 样本 | 825 |
| 分辨率 / T | 480×640 / T=10 |
| BN 策略 | `no_running`（78 个模块改动） |
| 全局发放率 | 0.056709 |
| **精度锚** | **AEE 1.19951 / AEE_PE1 0.36380 / DSEC_Fl 5.31336 / AAE_Benchmark 5.10636** |
| 封存 | `SHA256SUMS` + `RUN_COMPLETE.txt` |

所以 T45b = **正确 join**，不是重跑。活动率来源标签改为
`ep34_m2041_spike_profile(480x640,825samples,T=10,no_running_bn)`。

代码：`t45_mapping_cost.py --rate-src ep34` → `results/t45b_mapping_cost_ep34.json`
（同时修掉了原脚本无论 src 都覆写 `t45_mapping_cost.json` 的 bug；
旧 foreign 结果备份在 `results/t45_mapping_cost.foreign_backup.json`）。

---

## 2. 修正一：`attn.proj` 的输入不是 `attn_sn`（12 处全错）

`Spiking_swin_transformer3D.py:709-712`：

```python
x = (attn).reshape(...).permute(...).reshape(T, B_, H, W, C).float()
attn = self.attn_sn(x)      # ← 只作为第二个返回值进调试路径
x = self.proj(x)            # ← proj 吃的是上面 reshape 的 x，不是 attn
```

生产树里已有定论（`hw_autoresearch_nts07/scripts/analyze_encoder_storage_contract.py:307`、
`audit_h67_h68_atlif_coverage.py:64`）：**12 个 `attn_sn` 全部是 dead_debug**，
结果不进正常推理。profile 里它们的 `spikes = 0 / elements = 15.68G` 因此是**真实的**，
不是 profiler 抽风。

真正的 proj 输入是 `k.mul(att_token)` = **`sn_k` ⊙ `sn2_q` 两个 0/1 脉冲张量的逐元素积**，
即 `P = P(sn_k ∧ sn2_q) ≤ min(r_k, r_sn2q) ≤ r_k`。

- 原 T45 把 proj 挂到 `attn_sn`（rate 0）→ 24.88G dense MAC 记 0 SOP，
  等于**把这部分成本藏起来了**；
- `sn2_q` **未被 profile 收录**（93 层里有 `sn_q` 没有 `sn2_q`），
  所以真值算不了。本步退回**上界** `r(sn_k)`，在 json 里标 `ok_upperbound:<key>`。

修正后（上界）`enc.stage*.attn.proj` 全部退到表格尾部：

| stage | MAC(G) | r(sn_k) 均值 | SOP 上界(G) | 占全网 |
|---|---|---|---|---|
| stage2 | 11.944 | 0.0395 | 0.472 | 0.46% |
| stage3 | 5.308 | 0.0450 | 0.239 | 0.24% |
| stage0 | 3.650 | 0.0158 | 0.058 | 0.06% |
| stage1 | 3.981 | 0.0030 | 0.012 | 0.01% |

→ **注意力分支（proj+linear_q+linear_k）合计只占 ~3.7% SOP**。
结合 T45 已确认注意力是线性注意力（无 QK^T），
**"注意力是 SNN 能耗瓶颈"这条（T12 里 Bishop/GustavSNN 的说法）在本模型上不成立**，
因为本模型的注意力已被 Q/K-only 的线性形式抽干。

---

## 3. 修正二（影响最大）：decoder 的 stride-2 转置卷积差 4 倍

`Spiking_modules.py:467-474` 的 `MS_SpikingTransposeDecoderLayer.forward`：

```python
x = self.sn(x)           # 0/1 脉冲（所以 T45b 的 "decoders.N → decoders.N.sn" 映射是对的，没有错位）
x = self.deconv(x)       # ConvTranspose2d, stride=2, output_padding=1
x = self.norm_layer(x)
```

deconv 是 **stride-2 转置卷积**，MAC 数取决于实现口径：

- **口径 A｜零插值 + 普通卷积**（cuDNN / ptflops 的标准数法，也是 T45 hook 的数法）
  `MAC = in_pos × s² × in_ch × out_ch × k²`
- **口径 B｜输入散射/驻留**
  `MAC = in_pos × in_ch × out_ch × k²`
- 两者恒定差 `s² = 4` 倍。A 里 **3/4 的抽头乘的是零插值出来的零**。

`t45b3_deconv_convention.py` → `results/t45b3_deconv_convention.json`：

| stage | in_shape | MAC_A(G) | MAC_B(G) | A/B | rate |
|---|---|---|---|---|---|
| decoders.0 | [10,1,1536,15,20] | 63.701 | 15.925 | 4.000 | 0.1299 |
| decoders.1 | [10,1,770,30,40] | 63.867 | 15.967 | 4.000 | 0.1501 |
| decoders.2 | [10,1,386,60,80] | 64.033 | 16.008 | 4.000 | 0.1542 |
| decoders.3 | [10,1,194,120,160] | 128.729 | 32.182 | 4.000 | 0.2597 |
| **合计** | | **320.330** | **80.082** | 4.000 | |

**口径 A 与 T45 hook 逐项吻合**，所以这不是新口径引入的偏差，而是 T45 一直在用 A。

spike-driven 阵列本来就跳过零输入，因此 decoder 的**真实**活动加权操作数：

| | 口径 A | 口径 B |
|---|---|---|
| decoder SOP | 61.165G | **15.291G** |
| 全网 SOP | 101.585G | **55.712G** |
| decoder 占比 | 60.2% | **27.4%** |

→ 62 个百分点里的 32.8 个百分点是**零插值浪费**，不是真工作量。
纯稠密映射也能白拿 4×：dec.3 的 MAC 从 128.7G 降到 32.2G。

---

## 4. 修正后的成本分布（口径 B，活动率 = ep34 实测）

全网 946.38G dense MAC（口径 A）/ 54.116M 参数 / 活动加权 SOP **55.71G**（口径 B）

| 组 | MAC(G) | MAC% | SOP(G) | SOP% | 参数(M) | 参数% |
|---|---|---|---|---|---|---|
| encoder.swin3d（含 patch_embed 之外的全部 swin） | 255.14 | 26.95% | 17.856 | **32.1%** | 25.278 | 46.7% |
| frontend（patch_embed：head/conv/residual_encoding/proj） | 307.00 | 32.44% | 16.144 | **29.0%** | 0.466 | 0.9% |
| decoder（4 × ConvTranspose2d） | 320.33 | 33.85% | 15.291 | **27.4%** | 7.141 | 13.2% |
| bottleneck.resblocks | 63.70 | 6.73% | 6.400 | 11.5% | 21.234 | **39.2%** |
| pred heads | 0.21 | 0.02% | 0.020 | 0.04% | 0.001 | ~0% |

**结论翻转**：口径 A 下"decoder 一家独大 60%"是**零插值假象**。
按真实工作量，成本是**三足鼎立**：encoder 32.1% / frontend 29.0% / decoder 27.4%，
再加 bottleneck 11.5%。没有单一 60% 的热点可以直接切。

内部结构（口径 B 下的 SOP）：

- **encoder**：MLP `fc1` 合计 9.512G（stage2 5.407 / stage0 2.195 / stage3 1.124 / stage1 0.786）
  —— **这就是 C1 那条链路所在的位置，占全网 17.1%**；
  MLP `fc2` 合计 2.837G；attention 3.727G（其中 proj 上界 0.781G）；downsample 1.781G。
- **frontend**：`residual_encoding` 4×Conv2d(96→96,k=3)@240×320 是主体，
  MAC 254.80G（占全网 26.9%！）但发放率只有 0.032–0.048 → SOP 10.580G。
  `head` 2.654G 按 r=1.0 计入（保守，见 §5）。`conv` 2.337G，`proj` 0.572G。
- **decoder**：口径 B 下四级几乎均衡（15.9/16.0/16.0/32.2G MAC），
  但 dec.3 的发放率 0.2597 是**全网最高**（这也是原来它独占 32.9% 的原因）。

### C1/BitFair 的覆盖率（重算）

C1 挂在 12 个 swin-MLP 的 `fc1→sn2` 门路上，对应 fc1 的 9.512G SOP：

- 口径 A 分母：9.512 / 101.585 = **9.4%**
- 口径 B 分母：9.512 / 55.712 = **17.1%**

**即"抄 BitFair 抄全"最多动到全网 1/6 左右的操作数**（T45 里算的 ≤23% 是 A 口径下的
上位估计）。这条仍然成立且更强：**只优化 encoder 的 FFN 门，天花板就是 ~17%**。

---

## 5. 诚实的边界（不可外推的部分）

1. **`attn.proj` 是上界，不是实测**。`sn2_q` 未进 profile，真值 `P(sn_k ∧ sn2_q)`
   需要把 `sn2_q` 补进 profiler 重跑。因为 proj 总量只占 ~1.4%，对结论影响 <0.5%。
2. **`patch_embed.head.conv` 按 r = 1.0 计入**（2.654G，占口径 B 的 4.8%）。
   它的输入是连续 voxel 不是脉冲，profile 里没有对应 neuron。
   真实事件体素占用率远低于 1，所以这一项**被高估**，实际 frontend 占比会更低。
3. **口径 B 是"若硬件支持跳过零插值抽头"的代价**，是设计目标不是既成事实。
   论文里必须同时给 A 和 B 两个数，并说明 B 需要什么样的数据流
   （输入驻留/散射式，或在零插值缓冲上做结构化零跳过）。
4. 活动率是 **825 样本、480×640、T=10、no_running BN** 下的均值；
   逐样本/逐场景方差未统计。
5. dense MAC 列来自 **randn 前向的 hook**，只用来定形状权重，
   与真实数据的数值无关（形状与数据无关，这点成立）。
6. **不能宣称 decoder 是"最贵的"**，也不能再引用 T45 的 55.6%/60.2% 作为论文数字。
   旧数字的出处和错因见 §3。

## 6. 对 T45c 的直接影响（下一步怎么做）

靶点排序变了，且"躲开零插值浪费"本身就是一个可写进论文的映射决策：

1. **下界先拿**：口径 B 的 4× 是映射决策，不需要新机制 → 先把它作为基线。
2. **真正要攻的三块**（口径 B 下 32.1/29.0/27.4%）：
   - encoder MLP `fc1` 9.512G —— C1/BitFair 已有积累但天花板 17.1%；
   - frontend `residual_encoding` 4×Conv2d(96→96,k=3)：**MAC 254.8G（26.9%）但发放率仅 3–5%**
     —— 高 MAC × 低发放率 = 最典型的"spike-driven 省不动"的结构，
     值得单独查它是不是可以降分辨率/降通道（受 T40f 纪律约束：必须带精度约束）；
   - decoder 的 `sn→deconv`：口径 B 后 15.3G，其中 dec.3 发放率 0.26 最高。
3. **下一刀先补 `sn2_q` 进 profiler**（把 §5.1 的上界变成实测），
   同时把 `head.conv` 的 voxel 占用率量出来（§5.2），这两项加起来影响 <5%，
   但能把表里所有"保守计入"清干净。
