# Round 4 报告：顶会顶刊广扫（2026-09-11）

**性质：** 文献核对。不是测量、不是标题。  
**身份：** `../IDENTITY_ATLIF.md`。  
**范围：** ISSCC/JSSC/CICC、ISCA/MICRO/HPCA、DATE/DAC/ICCAD/FPGA、TCAS-I/II/TVLSI/TCAD、NeurIPS/ICML/ICLR、CVPR/ICCV、ICRA 及本地 150 篇 `p0_txts` 中 2024–2026 未进 round-2/3 的命中。

**执行限度：** 十路 venue 子代理后半段因额度 402 中断。**落盘且可用：** `V0` `V1` `V3` `V4` `V7` `V10`。未落盘：V2/V5/V6/V8/V9。deep-research-5/6 的 verifier 全部失败，**不要当证据**。本报告只用已打开的本地全文 / 作者 PDF / 标明 abstract 的 IEEE 记录。

---

## 一句话

广扫之后仍然：**没有** 位于吸收切口上、「同一生产者的二值 GeMM last-use ∧ 连续 PED last-use」的硅或编译器 IR。变多的是必须抄全的 **A** 和同刊/同对象碰撞。H1/G1 继续修订，不是信件。Codex 下一步仍是双消费者核上拆 8088。

---

## 硅 / FPGA 动物园（抄 A，信件不能长这样）

| 对象 | 代表 | 信件若长这样就死 |
|---|---|---|
| 二值 SSA skip + overlay + 分类 mW | **ESTU TCAS-II 2025**（同刊 5 页） | 任何「spikeformer FPGA + skip + mW」 |
| 稀疏 conv + 二值注意力双引擎 | FireFly-T IEEE TC 2026 | dual-engine ≠ dual-consumer |
| 地址编码跳零 + 双脉冲 SDSA | Li et al. 2501.07825 FPGA | 又一个 SDSA skip FPGA |
| IAND 删残差保全脉冲 | Spike-IAND ISCAS 2025 | 我们**保留 PED**，这是负对照 |
| 层切分 dense 输入 / sparse 其余 | DATE 2025 hybrid | 不是同一生产者双 last-use |
| token 级跳过 + ReRAM-CIM | SPARTA ICCAD 2025（摘要） | CIM 禁标题；token skip ≈ ESTU |
| 3D 堆叠 spikeformer | ICCAD 2024/2025 Xu | 封装 PPA 不是机制 |
| 事件 OF 空间局部稀疏 | ASNA-Flow TVLSI 摘要 | G2 已停；禁「首个」 |
| occupancy / FireNet / TrueNorth / plane-fit | EventShiftFlow, SENECA, TBioCAS, ISCAS 2018 | 禁「首个事件 OF 硬件」 |
| 帧 OF FPGA | TCAS-I 2025 Liu 405 FPS；RT-FLOW；ERAFT FPGA | 不是事件、不是 SNN |

## 编译器（V10）

LoopTree / Timeloop / ZigZag / SparseTIR / da4ml / ConvFormer LFS / CFMP / RISCSparse 冻结 BN：**都没有** `{binary, continuous}` 从同一生产者分型 last-use。LFS 是「一类张量用完换槽」；CFMP **不是** layer fusion；RISCSparse 的 BN 折叠是冻结 μ/σ，与本网 live `10×96×120×160` **相反**。

## 算法侧新钉子

- **SymbolicLight V1 (2605.21333)：** 软件里已经写「二值脉冲门 + 连续残差流」。禁止信件说「没人这么切」。剩余问题是 **硬件 last-use / 同端口服务**，不是切法本身。实现仍是 dense kernel。
- **ELSA (2605.20802)：** token 一经产出立刻前传 + mini-batch spiking Gustavson。A 给二进制流式 last-use；残差仍是旁路 ADD。
- **ST-FlowNet / EDCFlow / STSSM：** 事件 OF 算法 A；GPU 或理论能耗，不是硅。EDCFlow 不得当 skip 神谕。
- **2501.14490：** 乘免费并行 PSN（因果膨胀+移位），阈值前 mix 的 A，不是非因果 T10。

## 相对 round-3 的增量

1. spikeformer 硬件对照从「ESTU+FireFly-T」扩成一整个动物园。  
2. ELSA / SegFold 把 Gustavson 从 HPCA 论文扩到 2026 数据流。  
3. SymbolicLight 打掉「双路径切法本身是 X」。  
4. 编译器 IR 已查：缺口仍在硬件/排程测量，不在「发明一种 IR 名词」。

详细表：`literature/V0_seed_catalog.md` `V1_*.md` `V3_*.md` `V4_*.md` `V7_*.md` `V10_*.md`。
