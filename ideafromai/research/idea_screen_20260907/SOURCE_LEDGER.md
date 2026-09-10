# 文献台账（有界，2026-09-07）

状态：`support-located` = 摘要/HTML/本地笔记已对上机制；`search-incomplete` = 未读全文每一个附录。  
**没有**声称穷尽。不是 kill list。

## 算法 / 注意力

| 工作 | 出处 | 对本课题 | 状态 |
|---|---|---|---|
| SDformerFlow | arXiv:2407.15801 / ICPR 2024（笔记里也有 2409.04082） | 公开 SDSA/线性 QK，**不是** H67 Motion-XOR | support-located |
| QKFormer | NeurIPS 2024，arXiv:2403.16552 | 线性脉冲 Q-K，无 V/softmax；分类不是光流 | support-located |
| Spike-driven Transformer | NeurIPS 2023 / V2 ICLR 2024 | mask+add 下游合同 | support-located |
| α-XNOR SSA | CVPR 2025 | 共静默项；α≠0.125；无 K_peer | support-located |
| Spiking ST-former / STSA | Visual Computer 2025 | 时空注意力 | search-incomplete |
| LRF-SSA | ICLR 2026，arXiv:2603.19290 | SSA 插件 | search-incomplete |
| SLI+ACF | arXiv:2608.19238 | 邻域通路 + SSA | search-incomplete |
| SQKformer | Neurocomputing 2026 | 膜 BN + 通道 QK | search-incomplete |
| SpikePool | arXiv:2510.12102 | pooling 换 SSA → 违反脉冲 QK | support-located（排除标题） |
| Spiking Patches | arXiv:2510.26614 | 事件 tokenization | search-incomplete |
| AT-LIF | NeurIPS 2025 | `{0,θ}` 训练；推理 θ 静态 | support-located |
| SpiLiFormer | ICCV 2025，arXiv:2503.15986 | 侧抑制；分类不是光流；I010 训法类比 | search-incomplete |

## 硬件

| 工作 | 出处 | 对本课题 | 状态 |
|---|---|---|---|
| Prosperity | HPCA 2025 | C1 林的强对照 | support-located |
| Phi | 笔记 ISCA 2025 / arXiv 2505.10909 | PAFT pattern | support-located |
| LoAS | MICRO 2024，arXiv:2407.14073 | FTP + inner-join；官方无完整 ASIC RTL | support-located |
| FireFly-T | IEEE TC 2026，arXiv:2505.12771 | AND-PopCount；Zynq overlay | support-located |
| FireFly-S | arXiv:2408.15578 | 双端稀疏，要重训 | search-incomplete |
| Bishop | ISCA 2025，arXiv:2505.12281 | TTB/ECP/AAC；无运动 XOR | support-located |
| 稀疏 SDT 加速器 | arXiv:2501.07825 | 双 spike 跳零 | support-located |
| Chen/Chang Spike-IAND-Former | arXiv:2503.19643 | 28 nm mux T=4/2/1 | support-located |
| ASTER | arXiv:2511.06770 | 模拟 PIM，不当 PPA | support-located（排除主线） |
| SpAtten / A³ / Sanger / ELSA / DOTA / FACT | HPCA/ISCA/MICRO/ASPLOS 2020–23 | ANN 注意力加速，可借鉴预测/级联 | 见 `research/07_transformer_attention_accelerators.md` |
| ASNA-Flow | TVLSI 2025 | 事件光流神经形态；勿重讲空间局部性 | support-located |
| Zhang CICC 2026 | 28 nm 光流 DLSS | 只借时间相似，不借 MaxPool 猜测 | support-located |
| Gist | ISCA 2018 | 消费者编码 / A8 先验 | support-located |
| SumMerge | ICS 2021 | 共享因式分解 | support-located |
| RSR / RSR++ | 本地对照 | C2 对照 | search-incomplete 全文 |

## 本地笔记（先读这些再上网）

- `ideafromai/research/grok46_20260905/04_paper_survey.md`
- `ideafromai/research/07_transformer_attention_accelerators.md`
- `rtl_h67/h67_motionxor_score_q7.sv` 公式与硬件一致

## 方法（本轮 ideation skill）

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065  
检索：2026-09-07，https://arxiv.org/abs/2609.00065 （v2, 2 Sep 2026）。用于 K-Dense `scientific-brainstorming` 流程，不是 OF 证据。
