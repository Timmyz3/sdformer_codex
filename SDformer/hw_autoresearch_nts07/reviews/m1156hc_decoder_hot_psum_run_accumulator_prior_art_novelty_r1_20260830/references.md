# Primary-source reference ledger

检索日期：2026-08-30。以下只列论文原文、作者/机构项目页或官方 artifact。

| 工作 | Primary source | 本项目必须承认的 prior-art 边界 |
|---|---|---|
| F. G. Gustavson, “Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition,” TOMS 1978 | [IBM Research publication](https://research.ibm.com/publications/two-fast-algorithms-for-sparse-matrices-multiplication-and-permuted-transposition) | 稀疏结果的坐标聚合、unordered merge/Gustavson accumulation 是经典算法，不可声称发明 sparse accumulator。 |
| Y.-H. Chen et al., “Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow,” ISCA 2016 | [paper PDF](https://www.cs.cmu.edu/~15740-f20/papers/isca16-chen-eyeriss.pdf), [official project](https://eyeriss.mit.edu/) | 本地 psum 累积与减少 psum movement 是 dataflow 基础；one-entry output holding 不是 novelty。 |
| A. Yazdanbakhsh et al., “GANAX,” ISCA 2018 | [arXiv paper](https://arxiv.org/abs/1806.01107) | transposed convolution 的 inserted-zero、相邻零模式重排、重复 microprogram 是直接 prior。注意 venue 是 ISCA，不是 MICRO。 |
| Z. Zhang et al., “SpArch,” HPCA 2020 | [MIT Han Lab official project](https://hanlab.mit.edu/projects/sparch) | streaming merge partial matrices、输出/输入 locality 协同是 prior；不能把在线 partial merge 当首次提出。 |
| G. Zhang et al., “GAMMA,” ASPLOS 2021 | [NVIDIA Research publication](https://research.nvidia.com/publication/2021-04_gamma-exploiting-gustavsons-algorithm-accelerate-sparse-matrix-multiplication), [author PDF](https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf) | Gustavson PE、high-radix merge、cache/buffer hybrid 与 irregular fiber reuse 是最直接 sparse-accumulator 架构边界。 |
| C. Wei et al., “Prosperity,” HPCA 2025 | [paper](https://arxiv.org/abs/2503.03379), [official artifact](https://github.com/dubcyfor3/Prosperity) | binary SNN product reuse与在线识别是 prior；H67 run accumulator不能称新 sparsity paradigm。 |
| C. Wei et al., “Phi,” ISCA 2025 | [paper](https://arxiv.org/abs/2505.10909) | pattern hierarchy、预计算与 residual element processing 是 prior；本机制不应借 pattern-sparsity 语言包装。 |
| T. Li et al., “FireFly-T,” arXiv 2025 | [paper](https://arxiv.org/abs/2505.12771) | multi-lane sparse decoder、bank-conflict-aware load balance、SNN event dataflow 是 prior；typed K8 的协议差必须具体说明。 |
| K. You et al., “ELSA,” ISCA 2026 | [paper](https://arxiv.org/abs/2605.20802) | bundled AER、fine-grained streaming、mini-batch spiking Gustavson-product 是直接 SNN prior；不能声称首次将 Gustavson 用于 SNN。 |
| J. Cuadrado et al., “Optical flow estimation from event-based cameras and SNNs,” Frontiers in Neuroscience 2023 | [paper](https://arxiv.org/abs/2302.06492), [journal](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1160034/full) | event-camera SNN U-Net、decoder upsampling/多尺度光流是应用先例；H67 claim 应限定为具体 descriptor/resource protocol。 |

Prosperity/Phi/FireFly-T 的相关性主要在 binary event/source 侧；它们不是 1-entry run accumulator 的直接来源。直接 novelty 边界应以 Gustavson/GAMMA/SpArch、Eyeriss 与 GANAX 为主，SNN 工作用于界定对象迁移而非制造“first”。
