# 硬件 idea 线：跳零之后的 Stage0 FC1 word-add + PSN T×T mix

身份：官方 AT-LIF `{0,θ}`，θ 吸入下一层 W，层间二值 GeMM。残差/PED 另计。  
**不是** skip-family 2× 标题，不是 ESTU/Jung SSA skip，不是 Prosperity-on-our-net，不是 CIM，不是编造 PPA。  
**不是** BitWave 相对 7-serial 的 60%。  
**不是** Scrooge tot-oracle（`rest=Σ|A||Y|`，tot 本身就是 leftover T×T 工作，tax=1 后 **KILL −7.0%**）。

## 公平分母（已跳 S=0）

Stage0.b0.fc1，10 帧：元素 nnz **17.7%**，空行 **3.0%**，AAC 剩余 **1.25e9 word-adds/帧**（nnz(S)×Cout，1 拍/次整字加）。unique 行合并 **4.6% → 不到 15%**。  
同一层 PSN leftover：**n_tokens × 384 × 10 × 10 ≈ 7.4e8 mix-MACs/帧**（Y 已由 FC1 产出）。r1.conv1 1.29e7；stem 4.21e7。跳零家族不再计入 extra。

## 针对性抄（KEEP：extra≥15% after tax）

| 机制 | 论文 | 目标 | extra | tax | after tax | 判定 |
|---|---|---|---:|---:|---:|---|
| 对偶 W=0 | FireFly-S | FC1 word-add | 0 | 1/32 | −3.1% | **KILL** |
| BitWave vs **整字加** | SparseCol | 同上 | −157% | 1/32 | **−160.5%** | **KILL** |
| BitWave vs 7-serial | 同上 | 对照 | +63% | 1/32 | +60.1% | **KILL 不当 bar** |
| unique 行 | Prosperity | 同上 | 4.6% | 1/32 | +1.4% | **KILL** |
| CGNet/SnaPEA skipped/total | MICRO'19 | 同上 | 81.6% | 1/32 | +78.5% | **KILL live BN** |
| 静态 85% L1 前缀 | CGNet 改进 | nnz 分数 | 13.7% | 0 | +13.7% | **KILL** |
| PSN 跳零 T | Y=0 列 | PSN mix | 3.1% | 0 | +3.1% | **KILL** |
| tot-oracle Scrooge | `Σ\|A\|\|Y\|` | PSN mix | 93.0% | **1.0** | **−7.0%** | **KILL**（tot 是 leftover 本身） |
| **LUT-GEMM G=4 packed** | LUT-GEMM ICLR'24 | FC1 word-add | 29.2% | 0 | **+29.2%** | **KEEP** 无损 |
| LUT-GEMM G=4 + 1/4 inspect | 同上 | 同上 | 29.2% | 0.25 | +4.2% | KILL（敏感度） |
| **Scrooge `‖A_rest‖₁·max\|Y\|`** | Scrooge DATE'26 | PSN mix | 70.4% | **1/T=0.10** | **+60.4%** | **KEEP** 无损 |
| support-code K=256 | LUT-DLA | FC1 | 88% 计算 | encode | — | **KILL**（非无损） |
| PSN rank-4 SVD | ATLIF factor | PSN | 20% | 0 | +20% | **KILL**（无 AEE） |
| 静态 Cin k=79 | CGNet 改进 | FC1 nnz | 15.7% | 0 | **+15.7%** | **KEEP** AEE **0.711** ≪ NB0 |
| 静态 Cout 砍 15% | 隐层剪枝 | FC1 | 15.1% | 0 | **+15.1%** | **KEEP** AEE ~0.71 |

Scrooge 的合法界 **不读跳过项的 |A|·|Y| 乘**：`rest = ‖A_rest‖₁ · max|Y|`。max|Y| 是 T 次比较，相对 T×T mix 的 inspect tax = **1/T**。禁止把 tot-dot 收成 1/T（会少收 T 倍）。

## 拼在一起 / 改进 / 新 idea

- **无损主线（同算子 max，跨算子加权和，禁止独立乘）**：FC1 LUT-GEMM G=4 **29.2%** + PSN Scrooge l1-maxabs **60.4%** → 链 extra **40.8%**。KEEP。
- **改进**：静态 Cin 收到 nnz extra≥15%（k=79），+15.7%，AEE 0.711（全跑 0.715，NB0 1.45）。与 LUT-GEMM 同算子，stack 取 max。
- **新 idea**：把 Scrooge 挂在 leftover PSN 上，用 **不乘跳过 Y** 的 L1-max 界，而不是 FC1 尾巴（live BN 要完整 Y）。tot-oracle 只作 KILL 对照。

## 存活线

在已经跳过 S=0 的 Stage0 剩余工作上：

1. **FC1 剩余整字加**：96-d 二值 S 按 4-bit 组 LUT-GEMM，非空组一次 table-add。相对 `nnz(S)×Cout` **+29.2%**，无损。
2. **PSN 剩余 T×T mix**：`‖A_rest‖₁·max|Y|` 区间证书早停。相对 `n_tokens×Cout×T×T` extra 70.4%，扣 max|Y| 的 1/T 税后 **+60.4%**，无损。Y 已存在，live BN 不挡。

链 extra **≈41%**。不是全网 dense 2×，不是 skip-family 再数，不是 BitWave-vs-7serial，不是 tot-oracle 85%。

收据：`results/remaining_copies.json`。两次入口 KEEP 元组一致。

## 阈值粒度（算法试完，反推硬件）

ep34 的 105 个 ATLIF-PSN **都是标量 θ≈1**。公式允许 \(θ_h\)，吸入仍是二值 GeMM。  
不训练、只把标量拆成逐神经元 / G=8/32 分块（rate-matched）：θ 的 std < 1e-3，LUT extra 仍 29.2%，Scrooge 仍 60.4%，AEE 在噪声里。  
**有用的用法是 θ=∞ 关掉最弱 15% 通道**（编译期表）：sn2 上 Scrooge 60.4%→64.6%；sn1 上等于 Cin drop（AEE 0.711）。硬件上 **G=32 tile 广播一个 θ 就够**，不必为每个神经元存阈值。详见 `THETA.md`。
