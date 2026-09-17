# Swin MLP 针对性照抄（2026-09-16）

MLP 名义 **156.5 G / 26.2%**。24 个 `fc1/fc2` 全挂钩。10 帧 GT，全跑 AEE **0.7158**。

## 抄什么

| 论文 | 落到 MLP 上的动作 |
|---|---|
| FireFly / 普通 AAC | 跳过全零 token 行 |
| Prosperity | 相同二值行只算一次（unique/nz） |
| C-STEP | T0–T1 静默的 token 不再算 T2–T9 |
| LUT-DLA | Codex 已在另一条 FC1→PSN 上抄过，本轮不重复 RTL |

## 测到的结构

**FC1（sn 后的发放 × W，才是大头）**

- Stage0：nnz 15–18%，**几乎每个空间位置每个 T 都有发放**（token_any 0.97），unique/nz **≈1.00**（没有可合并的相同行）
- Stage1 block0：nnz 5%，token_any **0.54**（约一半 T×token 行全零，AAC 有东西可跳），但 unique 0.76，且 **early_sil 47% 的 token 在后期全部都会发放**（late\|sil=1.0）
- Stage2/3 FC1：unique **≈1.00**，early_sil≈0

**FC2（隐层发放更稀）**

- nnz 1.7–6.4%；L0.b0 unique **0.57**（这是 MLP 里 Prosperity 最像样的一层）
- 多处 early_sil 高（L2.b3 达 77%），同样 **late\|sil≈1**：非因果 T10，前两拍静不等于后面静

## 质量实验：C-STEP 早静默 token 跳后期 T（所有 FC1）

AEE **0.7158 → 0.7257**。略差。平均 later-T 跳过份额只有 **3.2%**，因为多数 FC1 根本没有 early silent；有的那些后期还要发放。

## 结论（失败不扔）

1. **C-STEP 时间早退在 MLP 上不成立。** 非因果 T10：先静后响。适配：只做 **当前 T 的全零行跳过**（AAC），不要用早期 T 预测后期。
2. **Prosperity 行合并在 FC1 上几乎为空**（96-bit 发放向量各不相同）。适配：去 **FC2**（L0.b0 unique 0.57）或 **更短的 packed 码**（支持码/LUT，Codex 已走）。不要指望原始 96-d 行森林吃满 Stage0 FC1。
3. **真能省的是标准行稀疏：** Stage1.b0 FC1 有 46% 全零 token 行；Stage2 若干 FC2 token_any 0.49。这就是 FireFly AAC，不是新机制。
4. LUT/证书继续对着 **FC1→PSN 连续混叠**（Codex 整链 −4.3%），不是对着已经很稀的 FC2 发放行。

下一步若还抄 MLP：把 AAC 零行跳过做成 **所有 24 个 Linear 的同资源对照**（无损），再只对 unique≪1 的 FC2 试森林。不要再做跨 T 早退。
