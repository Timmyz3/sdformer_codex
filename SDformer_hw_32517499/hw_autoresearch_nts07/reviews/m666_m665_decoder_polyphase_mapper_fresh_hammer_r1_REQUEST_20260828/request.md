# M666 request｜M665 decoder polyphase mapper fresh hammer

请以全新上下文独立审查 M665，作者结果不可当 oracle。

重点核验：

1. 从 M514 RTL/contract 自行推导 destination parity、tap order 与 `destination=2*source-1+kernel`；
2. 用新的随机种子和非方形输入，把四个 `[T,M,K]` phase 矩阵拼回输出，与 PyTorch `ConvTranspose2d(K3,S2,P1,OP1)` 逐元素比较；
3. 独立重算 valid-tap、active-tap、popcount、product 守恒；
4. 攻击 bit order、C-order、shape/batch/length/tail、phase/K permutation、所有 ConvTranspose 参数、weight layout、symlink/traversal 和 M660 schema/route；
5. 确认源码未绑定未完成 M660 SHA，未把 input mapping 写成 cycles/speedup。

只有 `P0=0`、`P1=0` 才能给 `GO_M660_PAYLOAD_INTEGRATION_ONLY`。本评审不授权 GPU、EDA、生产周期模拟、性能数字或 DATE headline。
