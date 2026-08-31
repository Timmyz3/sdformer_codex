# M161 独立评审收口 overlay r2

M161 的实数代数、12 个 FFN 几何、BN1/BN2 extent 和 moment width census 保留；原硬件 DSE 不接纳。独立评分为 58/100，P0/P1/P2=`2/4/2`。

两个 P0 已由前一 correction overlay 覆盖：raw fc1 是 14–16b，不能继承 M31 的 8x8 资源/周期；五移动 BN 基线不公平。DATE 安全口径改为：

- Q24 rank state 相对 streaming dense BN1：`2.0614x` 本地 bit-movement sensitivity；连共同 BN2 后 `1.6823x`。
- Q8 训练候选分别为 `6.1842x` 与 `2.9441x`。
- 均不是 transactions、cycles、energy 或 system speedup。

Q24 也不能称“定点精确”：当前没有 factor sumabs、binary point、动态 alpha 范围及 correction overflow 证明；独立保守界在 correction 前已需要 24–27b。

下一硬件里程碑限定为 Q8 early-requant streaming frontend：复用 96 个 signed-INT8 product slots，同时做 32-value/cycle moments 与 rank-3 right projection。它可以独立做 RTL/VCS/DC，但在 PAFT/valid825 前只证明模块行为，不接纳网络精度或系统倍率。
