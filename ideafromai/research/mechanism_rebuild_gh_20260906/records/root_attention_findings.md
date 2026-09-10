# 本轮原生注意力反查记录

状态：只读研究；未启动 RTL、EDA、训练或新 profile。2026-09-06。

曾考虑从 PADE 的部分执行迁移到“所有可能补全均产生相同 Q1.7 gate 才提交”的精确消费者证书。PADE 的低贡献剪枝保持的是其稀疏化规则，而非完整 dense attention 数值；本候选若保留，只能另定整数部署合同，不能借冻结 FP ep34 的 AEE。

## 直接读到的仓库反证

- `SDformer/hw_autoresearch_nts07/rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv` 已实现时间 pair、score class、多重度和按 active K 输出描述符。K 空仍参与归一化。直方图、同分数聚合和 K 空不输出都不能重新命名为本轮新意。
- `rtl_h67/h67_row_qmax_denominator_certificate.sv` 已有 Q 计数上界、Q/K/motion 紧上界、原始 Q/K wrapper、分母指数证书。其默认 T450 与最大分数界可证明分母指数 17；接收次序及有效位检查也已有。
- `docs/399_双线OUT32与整帧Attention模型收口_20260814.md:105–165` 记录旧分母证书与 class-scan 模型。旧 profile 的命中率不能移给 ep34；该文的 class-scan-only 上界约 1.014，也不能写成新 RTL 收益。
- `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py:5329–5392` 的 Q 与 K 投影后都有 BN。当前 no_running 策略意味着只省某些 Q 行的投影会影响整域统计，不能仅凭 K 空或分数区间绕过 Q 计算。
- 同文件 `:301–365` 定义部署 Shiftmax：Q7，差分查 16 项 LUT、整数移位、全行 sumexp、下一 2 的幂分母、preserve_mean 时乘 450，再 RNE 至 unsigned Q1.7。共同最大值、分母、舍入均是依赖；不是“分数为整数幂所以只移位”。

## 本轮判定

纯分母证书/score histogram 已有，纯 32-bit popcount 的部分通道早停可能得不偿失。扩大到省 Q 投影又遇到动态 BN 的完整统计生产者。现阶段不把这一条列为强主候选，不用 first-H67-HW 提高评分。

## 一手外部来源

- PADE，HPCA 2026 稿件 / arXiv 2025-12-16：<https://arxiv.org/html/2512.14322v1>，本轮读 §IV-A/B 的 bit-plane bounds 与动态阈值。其 bounds 来自实际 Q 与未读取 K 位；镜像 BUI 的外部 ub/lb 接口不能替代该生产者。

以上是研究判断，不是发表准入或已测性能结论。
