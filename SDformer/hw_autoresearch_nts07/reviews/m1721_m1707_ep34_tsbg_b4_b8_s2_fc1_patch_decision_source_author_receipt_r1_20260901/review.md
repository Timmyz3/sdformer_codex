# M1721 作者源码回执

M1721 只准备 M1707 完成后的离线判决链，未等待或执行 capture/analyzer，也未启动 GPU、RTL 或 EDA。

TSBG 仅测试 B4/B8。普通 persistent same-capacity LRU-B row buffer 是强基线；取权重、计算与 roofline cycle 分开输出，取数倍率不得改称周期倍率。

S2 仅允许 FC1 使用真实 retained signed codeword。PATCH 在 M1707 格式中只有 histogram/debt，没有 token/block retained values，因此 fail closed；FC2 继承 M1713 数学 NO-GO。任何 S2 正 epsilon 点仍必须完成同一四十样本 paired AEE 与同资源周期重放。

作者阶段 14/14 synthetic/source tests 通过；未创建 M1721 result、release 或 paper result。
