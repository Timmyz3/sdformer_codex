# M371：M370 bottleneck magnitude gate 独立打铁

结论：**96/100，P0/P1/P2 = 0/0/3。确认 fast-kill G7 bottleneck scalar magnitude gate。** 这只终止四个 bottleneck Conv 输入上的逐层标量 `|x|<theta` 方案，不影响 G11，也不能外推 FC、patch embed 或 attention。

## 独立重算

M371 没有把 M370 result 当计算 oracle，也没有 import 或执行 M370 analyzer。独立 strict-JSON parser 读取四个冻结 manifest 的全部 248 条 record，检查完整 codebook、float32 bit pattern、count/elements/nonzero/sign/value-audit 一致性，并输出 248 行逐 record CSV。

总计审计 571,392,000 个 manifest-codebook value，其中 active source 63,865,851 个：H67 ep35 S10 为 11,010,375，PAFT ep4 S10 为 8,027,179，control ep4 S10 为 9,328,503，H67 ep35 train S32 为 35,499,794。

每条 record 都严格为 `{+0,a_l}`；同一 operator 的 `a_l` 在 H67 ep35、PAFT ep4、control ep4 和 train S32 间 bit-identical：

- `resblocks.0.conv1.0`: `3f7fff87`，0.9999927878；
- `resblocks.0.conv2.0`: `3f7fff70`，0.9999914169；
- `resblocks.1.conv1.0`: `3f7fff9f`，0.9999942183；
- `resblocks.1.conv2.0`: `3f7ffdb4`，0.9999649525。

独立重算 `theta={0,1/64,1/32,1/16,1/8}` 的所有 record/layer/cohort 点，新增 active drop 全为 0。严格规则下 `theta=a_l` 仍是 0%；把 theta 提到紧邻的下一个 float32，四层各自立即变成 100% active drop。

## fast-kill 是否充分

对 G7 的既定目标——用逐层标量幅度阈值获得非零但非全删的 sparsity/accuracy Pareto——这个证据充分。因为输入只有一个非零幅值，阈值不可能区分任意两个 active source：要么全保留，要么全删除。因此不值得为该 G7 bottleneck 机制排 RTL 或 A800 accuracy job。

但 0%→100% 是 source-drop cliff，不是已经测得的 accuracy cliff；M371 没有声称全删必然导致精度失败。

## 三个 P2 与边界

1. 本次验证到 manifest 完整 codebook，没有重新解压原始 float payload 并逐值计数。
2. 结论只覆盖四个身份与四个 bottleneck Conv 输入，不能推广到其他 sequence/checkpoint/operator。
3. 没有实际测量全删点的任务精度；fast-kill 的充分性来自“没有部分幅度点”，不是 accuracy 结果。

G11 按 `|w*x|` 区分 product，权重仍可为相同 `a_l` 的 source 提供不同预算，所以本结果不否定 G11。FC、patch embed、attention 没有出现在四份 manifest 中，不能外推。无 cycle、RTL、system speedup、paper PPA 或 DATE headline 准入。
