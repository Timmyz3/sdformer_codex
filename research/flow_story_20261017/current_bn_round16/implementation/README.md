# R16 CPU 算术验证入口

这是一套共享普通执行器的有限 CPU 原型，不使用真实光流质量、checkpoint、GPU 或训练。固定矩阵与否决门见 [PROTOCOL_V1.md](PROTOCOL_V1.md)。源审通过后写入 `freeze_v1.json` 并绑定本目录四个源文件，首次执行使用 fresh `actual_v1/`；失败保留，不在原版本上修改数值合同。

```sh
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B run_cpu_v1.py \
  --freeze freeze_v1.json --freeze-sha256 <actual-frozen-sha256> \
  --output-dir actual_v1
```

入口先核 63 个历史 native 文件、26 个本地 CPU 依赖与本目录源 SHA；拒绝覆盖输出。276 个 case、644 个逻辑 MLP 分支调用、368 次实际 native MLP 方法调用，每个 FC 总计473300行。每个 case 保存独立 JSON 和无损 NPZ，含真实输入、初始模型 state、mask/面积/cell、FC/PSN/BN/branch/residual、实际 surrogate prethreshold、数组 payload SHA、文件 SHA、state/input/hook 闭合。两个原生 BN 明确设置多步模式、移除 running buffers；不加载任何 CMA。

`compact_current_mlp.compact_mlp(native,u,mask)` 返回 `(branch,audit)`。`audit` 中的 `packed_input`、`areas`、`BN1/BN2` 矩及 `BN1/BN2_output` 是诊断张量，不是 JSON 元数据；未来调用方应显式处理/释放，不要把它们长期累积在全网记录。PSN/FC 实际调用与原始中间量在 runner 外围观察，core 不额外重算 PSN。`expanded_mlp` 独立构造 dense 输入后运行真实 native MLP；全fine另有直接 native 原输入对照。

数学来源是原生 `SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:164` 的 MS_Spiking_Mlp 顺序、`Spiking_submodules.py:183` 的完整时间 PSN 及 `Spiking_modules.py:101` 的原生归一化。沿用的是本仓库已固定来源，不声称覆盖面积加权统计具有光流专属新意。相关设计边界见 `research/flow_story_20261016/bn_conversion_round14/WEIGHTED_BN_CANDIDATE.md` 与 `selected16_bn_round15/independent_review/NEXT_CPU_OPTION_REVIEW_V1.md`。

本轮保留两个 FC 生产张量后归约/回读，明确两个全域屏障；容量不是实际 DRAM 流量，CPU调用和行数不是 PE 周期。强普通策略与未来光流评分须共享相同执行接口及资源约束。即使全部通过，也只有有限 synthetic CPU 证据，尚未证明真实权重、GPU、全网光流精度或周期收益。
