# M501 exact adjacent-overlap 结果独立打铁评审 r1

日期：2026-08-27  
范围：独立审查 M501 result、contract、analyzer、runner、两份冻结 manifest 和既有 preflight/runtime review；未修改生产文件或 `docs/359`，未启动 VCS/DC/PT/GPU。本评审的 raw replay 不 import、不调用生产 M501 analyzer。

## 裁决

**`ALLOW_ONE_SAME_RESOURCE_CYCLE_FASTKILL_ONLY_NO_RTL`。机会证据 95/100，独立 RTL novelty 0/100。**

M501 的预声明 validation horizontal-G2 点成立：独立解压并重算 40 条 validation record 得到 `11,010,375 / 7,980,680 = 1.379628678x` event-work reduction，与封存 result 逐位一致。代入四层 Conv 冻结份额 `79,630,957 / 620,302,905 = 12.837431%`，理想 envelope sensitivity 为 `1.036617917x`，也一致。

这两个数字只足以授权一次同资源 cycle fast-kill，不足以授权 RTL。原因很明确：ExSpike APEC 是直接 prior art；而本次 168 条冻结 record 全部只有零和一个 operator-constant 正幅值，所以 bit-exact overlap 在这个 trace 上退化为 support intersection，没有激活 signed/multi-amplitude H67 差异化。

## 1. 封存与身份

结果目录 `SHA256SUMS` 的 8 个条目全部通过，manifest SHA256 为：

```text
aafb53027931ec3f49f67b3dc18bd130e2fd3f30876fa418e880baadb04c4a7f
```

关键身份：

| 对象 | SHA256 |
|---|---|
| M501 result | `37ce6d66a73c5dc3c19e887497ac85b473bc4789c0c241b4073d6af5d4c6cd18` |
| contract | `bbb7bce5015ab3a3a5772b86d594853da353380df8dcd85a295e480d422eb2d6` |
| analyzer | `5bdfa6f6fa81510d11751d6867748515763d3d4b31927b8cfe03e03ee597b7e7` |
| Python-3.10-fixed runner | `51b1011abd31fb31ba9049d06695ff46f1bd3a6c3369c5ba721f574b8368f02a` |
| `docs/359` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

result 自身的 manifest 没有收进 runner，这是一个低风险交接缺口；但 runtime hammer 的封存 manifest `f5227e0c...` 已锁定该 runner，复合证据链可用。

## 2. 独立复算

我做了两条不同于生产 analyzer 的检查路径：

1. 对已封存的 1,008 条 detailed row 重新分组，重算 overall/per-operator/per-sequence 的整数守恒、ratio 和 fraction；
2. 逐个解压 168 份 float32 payload，直接对 `x=0/1,2/3,...` 成对，只重算预声明 horizontal-G2，不 import 生产实现。

| Cohort | Records | Sequences | Baseline events | Exact overlap | Candidate events | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| validation S10 | 40 | 1 | 11,010,375 | 3,029,695 | 7,980,680 | **1.379628678x** |
| train calibration S32 | 128 | 18 | 35,499,794 | 9,966,776 | 25,533,018 | **1.390348528x** |

train-only 18 序列的 horizontal-G2 reduction 分布是 `min/median/max = 1.353935810x / 1.393535943x / 1.418593061x`。这说明机会在 train calibration 序列上不是单一场景偶然，但不能写成 heldout multi-sequence。

两个 cohort 的四个非零码字完全一致：

```text
resblocks.0.conv1.0  0x3f7fff87
resblocks.0.conv2.0  0x3f7fff70
resblocks.1.conv1.0  0x3f7fff9f
resblocks.1.conv2.0  0x3f7ffdb4
```

全部 record 的第二个 codeword 有限且为正，`negative_count=0`。因此本轮能证明的只是对该冻结 trace 的 support-overlap 机会。

## 3. 16.03125 KiB 仍是未定价容量

selected signed19 proxy 算术正确：

```text
768 * 3 * 3 * 19 = 131,328 bit = 16,416 byte = 16.03125 KiB
```

但它不是生成的 SRAM macro，也不是完整物理代价。四档容量只是 `16/19/24/32 bit -> 13.5/16.03125/20.25/27 KiB`。下列成本尚未进入周期：group input buffer/comparator、macro 组织/端口/同步返回、weight readiness、overlap 结果向两个相邻 destination 的扇出与 commit、border/tail，以及与 parent/PWP reuse 的争用。

## 4. 只准一次的 same-resource cycle fast-kill

可以做一次，且只做一次。它的任务是杀掉或保留一个 **ExSpike-derived supporting mechanism**，不是发明新的 APEC RTL。

最小合法模型必须：

- 固定 H67 ep35 validation S10、四层 bottleneck Conv、horizontal-G2，禁止改轴或从 train 选点；
- baseline 与 candidate 使用同一个 Conv engine、task order、lane/bank 数、频率、8-bank weight path、128 B/cycle 带宽、output-bank/row-tile/CAM 资源和相同 psum-sink 假设；
- 运行时实现 pair formation/exact comparator，将 overlap 与两份 residual 经真实 bank mapping 发射，统计 weight-read transaction；
- 对 overlap scratch 收进容量、端口、延迟和位宽敏感性；16.03125 KiB 不得免费叠加，要么从同一总 SRAM 预算中挪出，要么进行 area-normalized 对比；
- 对一份 overlap partial sum 向两个 shifted destination 的扇出、bank conflict、commit/update cycle、border/tail 全部收费；
- 报 queue occupancy/backpressure、scratch-port conflict、weight-not-ready、destination-commit stall 和 zero-descriptor weight-fetch suppression；
- 若叠加 M473/M504，两臂必须使用同一已准入的 storage/port 组织，禁止一臂用 ideal 1R1W、另一臂收物理税，也禁止相乘独立倍率。

硬门如下：

| Gate | 阈值 |
|---|---:|
| 数值/event 守恒 mismatch | **0** |
| validation 四 Conv same-resource cycle speedup | **>=1.20x** |
| 对 620.303M envelope 的 charged sensitivity | **>=1.02x** |
| train-only 18 序列最差 cycle speedup | **>=1.15x** |
| 免费 SRAM/免费端口 | **禁止** |

任意一门失败：`KILL_M501_HARDWARE_LINE`，机会数字只留 DSE/prior-art 对照。全部通过：也只能作为明确引用 ExSpike 的 supporting mechanism，需要与另一个真正 H67-native 的物理/算法贡献绑定并再经独立审查；仍不允许 standalone APEC RTL novelty。

## 5. DATE 口径

该点当前可引用的最强表述是：

> 在冻结 H67 四层 bottleneck Conv 输入上，ExSpike-APEC-style horizontal-G2 exact overlap 将 event work 减少 `1.3796x`；该机会在 train-only 18 序列上的分布为 `1.3539x–1.4186x`。由于 trace 是 positive two-codeword，该点是直接 prior 机制的 workload audit，而非新的 signed-analog 压缩。

禁止表述：`1.3796x throughput`、`1.0366x system speedup`、“通用 signed-analog overlap”、“新 APEC RTL”、heldout multi-sequence、PPA/energy 或 DATE headline。即使事件工作完整转化，该点的理想 envelope sensitivity 也只是 `1.036618x`，它不会成为 Best Paper 级系统倍率主轴。

## 6. 可复现性

独立 checker：`audit_m501_result_independent.py`。它默认重聚合封存 detailed rows，增加 `--raw-replay` 时会独立重放全部 168 份 horizontal-G2 payload。结构化裁决见同目录 JSON；封存见 `SHA256SUMS` 和 `SHA256SUMS.seal.sha256`。
