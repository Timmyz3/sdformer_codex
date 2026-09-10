# C1 / C2 完整迁移与重构进度

**先看：[可读报告](report.html)。C1、C2 均未完成生产重构，当前也没有已建立的强接收贡献。** 本包完成完整路径审阅、一个真实 C1 完整算子的官方外层参考及两项 C2 机会筛查，保留明确的失败和接口限制。

## 实际完成

- C1：恢复 sample0 的一个完整 `3000×6912×768` 算子，独立输入重建核对20,736,000位。29个子集回调检查、6个小矩阵官方外层检查通过；四次完整运行的全部13项计数经独立复算。执行原 `run_fc` 外层及已核对的加速关系回调；不是完全未改官方 API、物理 PPU 或 RTL 周期。
- C2：完整源等价类显式字典计入键、索引、次数后，stage0仅省2.9235%状态，stage3增加12.2875%，未计哈希/端口。停止该版本。
- C2：每 bank 一个选定时间模式，用最后成员提交局部和。排除同帧校准位置后，每输出通道加法减少9.3387%/15.1117%；18个记录的数值诊断核对360个合成系数输出。无周期、真实权重、FP32、BN2/shortcut或PPA证明。

独立审阅发现：所选 `c%8` 符合 m803 通道拆分接口，但捕获权重按 source-group 行地址分 bank，物理映射尚未接通。两个层全部bank分别选中相同的0x140/0x1c3，所以没有bank专属选择的额外收益证据。逻辑系数读取不变不能写成物理节拍不变。

ATLIF 始终保留 `z=theta*g` 的连续值阈值幅值。诊断有理数不代替冻结FP32或Acc24合同。旧 C2 功能叶、旧 C1 弱增量停止结论均保持，未修改生产RTL、模型、主稿或docs/359。

## 文件

| 文件 | 用途 |
|---|---|
| [complete_path_contract.md](complete_path_contract.md) | 完整迁移边界、替换位置、数值/存储/消费者与晋级条件 |
| `c1_full_layer_plan.json`、`c1_full_layer.py`、`c1_full_layer_r1.json` | 完整单算子官方外层参考与输入身份 |
| `c1_attempt_1_runtime_failure.json` | 首次启动的运行库错误；捕获解析前失败，原样保留 |
| `c2_equivalence_plan.json`、`c2_equivalence.py`、`c2_equivalence_r1.json` | 完整非零源去重、显式状态、整数统计诊断 |
| `c2_bank_mode_plan.json`、`c2_bank_mode.py`、`c2_bank_mode_r1.json` | 单一预声明模式选择、同帧剩余位置机会及最后成员诊断 |
| `independent-reviews.json` | 独立子代理复算/机制审阅；不是外部期刊意见 |
| `source-ledger.json`、`artifact-qa.json`、`SHA256SUMS` | 来源、报告核验和本包完整性 |

## 复核与复现

Python 必须3.12。C2用 `/opt/anaconda3/bin/python3.12` 的 NumPy；C1另用进程内隔离的 Torch2.8.0+cpu，目录 `/tmp/tcasii_complete_baseline_20260907/runtime`。首次运行缺少正确libstdc++，已保留失败收据；随后仅用进程环境 `LD_LIBRARY_PATH=/opt/anaconda3/lib` 修正，未修改系统库。

脚本拒绝覆盖已有结果。复现应创建本目录的**相邻运行目录**，只复制对应计划和脚本，保留上一级的共享只读解析器路径，输出独立收据；不要删除或改写本包结果。C1命令为：

```bash
env LD_LIBRARY_PATH=/opt/anaconda3/lib MPLBACKEND=Agg /opt/anaconda3/bin/python3.12 c1_full_layer.py
```

若隔离 Torch 目录已不存在，按[PyTorch 官方历史版本文档](https://pytorch.org/get-started/previous-versions/)恢复相同CPU版本；运行时Python/Torch/NumPy和所有输入/源码SHA均写入结果。C1参考计数器只覆盖本计划的8-bit、8个popcount、issue2参数，不宣称通用。

报告HTML已核对全部内部锚点和本地链接，并用Chromium在1280×1800、比例1采样检查开头、完整C1、C2图和密集表格、结尾；未声称所有设备或打印布局均已视觉检查。
