# TP 官方 SDT 评分的 v2 MLP 适配

已实现 **CPU 机制适配**，尚未进行 DSEC 推理、恢复训练或 GPU 测量。不能由本目录宣称精度保持、TP 全方法复现或硬件加速。来源与方法取舍见 [SNN_TRANSFER_SOURCES.md](../../idea_and_protocol/SNN_TRANSFER_SOURCES.md)。本目录不读取 checkpoint、GT、数据或远端服务器，也不修改原生模型源码。

## 固定来源

- [官方仓库](https://github.com/xlzhou126/TP-Spikformer)，固定 revision `bcc915ad62254efa45800245d8abcf40905618fc`。
- [固定文件](https://github.com/xlzhou126/TP-Spikformer/blob/bcc915ad62254efa45800245d8abcf40905618fc/sdt_drop/model/sdt_drop.py)，SHA256 `e4a688393b34e9bf2c15a4ee783cabe8427ff1cfab22429bad95a1cc8e91049d`，22,260 bytes。原始 bytes 保留在 [upstream/sdt_drop.py](upstream/sdt_drop.py)，未修改、未整体导入；[来源清单](source_manifest.json)保留获取方式与状态。
- GitHub REST API 首次网络访问受沙箱限制，获准重试后遇 403 限流；之后通过只读 `git ls-remote` 固定 revision，再取得该 revision 的 raw 文件。不存在以可变 `main` 充作固定版本的情况。上游许可证尚未独立核实，当前仅保留有明确归属的内部研究快照。
- [论文](https://openreview.net/pdf?id=L5llQD0nMf)式10–20与该 SDT 代码有差异。本模块以具体官方代码的 `TemporalChangeScorer` / `LocalSpatialSimilarity` 为准，并明确限定 `prev_drop_mask=None`，不声称实现跨 block 永久删除。

## 保留的评分与明确的改动

[tp_score.py](tp_score.py) 的 `official_sdt_scores(x)` 接受 FP32 `[T,B,C,H,W]`，T 至少2：

1. 时间项先均值通道，再对相邻时间差取绝对值、跨时间均值和空间 softmax。不是先对各通道绝对差取均值；不补论文中的首时步项。
2. 空间项先均值时间，再用 `[C,C,3,3]` 全一／9卷积，`groups=1,padding=1`，随后通道余弦相似度（eps=1e-6）、取负与空间 softmax。**保留原代码通道混合卷积**，不静默替换成 depthwise 局部均值。
3. 总分是两项相加，输出 `[B,H,W]`；所有 T 共用空间支持。全零／单空间位置也定义良好；T1 的空时间差、非有限输入／输出和非FP32明确拒绝。

适配的决策位置是原生 `MS_Spiking_Mlp` 已收到的 `u = x + Attention(x)`。`u` 是可能带符号的实值残差特征，不要求二值 spike。官方 SDT 原 block 在其入口 x 评分，并选择后运行 attention＋MLP；本适配改为 attention 后评分、只取消后续 MLP。因此即使公式相同，输入分布、可用时机和可取消范围也不同，应使用 **TP-score / v2 MLP adaptation** 标签。

`select_spatial(total, mode, keep_count=... / keep_fraction=...)` 提供：

- `token`：对原空间位置选 TopK，输出完整空间 mask。`mask_dense_branch` 仅遮挡**已经执行的** MLP 输出，用于精度诊断，不能据此声称省周期。当前 TileMLP 执行器不接受碎 token mask。
- `tile9`：按原位置的9×9网格聚合分数，tail 只对实际点取均值，再选整 tile。空间扩展后与完整 T10 共享。**这个分组和均值是本项目适配，不是原 TP 公式。** 不将任意 token 打包成假平方网格去执行 Swin attention。
- 显式 `keep_count`，或 `floor(keep_fraction × 候选单元数)`。这是每层／每图的**单元数量预算**，不是跨层严格 PE 服务预算；不同 stage、tail 的成本不等价。按实际行数／硬件服务另匹配比较点。
- 默认 `torch_topk` 保留官方 tie 行为，等分时索引可能随设备／运行时变化。可选 `stable_index` 是明确标注的确定性适配，不冒充官方 tie 规则。0／全部保留不调用 TopK。

返回的 `Selection` 包括单元分数、`unit_keep`、完整 `spatial_keep`、每 tile 实际点数；`complete_t_mask(10)` 给 `[10,B,H,W,1]`，通道轴广播。`json_record()` 给可序列化的完整空间与单元 mask。无需GT或最终光流；未实现新的区域风险评分。

## 接入现有执行接口

`TPTilePolicy` 与 [TileMLP](../../deployment_contract/tile_mlp.py) 的 `policy(layer_name, u)` 接口一致：只接受 `[10,1,H,W,C]`，返回 `[ceil(H/9),ceil(W/9)]` bool grid。`TileMLP` 返回 MLP branch，外层原生 ADD 残差负责输出 `u + branch`，跳过时 branch=0。

```python
# 调用方先用严格的新 checkpoint / data / source-hash 入口完成加载。
# sys.path 分别包含本目录与 deployment_contract 后：
from tp_score import TPTilePolicy
from tile_mlp import tiled_mlp_model

policy = TPTilePolicy(keep_fraction=0.5)
with torch.inference_mode():
    # 整个 model 须处于 eval，所有 BN 有有效冻结统计。
    with tiled_mlp_model(model, mode="actual_skip", policy=policy) as wrappers:
        prediction = model(input_tensor)
    masks_and_score_cost = policy.records
    actual_executed_rows = [wrapper.records for wrapper in wrappers]
```

本目录不放宽 [独立评估入口](../../evaluation/README.md) 的 checkpoint/schema、来源与数据守卫，也没有修改其 CLI。未来全网评估应接入新原生编码 checkpoint 的显式 schema，不能拿旧工程权重结果充作正式 baseline。当前统计 `P-paper` BN 不满足实际 tile 跳过条件，会被现有执行器拒绝；冻结 `P-code` 或校准后的 D 可以检查执行等价，但冻结本身不证明校准精度。正式精度比较应与同一 BN 语义的 dense 图成对报告。

`score_cost` 给实现的算子量与缓冲形状，包括完整 T 特征可用条件、通道混合卷积名义 MAC、核缓冲、余弦向量、两个 softmax、选择和 mask；这些**不是实测 PE 周期／访存或峰值内存**。当前代码保留昂贵的官方卷积并为诊断复制 JSON。即便全关，仍计算分数；attention 和生成 u 的生产者始终已经执行，不得计算为被取消的服务。记录的候选 FC 行数须与 `TileMLP.records.observed` 对照，控制／评分成本仍须后续硬件建模。CPU wall 仅为带守卫的评分工程时间，CUDA 不提供未同步 enqueue 时间冒充延迟。

## 实际 CPU 检查

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /opt/anaconda3/envs/pytorch310_cpu/bin/python \
  /home/zhumd/work/sdformer_codex/research/flow_story_20261012/transfer_experiments/tp/test_tp_cpu.py
```

[cpu_checks.json](cpu_checks.json) 是实际执行结果，包含测试与依赖文件 SHA。32项检查通过，CUDA 未初始化：

- 从固定 raw 文件仅抽取两段已审读 scorer AST 执行，避免整体导入上游依赖；适配输出与原类逐值完全相等。另用 NumPy FP64 标量邻域循环／余弦参考验证，不复用 PyTorch 卷积评分；覆盖边界、单点、B2与全零。
- 验证通道正负抵消行为，防止静默换成另一公式；验证 token TopK、全开／全关、tail 面积81/18/9/2及真实点均值、全 T 共享 mask、预算与数值拒错。
- 用**原生 MLP**合成参数／实值 u，接已有 TileMLP。4 tile 中分别保留0/1/3/4时，dense-masked 与 actual-skip 的 MLP 分支最大差均为0；每层 FC 实际执行行数分别0/90/1080/1100。部分保留分支非零；全关仍产生 u 且输出残差 u，FC 不执行。验证冻结 BN 要求与梯度开启拒绝。

这证明当前合成输入下的评分与执行接口；没有读取 checkpoint 或数据，没有完整网络精度结果，也没有恢复训练、GPU吞吐或周期／能耗结论。
