# sd5ai 资产与死块旁路复核（2026-09-19）

## 已完成的接管

两份远端交接已保存到 `remote_handoffs/`。指定的四个脚本/mask 与十四份 `deadprobe_*.json` 已复制到远端持久目录，并取回本目录 `remote_assets/`，合计十八件；没有删除 `/tmp` 原件。

远端持久目录：`/root/private_data/work/hardware_innovation_20260908/handoff_review_20260919/remote_assets/`。

ep34 恢复后的 checkpoint 与两份独立文件**全文件逐字节相同**，三者 inode 不同，每份 225,504,447 字节：

1. 当前 `dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth`；
2. `t49_ternary_smoke/warmstart/checkpoint_epoch34.pth`；
3. 事故前已有的 `hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth`。

所以，本次确认恢复了包括 T 维张量在内的文件内容。原检查“非 T 张量相等 + T 张量形状对”不足以证明 T 张量值；本次补上了这一缺口。mtime 改变本身不会改变内容摘要。**权重恢复不证明当前 overlay 与历史评测完全相同**；Git 中训练和安装路径另有变化，仍需绑定实际入口。

记录：`remote_assets/PRESERVATION.json`。全程没有写 checkpoint，也没有跑新 AEE。

## “19 个真死模块、无损旁路已过”不成立

### 1. 原始发放率已经反驳 K 恒零

`binary_deadmask.json` 的 19 是两个模型的观测数：ep35 九个、local5 ep44 十个，位置并集十个、交集九个。不是同一网络十九个独立模块。

这十九条中，**十八条 r>0，只有 local5 的一条 sn_k 在这三批数据上 r=0**。r<0.001 是近静默阈值，不能推导 K≡0。被列为可无损整块删除的 K 包括：

| 模型/块 | 原始 r（比例，不是百分比） |
|---|---:|
| ep35 stage1 block0 sn_k | 0.0000011413323 |
| ep35 stage2 block3 sn_k | 0.0003600823111 |
| local5 stage0 block1 sn_k | 0.0000798172672 |
| local5 stage2 block3 sn_k | 0.0004842464114 |

即使三批恰好全零，也只能证明样本内静默；不是跨输入架构不变量。将所有 θ 同时乘 0.1 还改变了上游发放和幅度，不能作为逐模块独立反事实证明。

### 2. 原旁路安装函数实际匹配零个模块

已从交接脚本提取**原 `_install_skip` 函数**，对同一探针 JSON 中的真实模块名执行。没有另写一套替代匹配算法冒充复现。

原代码：

```python
key = _bn.split('sttmultires_unet.')[-1]
if any(key.startswith(sb) or key == sb for sb in _SKIP):
    ...
```

真实名字剥掉此前缀后仍是 `encoders.swin3d.layers.1.swin_blocks.0`，目标却是 `layers.1.swin_blocks.0`。

| 名称输入 | ep35 安装数 | local5 安装数 |
|---|---:|---:|
| 取证文件中的真实名称 | **0** | **0** |
| 仅在复核小模型中规范化名称的阳性对照 | 2 | 3 |

复现：`/usr/bin/python3.12 check_deadskip_match.py`；结果 `DEADSKIP_MATCH_REVIEW.json`。这不是新的网络等价实验，足以表明现有脚本无法验证报告声称的旁路。

### 3. 修好名字之后仍有接口与数学缺口

- 当前远端 `bsa_attention.py:6634` 返回 `(x, attn)` 元组；钩子 `torch.zeros_like(out)` 将整个返回值当张量，接口不匹配。
- 钩子是 forward **执行之后**替换输出，不能据此测得跳过计算的周期。
- K=0 只能先推出 gated-K 乘积为零。后续 `self.proj` 的 bias、`proj_bn` 的 affine/统计可能产生非零值；整段返回零还需额外条件或保留正确常量路径。
- 本次 `/tmp` 未找到报告声称的 `deadskip_verify_*.json`；不能用缺失结果补证已经确认的脚本问题。

**裁决：撤回当前“位级通过/架构真死”的资格，不删研究资产。** 这些位置可以成为有损近静默剪枝候选；若考虑无损跳过，先保留完整消费者、在真正施加扰动的检查点上对拍。ep35/local5 的 mask 不能直接装进 ep34。

## 其余远端结论的处理

| 项目 | 处理 |
|---|---|
| local5 ep44 AEE 1.2819 | 单独历史分支；不是 ep34 基线，不与 ep34 1.199514 混写 |
| H12 四个 backward 已全部修好 | 不接收；官方变体没有 `.abs()`，H12 负分支和 θ 聚合另有问题，见 `THETA_BN_REVIEW.md` |
| 三元 warm start 失败 | 保存已测配置的负结果，不外推“任何三元/任何 θ 重训均不可能” |
| θ-fold 与 BN | 线性吸收恒等式不受后继 BN 阻止；仍需逐消费者和数值边界导出 |
| attention 0.59% / 99.4% | 历史工作量信封，不是当前变更网络实测周期份额 |
| valid825 profile 可免 GPU 作分布复核 | 可以描述评测集分布；用于选择剪枝/调参就是验证集参与设计，不能冒充独立验证 |
| MVSEC dt1/dt4 不同 | 接受不可直接排名；不接受凭经验换算后声称“真实只差 1.3×” |
| T 焊在参数形状，因此 MTT 必需 | 前半解释直接改 T 配置为何失败；后半不能推出只有一种迁移算法可行 |
| CICC 必须先拿全文，阻塞一切 | 本地已有原始 PDF，已确认混合 SNN/ANN，见 `HARDWARE_PRIOR_REVIEW.md` |

本轮不把死块 mask 配置化到生产、不跑其 DC、不启动 H12 全训。先修正被试对象和执行入口，避免对没有实际发生的旁路做硬件优化。

## 新列文献的有限核验

这不是把 Top10 全部复现；只核了本轮会直接改变执行排序的三项。

- **LoAS（MICRO 2024）**：作者仓库明确提供 PyTorch 稀疏画像、微调、LTH 剪枝和 artifact。09-19 查询 `main` 完整树（37 个文件、未截断）未见 `.v/.sv/.vhd/.vhdl`；artifact README 说明复现 Table II / Fig11。故不能直接写“开源 RTL 可套”；也不能将单次发放剪枝/恢复训练解释成近静默模块无损删除。[作者仓库](https://github.com/RuokaiYin/LoAS)、[artifact 说明](https://github.com/RuokaiYin/LoAS/blob/main/artifact/README.md)。没有据此断言作者其他地方绝无 RTL。
- **MM2IM（FPL 2025）**：论文身份成立，不仅是未接收 arXiv；MatMul+col2IM 针对输出映射、重叠累加及 cropped 无效结果。保留为 decoder 强基线/边界迁移候选；其 FPGA 对 CPU 的倍率不能当本模型增量。先与本库已跳结构零的 scatter/polyphase 比较，不能再向基线收零插值税。[原论文](https://arxiv.org/abs/2507.07683)。
- **MI-TRQR（NeurIPS 2025）**：作者实现可用，基于 PSN，处理时间冗余，值得放算法第二队列；它不是已验证的本网络无损时间跳过。需要实际 mask/重标定对 T10 门与事务的影响，并在同协议最终光流头验证精度。[作者实现](https://github.com/dfxue/MI-TRQR)、[论文](https://openreview.net/pdf/022289abe80c122ccd71e42d2cb60fe04fcf07ad.pdf)。

`2501.07825` 确为 Spike-Driven Transformer 稀疏硬件论文，摘要的 13.24× 是论文原对照的吞吐；不能归给本工作 conv。其双脉冲注意力与当前 Motion-XOR 仍需单独适配。[原论文](https://arxiv.org/abs/2501.07825)。
