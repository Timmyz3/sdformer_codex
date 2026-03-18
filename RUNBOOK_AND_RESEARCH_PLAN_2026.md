# SDformerFlow 跑通手册与 2026 研究计划

## 1. 这份文档解决什么问题

这份文档解决两个现实问题：

1. 你当前不知道如何把 upstream 原版 `SDformerFlow` 和本地模块化版本真正跑起来。
2. 你需要一条可执行的研究路线，把“跑通 -> 消融 -> 深层改进 -> 硬件映射 -> 投稿”串起来。

本文档会给出：

- 运行原版 `SDformerFlow` 的详细步骤
- 运行本地 baseline 和可插拔模块版本的详细步骤
- 最容易卡住的依赖和路径问题
- 适合继续借鉴的论文与开源代码
- 面向 `BioCAS / ICCAD / ISCAS` 的具体研究安排

---

## 2. 先给结论：最稳的执行策略

不要一开始就试图用同一个环境同时跑：

- upstream 原版 `SDformerFlow`
- 本地 wrapper 版本
- 本地插件版本

最稳的做法是两套环境、两阶段验证：

### 阶段 A

先用独立环境复现 upstream 原版。

目的：

- 验证数据路径、预处理和官方脚本没问题
- 确认原版模型、loss、metrics 都能跑通

### 阶段 B

再用本地工程环境跑：

- `sdformer_baseline`
- `variant_a / variant_b / variant_c`
- `variant_modular`

目的：

- 验证本地 adapter、数据 wrapper、插件管线、profiler 路径

这是因为两边的依赖假设并不完全相同：

- upstream README 写的是 Ubuntu 22.04、Python 3.7.3
- 本地 `environment.yml` 是 Python 3.10、PyTorch 2.2、CUDA 12.1

因此，先把 upstream 单独跑通，再迁移到本地版本，是风险最低的路径。

---

## 3. 当前仓库的真实运行入口

### 3.1 upstream 原版入口

位置：

- `third_party/SDformerFlow/`

关键脚本：

- `eval_DSEC_flow_SNN.py`
- `eval_MV_flow_SNN.py`
- `train_flow_parallel_supervised_SNN.py`
- `train_mdr_supervised_SNN.py`

### 3.2 本地工程入口

位置：

- `scripts/run_train.sh`
- `scripts/run_eval.sh`
- `scripts/profile_latency.sh`

但你现在在 Windows + PowerShell 环境下，最好不要直接依赖 `.sh`，而是直接运行 Python module：

- `python -m src.trainers.train`
- `python -m src.trainers.eval`
- `python -m src.utils.profiler`

---

## 4. 当前最关键的风险点

### 4.1 你现在机器里 `python` 还不可用

在当前工作区环境下，直接执行 `python --version` 会失败。

这意味着：

- 当前仓库虽然代码结构已整理好
- 但实际训练、评估、仿真都还没有在这台机器上被真正跑过

所以第一优先级不是继续加功能，而是先把环境跑通。

### 4.2 upstream README 的部分命令和文件名已经过时

不要只照抄 upstream README。

以本仓库当前真实文件为准：

- DSEC 训练 config 应该优先看：
  - `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`
- DSEC 验证 config 应该优先看：
  - `third_party/SDformerFlow/configs/valid_DSEC_supervised.yml`

README 里提到的一些文件名和当前目录下真实存在的 yaml 文件名并不完全一致。

### 4.3 upstream 数据相对路径和本地工程数据相对路径不一样

当前数据目录在本地工程根目录下是：

```text
SDformer/data/Datasets/DSEC/saved_flow_data
```

而 upstream config 里的路径写法通常是：

```text
data/Datasets/DSEC/saved_flow_data
```

如果你进入 `third_party/SDformerFlow/` 再直接跑 upstream 脚本，这个相对路径会解析成：

```text
third_party/SDformerFlow/data/Datasets/...
```

这通常是错的。

所以你需要二选一：

1. 修改 upstream yaml 中的 `data.path`
2. 在 `third_party/SDformerFlow/` 下建立指向根目录 `data/` 的软链接

推荐做法：

- 直接复制一份 upstream yaml
- 把 `data.path` 改成相对于 upstream 目录正确的路径，例如 `../../data/Datasets/...`

---

## 5. 推荐借鉴的论文与开源代码

下面不是“泛泛参考文献”，而是适合继续拿来做可拔插涨点的候选项。

| 方向 | 论文/项目 | 代码情况 | 建议提炼成什么插件 | 更适合软件还是硬件 |
| --- | --- | --- | --- | --- |
| 低成本时序混合 | [TSM](https://arxiv.org/abs/1811.08383), [official code](https://github.com/mit-han-lab/temporal-shift-module) | 有官方代码 | `temporal_shift` / `temporal_shuffle` | 软件先做，硬件也友好 |
| token 动态裁剪 | [DynamicViT](https://arxiv.org/abs/2106.02034), [official code](https://github.com/raoyongming/DynamicViT) | 有官方代码 | `token_budget` / `saliency_gate` | 软件为主 |
| token merging | [ToMe](https://arxiv.org/abs/2210.09461), [official code](https://github.com/facebookresearch/ToMe) | 有官方代码 | `token_merge` / `window_merge` | 软件为主 |
| 结构化窗口裁剪 | [HeatViT](https://arxiv.org/abs/2211.08110) | 代码未在本轮确认 | `window_topk` / `window_budget` | 非常适合硬件 |
| head + token 联合稀疏 | [SpAtten](https://arxiv.org/abs/2012.09852) | 代码未在本轮确认 | `head_group` / `head_budget` | 非常适合硬件 |
| IO-aware attention | [FlashAttention](https://arxiv.org/abs/2205.14135), [official code](https://github.com/Dao-AILab/flash-attention) | 有官方代码 | `io_aware_window_kernel` | 偏硬件/系统 |
| 光流精度补偿 | [FlowFormer](https://arxiv.org/abs/2203.16194), [official code](https://github.com/drinkingcoder/FlowFormer-Official) | 有官方代码 | `refine_head` / `cost_memory_head` | 软件为主 |
| SNN Q-K 注意力 | [QKFormer](https://arxiv.org/abs/2403.16552) | 本轮未确认官方代码 | `qk_gate` / `head_sparse_qk` | 软件与硬件都适合 |
| 时空 spike attention | [STAtten](https://arxiv.org/abs/2409.19764) | 本轮未确认官方代码 | `temporal_spike_qk` | 软件与硬件都适合 |

建议你优先看三类：

1. `TSM / TokShift` 类，补更轻量的时序插件
2. `HeatViT / SpAtten` 类，补结构化稀疏插件
3. `FlowFormer` 类，补轻量 refinement head

---

## 6. 环境搭建建议

## 6.1 操作系统建议

如果你可以选，优先顺序如下：

1. Linux / WSL Ubuntu
2. Windows + Conda + PowerShell
3. 纯 Windows 且依赖 bash 脚本

推荐理由：

- upstream 明确是按 Ubuntu 路径开发的
- `.sh` 脚本、CuPy、SpikingJelly、Icarus Verilog、Yosys 在 Linux/WSL 下更稳

### 6.1.1 你当前的实际环境前提

你补充说明了：Python 环境是通过 Anaconda 配置的，而且路径在 `D:` 盘。

因此，后续最推荐的使用方式是：

- PowerShell
- 显式激活 `D:` 盘的 Anaconda
- 直接运行 `python -m ...`

如果你的 Anaconda 根目录是：

```text
D:\Anaconda
```

那么在 PowerShell 中建议先执行：

```powershell
& D:\Anaconda\shell\condabin\conda-hook.ps1
conda activate base
```

如果你的实际目录是：

```text
D:\Anaconda3
```

就把上面的路径替换成：

```powershell
& D:\Anaconda3\shell\condabin\conda-hook.ps1
conda activate base
```

如果你还没有让 PowerShell 认识 conda，可以先执行一次：

```powershell
& D:\Anaconda\Scripts\conda.exe init powershell
```

然后重开 PowerShell。

### 6.2 推荐环境拆分

#### 环境 1：upstream 复现环境

名称建议：

- `sdformer-upstream`

推荐：

- Python 3.10 先试
- 如果不兼容，再退到 Python 3.8 或 3.7

原因：

- upstream requirements 里已经写了 `cupy-cuda12x`
- 但 README 又写 Python 3.7.3
- 最快的策略不是一开始强行卡死版本，而是先用 Python 3.10 + CUDA 12 试最少安装集合

#### 环境 2：本地工程环境

名称建议：

- `sdformerflow-hw`

按本地 `environment.yml` 创建。

---

## 7. 如何跑通 upstream 原版 SDformerFlow

下面以 PowerShell 为例写命令。

### 7.1 创建 upstream 环境

```powershell
& D:\Anaconda\shell\condabin\conda-hook.ps1
conda create -n sdformer-upstream python=3.10 -y
conda activate sdformer-upstream
pip install -r D:\code\sdformer_codex\SDformer\third_party\SDformerFlow\requirements.txt
```

如果你用的是特定 CUDA 版本的 PyTorch，请先手动安装与你显卡匹配的 `torch` / `torchvision`，再装其余依赖。

### 7.2 修正 upstream 配置里的数据路径

先复制配置文件，不要直接改官方原件：

```powershell
Copy-Item `
  D:\code\sdformer_codex\SDformer\third_party\SDformerFlow\configs\valid_DSEC_supervised.yml `
  D:\code\sdformer_codex\SDformer\third_party\SDformerFlow\configs\valid_DSEC_supervised_local.yml
```

然后把其中的 `data.path` 改成你的真实数据位置。

如果你从 `third_party\SDformerFlow` 目录运行，推荐写成：

```text
../../data/Datasets/DSEC/saved_flow_data
```

### 7.3 先做最小验证

进入 upstream 目录：

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
```

先跑 DSEC 验证：

```powershell
python eval_DSEC_flow_SNN.py --config configs/valid_DSEC_supervised_local.yml
```

如果这一步能跑通，说明：

- upstream 模型能实例化
- DSEC 数据路径正确
- loss/metrics 可以计算
- SpikingJelly/CuPy 兼容性至少基本可用

### 7.4 跑原版训练

同样先复制并修正训练 yaml 的 `data.path`：

- `configs/train_DSEC_supervised_SDformerFlow_en4.yml`

然后运行：

```powershell
python train_flow_parallel_supervised_SNN.py --config configs/train_DSEC_supervised_SDformerFlow_en4_local.yml
```

### 7.5 upstream 最常见报错与处理

#### 报错 1：找不到数据

优先检查：

- `data.path`
- 当前工作目录
- DSEC 是否已经预处理为 `saved_flow_data`

#### 报错 2：CuPy 或 CUDA 版本不匹配

优先检查：

- `cupy-cuda12x`
- `torch.cuda.is_available()`
- 显卡驱动版本

#### 报错 3：MLflow 导致启动失败

upstream 训练脚本默认会走 MLflow。

处理方式：

- 先安装 `mlflow`
- 或者先改脚本里的 `use_ml_flow` 开关做最小复现

---

## 8. 如何跑通本地 baseline 和插件版

### 8.1 创建本地工程环境

如果你用 conda：

```powershell
& D:\Anaconda\shell\condabin\conda-hook.ps1
conda env create -f D:\code\sdformer_codex\SDformer\environment.yml
conda activate sdformerflow-hw
```

或者按 `requirements.txt` 自装。

### 8.2 PowerShell 下不要依赖 `.sh`

直接进入工程根目录：

```powershell
Set-Location D:\code\sdformer_codex\SDformer
```

然后直接运行 Python module。

### 8.3 先跑本地 baseline 评估

```powershell
python -m src.trainers.eval --config configs/sdformer_baseline.yaml
```

如果已经有 checkpoint：

```powershell
python -m src.trainers.eval `
  --config configs/sdformer_baseline.yaml `
  --checkpoint experiments/logs/train/sdformer_baseline_best.pth `
  --write-summary
```

### 8.4 跑本地 baseline 训练

先做 1 epoch smoke test：

```powershell
python -m src.trainers.train `
  --config configs/sdformer_baseline.yaml `
  --epochs 1 `
  --output-dir experiments/logs/baseline_smoke
```

如果这一步能跑通，再去跑完整训练。

### 8.5 跑本地插件版 `variant_modular`

先训练一个最小 smoke：

```powershell
python -m src.trainers.train `
  --config configs/model_variants/variant_modular.yaml `
  --epochs 1 `
  --output-dir experiments/logs/variant_modular_smoke
```

再评估：

```powershell
python -m src.trainers.eval `
  --config configs/model_variants/variant_modular.yaml `
  --checkpoint experiments/logs/variant_modular_smoke/variant_modular_best.pth `
  --write-summary
```

### 8.6 跑 profiler

```powershell
python -m src.utils.profiler --config configs/model_variants/variant_modular.yaml
```

输出重点看：

- `active_timestep_ratio`
- `active_window_ratio`
- `token_keep_ratio`
- `head_keep_ratio`

### 8.7 批量做 ablation

当前仓库有：

- `scripts/run_ablation.sh`

但在 PowerShell 下更稳的是手工按配置逐个跑：

```powershell
python -m src.trainers.eval --config configs/sdformer_baseline.yaml --checkpoint <ckpt> --write-summary
python -m src.trainers.eval --config configs/model_variants/variant_a.yaml --checkpoint <ckpt> --write-summary
python -m src.trainers.eval --config configs/model_variants/variant_b.yaml --checkpoint <ckpt> --write-summary
python -m src.trainers.eval --config configs/model_variants/variant_c.yaml --checkpoint <ckpt> --write-summary
python -m src.trainers.eval --config configs/model_variants/variant_modular.yaml --checkpoint <ckpt> --write-summary
```

---

## 9. 你应该按什么顺序跑

推荐严格按下面顺序，不要跳步：

### 第 0 步

先确认工具：

- Python
- CUDA
- PyTorch
- CuPy
- MLflow

### 第 1 步

跑 upstream DSEC eval。

这是为了确认“原版可复现”。

### 第 2 步

跑本地 `sdformer_baseline` eval。

这是为了确认“本地 adapter 没有把原版跑坏”。

### 第 3 步

跑本地 `sdformer_baseline` 1 epoch smoke train。

这是为了确认“训练环路打通”。

### 第 4 步

跑 `variant_modular` 1 epoch smoke。

这是为了确认“插件管线打通”。

### 第 5 步

跑 `variant_modular` profiler。

这是为了确认“mask 统计和硬件接口语义打通”。

### 第 6 步

再进入长训练和正式消融。

---

## 10. 投稿与研究方向建议

下面基于 2026 年 3 月 11 日可确认到的官方信息给你建议。

### 10.1 BioCAS 2026

官方站点：

- [BioCAS 2026 official site](https://biocas2026.org/)

官方页面可确认：

- Paper submission deadline: **2026-06-05**

适合你的原因：

- 你做的是 event camera + SNN + 低功耗/硬件友好
- 如果你能给出能耗 proxy、稀疏执行率、SNN 合理性，BioCAS 很匹配

### 10.2 ISCAS 2026

官方站点：

- [ISCAS 2026 official site](https://2026.ieee-iscas.org/)

官方页面可确认：

- Paper submission deadline: **2025-10-12**

结论：

- 以当前日期 `2026-03-11` 来看，ISCAS 2026 已经错过

所以如果你想投 ISCAS，只能看：

- ISCAS 2027

### 10.3 ICCAD 2026

官方站点：

- [ICCAD official site](https://iccad.com/)

当前这次检索中，我确认了官方站点存在，但没有从官方页面稳定抓到清晰的 2026 paper deadline 文本。

因此建议：

- 不要先把计划完全押在 ICCAD 2026
- 在正式排期前，再去官方 CFP 页面核一次

如果到时 ICCAD 2026 截止还来得及，它会非常适合“软硬件协同 + accelerator + sparse scheduler”这条线。

---

## 11. 哪个会议最适合你当前阶段

### 当前最现实的首选：BioCAS 2026

理由：

- 你现在还在“跑通 + 模块消融 + 早期硬件映射”阶段
- BioCAS 对 neuromorphic / bio-inspired / low-power SNN 系统比较友好
- 只要你能在 6 月前给出：
  - baseline 复现
  - 2 到 3 个可插拔模块的有效消融
  - 能耗/稀疏执行 proxy
  - 一版清楚的 accelerator 架构图

就有机会形成一篇像样的稿子

### 中期最强目标：ICCAD

前提是你后面补齐：

- block-level attention hook
- 更可信的 memory/cycle model
- `attention_unit.v` 的真实实现
- synthesis / area / throughput 结果

如果这些能做完，ICCAD 会比 BioCAS 更适合承载硬件故事。

### 备用与延长线：ISCAS 2027

如果 2026 年节奏来不及，ISCAS 2027 是更稳的备选。

---

## 12. 三条可选研究路线

### 路线 A：快发 BioCAS

目标：

- 赶 BioCAS 2026

核心工作：

- 跑通 upstream 和本地 baseline
- 完成 `temporal_shift + timestep_budget + window_topk`
- 给出 ablation + profiler + energy proxy
- 画出 accelerator 体系结构图

优点：

- 时间最现实
- 风险最低

缺点：

- 硬件部分可能深度不够

### 路线 B：冲 ICCAD

目标：

- 面向更强的 HW/SW co-design 稿件

核心工作：

- block-level attention hook
- head/window/timestep 显式调度
- 更准确的 perf model
- RTL `attention_unit` 实现
- synthesis 数据

优点：

- 论文故事更完整

缺点：

- 时间和工程难度都更高

### 路线 C：先 BioCAS，再扩展到 ICCAD/ISCAS

目标：

- 先用软件+系统稿子拿结果
- 再把硬件深做成下一篇

这是我最推荐的路线。

---

## 13. 具体时间安排

下面给你一份更细的执行计划。

### 第 1 周

- 把 Python / CUDA / CuPy / MLflow 环境跑通
- 跑通 upstream DSEC eval
- 记录所有报错和修复方式

交付物：

- upstream eval 成功日志
- 可复现环境说明

### 第 2 周

- 跑通本地 `sdformer_baseline` eval
- 跑通本地 `sdformer_baseline` 1 epoch smoke train
- 跑通 profiler

交付物：

- baseline smoke ckpt
- baseline metrics
- baseline profile csv/json

### 第 3 周

- 跑 `variant_modular`
- 分别启停：
  - `temporal_shift`
  - `timestep_budget`
  - `window_topk`
  - `structured_token`

交付物：

- 单模块 ablation 表
- 训练/验证稳定性记录

### 第 4 周

- 做模块组合：
  - `temporal_shift + timestep_budget`
  - `timestep_budget + window_topk`
  - `window_topk + structured_token`
  - `full modular bundle`

交付物：

- 组合消融表
- 稀疏率 vs 精度表

### 第 5-6 周

- 进入 block-level hook 设计
- 优先把 `window_topk` 和 `head_group` 向 upstream block 内部推进

交付物：

- block hook 设计图
- 第一版内部 attention 改造 patch

### 第 7-8 周

- 做硬件映射
- 更新 perf model
- 明确：
  - controller metadata
  - active window scheduler
  - head selector
  - SRAM usage

交付物：

- accelerator 图
- perf model 表
- 软硬件接口说明

### 第 9 周起

- 根据结果选择会议线
- 开始写论文

---

## 14. 论文实验最少应该有哪些表

无论投哪个会，至少要准备：

### 表 1：baseline 对齐

- upstream original
- local baseline

指标：

- AEE
- AAE
- 参数量
- profile proxy

### 表 2：单模块消融

- baseline
- `+ temporal_shift`
- `+ timestep_budget`
- `+ window_topk`
- `+ structured_token`

### 表 3：模块组合消融

- baseline
- 时间稀疏组合
- 空间稀疏组合
- 全量模块组合

### 表 4：硬件相关指标

- `active_timestep_ratio`
- `active_window_ratio`
- `token_keep_ratio`
- `head_keep_ratio`
- latency proxy
- activation/weight bytes proxy

### 表 5：如果有 RTL

- area
- frequency
- cycles
- energy estimate

---

## 15. 你现在最应该做什么

如果只看一条最关键建议，就是：

1. 先建两套环境
2. 先跑通 upstream eval
3. 再跑本地 baseline eval/train
4. 最后再跑 `variant_modular`

不要反过来。

因为只有先拿到一个“确定能跑”的原版基线，你后面所有插件涨点、硬件映射、论文结果才有可信锚点。

---

## 16. 如果我继续帮你，最值得做的下一步

接下来最值得我继续替你做的，是下面三件事里的第一件：

1. 直接给你补一份“Windows/PowerShell 可执行版”的环境与运行脚本说明，甚至替你生成本地专用 yaml
2. 继续往 upstream `Spiking_SwinTransformerBlock3D` 里做 block-level hook
3. 开始搭建论文实验表格模板和日志规范

如果你的目标是尽快发文，建议顺序是：

- 先做 1
- 再做 2
- 最后做 3
