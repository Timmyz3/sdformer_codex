# 原版 SDformerFlow 跑通方案文档（中文）

## 1. 文档目标

这份文档只回答一件事：
如何先把 `third_party/SDformerFlow` 里的原版 SDformerFlow 跑起来。

这里不讨论本地 wrapper、可拔插模块、硬件加速器，也不直接进入你后续的改进版实验。  
目标是先把 upstream 原版链路打通，确认：

1. 环境能正常导入依赖
2. 数据目录结构正确
3. 原版训练脚本能启动
4. 原版评估脚本能读取训练结果

只有这一步跑通，后面的插件消融和软硬件协同实验才有一个可信的 baseline。

---

## 2. 先理解原版工程的运行逻辑

原版 SDformerFlow 的入口不在本地 `src/`，而在：

- `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`

它的基本执行链路是：

```text
YAML 配置
  -> YAMLParser
  -> DSECDatasetLite
  -> MS_SpikingformerFlowNet_en4
  -> loss.flow_supervised
  -> MLflow 记录训练结果
  -> 训练结束后通过 runid 恢复模型做评估
```

这里有两个关键点必须先弄清：

1. 训练和评估都依赖 YAML 配置。
2. 评估默认不是直接吃一个本地 `.pth` 路径，而是通过 `mlflow runid` 找训练产物。

所以原版跑通不是“执行一个脚本”这么简单，而是一个顺序问题：

1. 先配环境
2. 再准备预处理数据
3. 再启动训练
4. 训练成功后拿到 `runid`
5. 再用 `runid` 跑评估

---

## 3. 你现在真正要先跑哪一条链路

建议你不要一上来就正式训练 60 个 epoch。  
正确顺序是下面 4 步：

1. 先做环境导入检查
2. 再做数据目录检查
3. 再做训练脚本启动检查
4. 最后再做完整训练和评估

也就是说，先追求“能启动”，再追求“跑完整”。

---

## 4. 当前仓库里与 upstream 相关的真实情况

### 4.1 实际可用的训练配置名

upstream README 里写的训练命令还是旧名字：

```text
configs/train_DSEC_supervised_MS_Spikingformer4.yml
```

但当前仓库里真正存在、并且 `train_flow_parallel_supervised_SNN.py` 默认使用的是：

- `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`

所以你后面运行时，应该以这个配置为准，不要照 README 的旧配置名直接跑。

### 4.2 数据目录当前是空的

你当前仓库里的：

- `D:\code\sdformer_codex\SDformer\data`

现在只有 `.gitkeep`，没有可训练的数据。  
这意味着你现在还没有满足 upstream 运行前提。

### 4.3 DSEC 预处理脚本不是“全自动可直接训练”

`DSEC_dataloader/DSEC_dataset_preprocess.py` 默认会处理事件体素，但它有三个现实问题：

1. 它默认写入 `../data/Datasets/DSEC`
2. 它默认把 voxel 存到 `event_tensors/10bins_pol/left`
3. 而 `DSECDatasetLite` 在 `loader.polarity: True` 时默认会去读 `event_tensors/10bins/left`

也就是说，脚本输出目录名和 loader 读取目录名之间存在不一致。  
如果你直接照脚本跑，很容易出现“预处理好了但训练仍找不到文件”的情况。

### 4.4 `sequence_lists` 不是自动生成的

`DSECDatasetLite` 会强依赖下面这些 split 文件：

- `sequence_lists/train_split_seq.csv`
- `sequence_lists/valid_split_seq.csv`

但当前 upstream 仓库里没有自动生成这些 csv 的脚本。  
因此对你来说，最稳的起步方式不是“从零自己生成全部 DSEC 预处理产物”，而是先准备一份已经包含：

- `event_tensors`
- `gt_tensors`
- `mask_tensors`
- `sequence_lists`

的完整预处理数据目录。

---

## 5. 最稳的总体策略

如果目标是“先跑通原版”，我建议你采用下面的策略，而不是马上自己重做数据预处理：

### 策略 A：最稳，优先推荐

直接准备一份已经预处理好的 DSEC `saved_flow_data` 目录，然后只解决环境和配置问题。

你需要最终拥有这样的目录：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\
  |- event_tensors\
  |- gt_tensors\
  |- mask_tensors\
  |- sequence_lists\
```

这是最省时间、最少踩坑的方案。

### 策略 B：自己从原始 DSEC 开始生成

只有在你拿不到现成的 `saved_flow_data` 时，再考虑这条路。  
但这条路当前并不适合作为“先跑通原版”的第一步，因为它涉及：

1. 原始 DSEC 下载
2. ground truth / mask 文件准备
3. 事件体素生成
4. split csv 生成
5. `10bins` / `10bins_pol` 目录名对齐

工作量更大，也更容易把问题混在一起。

---

## 6. 环境配置方案

## 6.1 推荐思路

你的 Python 环境通过 Anaconda 配置，路径在 D 盘。  
所以后续所有命令建议都先用 PowerShell 激活 conda。

你机器上的实际根目录是：

```text
D:\anaconda
```

所以后续命令统一按这个路径写。

如果直接在 PowerShell 中激活 conda：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda activate base
```

如果第一次激活失败，可以先执行一次：

```powershell
conda init powershell
```

然后重开 PowerShell。

## 6.2 推荐新建一个只服务 upstream 的环境

不要直接复用你后面做插件和硬件协同的环境。  
建议给 upstream 单独建环境，避免依赖互相污染。

### 严格复现路线

更接近 upstream README 的版本：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda create -n sdformer-upstream python=3.7.3 -y
conda activate sdformer-upstream
```

### 实用路线

如果 `python=3.7.3` 在你当前机器上装包太困难，再退一步试：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda create -n sdformer-upstream python=3.10 -y
conda activate sdformer-upstream
```

但要明确一点：  
这是工程上更方便的折中，不是 upstream 官方声明的严格环境。

## 6.3 安装依赖

进入 upstream 目录：

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
pip install -r requirements.txt
```

如果 `cupy-cuda12x` 安装失败，先确认你的显卡驱动和 CUDA 运行时版本。  
因为 upstream 代码在 GPU 上会调用：

```python
functional.set_backend(model, "cupy", neurontype)
```

如果你暂时不具备可用的 CuPy 环境，训练大概率会慢很多，甚至在某些配置下直接失败。  
所以最稳的做法仍然是优先准备好可用 GPU 环境。

---

## 7. 先做环境 smoke test

在真正训练前，先跑导入检查。  
这一步的目标只有一个：确认依赖能 import。

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
python -c "import torch; import spikingjelly; import timm; import mlflow; print('imports_ok')"
```

如果这里失败，不要直接进训练。  
先把缺失包或版本冲突解决掉。

然后再确认 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果返回 `False` 或 `0`，那就说明你当前不是可用 GPU 训练状态。

---

## 8. 数据准备要求

## 8.1 upstream DSEC loader 真实依赖的目录

对于当前默认配置：

- `model.encoding: voxel`
- `data.num_frames: 10`
- `data.preprocessed: True`
- `loader.polarity: True`

loader 最终会去读取：

```text
data.path/
  |- gt_tensors/
  |- mask_tensors/
  |- sequence_lists/train_split_seq.csv
  |- sequence_lists/valid_split_seq.csv
  |- event_tensors/10bins/left/<sequence>/<file>.npy
```

注意这里是 `10bins`。

## 8.2 你最该优先检查的几项

假设你的数据根目录准备成：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data
```

那你必须确认下面这些路径真实存在：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\gt_tensors
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\mask_tensors
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\sequence_lists\train_split_seq.csv
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\sequence_lists\valid_split_seq.csv
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\event_tensors\10bins\left
```

只要缺一项，原版训练就会在 dataloader 之前或过程中失败。

## 8.3 关于 `10bins_pol` 的处理建议

如果你手头只有：

```text
event_tensors\10bins_pol\left
```

而没有：

```text
event_tensors\10bins\left
```

那你至少要做一件事来对齐：

### 方案 1：重命名目录

把 `10bins_pol` 改成 `10bins`。

### 方案 2：改 loader

修改 `DSECDatasetLite` 中 voxel 的读取目录逻辑。

### 方案 3：改配置

把 `loader.polarity` 改成与当前数据实际格式匹配的设置。

如果你的目标是“先跑通原版”，优先建议方案 1，因为改动最小、最不容易引入新变量。

---

## 9. 配置文件该怎么处理

## 9.1 不建议直接改官方原始配置

建议你复制一份训练配置和验证配置，做本地专用版本。  
这样后续出问题时，官方配置和你的本地配置不会混在一起。

例如你可以复制：

- `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`
- `third_party/SDformerFlow/configs/valid_DSEC_supervised.yml`

变成你自己的本地版本，例如：

- `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_local.yml`
- `third_party/SDformerFlow/configs/valid_DSEC_supervised_local.yml`

## 9.2 训练配置最少要改哪几项

最少只改下面几类内容：

### 1. 数据根目录

把：

```yaml
data:
    path: data/Datasets/DSEC/saved_flow_data
```

改成相对于 `third_party/SDformerFlow` 工作目录真正有效的路径。  
如果你的数据放在：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data
```

而你又是在：

```text
D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
```

执行训练，那推荐写成：

```yaml
data:
    path: ../../data/Datasets/DSEC/saved_flow_data
```

### 2. GPU 编号

根据你的设备设置：

```yaml
loader:
    gpu: 0
```

### 3. worker 数量

Windows 下如果 dataloader 有兼容问题，可以先从：

```yaml
loader:
    n_workers: 0
```

起步，先保证能跑，再逐步增大。

### 4. batch size

显存不够时，先用：

```yaml
loader:
    batch_size: 1
```

### 5. AMP

如果混合精度引发异常，可以先关掉：

```yaml
optimizer:
    use_amp: False
```

---

## 10. 建议的第一次训练流程

## 10.1 第一次训练不要直接追求完整收敛

第一次训练的目标只有两个：

1. 模型成功开始迭代
2. MLflow 成功生成 run

也就是说，你的第一次运行是“连通性验证”，不是“最终实验”。

## 10.2 建议命令

进入 upstream 根目录：

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
```

执行训练：

```powershell
python train_flow_parallel_supervised_SNN.py --config configs/train_DSEC_supervised_SDformerFlow_en4_local.yml
```

如果你想显式指定本地 MLflow 目录，也可以这样：

```powershell
python train_flow_parallel_supervised_SNN.py `
  --config configs/train_DSEC_supervised_SDformerFlow_en4_local.yml `
  --path_mlflow file:///D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/mlruns
```

如果不指定 `--path_mlflow`，通常会在当前工作目录下使用默认 MLflow 路径。

## 10.3 训练启动后你该看什么

只看下面几件事：

1. 能否成功打印 `device: cuda:0`
2. 能否成功创建 `Training Dataset ...`
3. 能否成功创建 `Validation Dataset ...`
4. 能否开始进入 epoch / iteration
5. 是否打印出 MLflow 目录或 run 信息

只要这 5 件事成立，就说明原版主链路基本已经通了。

---

## 11. 训练完成后怎么接评估

## 11.1 评估脚本的核心不是 checkpoint 路径，而是 runid

评估命令的重点是：

- `--config`
- `--runid`

它会通过 `mlflow.get_run(args.runid)` 找到训练保存的模型。

所以你训练成功后，先记录 runid。

## 11.2 建议命令

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
python eval_DSEC_flow_SNN.py `
  --config configs/valid_DSEC_supervised_local.yml `
  --runid 你的训练runid
```

如果训练时指定了 `--path_mlflow`，评估时也要保持一致：

```powershell
python eval_DSEC_flow_SNN.py `
  --config configs/valid_DSEC_supervised_local.yml `
  --path_mlflow file:///D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/mlruns `
  --runid 你的训练runid
```

---

## 12. 最推荐的实际执行顺序

这是我建议你真正照着做的顺序：

### 第一步：确认 conda 激活正常

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda activate sdformer-upstream
```

### 第二步：确认依赖可导入

```powershell
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
python -c "import torch; import spikingjelly; import timm; import mlflow; print('imports_ok')"
```

### 第三步：确认数据目录齐全

至少手动检查：

- `gt_tensors`
- `mask_tensors`
- `sequence_lists`
- `event_tensors/10bins/left`

### 第四步：复制配置并改 `data.path`

不要直接改原配置。

### 第五步：把 `n_workers` 先设成 0

先解决 Windows dataloader 稳定性问题。

### 第六步：先启动训练

只验证能否进 iteration。

### 第七步：记录 runid

后续评估靠它恢复模型。

### 第八步：再跑 eval

确认 `AEE/AAE` 能正常输出。

---

## 13. 你最可能遇到的错误，以及怎么判断

## 13.1 `ModuleNotFoundError`

说明依赖没装好。  
先停在环境层解决，不要继续调模型。

## 13.2 `FileNotFoundError` 指向 `sequence_lists` 或 `event_tensors`

说明不是模型问题，而是数据目录不完整或目录名不匹配。

优先检查：

1. `data.path` 是否写对
2. `10bins` / `10bins_pol` 是否一致
3. `train_split_seq.csv` / `valid_split_seq.csv` 是否存在

## 13.3 `mlflow.get_run(...)` 失败

说明评估时给的 `runid` 或 `path_mlflow` 不对。  
训练和评估必须指向同一个 MLflow tracking 目录。

## 13.4 dataloader 卡死或 Windows 多进程报错

先把：

```yaml
loader:
    n_workers: 0
```

跑通后再增加。

## 13.5 显存不足

优先降低：

1. `batch_size`
2. `crop`
3. AMP 开关和可视化开关

---

## 14. 一个现实判断

如果你现在的目标只是“先跑通原版 SDformerFlow”，那真正的卡点通常不是模型代码本身，而是这三件事：

1. 数据目录是否完整
2. `data.path` 是否相对当前工作目录正确
3. MLflow 的 `runid` 链路是否闭环

把这三件事理顺，原版就不难起。

---

## 15. 我对你当前阶段的具体建议

你现在不要同时做下面几件事：

- 改模型
- 改插件
- 改硬件
- 重写预处理
- 跑原版训练

正确做法是只做原版跑通：

1. 先准备完整的 `saved_flow_data`
2. 再复制本地训练配置
3. 把 `data.path` 改对
4. 把 `n_workers` 先设成 0
5. 先训练启动成功
6. 再拿 `runid` 跑 eval

只要原版这一条链路先通，后面的所有工作都会清晰很多。

---

## 16. 你看这份文档之后该立刻做什么

你下一步最应该做的是两件事：

1. 确认你是否已经有完整的 `saved_flow_data`
2. 按下面第 17 节的命令先把 upstream 环境和导入检查跑通

---

## 17. 按你机器路径整理好的 PowerShell 命令模板

下面这组命令只针对你的机器路径：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda activate base
```

如果你还没创建 upstream 专用环境：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda create -n sdformer-upstream python=3.7.3 -y
conda activate sdformer-upstream
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
pip install -r requirements.txt
```

如果你已经创建过环境，只需要：

```powershell
& D:\anaconda\shell\condabin\conda-hook.ps1
conda activate sdformer-upstream
Set-Location D:\code\sdformer_codex\SDformer\third_party\SDformerFlow
```

然后先做 import 检查：

```powershell
python -c "import torch; import spikingjelly; import timm; import mlflow; print('imports_ok')"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果这两条都通过，下一步再去确认数据目录：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data
```

确认下面这些路径存在：

```text
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\gt_tensors
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\mask_tensors
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\sequence_lists\train_split_seq.csv
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\sequence_lists\valid_split_seq.csv
D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\saved_flow_data\event_tensors\10bins\left
```

如果这些都齐，再进入训练配置阶段。
