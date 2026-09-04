# Colab 单序列 Smoke Test 迁移指南

这份文档只服务一个目标：

把当前已经在本地打通的 `zurich_city_09_a` 单序列 smoke test 迁到 Google Colab，继续验证原版 `SDformerFlow` 的训练链路。

当前本地状态：

- 原版 upstream 入口可用：`third_party/SDformerFlow`
- 单序列原始数据已对齐：
  - `data/Datasets/DSEC/train_events/zurich_city_09_a/events/left`
  - `data/Datasets/DSEC/train_optical_flow/zurich_city_09_a/flow`
- 单序列预处理脚本已就位：`tools/prepare_dsec_single_sequence.py`
- 单序列 smoke 配置已就位：
  - `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml`
  - `third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq.yml`

## 1. 为什么 Colab 侧不要直接上传 `saved_flow_data`

本地这份单序列 `saved_flow_data` 体积大约是 `9.6 GB`，上传慢，而且 Drive 上直接读大量小文件不稳。

Colab 官方 FAQ 明确建议：

- Colab 免费版提供 GPU，但资源和 GPU 类型是动态的，不保证固定硬件或固定配额。
- 免费版 notebook 最长大约 12 小时。
- 从 Drive 读取大量小文件容易超时；更推荐先把归档文件复制到 Colab 虚拟机本地，再在本地解压。

来源：

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

所以这里的推荐方案是：

1. 本地只上传 `raw 单序列`
2. Colab 端重新生成 `saved_flow_data`
3. 再跑原版 smoke train

## 2. 本地需要准备什么

推荐只准备这两份原始单序列目录：

- `SDformer/data/Datasets/DSEC/train_events/zurich_city_09_a`
- `SDformer/data/Datasets/DSEC/train_optical_flow/zurich_city_09_a`

你本地观察到的大致体积是：

- `train_events/zurich_city_09_a`: 约 `3.03 GB`
- `train_optical_flow/zurich_city_09_a`: 约 `0.18 GB`

这比直接传 `saved_flow_data` 更合理。

## 3. 本地打包推荐

推荐先把原始单序列打成两个 `tar` 文件，再上传到 Google Drive。

在 PowerShell 里执行：

```powershell
New-Item -ItemType Directory -Force D:\code\sdformer_codex\colab_upload | Out-Null

tar -cf D:\code\sdformer_codex\colab_upload\zurich_city_09_a_train_events.tar `
  -C D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\train_events `
  zurich_city_09_a

tar -cf D:\code\sdformer_codex\colab_upload\zurich_city_09_a_train_optical_flow.tar `
  -C D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\train_optical_flow `
  zurich_city_09_a
```

然后把这两个文件上传到 Google Drive，例如：

```text
MyDrive/sdformer_colab/raw/
  zurich_city_09_a_train_events.tar
  zurich_city_09_a_train_optical_flow.tar
```

## 4. Colab Runtime 选择

在 Colab 页面里：

1. 打开一个新 notebook
2. 选择 `Runtime -> Change runtime type`
3. `Hardware accelerator` 选 `GPU`

先执行：

```python
!nvidia-smi
```

如果能看到 T4 / L4 / P100 / V100 之类 GPU，就可以继续。

## 5. Colab 端目录约定

这个 runbook 假设你在 Colab 里使用下面这些路径：

```text
/content/sdformer_codex
/content/sdformer_codex/SDformer
/content/drive/MyDrive/sdformer_colab/raw
/content/drive/MyDrive/sdformer_colab/mlruns
```

## 6. Colab Cell 1：挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 7. Colab Cell 2：拉取当前仓库

如果你要直接使用 GitHub 上的当前仓库：

```python
%cd /content
!git clone --recurse-submodules https://github.com/Timmyz3/sdformer_codex.git
%cd /content/sdformer_codex/SDformer
```

如果你已经在 GitHub 上继续提交了新的改动，之后可以再执行：

```python
%cd /content/sdformer_codex
!git pull
!git submodule update --init --recursive
%cd /content/sdformer_codex/SDformer
```

## 8. Colab Cell 3：安装依赖

Colab 一般已经自带 GPU 版 `torch`，这里不要再用 upstream 原始 `requirements.txt` 去覆盖它。

直接安装我们补好的 `no_torch` 依赖清单：

```python
%cd /content/sdformer_codex/SDformer/third_party/SDformerFlow
!python -m pip install -q -r requirements_colab_no_torch.txt
```

然后做导入检查：

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import spikingjelly
import timm
import mlflow

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print("imports_ok")
```

理想结果：

- `torch.cuda.is_available()` 为 `True`
- `torch.cuda.device_count()` 至少为 `1`

## 9. Colab Cell 4：把 raw 单序列复制到本地 VM 并解压

Colab FAQ 明确建议不要直接在 Drive 挂载目录里高频读写大量小文件，所以这里先把归档复制到 `/content` 本地磁盘。

```python
%cd /content
!mkdir -p /content/dsec_raw_archives
!cp /content/drive/MyDrive/sdformer_colab/raw/zurich_city_09_a_train_events.tar /content/dsec_raw_archives/
!cp /content/drive/MyDrive/sdformer_colab/raw/zurich_city_09_a_train_optical_flow.tar /content/dsec_raw_archives/
```

然后解压到仓库标准目录：

```python
!mkdir -p /content/sdformer_codex/SDformer/data/Datasets/DSEC/train_events
!mkdir -p /content/sdformer_codex/SDformer/data/Datasets/DSEC/train_optical_flow

!tar -xf /content/dsec_raw_archives/zurich_city_09_a_train_events.tar \
  -C /content/sdformer_codex/SDformer/data/Datasets/DSEC/train_events

!tar -xf /content/dsec_raw_archives/zurich_city_09_a_train_optical_flow.tar \
  -C /content/sdformer_codex/SDformer/data/Datasets/DSEC/train_optical_flow
```

## 10. Colab Cell 5：检查原始单序列目录

```python
import os

checks = [
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/train_events/zurich_city_09_a/events/left/events.h5",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/train_events/zurich_city_09_a/events/left/rectify_map.h5",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/train_optical_flow/zurich_city_09_a/flow/forward_timestamps.txt",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/train_optical_flow/zurich_city_09_a/flow/forward",
]

for path in checks:
    print(path, os.path.exists(path))
```

4 条都应该是 `True`。

## 11. Colab Cell 6：在 Colab 端生成 `saved_flow_data`

```python
%cd /content/sdformer_codex/SDformer
!python tools/prepare_dsec_single_sequence.py \
  --root /content/sdformer_codex/SDformer/data/Datasets/DSEC \
  --sequence zurich_city_09_a \
  --num-bins 10 \
  --valid-stride 10
```

正常情况下会打印类似：

```text
Generated 638 samples for zurich_city_09_a
Train samples: 574
Valid samples: 64
```

## 12. Colab Cell 7：检查 `saved_flow_data`

```python
checks = [
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data/gt_tensors",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data/mask_tensors",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data/sequence_lists/train_split_seq.csv",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv",
    "/content/sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/zurich_city_09_a",
]

for path in checks:
    print(path, os.path.exists(path))
```

## 13. Colab Cell 8：启动原版单序列 smoke training

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
```

```python
%cd /content/sdformer_codex/SDformer/third_party/SDformerFlow
!python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///content/drive/MyDrive/sdformer_colab/mlruns
```

你要观察的成功信号：

1. 打印出 `device: cuda:0`
2. 正常创建 `Training Dataset ...`
3. 正常创建 `Validation Dataset ...`
4. 进入 `Epoch 0`
5. 不再像你本地 4GB 卡那样在第一批前向直接 OOM

## 14. Colab Cell 9：训练后跑评估

原版评估脚本依赖 `runid`，不是直接喂一个本地 checkpoint 路径。

训练结束后，先到 MLflow 目录里记下 run id，再执行：

```python
%cd /content/sdformer_codex/SDformer/third_party/SDformerFlow
!python eval_DSEC_flow_SNN.py \
  --config configs/valid_DSEC_supervised_single_seq.yml \
  --runid YOUR_RUN_ID \
  --path_mlflow file:///content/drive/MyDrive/sdformer_colab/mlruns
```

把 `YOUR_RUN_ID` 换成实际值。

## 15. Colab 端常见问题

### 15.1 GPU 没分到

现象：

- `torch.cuda.is_available()` 是 `False`
- 或 `!nvidia-smi` 看不到 GPU

处理：

1. 确认 runtime type 已选 GPU
2. 重连 runtime
3. 换一个时间段再试

这属于 Colab 免费资源动态分配问题，不是仓库代码问题。

### 15.2 `cupy` 安装失败

先看错误是不是网络临时失败。Colab 上通常重跑一次安装单元就够。

如果是 wheel 兼容性问题，再单独试：

```python
!python -m pip install cupy-cuda12x
```

### 15.3 Drive 读取很慢

不要直接在 Drive 目录里运行预处理和训练。  
按上面的流程，先把归档复制到 `/content`，再解压，再运行。

### 15.4 Colab 中断

免费版最长一般约 12 小时，而且 idle 也会断。  
所以建议：

1. 原始 tar 长期保存在 Drive
2. `mlruns` 放在 Drive
3. 每次 runtime 断开后，只重做“挂载 Drive -> clone repo -> 解压 raw -> 生成/复用 saved_flow_data -> 继续跑”

## 16. 这条 Colab 路线的定位

它适合做：

- 原版 `SDformerFlow` 单序列 smoke training
- 原版训练链路验证
- 后续模块改动的快速功能回归

它不适合做：

- 完整 DSEC 全量长训练
- 论文最终大规模统计实验

完整论文训练仍然更适合固定 GPU 服务器。
