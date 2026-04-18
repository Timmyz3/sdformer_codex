# SCNet 单序列训练运行指南

这份文档只解决一个问题：

如何把当前仓库里的 `SDformerFlow` 单序列 smoke test 放到 SCNet 上跑起来。

推荐路线不是一上来就提交正式“模型训练”任务，而是两阶段：

1. 先用 `Notebook` 把环境、代码、数据、启动命令交互式跑通
2. 再把完全相同的命令迁移到 `模型训练` 页面做正式任务

这样做的原因很简单：

- 你现在这个项目还在 bring-up 阶段
- 原版 upstream 对环境和数据目录都比较敏感
- 先在 Notebook 中调通，再转训练任务，排错成本最低

## 1. 为什么推荐 SCNet 上先用 Notebook

SCNet 官方“快速开始”文档明确给出了 Notebook 的基础流程：

- 登录控制台
- 进入人工智能服务功能区
- 在 `Notebook` 页面创建实例
- 开机后点击 `JupyterLab` 进入环境  

来源：

- [SCNet 快速开始](https://www.scnet.cn/help/docs/mainsite/ai/quick-start/)

SCNet 官方“模型训练最佳实践”也建议先基于 Notebook 构建并验证环境，再保存镜像，再去模型训练页面创建任务。并且文档明确建议把工程、数据、模型文件放在 `/root/private_data` 下。

来源：

- [SCNet 模型训练最佳实践](https://www.scnet.cn/help/docs/mainsite/ai/model-training/practice/)
- [SCNet Notebook 功能介绍](https://www.scnet.cn/help/docs/mainsite/ai/notebook/function-introduction/index.html)

## 2. 你这次在 SCNet 上要跑什么

不是完整 DSEC 全量训练，而是当前已经准备好的单序列 smoke 版本：

- 数据序列：`zurich_city_09_a`
- 预处理脚本：`tools/prepare_dsec_single_sequence.py`
- 训练配置：`third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml`
- 评估配置：`third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq.yml`

## 3. 资源怎么选

### 3.1 Notebook 阶段

建议选：

- 1 张 GPU 起步
- 优先选显存更大的卡
- 如果有 `AI-64GB` 这类规格，优先选它

原因：

- 你本地 `RTX 3050 Laptop GPU 4GB` 已经验证会在第一批前向时 OOM
- 这个原版 `MS_SpikingformerFlowNet_en4` 对显存不友好

### 3.2 模型训练阶段

单序列 smoke test 不需要多机多卡。  
先用：

- `实例数 = 1`
- `每实例加速卡数量 = 1`

把链路跑通以后，再讨论扩展。

## 4. SCNet 上推荐的工作目录

建议统一放在：

```text
/root/private_data/sdformer_codex
```

SCNet 官方文档明确建议把个人工程和数据放在 `/root/private_data` 下。

## 5. 先准备什么文件

你不需要把本地整个工作目录都上传到 SCNet。

推荐只准备两类内容：

### 5.1 代码

直接在 SCNet 实例里 `git clone` 你的 GitHub 仓库：

- 仓库地址：[Timmyz3/sdformer_codex](https://github.com/Timmyz3/sdformer_codex)

### 5.2 数据

只上传 raw 单序列归档：

- `zurich_city_09_a_train_events.tar`
- `zurich_city_09_a_train_optical_flow.tar`

如果你还没打包，本地 PowerShell 命令是：

```powershell
New-Item -ItemType Directory -Force D:\code\sdformer_codex\scnet_upload | Out-Null

tar -cf D:\code\sdformer_codex\scnet_upload\zurich_city_09_a_train_events.tar `
  -C D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\train_events `
  zurich_city_09_a

tar -cf D:\code\sdformer_codex\scnet_upload\zurich_city_09_a_train_optical_flow.tar `
  -C D:\code\sdformer_codex\SDformer\data\Datasets\DSEC\train_optical_flow `
  zurich_city_09_a
```

然后把这两个 tar 上传到 SCNet 的文件管理 `E-File`。  
SCNet 文件传输文档说明，E-File 支持上传软件安装包和计算文件，建议在个人目录下新建工作文件夹。

来源：

- [SCNet 文件传输与下载](https://www.scnet.cn/doc/1.0.6/30000/general-handbook/User-Guide/file-transfer.html)

## 6. 第一阶段：用 Notebook 跑通

### 6.1 创建 Notebook

在 SCNet 控制台：

1. 进入 `人工智能服务`
2. 点击 `Notebook`
3. 点击 `创建 Notebook`
4. 选择：
   - 区域
   - 1 张 GPU
   - 基础镜像中的 PyTorch 镜像

SCNet 文档示例里使用的是 PyTorch 2.4.1、Python 3.10、Ubuntu 22.04 一类的基础镜像。  
对你当前项目，这是合适的起点。

这一步是基于官方示例做的工程建议，不是平台唯一可用镜像。

### 6.2 进入 JupyterLab 终端

实例启动后，点 `JupyterLab`，打开终端。

先执行：

```bash
nvidia-smi
python --version
pwd
```

你要确认：

- GPU 可见
- Python 是 3.10 左右
- 当前能正常在终端里执行命令

### 6.3 克隆代码

```bash
cd /root/private_data
git clone --recurse-submodules https://github.com/Timmyz3/sdformer_codex.git
cd /root/private_data/sdformer_codex/SDformer
```

如果后面你本地又有新提交，可以更新：

```bash
cd /root/private_data/sdformer_codex
git pull
git submodule update --init --recursive
cd /root/private_data/sdformer_codex/SDformer
```

### 6.4 安装依赖

不要直接使用 upstream 原始 `requirements.txt` 去重装 `torch`。  
优先使用当前仓库里已经准备好的“保留平台 torch、只装其余依赖”的清单：

- [requirements_colab_no_torch.txt](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/requirements_colab_no_torch.txt)

在 SCNet 终端执行：

```bash
cd /root/private_data/sdformer_codex/SDformer/third_party/SDformerFlow
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_colab_no_torch.txt
```

然后做检查：

```bash
python - <<'PY'
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, spikingjelly, timm, mlflow
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print("imports_ok")
PY
```

你希望看到：

- `torch.cuda.is_available()` 为 `True`
- 打印 `imports_ok`

### 6.5 准备 raw 单序列数据

假设你已经把两个 tar 上传到了 SCNet 的个人文件区，并且它们最终位于：

```text
/root/private_data/raw/zurich_city_09_a_train_events.tar
/root/private_data/raw/zurich_city_09_a_train_optical_flow.tar
```

如果你的实际路径不同，把下面命令里的路径改掉即可。

先解压到项目标准目录：

```bash
cd /root/private_data/sdformer_codex/SDformer

mkdir -p data/Datasets/DSEC/train_events
mkdir -p data/Datasets/DSEC/train_optical_flow

tar -xf /root/private_data/raw/zurich_city_09_a_train_events.tar \
  -C data/Datasets/DSEC/train_events

tar -xf /root/private_data/raw/zurich_city_09_a_train_optical_flow.tar \
  -C data/Datasets/DSEC/train_optical_flow
```

### 6.6 检查原始数据目录

```bash
python - <<'PY'
import os
checks = [
    "data/Datasets/DSEC/train_events/zurich_city_09_a/events/left/events.h5",
    "data/Datasets/DSEC/train_events/zurich_city_09_a/events/left/rectify_map.h5",
    "data/Datasets/DSEC/train_optical_flow/zurich_city_09_a/flow/forward_timestamps.txt",
    "data/Datasets/DSEC/train_optical_flow/zurich_city_09_a/flow/forward",
]
for p in checks:
    print(p, os.path.exists(p))
PY
```

4 条都应该是 `True`。

### 6.7 生成 `saved_flow_data`

```bash
cd /root/private_data/sdformer_codex/SDformer

python tools/prepare_dsec_single_sequence.py \
  --root data/Datasets/DSEC \
  --sequence zurich_city_09_a \
  --num-bins 10 \
  --valid-stride 10
```

正常输出应接近：

```text
Generated 638 samples for zurich_city_09_a
Train samples: 574
Valid samples: 64
```

### 6.8 检查 `saved_flow_data`

```bash
python - <<'PY'
import os
checks = [
    "data/Datasets/DSEC/saved_flow_data/gt_tensors",
    "data/Datasets/DSEC/saved_flow_data/mask_tensors",
    "data/Datasets/DSEC/saved_flow_data/sequence_lists/train_split_seq.csv",
    "data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv",
    "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/zurich_city_09_a",
]
for p in checks:
    print(p, os.path.exists(p))
PY
```

### 6.9 启动 smoke training

```bash
cd /root/private_data/sdformer_codex/SDformer/third_party/SDformerFlow

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

你要观察的信号：

1. `device: cuda:0`
2. `Training Dataset ...`
3. `Validation Dataset ...`
4. `Epoch 0`
5. 训练不再像你本地 4GB 卡那样在第一批前向直接 OOM

### 6.10 如果 Notebook 阶段跑通

这时你就有两种选择：

1. 继续在 Notebook 里训练
2. 把环境 `保存镜像`，然后转到 `模型训练`

SCNet 官方 Notebook 文档支持保存镜像，但也提醒单层镜像数据不要超过 `15 GiB`；大文件应放在 `/root/private_data` 等文件存储路径。

来源：

- [SCNet Notebook 功能介绍](https://www.scnet.cn/help/docs/mainsite/ai/notebook/function-introduction/index.html)

## 7. 第二阶段：迁移到模型训练任务

当 Notebook 中这条命令已经确认可跑后，再进入 SCNet 的 `模型训练` 页面创建任务。

官方模型训练手册说明：

- 进入 `人工智能服务 -> 模型训练`
- 点击 `创建训练任务`
- 选择加速卡型号、每实例卡数、实例数、训练镜像
- 填写启动命令

来源：

- [SCNet 模型训练用户帮助手册](https://www.scnet.cn/help/docs/mainsite/ai/model-training/index.html)

### 7.1 推荐配置

对于你当前单序列 smoke test：

- `实例数 = 1`
- `每实例加速卡数量 = 1`
- 训练镜像：
  - 如果你在 Notebook 中已经保存镜像，就选 `我的镜像`
  - 如果没有保存镜像，就重新选同款 PyTorch 基础镜像，并在启动命令中补安装步骤

### 7.2 挂载建议

如果平台自动提供个人持久目录并能在容器中看到 `/root/private_data`，直接用它。  
如果没有自动看到，就在训练任务的“自定义挂载”里把你的个人文件区挂到 `/root/private_data`。

这是工程建议。平台官方手册明确支持通过自定义挂载把 E-File 中的文件挂到容器中。

### 7.3 启动命令模板

如果你已经保存好了带依赖的镜像，启动命令可以直接用：

```bash
cd /root/private_data/sdformer_codex/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

如果你没有保存镜像，而是直接用基础镜像，启动命令建议写成：

```bash
cd /root/private_data/sdformer_codex/SDformer/third_party/SDformerFlow
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_colab_no_torch.txt
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

但这会让每次任务启动都重新装依赖，所以仍然更推荐先保存镜像。

## 8. 训练结束后怎么评估

原版评估脚本用 `runid`，不是直接给 checkpoint 路径。

命令：

```bash
cd /root/private_data/sdformer_codex/SDformer/third_party/SDformerFlow
python eval_DSEC_flow_SNN.py \
  --config configs/valid_DSEC_supervised_single_seq.yml \
  --runid YOUR_RUN_ID \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

把 `YOUR_RUN_ID` 换成实际训练对应的 run id。

## 9. 你现在最应该怎么开始

最稳的顺序就是：

1. 在 SCNet 里先创建 1 张大显存卡的 Notebook
2. 在 `/root/private_data` 下 clone 当前仓库
3. 把两个 raw tar 上传到 E-File
4. 在 Notebook 中解压 raw 数据
5. 跑 `prepare_dsec_single_sequence.py`
6. 跑单序列 smoke training
7. 如果确认可用，再保存镜像并迁移到“模型训练”

## 10. 当前项目对应的关键文件

- 预处理脚本：[prepare_dsec_single_sequence.py](/D:/code/sdformer_codex/SDformer/tools/prepare_dsec_single_sequence.py)
- 训练配置：[train_DSEC_supervised_SDformerFlow_en4_single_seq.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml)
- 评估配置：[valid_DSEC_supervised_single_seq.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq.yml)
- 依赖清单：[requirements_colab_no_torch.txt](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/requirements_colab_no_torch.txt)

## 11. 异构卡兼容选择

如果你当前拿到的是 ROCm/HIP 或其他异构卡，而不是标准 CUDA 卡，不要改默认配置，直接使用新增的异构卡分支：

- 异构卡依赖：[requirements_hetero_no_cupy.txt](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/requirements_hetero_no_cupy.txt)
- 异构卡训练配置：[train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml)
- 异构卡评估配置：[valid_DSEC_supervised_single_seq_hetero.yml](/D:/code/sdformer_codex/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq_hetero.yml)

这套配置只把 `runtime.snn_backend` 切到 `torch`，并移除了 `cupy-cuda12x` 依赖；原来的 CUDA/4090 路线保持不变。
