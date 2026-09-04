# Codex 服务器接手说明

这份文档用于把当前本地 Codex 会话的关键上下文，转交给服务器上的另一条 Codex 会话。

## 1. 项目与仓库状态

- GitHub 仓库：`https://github.com/Timmyz3/sdformer_codex`
- 当前工作仓库根目录：`SDformer`
- 最新关键提交：
  - `998c172 Vendor upstream SDformerFlow into main repository`
  - `5ee39fa Add hetero backend path for ROCm smoke tests`

当前仓库已经不是子模块工程：

- `third_party/SDformerFlow` 已经从 submodule 收编成普通目录
- `.gitmodules` 已删除
- `upstream` remote 已删除
- `origin` 只指向用户自己的仓库

## 2. 当前研究目标

当前优先目标不是完整论文训练，而是：

1. 跑通原版 `SDformerFlow` 单序列 smoke test
2. 数据序列固定为 `zurich_city_09_a`
3. 在服务器环境中完成：
   - 依赖安装
   - 数据解压
   - `saved_flow_data` 生成
   - 原版训练脚本启动

## 3. 关键文件

### 3.1 原版训练与评估入口

- `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`

### 3.2 单序列预处理脚本

- `tools/prepare_dsec_single_sequence.py`

功能：

- 从 DSEC 原始单序列目录生成最小版 `saved_flow_data`
- 当前目标序列：`zurich_city_09_a`

### 3.3 标准 CUDA 单序列配置

- `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml`
- `third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq.yml`

### 3.4 异构卡兼容分支

这是新增但不影响原版 CUDA 路线的兼容层。

- backend helper：
  - `third_party/SDformerFlow/utils/runtime_backend.py`
- 异构依赖清单：
  - `third_party/SDformerFlow/requirements_hetero_no_cupy.txt`
- 异构训练配置：
  - `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml`
- 异构评估配置：
  - `third_party/SDformerFlow/configs/valid_DSEC_supervised_single_seq_hetero.yml`

## 4. backend 兼容改动说明

原版脚本默认在非 CPU 情况下强制：

```python
functional.set_backend(model, "cupy", neurontype)
```

这会在 ROCm/HIP 或其他异构卡上失败，因为没有 `cupy-cuda12x`。

现在已经改成通过：

- `utils/runtime_backend.py`

统一选择 backend。

支持：

- `cupy`
- `torch`
- `auto`

默认原版 CUDA 路线保持 `cupy`。  
只有在异构配置中显式设置：

```yaml
runtime:
    snn_backend: "torch"
```

才会走兼容分支。

## 5. 数据与包文件

本地已经打好的上传文件：

- 代码瘦身包：
  - `D:\code\sdformer_codex\server_upload\sdformer_code_slim.tar`
- 单序列原始事件包：
  - `D:\code\sdformer_codex\server_upload\zurich_city_09_a_train_events.tar`
- 单序列原始光流包：
  - `D:\code\sdformer_codex\server_upload\zurich_city_09_a_train_optical_flow.tar`

服务器预期目录：

```text
/root/private_data/raw/
```

服务器工作目录约定：

```text
/root/private_data/work/SDformer
```

## 6. 服务器侧标准解压路径

代码包解压：

```bash
mkdir -p /root/private_data/work
cd /root/private_data/work
tar -xf /root/private_data/raw/sdformer_code_slim.tar
cd /root/private_data/work/SDformer
```

数据包解压：

```bash
mkdir -p data/Datasets/DSEC/train_events
mkdir -p data/Datasets/DSEC/train_optical_flow

tar -xf /root/private_data/raw/zurich_city_09_a_train_events.tar -C data/Datasets/DSEC/train_events
tar -xf /root/private_data/raw/zurich_city_09_a_train_optical_flow.tar -C data/Datasets/DSEC/train_optical_flow
```

## 7. 已知运行路线

### 7.1 标准 4090 / CUDA 路线

依赖：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
python -m pip install -r requirements_colab_no_torch.txt
```

预处理：

```bash
cd /root/private_data/work/SDformer
python tools/prepare_dsec_single_sequence.py \
  --root data/Datasets/DSEC \
  --sequence zurich_city_09_a \
  --num-bins 10 \
  --valid-stride 10
```

训练：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

### 7.2 异构卡 / HIP 路线

依赖：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
python -m pip install -r requirements_hetero_no_cupy.txt
```

训练：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_single_seq_hetero.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

## 8. 已知问题

### 8.1 服务器 git / github.com DNS 不稳定

服务器端一度出现：

```text
Could not resolve hostname github.com
```

因此当前更可靠的路线是：

- 本地打包
- 上传 tar
- 服务器端解压

而不是依赖服务器直接 clone GitHub。

### 8.2 OpenCV 导入失败已确认可修复

服务器在 2026-03-24 上实际出现过：

```text
ImportError: libxcb.so.1: cannot open shared object file
```

现已验证下面这组命令可以修复：

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
python -m pip uninstall -y opencv-python
python -m pip install --index-url https://pypi.org/simple opencv-python-headless
```

修复后测试：

```bash
python - <<'PY'
import cv2
print(cv2.__version__)
PY
```

服务器端已成功打印：

```text
4.13.0
```

注意：

- 当前全局 `pip` 默认镜像是 `https://pypi.tuna.tsinghua.edu.cn/simple`
- 该镜像在安装 `opencv-python-headless` 时返回过 `403`
- 因此这里建议显式加 `--index-url https://pypi.org/simple`

### 8.3 当前最新阻塞点：`train_events` 原始 tar 已损坏

虽然 `cv2` 问题已经修复，但在服务器上运行：

```bash
cd /root/private_data/work/SDformer
python tools/prepare_dsec_single_sequence.py \
  --root data/Datasets/DSEC \
  --sequence zurich_city_09_a \
  --num-bins 10 \
  --valid-stride 10
```

时出现：

```text
OSError: Unable to synchronously open file (truncated file: eof = 583006208, stored_eof = 3024384503)
```

已确认问题不在代码，而在上传到服务器的原始事件包本身：

- `/root/private_data/raw/zurich_city_09_a_train_events.tar` 只有约 `556 MB`
- 但 `tar -tvf` 显示其中的 `events.h5` 应该是 `3024384503` bytes
- 当前服务器上解压后的 `events.h5` 也只有 `583006208` bytes
- `tar -tvf /root/private_data/raw/zurich_city_09_a_train_events.tar` 会报：

```text
tar: Unexpected EOF in archive
```

这说明：

- `train_optical_flow` 包是完整的
- `train_events` 包是截断包
- 在重新上传完整 `zurich_city_09_a_train_events.tar` 之前，服务器侧无法生成 `saved_flow_data`
- 因此也无法继续单序列 smoke 训练

## 9. 当前最优先下一步

服务器端新的 Codex 会话接手时，优先做下面这件事：

1. 重新上传完整的 `/root/private_data/raw/zurich_city_09_a_train_events.tar`
2. 重新解压 `data/Datasets/DSEC/train_events/zurich_city_09_a`
3. 再次执行 `tools/prepare_dsec_single_sequence.py`
4. 成功生成 `saved_flow_data` 后再启动单序列训练

也就是说，当前最应该盯的是原始事件包完整性，不是进一步改模型。
