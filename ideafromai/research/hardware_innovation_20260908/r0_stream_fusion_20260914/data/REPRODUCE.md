# 本阶段实际入口

全部脚本要求Python3.12。生产和前轮文件只读；输出为脚本所在本目录。没有模型训练、参数扫描、hash或环境安装步骤。

本机捕获、完整整数参考及统计使用已经存在的解释器：

```bash
cd /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/r0_stream_fusion_20260914/data
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$PWD/cupy_cache"
export TORCH_HOME="$PWD/torch_cache"
R0_PY312=/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/r0_execution_trials_20260913/data_and_quality/py312/bin/python
"$R0_PY312" capture_full.py
"$R0_PY312" make_gold.py
/opt/anaconda3/bin/python3.12 source_statistics.py
```

A800质量调用的位置为 `/root/private_data/work/hardware_innovation_20260908/r0_stream_fusion_20260914/data`。同目录已放model_access.py、evaluate_valid825.py、evaluate_integer_factor.py、三份固定weights和integer_factors.npz；只读远端原模型与完整数据。实际环境变量与命令：

```bash
cd /root/private_data/work/hardware_innovation_20260908/r0_stream_fusion_20260914/data
export PYTHONDONTWRITEBYTECODE=1
export R0_STREAM_BASE=/root/private_data/work/hardware_innovation_20260908
export R0_STREAM_REPO=/root/private_data/work/sdformer_codex
export CUPY_CACHE_DIR="$PWD/cupy_cache"
export TORCH_HOME="$PWD/torch_cache"
R0_A800_PY312=/root/private_data/work/hardware_innovation_20260908/env312/bin/python3.12
"$R0_A800_PY312" evaluate_valid825.py
"$R0_A800_PY312" evaluate_integer_factor.py
```

本次实际安排是在三臂825中间优先插入新链十帧，然后恢复原825进程；无需按此暂停安排才能复现函数。dense的已记录wall_seconds含暂停，不用于速度比较。A800报告保留其Python/Torch/GPU/TF32身份，与本地3090捕获环境区别明确。

最终JSON/逐帧文件同步本机后，以下入口只做已有结果与输入公式检查，不重新运行模型或RTL：

```bash
/opt/anaconda3/bin/python3.12 validate_quality.py
/opt/anaconda3/bin/python3.12 review_stream_counts.py
/opt/anaconda3/bin/python3.12 review_integer_counts.py
```

具体报告：[README](README.md)、[REVIEW_STREAM](REVIEW_STREAM.md)、[REVIEW_INTEGER_FACTOR](REVIEW_INTEGER_FACTOR.md)。
