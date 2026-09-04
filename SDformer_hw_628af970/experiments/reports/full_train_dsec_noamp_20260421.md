# DSEC 全量 Baseline 训练报告

## 当前状态

- 报告状态：进行中
- 数据集根目录：`data/Datasets/DSEC/saved_flow_data`
- 训练日期：`2026-04-21`
- 当前主配置：`third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml`
- 总实验台账：[experiment_master_log.md](/home/zhumd/code/sdformer_codex/SDformer/experiments/reports/experiment_master_log.md)

## 这份报告的用途

这份报告用于记录 SDformerFlow 在完整 DSEC 光流训练集上的 baseline 训练过程，目标是得到：

- 可复现的 baseline checkpoint
- 可复现的推理结果
- 后续优化实验可直接对比的参考记录

## 为什么会有多次训练尝试

完整 baseline 在最初的 `AMP` 配置下，训练到 `epoch 0` 中段时出现了 `NaN`。  
为了区分问题来源、同时尽量保住训练速度，我们依次尝试了几条路线：

- `AMP` 开启：速度快，但出现 `NaN`
- `AMP` 关闭：更稳，但训练速度明显下降
- `AMP` 开启 + 降低学习率：希望在保留速度的同时减少数值不稳定

当前判断：

- `CuPy` 问题属于启动/后端兼容问题
- `NaN` 更像训练过程中的数值稳定性问题
- `AMP` 很可能是诱发 `NaN` 的重要因素之一，但不一定是唯一因素

## 目前已经跑过的实验

### 1. 单序列 smoke run

- 目的：验证从数据读取到训练保存的整条链路能否跑通
- 结果：成功
- run id：`a417c8095b3f42e3a2762b3cf446ccb1`

### 2. 全量训练，AMP 开启

- 目的：直接跑完整 baseline
- 结果：失败，出现 `NaN`
- run id：`5e98b281af454dfd9e17b16099d329dc`
- 失败位置：`epoch 0` 约 `2461 / 7345` step

### 3. 全量训练，AMP 关闭

- 目的：优先验证稳定性
- 结果：能跑，但速度过慢，不适合作为当前主线
- run id：`76225a9bc6cb4e7099f6f69feceb57d7`

### 4. 全量训练，AMP 开启 + 学习率下调

- 目的：保留 `AMP` 的速度优势，同时降低 `NaN` 概率
- 结果：进行中
- run id：待训练完成后补充

## MLflow 目前会记录什么

这个项目当前会写入 MLflow 的内容至少包括：

- 实验名
- run id
- 配置参数
- 历史 run id（如果有）
- 模型权重 `model.pth`
- 训练状态 `state_dict.pth`

不是所有信息都会自动以结构化指标写进 MLflow。下面这些内容目前主要还是在日志文件里：

- tqdm 训练进度
- 每轮训练速度
- 显存峰值
- 警告和报错栈

## 数据集摘要

当前准备好的完整 DSEC baseline 数据如下：

- flow 序列数：`18`
- 训练样本数：`7345`
- 验证样本数：`825`

## 训练流程说明

训练结束后，这一节会补成完整中文说明，至少包括：

- 原始 DSEC 数据如何整理成 `saved_flow_data`
- 一个训练样本里到底包含什么
- 模型前向传播时发生了什么
- loss、反向传播、参数更新分别是什么
- `run id`、`model.pth`、`state_dict.pth` 三者的关系
- 后续如何用训练好的结果做推理

## 最终指标

训练完成后补充。

## 推理结果

训练完成后补充。
