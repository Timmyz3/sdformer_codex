# Markdown 分类导航

## 1. 维护规则

这份文件是当前项目所有重要 Markdown 文档的分类导航。

从现在开始，新增任何有长期使用价值的 `.md` 文件时，都必须同步更新这份导航，至少补这三项：

1. 文件路径
2. 它解决什么问题
3. 遇到什么情况时应该去看它

如果不更新这份导航，后续文档会继续堆积，查找成本会越来越高。

## 2. 快速入口

### 如果你刚接手项目，不知道整体做到了哪里

先看：

- [CODEX_SERVER_HANDOFF_ZH.md](/root/private_data/work/SDformer/CODEX_SERVER_HANDOFF_ZH.md)
- [README.md](/root/private_data/work/SDformer/README.md)

### 如果你想知道当前研究主线、阶段目标和论文级路线

看：

- [PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md](/root/private_data/work/SDformer/PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md)
- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)

### 如果你要跑单序列 smoke

看：

- [COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md](/root/private_data/work/SDformer/COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md)
- [SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md](/root/private_data/work/SDformer/SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md)

### 如果你要做 DSEC 小量实验 / 子集实验

看：

- [DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md](/root/private_data/work/SDformer/DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md)
- [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)

### 如果你要改模型结构、改注意力、插模块、做剪枝

看：

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
- [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
- [MODULAR_UPGRADE_TECHNICAL_DOC.md](/root/private_data/work/SDformer/MODULAR_UPGRADE_TECHNICAL_DOC.md)
- [MODULE_ZOO.md](/root/private_data/work/SDformer/MODULE_ZOO.md)

### 如果你想看训练记录、曲线、MLflow 对比

看：

- [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)
- [experiments/results/mlflow_compare/README.md](/root/private_data/work/SDformer/experiments/results/mlflow_compare/README.md)

### 如果你要看硬件文档

看：

- [arch.md](/root/private_data/work/SDformer/hw/docs/arch.md)
- [interfaces.md](/root/private_data/work/SDformer/hw/docs/interfaces.md)
- [perf_model.md](/root/private_data/work/SDformer/hw/docs/perf_model.md)

### 如果你要看论文/文献辅助材料

看：

- [efficient_transformer_survey.md](/root/private_data/work/SDformer/docs/literature/efficient_transformer_survey.md)
- [final_recommendations.md](/root/private_data/work/SDformer/docs/literature/final_recommendations.md)

## 3. 按用途分类

### 3.1 项目状态与总览

- [README.md](/root/private_data/work/SDformer/README.md)
  - 项目总览
- [CODEX_SERVER_HANDOFF_ZH.md](/root/private_data/work/SDformer/CODEX_SERVER_HANDOFF_ZH.md)
  - 服务器交接、已完成内容、当前阻塞点
- [REPORT.md](/root/private_data/work/SDformer/REPORT.md)
  - 项目阶段性汇总
- [RUNBOOK_AND_RESEARCH_PLAN_2026.md](/root/private_data/work/SDformer/RUNBOOK_AND_RESEARCH_PLAN_2026.md)
  - 中长期计划与研究安排

### 3.2 运行与复现

- [UPSTREAM_SDFORMERFLOW_RUNBOOK_ZH.md](/root/private_data/work/SDformer/UPSTREAM_SDFORMERFLOW_RUNBOOK_ZH.md)
  - upstream 路线怎么跑
- [PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md](/root/private_data/work/SDformer/PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md)
  - 论文级路线图
- [COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md](/root/private_data/work/SDformer/COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md)
  - Colab 单序列 smoke
- [SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md](/root/private_data/work/SDformer/SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md)
  - 单序列训练的详细流程
- [FULL_STACK_TECHNICAL_GUIDE_ZH.md](/root/private_data/work/SDformer/FULL_STACK_TECHNICAL_GUIDE_ZH.md)
  - 更偏工程链路的技术说明

### 3.3 实验流程与 SOP

- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
  - 改模型之后如何逐级筛选，不直接上 full train
- [DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md](/root/private_data/work/SDformer/DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md)
  - DSEC 小子集实验协议
- [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)
  - 可直接执行的小量实验模板

### 3.4 模型结构改动

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
  - baseline 从输入到输出的详细走读
- [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
  - 改注意力、插模块、换神经元、做剪枝的入口与接口
- [MODULAR_UPGRADE_TECHNICAL_DOC.md](/root/private_data/work/SDformer/MODULAR_UPGRADE_TECHNICAL_DOC.md)
  - 模块升级相关技术说明
- [MODULE_ZOO.md](/root/private_data/work/SDformer/MODULE_ZOO.md)
  - 可选模块与模块池

### 3.5 训练可视化与结果对比

- [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)
  - 如何看训练曲线、如何导出 run 对比
- [experiments/results/mlflow_compare/README.md](/root/private_data/work/SDformer/experiments/results/mlflow_compare/README.md)
  - 当前 MLflow 对比导出结果说明

### 3.6 论文与研究设计

- [PAPER_CO_DESIGN_PROPOSAL.md](/root/private_data/work/SDformer/PAPER_CO_DESIGN_PROPOSAL.md)
  - 论文 co-design 方向
- [TECHNICAL_DOCUMENTATION.md](/root/private_data/work/SDformer/TECHNICAL_DOCUMENTATION.md)
  - 综合技术文档

### 3.7 文献与外部参考

- [efficient_transformer_survey.md](/root/private_data/work/SDformer/docs/literature/efficient_transformer_survey.md)
  - 高效 Transformer 文献综述
- [final_recommendations.md](/root/private_data/work/SDformer/docs/literature/final_recommendations.md)
  - 文献方向建议

### 3.8 硬件设计

- [arch.md](/root/private_data/work/SDformer/hw/docs/arch.md)
  - 硬件架构说明
- [interfaces.md](/root/private_data/work/SDformer/hw/docs/interfaces.md)
  - 硬件接口说明
- [perf_model.md](/root/private_data/work/SDformer/hw/docs/perf_model.md)
  - 硬件性能模型

### 3.9 第三方仓库自带说明

- [third_party/SDformerFlow/README.md](/root/private_data/work/SDformer/third_party/SDformerFlow/README.md)
  - upstream 原始 README

## 4. 按“遇到什么情况”查文档

### 情况 1：我不知道当前项目做到了哪一步

看：

- [CODEX_SERVER_HANDOFF_ZH.md](/root/private_data/work/SDformer/CODEX_SERVER_HANDOFF_ZH.md)

### 情况 2：我想从头跑通一条最小链路

看：

- [COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md](/root/private_data/work/SDformer/COLAB_SINGLE_SEQ_SMOKE_RUNBOOK_ZH.md)
- [SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md](/root/private_data/work/SDformer/SCNET_SINGLE_SEQ_TRAIN_RUNBOOK_ZH.md)

### 情况 3：我想做正式的研究级实验

看：

- [PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md](/root/private_data/work/SDformer/PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md)
- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)

### 情况 4：我改了模型，不想直接跑 full train

看：

- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
- [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)

### 情况 5：我不知道该从哪里改注意力、加模块、做剪枝

看：

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
- [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)

### 情况 6：我想看不同 run 的训练曲线和指标对比

看：

- [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)
- [experiments/results/mlflow_compare/README.md](/root/private_data/work/SDformer/experiments/results/mlflow_compare/README.md)

### 情况 7：我想看硬件架构、接口和性能模型

看：

- [arch.md](/root/private_data/work/SDformer/hw/docs/arch.md)
- [interfaces.md](/root/private_data/work/SDformer/hw/docs/interfaces.md)
- [perf_model.md](/root/private_data/work/SDformer/hw/docs/perf_model.md)

### 情况 8：我想看 upstream 原始说明

看：

- [third_party/SDformerFlow/README.md](/root/private_data/work/SDformer/third_party/SDformerFlow/README.md)

## 5. 新增文档时的更新清单

以后每新建一个重要 `.md` 文件，至少要做这 4 件事：

1. 把它加到本导航
2. 写清楚它属于哪一类
3. 写清楚“什么情况下看它”
4. 如果它替代了旧文档，要在这里标明优先级

## 6. 推荐优先级与阅读顺序

### 6.1 新手第一次进项目

按这个顺序读：

1. [README.md](/root/private_data/work/SDformer/README.md)
2. [CODEX_SERVER_HANDOFF_ZH.md](/root/private_data/work/SDformer/CODEX_SERVER_HANDOFF_ZH.md)
3. [PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md](/root/private_data/work/SDformer/PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md)
4. [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
5. [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)

### 6.2 如果你准备改模型

按这个顺序读：

1. [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
2. [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
3. [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
4. [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)

### 6.3 如果你准备跑实验

按这个顺序读：

1. [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
2. [DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md](/root/private_data/work/SDformer/DSEC_SUBSET_EXPERIMENT_PROTOCOL_ZH.md)
3. [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)
4. [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)

### 6.4 如果你准备做硬件协同设计

按这个顺序读：

1. [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
2. [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
3. [arch.md](/root/private_data/work/SDformer/hw/docs/arch.md)
4. [interfaces.md](/root/private_data/work/SDformer/hw/docs/interfaces.md)
5. [perf_model.md](/root/private_data/work/SDformer/hw/docs/perf_model.md)

## 7. 三条独立阅读路线

### 7.1 研究路线

- [PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md](/root/private_data/work/SDformer/PAPER_GRADE_SDFORMERFLOW_RESEARCH_RUNBOOK_ZH.md)
- [RESEARCH_ITERATION_SOP_ZH.md](/root/private_data/work/SDformer/RESEARCH_ITERATION_SOP_ZH.md)
- [DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md](/root/private_data/work/SDformer/DEV_SUBSET_EXPERIMENT_TEMPLATE_ZH.md)
- [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)

### 7.2 模型改动路线

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
- [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
- [MODULAR_UPGRADE_TECHNICAL_DOC.md](/root/private_data/work/SDformer/MODULAR_UPGRADE_TECHNICAL_DOC.md)
- [MODULE_ZOO.md](/root/private_data/work/SDformer/MODULE_ZOO.md)

### 7.3 工程/硬件路线

- [FULL_STACK_TECHNICAL_GUIDE_ZH.md](/root/private_data/work/SDformer/FULL_STACK_TECHNICAL_GUIDE_ZH.md)
- [TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md](/root/private_data/work/SDformer/TRAINING_VISUALIZATION_AND_COMPARISON_ZH.md)
- [arch.md](/root/private_data/work/SDformer/hw/docs/arch.md)
- [interfaces.md](/root/private_data/work/SDformer/hw/docs/interfaces.md)
- [perf_model.md](/root/private_data/work/SDformer/hw/docs/perf_model.md)

## 8. 当前新增的关键文档

这次新增并已纳入导航的是：

- [BASELINE_MODEL_WALKTHROUGH_ZH.md](/root/private_data/work/SDformer/BASELINE_MODEL_WALKTHROUGH_ZH.md)
- [MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md](/root/private_data/work/SDformer/MODEL_MODIFICATION_AND_PRUNING_GUIDE_ZH.md)
- [MD_CLASSIFICATION_NAVIGATION_ZH.md](/root/private_data/work/SDformer/MD_CLASSIFICATION_NAVIGATION_ZH.md)
