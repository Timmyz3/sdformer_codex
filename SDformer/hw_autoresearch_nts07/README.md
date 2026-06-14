# NTS-07b 硬件加速器 Autoresearch

面向 **DATE 2027** 的软硬件协同加速器研究包，锚定已验证软件主线：

- **模型**：`MS_SpikingformerFlowNet_en4` + **H60 无 carrier** 注意力（NTS-07b）
- **软件主线**：**11bc/11bd 统一 H60 全线 12 block**（短测中）；废弃 11aa Legacy+H60 混用
- **硬件目标**：480×640 @ 30 FPS，片上 SRAM < 2MB，28nm 能效优于通用 Dense MAC

**Autoresearch 状态：目标已达成** → 见 [`GOAL_REACHED.md`](GOAL_REACHED.md)

---

## 文档（推荐阅读顺序）

| 顺序 | 文档 | 适合谁 |
|------|------|--------|
| 0 | `docs/16_统一H60注意力硬件方案.md` | **Step1 完整教程 + 硬件主线（统一 H60）** |
| — | `docs/15_*` | ~~废弃（混用方案）~~ |
| 1 | `docs/14_硬件小白入门路线图.md` | **硬件新手总路线** |
| 2 | `docs/02_end_to_end_dataflow.md` | 端到端数据流 |
| 3 | `docs/13_扩展文献库与可借鉴清单.md` | 扩展文献与 idea 池 |
| 4 | `docs/05_module_interface_spec.md` | 数据流清楚后再读接口 |

## 文档索引

| 文件 | 内容 |
|------|------|
| [docs/00_executive_summary.md](docs/00_executive_summary.md) | 执行摘要、创新点、交付清单 |
| [docs/01_nts07b_architecture_profile.md](docs/01_nts07b_architecture_profile.md) | NTS-07b 网络架构与张量剖面 |
| [docs/02_end_to_end_dataflow.md](docs/02_end_to_end_dataflow.md) | 事件→光流全链路数据流 |
| [docs/03_operator_hardware_mapping.md](docs/03_operator_hardware_mapping.md) | 算子清单与硬件引擎映射 |
| [docs/04_optimization_strategies.md](docs/04_optimization_strategies.md) | 可优化硬件策略（搜索空间） |
| [docs/05_module_interface_spec.md](docs/05_module_interface_spec.md) | 模块划分、总线、寄存器协议 |
| [docs/06_literature_review.md](docs/06_literature_review.md) | 文献对标与借鉴 |
| [docs/07_date2027_innovations.md](docs/07_date2027_innovations.md) | DATE 论文硬件创新点 |
| [docs/08_perf_area_energy_model.md](docs/08_perf_area_energy_model.md) | 周期/面积/能耗模型 |
| [docs/09_implementation_roadmap.md](docs/09_implementation_roadmap.md) | RTL→综合→论文 6 周路线 |
| [docs/10_autoresearch实验结果.md](docs/10_autoresearch实验结果.md) | Segment-0 实验全表 |
| [docs/11_统一ATLIF算子与数据流.md](docs/11_统一ATLIF算子与数据流.md) | 双模 ATLIF 统一算子 |
| [docs/12_文献启发_autoresearch.md](docs/12_文献启发_autoresearch.md) | Segment-1 文献实验 |
| [docs/13_扩展文献库与可借鉴清单.md](docs/13_扩展文献库与可借鉴清单.md) | **20+ 篇扩展文献** |
| [docs/14_硬件小白入门路线图.md](docs/14_硬件小白入门路线图.md) | **入门：先数据流后接口** |
| [docs/16_统一H60注意力硬件方案.md](docs/16_统一H60注意力硬件方案.md) | **统一 H60 硬件主线** |
| [docs/15_11aa_step1_层映射与硬件迁移.md](docs/15_11aa_step1_层映射与硬件迁移.md) | ~~废弃（混用注意力）~~ |

---

## RTL 索引

| 模块 | 文件 |
|------|------|
| 常量包 | `rtl/nts07_pkg.vh` |
| 统一 ATLIF 编码 | `rtl/atlif_unified_encode_unit.v` |
| 三值封装（兼容） | `rtl/ternary_encode_unit.v` |
| TX/SC 打分 | `rtl/tx_sc_score_unit.v` |
| Shiftmax | `rtl/shiftmax_unit.v` |
| H60 注意力引擎 | `rtl/h60_attention_engine.v` |
| 稀疏 MAC | `rtl/sparse_mac_pe.v` |
| 阶段控制器 | `rtl/nts07_controller.v` |
| 顶层 | `rtl/nts07_top.v` |

---

## 快速命令

```bash
cd hw_autoresearch_nts07

# 跑完全部 11 轮实验并更新仪表盘
python3 scripts/run_all_experiments.py

# 单次 perf model
./autoresearch.sh

# RTL 语法检查（需 iverilog）
bash scripts/run_nts07_sim.sh
```

**推荐硬件配置（11bc  provisional）：** `scripts/configs/nts11bc_anchor.json`（~10.4 mJ，~91 FPS，388KB SRAM）