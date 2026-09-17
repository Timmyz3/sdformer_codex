# P0 精读诚实进度表

更新：2026-09-11（与交接稿对齐）

## 目标
P0 = **246**（`literature_precision_732.json` 中 `precision_tier=P0_high_rel`）全文/方法级 idea 提取 + 卡

## 交付计数（当前盘点）
| 项 | 数 |
|---|---:|
| idea 卡 md | 245 |
| CSV 数据行（unique uid） | 252 |
| P0 uid 已覆盖 | 246 / 246 |
| P0 仍缺口 | 0 |
| gap01–08 excerpt 包已提卡 | 90（12×7+6） |

## 诚实边界（必读）
- **覆盖 ≠ 每篇 PDF 全文精读**。含：方法摘录窗、开源诚实盘点、unresolved（无全文不虚报）、以及 uid 别名行。
- 别名（不重复深读）：`MAIN-R276`→`ARX-012`/`arxiv_2407.08356.md`；`MUSHA-SP001`→`MAIN-R002`/`GustavSNN.md`。
- 不抢 Stage B；不碰精度恢复训练；融合第二队列。
- 落盘：`/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/`（`/home/zhumd/work/ideafromai` 软链同路径）。

## 粗分类（CSV source 关键字，可重叠口径以外的粗数）
| 类 | 约数 |
|---|---:|
| unresolved* | 62 |
| opensource* | 28 |
| alias_* | 2 |
| 含 arxiv_ | 145 |
| 含 audit | 10 |

## 来源分布 Top
- `unresolved_no_fulltext`: 62
- `opensource_inventory_honest`: 28
- `audit_deep_verified`: 10
- `local_author_fulltext`: 6
- `arxiv_1708.04485+本地excerpt`: 1
- `arxiv_1802.03806+本地excerpt`: 1
- `arxiv_1908.08976+本地excerpt`: 1
- `arxiv_2005.03842+本地excerpt`: 1
- `arxiv_2012.09852+本地excerpt`: 1
- `arxiv_2203.07516+本地excerpt`: 1
- `arxiv_2209.09570+本地excerpt`: 1
- `arxiv_2209.11741+本地excerpt`: 1
- `arxiv_2211.08110+本地excerpt`: 1
- `arxiv_2304.07493+本地excerpt`: 1
- `arxiv_2304.12760+本地excerpt`: 1

## 关联交付（同目录，供交接引用）
- `idea_cards/` · `idea_extract_per_paper.csv` · `idea_synthesis.md`
- `ab_fusion_candidates.md` · `ab_fusion_priority.md` · `fusion_status_overnight.md`
- `rtl_microprobes_for_box.md`（探针规格；实测在管理侧 box）
- `p0_excerpt_batches_gap/gap_01.json` … `gap_08.json`
