# 文献合并盘点（≥300｜含开源 + arXiv 增量）

日期：2026-09-10

## 去重计数

| 量 | 数 |
|---|---:|
| 唯一条目总计 | 732 |
| paper | 653 |
| open_source | 60 |
| 其他 entity | 19 |
| already_inventoried（主审计/musha） | 445 |
| new_this_round | 287 |

### depth_bucket（诚实深度，非全文精读计数）

- `metadata_or_list`: 331
- `unclear`: 226
- `method_or_partial`: 94
- `open_source_inventory`: 60
- `deep_or_method`: 21

## 会场 / 开源覆盖（字符串命中）

| 目标 | 命中 |
|---|---:|
| neurips | 16 |
| icml | 6 |
| cvpr | 17 |
| iccv | 9 |
| eccv | 5 |
| isca | 45 |
| micro | 19 |
| hpca | 13 |
| asplos | 4 |
| dac | 9 |
| iccad | 9 |
| tcas | 22 |
| jssc | 13 |
| tvlsi | 9 |
| tpami | 1 |
| isscc | 35 |
| arxiv | 281 |
| open_source | 60 |
| other | 172 |

## 与主战场关键词桶（可重叠）

| 桶 | 条数 |
|---|---:|
| lifting_temporal | 27 |
| product_sparsity_psn | 37 |
| gustavson | 14 |
| structured_pruning | 16 |
| conditional_gate | 11 |
| optical_flow_snn | 69 |
| sparse_transformer_hw | 14 |
| shared_completion | 7 |

## 诚实声明

- 达成 ≥300 靠「已盘点合并 + 开源仓盘点 + arXiv 题名/摘要增量」。
- **不等于 300 篇全文精读**；`new_this_round` 的 arXiv 默认为 `metadata_or_list`。
- 可复用 `literature_audit_20260909` / mushaolong；本目录标注 `read_flag`。
- 不抢 Stage B；负结果只停布局不杀全家。
