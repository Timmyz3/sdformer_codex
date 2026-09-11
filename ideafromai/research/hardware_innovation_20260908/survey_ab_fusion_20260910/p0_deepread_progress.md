# P0 精读诚实进度表（过夜收口）

更新：2026-09-10 夜末

## 目标
P0 = **246** 全文/方法级精读 + idea 卡

## 交付计数
| 项 | 数 |
|---|---:|
| idea 卡 md | 155 |
| CSV 行 | 161 |
| arXiv 方法摘录卡 | 55 |
| 开源诚实盘点卡 | 28 |
| unresolved（无全文，不虚报） | 56 |
| 其他（audit/本地全文等） | 16 |
| P0 有非 unresolved 覆盖（按 uid/名） | 100 |
| P0 标 unresolved | 62 |
| P0 仍缺口（应继续补） | 91 |

## 来源分布（CSV source）
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
- `arxiv_2306.02960+本地excerpt`: 1
- `arxiv_2310.02065+本地excerpt`: 1
- `arxiv_2310.07707+本地excerpt`: 1
- `arxiv_2402.11662+本地excerpt`: 1
- `arxiv_2403.00849+本地excerpt`: 1
- `arxiv_2404.10597+本地excerpt`: 1
- `arxiv_2407.10416+本地excerpt`: 1
- `arxiv_2408.08794+本地excerpt`: 1
- `arxiv_2408.15578+本地excerpt`: 1
- `arxiv_2409.04082+本地excerpt`: 1
- `arxiv_2409.05227+本地excerpt`: 1
- `arxiv_2411.14733+本地excerpt`: 1
- `arxiv_2412.16757+本地excerpt`: 1
- `arxiv_2501.07825+本地excerpt`: 1
- `arxiv_2501.11554+本地excerpt`: 1
- `arxiv_2501.13610+本地excerpt`: 1
- `arxiv_2501.14490+本地excerpt`: 1
- `arxiv_2503.03379+本地excerpt`: 1
- `arxiv_2503.15986+本地excerpt`: 1
- `arxiv_2503.19643+本地excerpt`: 1
- `arxiv_2505.12281+本地excerpt`: 1
- `arxiv_2505.12292+本地excerpt`: 1
- `arxiv_2505.12771+本地excerpt`: 1
- `arxiv_2507.09780+本地excerpt`: 1
- `arxiv_2510.12102+本地excerpt`: 1
- `arxiv_2510.26614+本地excerpt`: 1
- `arxiv_2511.06770+本地excerpt`: 1
- `arxiv_2603.15184+本地excerpt`: 1
- `arxiv_2604.03626+本地excerpt`: 1
- `arxiv_2605.20802+本地excerpt`: 1
- `arxiv_2605.28312+本地excerpt`: 1
- `arxiv_2607.05445+本地excerpt`: 1
- `arxiv_2608.19238+本地excerpt`: 1
- `arxiv_2505.07556+本地excerpt`: 1
- `arxiv_2506.03512+本地excerpt`: 1
- `arxiv_2508.12637+本地excerpt`: 1
- `arxiv_2512.17555+本地excerpt`: 1
- `arxiv_2512.20073+本地excerpt`: 1
- `arxiv_2602.23204+本地excerpt`: 1
- `arxiv_2407.20421+本地excerpt`: 1
- `arxiv_2410.23082+本地excerpt`: 1
- `arxiv_2412.09105+本地excerpt`: 1
- `arxiv_2412.11284+本地excerpt`: 1
- `arxiv_2503.03256+本地excerpt`: 1

## 管道状态
- arXiv excerpt batch_01–06 + residual：**完成**
- 本地作者全文 6 + audit_deep 10：**完成**
- 开源 28：**完成（仓级，非论文全文）**
- need_locate：已标 unresolved，不虚报精读
- MP1/MP2：管理已 PASS，并进 fusion_status

## 诚实声明
1. **未达到**「246 篇全部全文精读」；达到的是：可获得全文/摘录的批次已方法级提卡 + 无法定位者诚实 unresolved。
2. arXiv 卡为 **方法摘录窗口**，不是逐章精读全书。
3. 开源卡不是论文全文。
4. 不抢 Stage B；RTL 粗验≠净服务%/TCAS PPA。

## 文件
- `idea_cards/`
- `idea_extract_per_paper.csv`
- `idea_synthesis.md`
- `fusion_status_overnight.md`
- `rtl_microprobes_for_box.md`
- `ab_fusion_candidates.md` / `ab_fusion_priority.md`
