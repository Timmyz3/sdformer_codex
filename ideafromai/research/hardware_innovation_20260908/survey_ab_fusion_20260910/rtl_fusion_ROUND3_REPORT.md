# Fusion Screen ROUND3 报告

日期：2026-09-13  
杀门：`fusion_killcards_round3.md`  
执行器中途资源耗尽，但 cand10–14 的 `make sim && make synth` 已落盘；本页按日志收口。

## 范围诚实
iverilog + yosys techmap 玩具 RTL。**不是** Codex Stage B / ASIC PPA / 标题 X。  
verdict 仅 `KEEP_PROBE` / `BASE_ONLY` / `KILL_LAYOUT`。

## 结果
1. **cand10 弹性直供**：always-ready/burst 赢，**long896 903=903** → 公平长背压不降 job → **KILL_LAYOUT**。SRAM 事务减少不单独晋级。
2. **cand11 小整数基底+例外**：8→10（例外/tag/fallback）→ **KILL_LAYOUT**。
3. **cand12 gate-summary×NRV**：并集 14=14 满读，摘要税 +2 拍 → **KILL_LAYOUT**（不杀 Gustav）。
4. **cand13 驻留 R24**：64→48，同 RF=24 → **BASE_ONLY**（普通驻留，非 lifting X）。
5. **cand14 例外打包**：三密度全更慢 → **KILL_LAYOUT**。

## 幸存
**KEEP_PROBE：无。** 仅再添一条 BASE_ONLY（cand13）。三轮筛下来没有可抬的创新探针。

下一步若继续：换**真实改变并集读集合或长背压 job** 的接口；不要复活已杀布局。
