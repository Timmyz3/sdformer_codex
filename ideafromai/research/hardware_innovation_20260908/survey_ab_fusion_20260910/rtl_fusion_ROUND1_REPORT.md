# Fusion Screen ROUND1 报告

日期：2026-09-13  
执行：box 上 RTL→`make sim`→`make synth`→杀门卡→下一个  
杀门源：`/workspace/rtl_fusion_screen/fusion_killcards_round1.md`

## 范围诚实
- iverilog + yosys techmap @ `/workspace/rtl_fusion_screen/`
- 周期 = 玩具/同资源自检；cells = techmap 代理
- **NOT** Codex Stage B service%、**NOT** ASIC PPA、**NOT** 生产 nts07/main.tex
- 不宣称标题级 X；优先杀布局

## 筛序与结果

1. **cand1 source_commit_preview_bypass**（NEXT_INTERFACE）  
   - 测：1-port 下 commit 旁路 vs store-then-reload  
   - 结果：16→8 周期胜 → **KEEP_PROBE**  
   - 停哪层：不停；**勿**抬为 CSE/Gustav/lifting 标题 X  

2. **cand2 dual_consumer_early_release**（F5 / Grok H1）  
   - 测：gate-first early release vs last-read  
   - 结果：job 周期同为 16 → **KILL_LAYOUT**  
   - 停：该 early-release 布局  

3. **cand3 group_accept_predict**（F2）  
   - 测：组预测接受+误预测税 vs 始终重算；TB 接受率 0/8、3 miss  
   - 结果：51≥48 → **KILL_LAYOUT**  
   - 停：该组接受/检查点布局（本场景）  

4. **cand4 partial_rne_merge_cert**（F4）  
   - 测：末5+证书（2 accept/3 rollback）vs 全合并  
   - 结果：49≥40 → **KILL_LAYOUT**  
   - 停：该部分合并+证书布局  

5. **cand5 sparse_source_joint_delete**（F1）  
   - 测：共同删字 vs 双消费者并集义务  
   - 结果：union=W 满读，scan 仍 8 → **KILL_LAYOUT**  
   - 停：该并集满读下的共同删字布局  

## 幸存
仅 **cand1 KEEP_PROBE**（同端口旁路相对存后读）。其余四刀均 **KILL_LAYOUT**。  
无候选晋级为标题 X；Codex Stage B 主线未被 divert。

## 路径速查
```
/workspace/rtl_fusion_screen/SCOREBOARD.md
/workspace/rtl_fusion_screen/ROUND1_REPORT.md
/workspace/rtl_fusion_screen/fusion_killcards_round1.md
/workspace/rtl_fusion_screen/cand1_source_commit_preview_bypass/
/workspace/rtl_fusion_screen/cand2_dual_consumer_early_release/
/workspace/rtl_fusion_screen/cand3_group_accept_predict/
/workspace/rtl_fusion_screen/cand4_partial_rne_merge_cert/
/workspace/rtl_fusion_screen/cand5_sparse_source_joint_delete/
```

## 同步副本
已复制 SCOREBOARD + ROUND1_REPORT → `/workspace/overnight_20260911/`
