# Fusion Screen ROUND2 报告

日期：2026-09-13（Asia/Shanghai）  
执行：box 上 RTL→`make sim`→`make synth`→杀门卡  
杀门源：`fusion_killcards_round2.md`（接 ROUND1）

## 范围诚实
- iverilog + yosys techmap @ `/workspace/rtl_fusion_screen/`
- 周期 = 玩具/同资源自检；cells = techmap 代理
- **NOT** Codex Stage B service%、**NOT** ASIC PPA、**NOT** 生产 nts07/main.tex
- 不宣称标题级 X；已杀 cand2–5 不复活

## 筛序与结果

### R2-A 深化 cand1
1. **cand1b bp_contention**  
   - 测：公平 out_ready 波形（早 stall + 50% duty）；保序旁路 vs store-drain  
   - 结果：23 vs 23 → **KILL_LAYOUT**  
   - 停：背压下旁路优势；ROUND1 KEEP_PROBE 不得再当真依赖证据

2. **cand1c multibeat_raw**  
   - 测：2-beat×4 词 RAW forward vs 存后读  
   - 结果：16→8 → **BASE_ONLY**（always-ready 同构，不救 cand1）

**cand1 总判**：公平 BP 下优势消失 → **整体不保留 KEEP_PROBE**；仅普通 bypass 教学底座。

### R2-C F7
3. **cand7 gustav_pack_align**  
   - 测：因子对齐序 vs ordinary 交错；同 pack 宽  
   - 结果：物理 cycles 9=9、packs 4=4；nav 7→3 仅代理 → **KILL_LAYOUT**

### R2-B F3
4. **cand6 shared_parent_dag**  
   - 测：有限共享父 lane + 多父等待 vs CSE 一次算  
   - 结果：5 ≥ 4 → **KILL_LAYOUT**

### R2-D BN
5. **cand8 bn_deferred_v**  
   - 测：单遍 deferred-V vs 两遍 BN+V  
   - 结果：8 < 16 但门语义非同构 → **BASE_ONLY**（普通融合，不是 X）

### 附加
6. **cand9 encode_u_q8**  
   - 测：Q8+decode tax vs expanded 2cyc  
   - 结果：16 ≥ 16 → **KILL_LAYOUT**

## 幸存
- **KEEP_PROBE**：无  
- **BASE_ONLY**：cand1c、cand8（均禁止抬标题 X）

## 路径速查
```
/workspace/rtl_fusion_screen/SCOREBOARD.md
/workspace/rtl_fusion_screen/ROUND2_REPORT.md
/workspace/rtl_fusion_screen/fusion_killcards_round2.md
/workspace/rtl_fusion_screen/cand1b_bp_contention/
/workspace/rtl_fusion_screen/cand1c_multibeat_raw/
/workspace/rtl_fusion_screen/cand6_shared_parent_dag/
/workspace/rtl_fusion_screen/cand7_gustav_pack_align/
/workspace/rtl_fusion_screen/cand8_bn_deferred_v/
/workspace/rtl_fusion_screen/cand9_encode_u_q8/
```

## 同步副本
已复制 SCOREBOARD + ROUND2_REPORT + killcards_round2 → `/workspace/overnight_20260911/`
