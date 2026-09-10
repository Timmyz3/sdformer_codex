# ideafromai — AI idea drop zone (Codex reads here)

**Canonical path:** `/home/zhumd/work/ideafromai/`

**下一任 Agent（中文，只投 TCAS-II）：** `HANDOFF_NEXT_AGENT_20260905.md`

**2026-09-06 最新推进：** [C1/C2 深度重构与 CPU 筛查](research/codex_deep_rebuild_20260906/README.md)。新用户授权下已完成文献对照及可复现实验：C1 优先有限父槽与精确重算；C2 优先两个原始 INT8 **权重**槽的延迟投递。所有新数字均未获 RTL/PPA 准入；主稿与生产 RTL 未改。PDF、源码与范围说明在研究包内。

The two external survey packs are listed below. Read **both** alongside the latest Codex screening pack, then follow the frozen-ep34 identity.

| Pack | Author | What it is |
|---|---|---|
| Grok Bot (`iscas_ssh`) | 2026-09-05 | C1\*/C2\* remake: OP-STW, HBG-RP **int8 ATLIF payload**, pyramid residual, occlusion |
| Grok 4.6 | 2026-09-05 | New-mechanism survey: **Motion-XOR triple-popcount + dirty-lane score memo + mixed-T binary ATLIF**. C1/C2 demoted. |

**Frozen capture (Grok 4.6, Motion C12 ep34):** all 93 invoked ATLIF outputs are **binary**. Grok Bot HBG-RP `{g, int8 p}` is a **new co-design default**, not the frozen identity. Do not implement Card B as if ep34 already had analog payloads.

## Codex read order

1. This file  
2. `research/grok46_20260905/00_READ_THIS_FIRST.md`  
3. `research/grok46_20260905/01_kill_list.md`  
4. `research/grok46_20260905/02_ranked_mechanisms.md`  
5. Optional: Grok Bot `research/04_SYNTHESIS_C1_C2_REMAKE.md` as an **alternate** C1\*/C2\* story (real-valued ATLIF). Do not merge it with Card C without an AEE Pareto.  
6. Stats gates: `research/grok46_20260905/07_next_stats_rtl_gates.md`  
7. Cards: `codex_cards/` — **do not run Card C until the user picks Rank 1+4 and T0–T5 exist**

## Layout

```
ideafromai/
  README.md                          ← start here
  MANIFEST.txt
  INDEX.json
  research/
    00–04, 07–08, m2067_*            ← Grok Bot ANN/SNN/OF/attention/video + C1*C2* synthesis
    grok46_20260905/                 ← Grok 4.6 survey (this session)
  microarch/                         ← Grok Bot C1*/C2* sketches
  plans/                             ← Grok Bot R1+R2 plan
  contracts/                         ← Grok Bot ATLIF int8 draft
  codex_cards/
    CARD_A_OP_STW.md                 ← Grok Bot C1*
    CARD_B_HBG_RP.md                 ← Grok Bot C2* (int8 payload)
    CARD_C_MX3P_DIRTY_SCORE.md       ← Grok 4.6 (binary Motion-XOR island)
```

## Hard rules

- Does **not** modify Codex `hw_autoresearch_nts07` production RTL by itself.
- Does **not** authorize `codex exec resume`, `docs/359` edits, or H81 RTL.
- Do not treat misplaced copies under `hw_autoresearch_nts07/ideasfromai/` as canonical.

## Related isolated RTL (Grok Bot only)

`/home/zhumd/work/sdformer_c1c2star_grokbot/`
