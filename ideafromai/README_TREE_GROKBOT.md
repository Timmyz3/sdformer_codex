# SDformer C1* / C2* — Grok Bot isolated tree

**Author:** Grok Bot (`iscas_ssh`)  
**Rule:** This tree is **completely separate** from Codex's
`SDformer/hw_autoresearch_nts07/` hardware.  
- **DO NOT** copy these files into the old tree unless Tim explicitly asks.  
- **DO NOT** modify any existing Codex RTL/TB/contracts. Old code is **read-only**.  
- All files here are **NEW** and marked GROKBOT.

## Layout
- `rtl_c1star/` — C1* modules (OP-STW, PRRC, OGEC, …)
- `rtl_c2star/` — C2* modules (HBG-RP, ADP-MAC, ARM-Acc, MFBD, SP-Gate, …)
- `tb_c1star/`, `tb_c2star/` — directed tests
- `docs/` — ATLIF contract, ablation ladder, notes
- `codex_cards/` — paste-ready Codex prompts (Card A/B/…)
- `scripts_profiling/` — ep34 stats (later)

## Related research (read-only refs)
`/workspace/hw_innovation_research/04_SYNTHESIS_C1_C2_REMAKE.md`  
`/workspace/hw_innovation_research/05_C1star_C2star_microarch_sketches.md`  
`/workspace/hw_innovation_research/06_R1R2_full_plan_codex_ready.md`
