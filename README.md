# SDformerFlow HW/SW Co-Design

Spiking Transformer optical-flow research stack built around the upstream `SDformerFlow` baseline.

## Layout

- `third_party/SDformerFlow/`: locked upstream baseline submodule
- `configs/`: baseline, variants, and quantization specs
- `src/`: Python adapters, datasets, trainers, modules, and profilers
- `scripts/`: setup, data, train, eval, ablation, and profiling entrypoints
- `hw/`: RTL, testbench, synthesis, and architecture docs
- `tools/`: Python golden simulator and quant export utilities

## Documentation

- `TECHNICAL_DOCUMENTATION.md`: complete technical write-up of the current implementation
- `REPORT.md`: stage progress and status tracking
- `PAPER_CO_DESIGN_PROPOSAL.md`: paper-driven upgrade path for sparse attention, HW/SW co-design, and accelerator architecture
- `MODULE_ZOO.md`: pluggable module library for small ablations and future HW-aware variants
- `MODULAR_UPGRADE_TECHNICAL_DOC.md`: detailed technical write-up of the modular sparse plug-ins and their integration points
- `RUNBOOK_AND_RESEARCH_PLAN_2026.md`: step-by-step execution guide for upstream/local runs plus a conference-oriented research plan
- `FULL_STACK_TECHNICAL_GUIDE_ZH.md`: Chinese master document covering runtime logic, plug-in schemes, experiment configs, and execution guidance
- `UPSTREAM_SDFORMERFLOW_RUNBOOK_ZH.md`: Chinese runbook focused only on bringing up the original upstream SDformerFlow baseline

## Quick Start

```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh --dataset all
bash scripts/run_train.sh configs/sdformer_baseline.yaml --output-dir experiments/logs/train
bash scripts/run_eval.sh configs/sdformer_baseline.yaml --checkpoint experiments/logs/train/sdformer_baseline_best.pth
```

## Baseline

- Upstream repo: `https://github.com/yitian97/SDformerFlow.git`
- Submodule path: `third_party/SDformerFlow`
- Locked commit: `13088516440ab3faba4142c986d162cf5dd7c299`

## Status

- Phase 0: repository rebuilt around the optical-flow baseline
- Phase A: baseline integration scaffolded through adapter scripts
- Remaining model-variant integration, quantitative validation, and RTL closure continue in this repo
