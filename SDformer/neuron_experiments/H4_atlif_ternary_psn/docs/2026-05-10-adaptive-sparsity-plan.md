# H4 Adaptive Sparsity Follow-Up Plan

Goal: test weaker, stage-aware, and target-rate adaptive sparsity for ATLIF ternary PSN without touching the third-party baseline.

Files:
- overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py: add target-rate and per-module control fields.
- overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py: parse per-stage settings and install module-specific settings.
- tests/test_atlif_ternary_psn.py: verify backward compatibility, per-stage overrides, and target-rate update direction.
- configs/h4j/h4k/h4l yml files: short-run sweeps.

Tasks:
1. Extend config with optional stage_activity_eta, stage_max_threshold, stage_negative_threshold_scale, target_rate, target_rate_eta.
2. Preserve old scalar config behavior for H4h.
3. Add target-rate threshold update: threshold can increase or decrease based on module firing minus target_rate.
4. Run unit tests.
5. Run short 80-step training + valid10 profile for H4j/H4k/H4l.
