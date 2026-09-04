# Local5 joint-head 完成后 formal 就绪记录

> 2026-08-10T17:43:06.942667+00:00

## 产物文件
- `checkpoint_projection_contract.json`
- `checkpoint_projection_contract.npz`
- `gpu_exclusivity_audit.json`
- `joint_head_run_identity.json`
- `joint_window_selection_plan.json`
- `local5_hardware_features.json`
- `local5_hardware_features.md`
- `ordered_cohort.json`
- `ordered_term_items.npz`
- `ordered_term_manifest.json`

## 下一步（严格）
1. 用正式 producer 写 admission/manifest
2. HxH preflight
3. 三段 adapter（软件金参考 / DUT / merge）
4. formal archive G0
5. 通过后才允许 EREP candidate RTL

formal G0 在 admission 前保持 DENY。
