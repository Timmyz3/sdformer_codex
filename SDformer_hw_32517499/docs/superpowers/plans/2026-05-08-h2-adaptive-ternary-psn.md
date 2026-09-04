# H2 Adaptive Ternary PSN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experiment-local attention-only fusion of PSN, AT-LIF-style adaptive thresholding, and ternary spikes.

**Architecture:** Keep SDFormerFlow baseline files untouched. Run baseline entrypoints through H2 wrappers that install `AdaptiveTernaryPSN` on attention `sn_q/sn_k` after loading the PSN checkpoint, preserving PSN temporal weights. Configs control target scope, threshold learning, output scaling, and optional activity regularization.

**Tech Stack:** PyTorch, SDFormerFlow, SpikingJelly, experiment-local Python overlays.

---

### Task 1: Adaptive Ternary PSN

**Files:**
- Create: `neuron_experiments/H2_adaptive_ternary_psn/overlay/models/STSwinNet_SNN/adaptive_ternary/adaptive_ternary_psn.py`
- Test: `neuron_experiments/H2_adaptive_ternary_psn/tests/test_adaptive_ternary_psn.py`

- [x] **Step 1: Implement ternary activation**

Use `sign(x)`, map `abs(x) < 0.5` to zero, and use clamp STE in backward.

- [x] **Step 2: Implement PSN temporal mixer reuse**

Clone `weight` and `bias` from a loaded baseline PSN into `AdaptiveTernaryPSN`.

- [x] **Step 3: Implement adaptive threshold output**

Use positive `theta = softplus(raw_theta) + min_threshold`; emit `ternary * theta`.

- [x] **Step 4: Test activation and parameter migration**

Run:

```bash
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H2_adaptive_ternary_psn/tests/test_adaptive_ternary_psn.py
```

Expected: all tests pass.

### Task 2: Attention Q/K Installer

**Files:**
- Create: `neuron_experiments/H2_adaptive_ternary_psn/overlay/models/STSwinNet_SNN/adaptive_ternary/installer.py`
- Test: `neuron_experiments/H2_adaptive_ternary_psn/tests/test_adaptive_ternary_psn.py`

- [x] **Step 1: Enumerate Swin attention blocks**

Support `stage_selection: all`, `layer0_only`, and `stage{index}`.

- [x] **Step 2: Replace only selected Q/K spiking neurons**

Replace `attn.sn_q.spiking_neuron` and `attn.sn_k.spiking_neuron`; do not touch `proj_sn`, MLP, downsample, or decoder neurons.

- [x] **Step 3: Add summary and optional regularization**

Expose module count, threshold statistics, activity statistics, and optional activity-rate penalty.

### Task 3: Entrypoints And Configs

**Files:**
- Create: `neuron_experiments/H2_adaptive_ternary_psn/entrypoints/train.py`
- Create: `neuron_experiments/H2_adaptive_ternary_psn/entrypoints/eval.py`
- Create: `neuron_experiments/H2_adaptive_ternary_psn/configs/smoke.yml`
- Create: `neuron_experiments/H2_adaptive_ternary_psn/configs/short.yml`
- Create: `neuron_experiments/H2_adaptive_ternary_psn/configs/full.yml`

- [x] **Step 1: Patch training after baseline checkpoint load**

Install H2 after `load_model` so Q/K starts from baseline PSN weights.

- [x] **Step 2: Patch eval before checkpoint load**

Install H2 before `load_model` so H2 checkpoints can be loaded.

- [x] **Step 3: Add smoke-only early stop**

Use `runtime.max_train_steps` in H2 entrypoint so smoke does not run the full training epoch.

### Task 4: Verification

**Files:**
- Test: `neuron_experiments/H2_adaptive_ternary_psn/tests/test_entrypoint_patch.py`
- Log: `neuron_experiments/H2_adaptive_ternary_psn/results/h2_smoke_20260508.log`

- [x] **Step 1: Run unit tests**

Expected: `3 tests OK` and `1 test OK`.

- [x] **Step 2: Run H2 smoke**

Expected: installs 4 Q/K modules for `layer0_only`, completes 2 train steps, validates 1 sample, and saves epoch 0 checkpoint.

