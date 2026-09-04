# H11 Ternary Event Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent H11 experiment that replaces H9a's floating Q*K Shiftmax gate with sign-only ternary event compatibility scoring.

**Architecture:** H11 copies the H9 overlay/entrypoint pattern into a new `neuron_experiments/H11_ternary_event_attention` folder. Q/K neurons remain PSN+ATLIF+ternary, baseline QKFormer carrier is preserved for stability, and the extra gate uses sign-only counts: `pp + alpha*nn - beta*mismatch`, optionally normalized by Shiftmax.

**Tech Stack:** PyTorch, SDFormerFlow overlay modules, unittest, YAML configs.

---

### Task 1: Isolate H11 Files

**Files:**
- Create: `neuron_experiments/H11_ternary_event_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`
- Create: `neuron_experiments/H11_ternary_event_attention/configs/h11a_event_score_h9a_core_guard120.yml`
- Create: `neuron_experiments/H11_ternary_event_attention/tests/test_event_attention.py`

- [ ] Copy only the H9 modules needed for independent training: overlay, entrypoints, tests, and selected configs.
- [ ] Keep `third_party/SDformerFlow` unchanged.

### Task 2: Add Sign-Only Event Score

**Files:**
- Modify: `neuron_experiments/H11_ternary_event_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`

- [ ] Add config fields `event_alpha`, `event_beta`, `event_use_threshold_scale`, and mode `ternary_event_compat`.
- [ ] Implement `ternary_event_score(q, k, alpha, beta)` using signs only:

```python
q_pos = q.gt(0).float()
q_neg = q.lt(0).float()
k_pos = k.gt(0).float()
k_neg = k.lt(0).float()
pp = (q_pos * k_pos).sum(dim=-1, keepdim=True)
nn = (q_neg * k_neg).sum(dim=-1, keepdim=True)
mismatch = (q_pos * k_neg + q_neg * k_pos).sum(dim=-1, keepdim=True)
score = pp + alpha * nn - beta * mismatch
```

### Task 3: Verify Behavior

**Files:**
- Modify: `neuron_experiments/H11_ternary_event_attention/tests/test_event_attention.py`

- [ ] Test that negative-negative agreement is weaker than positive-positive when `alpha < 1`.
- [ ] Test that opposite signs reduce score.
- [ ] Test that the patched forward preserves output shape.

### Task 4: Configs and Smoke

**Files:**
- Create: `neuron_experiments/H11_ternary_event_attention/configs/h11a_event_score_h9a_core_guard120.yml`
- Create: `neuron_experiments/H11_ternary_event_attention/configs/h11a_event_score_h9a_core_full.yml`

- [ ] Start from H9a core replacement set.
- [ ] Set `bsa_attention.mode: ternary_event_compat`, `event_alpha: 0.25`, `event_beta: 1.0`.
- [ ] Add lightweight angular loss only if the existing loss path supports it safely.
