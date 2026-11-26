# ROUND10 block2 allocation preparation

Read `PROTOCOL.md` and `freeze_v1.json` for the fixed scope. The collector privately imports frozen ROUND6 machinery and changes only the target to stage2/block2; old source files are unchanged. Root owns the separate analysis adapter, launch and independent review.

CPU contract command (no full model/GPU/data run):

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B research/flow_story_20261015/block2_allocation_round10/test_block2_cpu.py
```

`launch.json.remote_tokens` contains the complete preflight CLI and all required input SHA arguments. The default writes a fresh preflight receipt without inference. The execution command is the identical token list with `--execute` appended by root. Budget: 28×13 actual + 2 first-member dual = 366, maximum 30 minutes. Native/full-fine anchors and joint k2 masks are outside this run.

`membership_v1.json` is the authoritative stronger exclusion/selection manifest. `selection_manifest_v1.json` is the frozen ROUND6 API compatibility view and must not be used to describe the full ROUND10 exclusion rule on its own.

Preparation receipts record only tiny CPU execution and synthetic analysis. No experimental result or hardware speedup is implied by a receipt PASS. See the protocol for fixed primary/controls and the limits of the train-side, previously exposed sequences.
