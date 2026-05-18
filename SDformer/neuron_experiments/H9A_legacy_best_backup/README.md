# H9A Legacy Best Backup

This folder freezes the reproducible H9a legacy setup that produced the best H-series trade-off so far.

## Reference Result

- Source run: `neuron_experiments/H9_bipolar_self_attention/results/h9a_shiftmax_compat_h8m_full_bs8_20260512_200523_setsid`
- Checkpoint: `checkpoint_epoch29.pth`
- Profile copy: `results_reference/sops_summary_valid40.json`
- valid40: AEE `1.5043755`, AAE `7.6364652`, SOPs `3.0847G`, firing `0.0723596`

## Code Path

- Config: `configs/h9a_shiftmax_compat_h8m_full.yml`
- Entrypoint snapshots: `entrypoints/train.py`, `entrypoints/profile_sops.py`
- Attention snapshot: `overlay/models/STSwinNet_SNN/bsa_attention.py`
- Neuron snapshot: `overlay/models/STSwinNet_SNN/atlif_ternary_psn/`

The H9a behavior is the `compat_qk_product` / default mode in `bsa_attention.py`.
It keeps baseline `sn2_q(sum(q))` token spike gating and applies an additional Shiftmax compatibility gate.

This backup intentionally does not copy the 13G checkpoint directory. The checkpoint remains in the original H9 result folder, and the exact command is copied to `results_reference/run_command.txt`.
