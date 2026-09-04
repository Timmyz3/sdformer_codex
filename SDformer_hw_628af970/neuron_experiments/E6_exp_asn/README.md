# E6_exp_asn

Neuron type: `exp_nasn`.

This experiment implements E6a, a normalized Adaptive Spiking Neuron inspired by
`Adaptive Spiking Neurons for Vision and Language Modeling` (`arXiv:2604.12365`).

Run smoke:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/E6_exp_asn/entrypoints/train.py \
  --config neuron_experiments/E6_exp_asn/configs/smoke.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E6_exp_asn/results/e6a_nasn_smoke_20260506_checkpoint_epoch{}.pth
```
