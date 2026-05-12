# Candidate Spiking Neurons

This folder keeps experimental neuron operators for SDformerFlow ablations.
They are not wired into the upstream `Spiking_neuron` wrapper yet.

- `LMHNode`: compact LM-H/LM-HT style learnable temporal mixing, adapted from `hzc1208/LMHT_SNN`.
- `TSLIFNode`: two-compartment temporal-segment LIF, adapted from `kkking-kk/TS-LIF`.
- `ATLIFNode`: adaptive-threshold LIF, adapted from `putshua/Activity-Pruning-SNN`.
- `SNNode`: simple binary spiking neuron baseline.
- `TSNNode`: ternary spiking neuron, adapted from `yfguo91/Ternary-Spike`.

All candidates accept time-first tensors shaped `[T, B, ...]` and preserve the
input shape.
