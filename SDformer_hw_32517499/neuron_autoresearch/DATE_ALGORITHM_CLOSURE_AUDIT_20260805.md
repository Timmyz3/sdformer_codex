# DATE Algorithm Closure Audit

Status: **PASS**

- Local-5 rank-1: epoch 29
- Local-5 convergence: `not_plateaued`
- H67 rank-1: epoch 35
- H67 training lineage: own Motion-XOR crop ep19 -> fullres ep30; no NB0/Local-5 initialization.
- H67 convergence: `operationally_plateaued_or_overfit`
- NB0 convergence: `operationally_plateaued_or_overfit`
- Selected mainline under AEE<=NB0+5% and spikes<=NB0-20%: `H67`
- RTL claim: checkpoint-bound component exact only; not full-network RTL-exact.

| model | AE-3D frame-equal | AE-3D pixel-global | AE-3D sequence-balanced |
|---|---:|---:|---:|
| Local5 | 5.6594 | 5.4728 | 5.5956 |
| H67 | 5.6509 | 5.4851 | 5.6015 |
| NB0 | 6.1803 | 5.9892 | 6.0925 |
