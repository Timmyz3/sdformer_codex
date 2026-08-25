# H67/H81 matched score-pair profile

This profile uses the first captured window of every attention block. It is real-Q/K trace evidence, not a full-valid825 population estimate.

| route | scope | pairs | equal | equality | ideal dual-slot reduction |
|---|---|---:|---:|---:|---:|
| H67 | all | 621000 | 559189 | 90.0465% | 45.0233% |
| H67 | non_empty | 372914 | 311103 | 83.4249% | 41.7124% |
| H67 | both_k_active | 114216 | 84139 | 73.6666% | 36.8333% |
| H81 | all | 621000 | 563968 | 90.8161% | 45.4081% |
| H81 | non_empty | 373653 | 316621 | 84.7366% | 42.3683% |
| H81 | both_k_active | 117453 | 88972 | 75.7512% | 37.8756% |

## Stage breakdown

| route | stage | all equality | non-empty equality | both-K-active equality |
|---|---|---:|---:|---:|
| H67 | S0 | 99.9111% | 99.6964% | 98.5521% |
| H67 | S1 | 99.8389% | 97.3057% | 98.2639% |
| H67 | S2 | 93.0682% | 87.7954% | 80.0499% |
| H67 | S3 | 81.8329% | 77.9246% | 68.3562% |
| H81 | S0 | 99.8630% | 99.4976% | 99.1228% |
| H81 | S1 | 99.9389% | 99.2449% | 98.8930% |
| H81 | S2 | 94.3978% | 90.0791% | 86.9706% |
| H81 | S3 | 82.0319% | 78.3133% | 68.1361% |

`both_k_active` means both temporal K slices contain at least one active lane. H81 uses the TTX score; H67 uses the Motion-TTX score. Cross-route differences are descriptive because the checkpoints are recipe-level matched, not step-paired.
