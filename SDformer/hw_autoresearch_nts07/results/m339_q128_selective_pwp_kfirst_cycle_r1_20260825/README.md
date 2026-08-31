# M339 q128 selective-PWP K-first cycle DSE

M339 replays the nested train-only M338 catalog on all 40 exact M248
PAFT-ep4 running-BN records. The runtime exact vector-operation speedups over
bit-sparse work are 1.540642x, 1.692877x, 1.857852x and 2.043940x for q16,
q32, q64 and q128.

Runtime working sets nearly saturate each catalog: q128 has mean 106.97,
p90 125 and maximum 128 used PWPs per phase. Selective fetch therefore reduces
q128 PWP traffic by only 1.1966x and a strict equal-half double buffer needs
512 KiB to fit every q128 phase.

The pinned K-first recurrence consumes every raw row, adds common commit to
both branches, and serializes next-phase match before its data-dependent PWP
DMA. Its best wide-port q128 point is 2.003053x versus bit sparse; shared96 is
1.429893x. These recurrence figures are deliberately unadmitted pending finite
queues, bank conflicts, cache chunking, area normalization and RTL cycle match.

