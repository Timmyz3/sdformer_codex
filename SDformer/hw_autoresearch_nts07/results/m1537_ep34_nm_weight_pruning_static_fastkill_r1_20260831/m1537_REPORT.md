# M1537 ep34 N:M static weight-pruning fast-kill

Status: **PASS_STATIC_OPPORTUNITY__RETRAIN_AND_AEE_REQUIRED__NO_HARDWARE_ADMISSION**.

The released ep34 checkpoint has no lossless N:M path. Across patch embed, FC1, FC2, bottleneck Conv and decoder, an oracle 4:8 magnitude mask removes 23.00%--25.00% of L1 weight mass; 8:16 removes 21.23%--23.01%. These are static FP32 opportunity measurements, not accuracy, cycle, traffic, energy or RTL results. Only a new hardware-aware retrained checkpoint with paired AEE and same-resource replay may promote this candidate.
