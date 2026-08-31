# M243 ATLIF decoupled CSD module performance

M243 reconciles the already admitted M31, M37, M38 and M39 evidence without
creating a new system-speedup claim.  On the frozen 7,318,350-tile T10
population, the finite recurrence changes from `10*N` to `5+5*N`: 73,183,500
cycles become 36,591,755 cycles, or 1.999999727x.  The five-cycle startup is
charged.

The architectural contribution candidate is a zero-multiplier CSD4
reconstruction sidecar overlapped with the sole 96-slot signed-INT8 multiplier
phase.  Its sealed standalone TSMC28 logic-only result is 63,114.407654 um2 at
3 ns with setup/hold met and successful Formality.  That area is the sidecar,
not the whole matched candidate.

The next gate is a same-boundary serial T10 Synopsys baseline.  Until that run
exists, throughput per area, trained accuracy, integrated RTL, energy, system
speedup, paper PPA and headline claims remain false.  The conditional 1.062627x
fixed-compute context is an Amdahl diagnostic only.
