# M199 decoupled scanner/compactor DSE

M199 independently varies raw bitmap scan width S and stable descriptor emit
width F.  Every fill duration is generated from causal per-cycle arrivals into
a finite reservoir; it is not `max(total scan,total emit)`.  The exact
S1/F1, S2/F2, S4/F4 and S8/F8 points reproduce M198.

The selected S4/F2/B2 point takes 92,464,838 cycles, or 1.239882x versus the
raw-scanned S1/F1/W1 baseline.  Pipeline widening alone contributes 1.196273x
and pair fusion adds 1.036454x against the iso-pipeline W1 path.  The observed
maximum post-emit backlog is four descriptors.

S4/F2 is only 2.4854% slower than S4/F4 while halving descriptor output and
window-write width from 384 to 192 bits.  Bypassing pair fusion in stage 0,
where it is slower, yields an exact derived 92,355,284-cycle stage-aware point
(1.241353x).  RTL, DC, SRAM timing, complete FC2/FFN and physical/system claims
remain open.
