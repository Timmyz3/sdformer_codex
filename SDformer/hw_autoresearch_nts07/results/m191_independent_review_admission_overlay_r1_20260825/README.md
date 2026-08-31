# M191 admission correction

M191's 120-payload arithmetic is retained exactly, but its `C` label is not.
The measured batch depth counts adjacent FC2 windows; it does not guarantee
that each window owns a distinct token accumulator.  The only admitted name is
**arrival-order W-window bank-disjoint replay opportunity**.

The exact W2 and W4 opportunity points are 1.114620273x and 1.181195453x over
W1 replay.  They are not RTL, physical, FC2, FFN or system speedups.  In
particular, 43.457207% of W2 batches contain only one token, while only
8.812051% of W4 batches contain four tokens.  A successor must preserve token
identity, merge same-token windows into one Acc24, charge simultaneous context
state/ports, and include finite fill plus tagged SRAM responses.

This overlay does not modify the sealed M191 result or `docs/359`.
