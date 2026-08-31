# M300 shared FC1 + patch-Conv bounded-group sensitivity

This milestone combines the independently reviewed ten-module binary FC1 and
six-module binary patch-Conv opportunities under one four-output destination
group mechanism.  It maps each module's detailed M51 task ratio to that same
module's frozen M221 cycle term; it does not use an aggregate ratio shortcut.

The selected `group=4, beta=48` point projects 100,895,624 eligible FC1 cycles
to 68,211,470.113 and 172,321,077 eligible patch-Conv cycles to
108,532,160.011.  On the frozen 620,302,905-cycle compute envelope this is an
ideal-compaction sensitivity of 1.184168722x.  FC1 and patch-Conv weighted task
removal are 32.3712% and 37.0201%, respectively.

This point is admitted only for a paired S10 accuracy screen.  M51 and M221 are
not the same population (maximum module population differences are 3.266% for
FC1 and 4.613% for patch Conv), and the model omits router, mask fetch,
bank-conflict, scan/commit, SRAM, and schedule overhead.  Therefore it is not
an executable cycle result, system speedup, RTL, PPA, power, energy, or
headline result.

`beta=0` preserves every source/destination-group task and remains the exact
hardware subset.  The frozen next action is a `no_running` paired S10:
`group4/beta0` first, followed by `group4/beta48`; any AEE increase above 0.02
stops this lossy axis before RTL.
