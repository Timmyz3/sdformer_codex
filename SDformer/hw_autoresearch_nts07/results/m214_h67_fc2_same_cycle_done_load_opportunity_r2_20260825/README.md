# M214 frozen-H67 opportunity replay (r2)

This directory supersedes r1 for the corrected causal same-cycle-done
diagnostic.  It replays all 5,580,000 frozen H67 FC2 tokens and changes only
the M212 rule that a previously terminal-hint-closed lone paired window may be
loaded on the authoritative `upstream_done_accept` edge.

The result is 90,196,785 cycles, saving 191,982 cycles over M212
(1.002128x).  Stage 0 is unchanged.  This artifact is opportunity evidence;
the separate exact VCS receipt binds the implemented M214 RTL.  It is not a
complete-FC2, FFN, physical, or system speedup result.
