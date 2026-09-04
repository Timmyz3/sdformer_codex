# M837 M836 source-author handoff

M836 is an additive source-only repair for the M835 publication-boundary P1. It leaves the frozen M832/M828/M819/M809 160-row decoder schedule and resource tuple unchanged.

The private stage FD remains open through publication. Exact member type/dev/inode/bytes are revalidated before and after `renameat2(RENAME_NOREPLACE)`, the current results pathname is rebound after publication, and a failed boundary rolls back only the exact directory/member inodes recorded by this process through the pinned results FD.

This handoff authorizes only a receipt-blind independent source hammer. It does not authorize a true release, formal runner invocation, production replay, or any cycle/speedup claim.
