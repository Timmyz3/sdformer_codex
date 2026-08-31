# M1334/R14 real-M935 runtime-witness source — author receipt

Status: **PASS source-only; fresh different-author hammer required**.

This is the single successor permitted by the M1333 C1 readiness audit.  It
does not alter M528, M935, M1162, R3 SVA, the R13 real-M935 TB, or the 214,912 B
ledger.  It adds one verification-only bound witness and an exact seven-file
UNIT_DELAY filelist.

The witness uses a resettable monotonic eight-stage FSM and independent exact
counters.  It observes real service and child pins from first weight+psum
request through first core accept, second weight-only request and core accept,
then psum commit, row completion, and task completion.  It rejects missing,
reordered, duplicate, wrong-tuple, wrong-address/row/epoch, active attack-mask,
and design/service-fault traces.  A final operand dump precedes the only fatal;
only a complete exact trace can print the R14 PASS token.

Static checker and 13 directed Python tests pass.  No VCS, simv, DC, PT, PTPX,
remote, or GPU work ran, and no release was authored.  Therefore wrapper
functional VCS, timing, cycles, speedup, PPA, power, energy, and headline claims
remain false.
