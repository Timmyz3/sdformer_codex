# M506 FC2 synthesis-portable onehot VCS regression

The frozen M492 and M497 Synopsys VCS suites were rerun after replacing the
synthesis-visible `$onehot` calls in M342 with an explicit one-of-eight
predicate. Both suites preserve their exact cycle rows, zero-mismatch
arithmetic/transaction/weight checks, protocol attacks, and SVA coverage.

This result only clears a synthesis-portability blocker for M496. It is not a
new hardware mechanism, performance admission, system speedup, or paper-ready
PPA result.
