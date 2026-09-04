# M1238 independent hammer of M1234 cross-run binder successor

Verdict: **NO-GO for release authoring**.  M1234 closes both defects reported
by M1231 in the ordinary read window: the profile SHA and strict JSON parser
consume one descriptor snapshot, and all eight optical-flow error metrics reject
negative and non-finite values.  The author's controlled suite passes 15/15.

The independent hammer nevertheless found two new fail-closed blockers.

1. A controlled pathname replacement performed while the already-read JSON is
   parsed, after the last path/fd equality check but before `path.resolve()` is
   returned, is accepted.  The returned record contains AEE 1.0, the SHA and
   inode of the old bytes, while the recorded pathname now names a different
   inode containing AEE 9.0.  The immutable byte binding itself is sound, but
   the returned path identity is not stable through publication.
2. A lexical legacy-run path that is a directory symlink to the resume run is
   accepted as a two-run policy.  All four candidates build, but every emitted
   `run_directory` resolves to the same physical directory.  `O_NOFOLLOW` on
   the final profile component does not reject a symlinked parent component.

Independent positive checks also reject string-encoded `-1E-1000` and
`Infinity` for every one of AEE, AAE, AAE_Benchmark, AEE_PE1, AEE_PE2,
AEE_PE3, AEE_outliers and DSEC_Fl.  A replacement at descriptor EOF is
rejected.  The canonical synthetic fixture preserves exact candidates
ep29/30/32/34, two run directories, two configuration SHA identities, 825
samples, the four typed zero load-audit counters, 105 ATLIF and 12 attention
modules, lower-epoch tie breaking, E0--E8, and the fixed M1234 schema/status.

Required successor changes are narrow: finish the path-identity transaction
with a final lstat equality after parsing (or avoid pathname resolution after
the final check), and establish two non-symlinked, distinct resolved run-root
device/inode identities before reading candidates.  A fresh different-author
hammer is then required.  No production path, real checkpoint, valid825,
remote host, GPU, EDA, checkpoint selection, or E0--E8 rebind was accessed.

