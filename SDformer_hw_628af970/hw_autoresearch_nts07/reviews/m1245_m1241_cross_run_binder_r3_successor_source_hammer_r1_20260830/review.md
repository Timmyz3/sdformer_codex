# M1245 independent hammer of M1241 binder-r3 successor

Verdict: **PASS; release authoring is allowed**.  This is a source-only
verdict.  It does not execute the production binder, select a real checkpoint,
authorize hardware rebind, or authorize remote, GPU, capture, or EDA work.

The exact pinned M1241 source passes its 18/18 controlled tests.  The fixed
M1234 consumer schema/status, exact ep29/30/32/34 population, 825-sample
profiles, four typed-zero load counters, 105 ATLIF and 12 attention modules,
all eight finite nonnegative error metrics, minimum-AEE/lower-epoch selection,
and E0--E8 remain intact.

Both M1238 P0 defects are closed.  A replacement during semantic validation is
rejected by the final lstat equality.  A replacement between the initial lstat
and descriptor open is rejected by path/descriptor identity.  Every component
of run, checkpoint, profile, config, and manifest walks is opened with
`O_NOFOLLOW`; final and parent-component symlinks are rejected.  A rename of a
frozen run root into a symlink is rejected.  Checkpoint/profile walks must
contain the exact frozen run-root device/inode/type identity.  The two run roots
must be physically distinct.  Configs must be both physically distinct and
SHA-distinct: hardlink aliasing and distinct-inode/same-byte aliasing are each
rejected.

The additional post-check attack is intentionally different from the M1238
defect.  Replacing the profile *after* `confirm_frozen_path` returns is accepted,
but the replacement AEE is not read or used: selection remains epoch 32 with
the old descriptor's AEE and SHA.  There is no pathname operation after the
final equality.  This is the correct boundary for a mutable filesystem result;
any later production/result-hammer consumer must reopen the recorded absolute
path and reverify the recorded identity.  M1241's unchanged status still
requires that fresh result hammer and grants no hardware-rebind authority.

The source, test, contract, sealed M1234 predecessor, recursive-double-sealed
M1238 review, and frozen docs/359 were independently pinned.  No source,
production path, remote host, GPU, checkpoint, valid825 data, capture, or EDA
flow was modified or executed.
