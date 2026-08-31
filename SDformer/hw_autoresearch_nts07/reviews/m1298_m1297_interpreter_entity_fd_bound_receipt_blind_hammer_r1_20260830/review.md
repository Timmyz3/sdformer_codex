# M1298 receipt-blind hammer: M1297 fd-bound interpreter successor

## Verdict

**91/100. Entity/fd mechanism PASS; exact transfer GO; production execution STOP.**

M1297 closes both M1294 findings. It measures runtime version/capability through
the retained interpreter fd, binds symlink/realpath and exact stat/SHA identity,
revalidates before O_EXCL consumption, records the entity digest, and executes
the child through `/proc/self/fd/N` with the exact three sealed source fds plus
the interpreter fd. Local actual fd execution passed; omitting the interpreter
from `pass_fds` failed. Path retarget, fd close/replacement, every identity/type
drift, and version spoof all fail before attempt consumption.

M1257/M1292 policy values, candidates, artifacts, execution pins, eleven
snapshots, three sealed children, F1--F4 and E0--E8 do not drift.

## Blocking P1

The M1297 contract is not an entity-only successor at the claim boundary. It
drops `checkpoint_selected_now` and `remote_execution_authorized` from M1292's
closed exact-false map and adds `paper_ppa_ready`. Even though the authorization
section separately says execution is false, exact closed-map preservation is a
release invariant and this semantic drift was not part of the reviewed change.

Required repair: an additive successor must pin M1297 source/test/contract,
restore the exact M1292 claim keys and exact booleans, and keep any PPA statement
in a separate closed scope. A fresh different-author hammer must then pass.

After that repair, one production execution is only conditionally eligible and
still requires root's immediate live read-only preflight to match the pinned
realpath, dev/inode/mode/size/mtime/SHA, Python 3.12.3 and memfd/seal capability.
No attempt may be consumed before those gates.

No remote connection, checkpoint selection, production execution, GPU or EDA
action occurred. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
