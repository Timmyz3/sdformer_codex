# M1037 — M1040 C1 full-replay collision/resource repair source receipt

**Verdict: source chain is ready for independent M1039 hammer only. M1040 is not authorized to run.**

The additive M1040 runner uses a fresh result and attempt namespace. Before attempt creation it exact-pins `/usr/bin/pgrep` and `/usr/bin/flock`, rejects the six frozen EDA process names with `pgrep -x`, acquires a nonblocking fixed global C1 replay lock, and checks both commit headroom and `MemAvailable` against 16 GiB floors. The lock is held by an open file descriptor for the runner’s lifetime.

The process blacklist is deliberately exact and does not scan generic CPU jobs. Sandboxed tests reject each of `vcs1`, `vlogan`, `dc_shell`, `dc_shell-t`, `fm_shell`, and `pt_shell` before attempt creation; lock contention, low commit headroom, and low `MemAvailable` also stop before attempt. An unrelated CPU process is allowed through the gates and reaches only the sandbox `/bin/false` engine, demonstrating that the resource policy does not kill work merely for being CPU-active.

M1038 binds the M1016 engine/contract, M1025 GO authority, M1036 FAIL authority and exact M1040 runner. It reserves a hardcoded independent M1039 directory/status, and M1040 requires caller-pinned M1025, M1036, and M1039 outer seals plus cross-bound M1039 identities.

The old M1028 chain remains prohibited and unconsumed. Static checker, `bash -n`, release JSON/double-sidecar verification and 6/6 tests pass under the host default Python 3.6 as well as the pinned Python 3.10 shebang. Tests execute only redirected runner clones with `/bin/false`; neither production M1040 nor the 51.84M replay or EDA/GPU is invoked.
