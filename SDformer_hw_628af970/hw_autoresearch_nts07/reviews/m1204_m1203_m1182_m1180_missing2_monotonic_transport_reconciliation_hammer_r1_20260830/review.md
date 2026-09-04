# M1204 independent monotonic transport hammer

Verdict: **FAIL_CLOSED / execution not authorized**.

The declared source, test, contract, author manifest, and outer seal identities
all match.  The recursive author seal verifies, all ten author tests pass, and
the independent state-machine tests confirm exact-subset idempotence,
wrong/symlink rejection, partial-publication recovery, strict archive-member
validation, and the final both-exact gate.  M1203/M1180 attempt and result
namespaces remain absent.  This hammer performed no SSH, SCP, remote mutation,
GPU, capture, or EDA action.

The release is nevertheless unsafe at the transport boundary.  The preflight
program does not carry or inspect `REMOTE_ARCHIVE` or `REMOTE_STAGE`.  `run()`
then consumes its local attempt and invokes SCP to a fixed `/tmp` pathname;
only afterwards does the remote reconciler perform archive lstat/SHA checks.
A pre-existing symlink/nonregular archive pathname can therefore be followed or
overwritten before rejection.  Tar-member attack tests do not cover this
pre-verification pathname mutation.

Required successor: allocate an exclusive unpredictable remote temporary
directory/path through authenticated SSH, validate the returned anchored path
and its lstat/ownership/mode, SCP only into that directory, and verify archive
lstat/size/SHA before extraction.  Streaming the archive over SSH without a
remote pathname is also valid.  The successor needs a new exact identity and a
fresh different-author hammer.
