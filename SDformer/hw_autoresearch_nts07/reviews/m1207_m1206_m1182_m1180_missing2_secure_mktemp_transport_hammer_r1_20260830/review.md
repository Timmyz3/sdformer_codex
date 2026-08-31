# M1207 independent secure-mktemp transport hammer

Verdict: **PASS / authorize exactly one M1206 transport attempt; automatic retry is forbidden**.

The exact source, test, contract, author seal, and frozen docs/359 identities
match.  All 12 author tests and the independent local-only attack oracle pass.
The successor closes the M1204 fixed-path blocker by creating an unpredictable
0700 UID0 directory through authenticated remote `mktemp`, accepting only one
anchored stdout pathname, and placing `exact2.tar` only below that validated
exclusive directory.  The remote reconciler independently verifies archive
lstat/type/owner/size/SHA and every member before destination-local exclusive
temporary publication.

The independent state matrix confirms exact-subset idempotence, wrong and
symlink rejection without mutation, safe recoverable partial publication,
cleanup-failure no-success behavior, and fail-closed capture-marker races.
This hammer performed no SSH, SCP, transfer, GPU, capture, or EDA action.

Authorization is limited to the exact two dependency files and the single
sealed M1206 transport invocation.  It does not authorize M1180 capture and it
does not authorize an automatic retry.
