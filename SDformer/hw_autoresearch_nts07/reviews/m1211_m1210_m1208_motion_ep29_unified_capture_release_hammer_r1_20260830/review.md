# M1211 independent secure-transfer and one-shot launch hammer

Verdict: **GO, 100/100, P0=0, P1=0**.

This was a strictly local review. No SSH, SCP, network, GPU, capture, EDA, or attempt-marker action occurred.

The hammer exact-bound the final M1210 wrapper and its source contract, M1208 launch contract, 21-file transfer inventory/list, the inherited 95-row dependency inventory, and the release-author double seal. All 21 transferred members match path, order, size, and SHA. All 95 inherited rows carry unique safe paths and exact size/SHA identities, and the remote helper verifies every row before staging or publication.

The transport protocol uses an authenticated remote `mktemp -d`, then independently enforces root ownership and mode 0700. The fixed-name tar archive is checked for size, SHA, exact member order, regular-file type, per-member size, and extracted SHA. Publication accepts only ABSENT or byte-exact regular targets; drift, symlinks, unsafe parents, or stale publication temporaries fail closed. New files use O_EXCL/O_NOFOLLOW temporaries, fsync, atomic replace, and post-publication SHA.

M1180 remains a read-only failed boundary: its exact consumed token is required and its result/log must remain absent. M1208 uses disjoint fresh attempt/result/log namespaces. The permanent local M1210 no-retry marker is created with O_EXCL after successful remote transfer preflight and before the sole launcher call. There is no retry loop or marker cleanup.

Controlled tests passed 8/8. The independent checker passed 2292 checks and rejected 12/12 mutations covering command quoting, preflight bypass, duplicated launch, marker reordering/non-exclusivity, inherited dependency bypass, namespace weakening, symlink acceptance, non-atomic publication, remote-temp relaxation, and archive-identity relaxation.

Authorization is limited to one secure transfer and exactly one M1208 launch. The resulting capture still requires a fresh result hammer and is not yet paper-citable.
