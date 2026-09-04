# M1198 additive R7 source-gate repair author receipt

Status: `PASS_R7_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA`.

M1194 is accepted in full. The exact R6 testbench is clean and closes M1192's
concrete nested-core-ready-force P0, but its author checker accepted six
adversarial relaxations. R7 therefore does not rewrite the functional TB or any
RTL. It creates an additive source-gate and future result namespace around the
exact frozen R6 TB bytes.

The R7 call-graph gate recognizes both legal task-call spellings,
`helper(...)` and bare `helper;`, at any statement position rather than only at
the beginning of a line. Its transitive closure remains exactly the service
root, the no-core-ready request helper, reset, release, and public-driver clear.
The dedicated helper's force multiset must equal all nine request valid/tuple
fields exactly; subset checks are insufficient. Both service oracles are
exact-matched as own fault one, peer fault zero, and composed protocol,
boundary, and core fault zero.

Sixteen mutations were rejected. They include all six M1194 bypasses, plus
generic-helper aliasing, an explicit indirect helper, aliased force, joined peer
responses, service mask removal, normal M935 removal, and assertion/cover
removal. The preserved functional corpus remains 16 assertions, six covers,
seven protocol attacks, two service attacks, 24 legal transactions, 29 legal
mask-clear cases, three reset states, completed II=2, and one normal M935
row/task.

Author testing passed 74 checks and rejected all 16 mutations. A fresh
different-author hammer is mandatory before any R7 launcher or release may be
authored. No VCS, `simv`, license checkout, network, GPU, or other EDA tool was
invoked. docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
