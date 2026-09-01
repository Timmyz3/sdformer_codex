# M1688 additive reducer-topology repair

M1688 leaves M1681 and its grid, scheduler and execution path immutable. It changes only the sealed-shard completion arbiter. A shard now contributes to reduction only when the exact sibling topology is `result=true`, `attempt=true`, `work=false`, `failure=false`; the attempt must also be a regular non-symlink with mode exactly `0400`.

The reproduced `result+attempt+failure` attack, `result+attempt+work`, attempt symlink and wrong-mode attempt are all rejected. The existing 15 metric/result attacks remain rejected, and the two-shard reducer preserves request totals 344/346/348.

Both Python versions pass nine tests and compilation. The failed M1683 release name remains permanently forbidden. No release, payload, replay, production reduction, GPU or EDA action occurred. M1689 independent review is required.
