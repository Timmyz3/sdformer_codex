# M1044/M1046 UCLI-power source receipt

Status: `PASS_M1044_M1043_M1046_UCLI_POWER_COMPLETE_SOURCE`.

M1033 remains consumed and is not retryable.  Its compile/link succeeded, but
the first UCLI case produced UCLI-117 because the production binary lacked
`-debug_access+r`; it completed zero gate cases and produced zero SAIF files.
M1043 double-seals that boundary.

The additive M1046 runner uses a fresh namespace.  Before the attempt is
consumed, a frozen tiny DUT is compiled with the same
`-full64 -sverilog -debug_access+r -lca` flags used by all three production
axes.  The preflight must execute UCLI power enable/disable/report and create a
nonempty SAIF containing both the frozen top and `dut` hierarchy with positive
duration.  Any failure is sealed separately and does not consume M1046.

The `-lca` flag is evidence-driven, not decorative: a noncanonical author
probe with `-debug_access+r` alone reached UCLI but was rejected with
`LCA_FEATURES_NEED_OPTION`; the same probe with `-lca` passed compile,
simulation, power commands, and emitted a 2106-byte, 24 ns hierarchy-correct
SAIF.  No license value was recorded.

Static checks and 11 fault-injection tests pass, including missing debug
access, missing LCA, UCLI failure, missing/empty SAIF, wrong hierarchy/duration,
and namespace collision.  This receipt does not authorize production:
`launch_now=false`, M1045 must be independently authored and its exact outer
seal pinned by the caller, and no M1046/PT/PTPX/DC run was performed.
