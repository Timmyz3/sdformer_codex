# M1264 independent hammer of M1263 final checker/tests

## Verdict

**84/100; FAIL CLOSED.** `GO_SEPARATE_RELEASE_AUTHORING=false`. No release,
VCS, `simv`, EDA, GPU, or remote action was run.

The frozen pins match the assigned tuple: checker
`fe09c20...1e53d`, tests `f0add1a...96fb`, R12 TB
`e13d630...d302`, and docs/359 remains
`dedde7c...dfc4`.

## What closed

The declared regression ran all 30 tests and passed. In particular, the four
M1262 defects are covered: ordinary strings/`$fatal` cannot impersonate
executable `$display`, suffix-shadow tokens are rejected, force/release
duplicates are counted, and force/release outside the authorized task
inventory is rejected.

## New fail-closed findings

Five independent mutants were all incorrectly accepted. They form four P1
classes:

1. A mandatory child-seam `force` guarded by `if (1'b0)` remains in the exact
   lexical inventory, so the checker accepts a source that never drives that
   field.
2. Exact phase and PASS `$display` calls guarded by `if (1'b0)` remain in the
   `$display` inventory. Thus observability/PASS reachability is not proven.
3. `if (1'b0) normal_m935_completion();` satisfies the lexical one-call rule
   while deleting integrated-normal execution.
4. The executable random loop can be changed from `<24` to `<0`; retaining
   `test_index < 24` only in a comment satisfies the inherited raw-text marker.

These are syntactically legal source mutations and directly weaken the
properties this checker claims to authorize. Exact SHA pinning would protect a
later release from arbitrary replacement, but the assigned gate explicitly
requires P0/P1/P2 all zero before authoring that release. Therefore M1263 is
not admitted.

## Bounded repair

Do not add a general SystemVerilog parser. The bounded repair is to pin exact
canonical statement forms and their enclosing initial-block sequence for:

- the force helpers (reject conditional/control prefixes),
- phase/PASS calls,
- the integrated-normal call, and
- the exact random-loop header from executable text.

After those four predicates receive negative tests, one fresh different-author
hammer is sufficient. The TB/RTL/SVA remain frozen.
