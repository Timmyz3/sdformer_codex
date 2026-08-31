# M1272 — M1271/R13 checker/tests-only repair

Date: 2026-08-30  
Mode: checker/tests-only source authoring; no VCS/simv/EDA/GPU/remote  
Decision: **SOURCE GO pending one fresh different-author source hammer**  
Score: **99/100**  
P0/P1/P2: **0/0/1**

## Outcome

Only the R13 checker and its mutation tests changed. The R13 TB remains
byte-exact at `b749c7d...54263`; the M1270 contract, M528, M935, M1162, R3 SVA
and `docs/359` also retain their frozen SHA256 identities.

The repaired checker closes all four M1271 P1 classes:

1. PHASE/PASS tokens are extracted only from executable `$display` calls in
   the sole authoritative initial flow and must match exact token boundaries.
2. The completion call must occur exactly once in that initial flow; the two
   real-M935 beat calls must occur exactly once in the completion task; static
   false guards, control escapes and conditional compilation are rejected.
3. Blocking/nonblocking assignments and force/release statements to bare,
   parent or child `issue_request_*` objects are rejected independent of line
   formatting.
4. The executable oracle operand display and flush must directly dominate the
   sole X-safe fatal; commented or runtime-guarded operand printing is rejected.

## Mutation evidence

The clean frozen source passes. All 16 attacks are rejected, including all six
M1271 attacks: commented PASS, commented phase-complete, commented completion
call, statically disabled real beat calls, bare request assignment, and
commented oracle operand printing. The additional attacks cover near-neighbor
phase tokens, missing non-first beat, blocking/nonblocking and child
assignments, parent/bare force/release formatting, missing flush, guarded
operand print and disabled completion.

The checker and tests report `17/17 PASS` (one clean source plus 16 rejected
attacks), and both files compile with Python's bytecode checker.

## Claim boundary

This is source-only checker evidence. It does not establish functional VCS,
runtime SVA behavior, timing, cycles, speedup, PPA, energy, system speedup, or
a paper headline. It does not authorize release authoring or launch.

The only authorized next action is a fresh different-author, read-only source
hammer over the exact M1272 checker/tests/TB/contract identities and all M1271
attack classes. No further checker expansion is authorized unless that hammer
finds a concrete P0/P1 defect.

P2: the source admission remains a deliberately conservative lexical/control
shape proof rather than a general SystemVerilog parser. Exact frozen identities
and the independent mutation hammer are therefore still mandatory.
