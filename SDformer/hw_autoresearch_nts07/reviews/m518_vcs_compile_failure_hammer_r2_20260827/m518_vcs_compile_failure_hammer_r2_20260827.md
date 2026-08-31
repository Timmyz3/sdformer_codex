# M518 r4 VCS compile-failure hammer review

Verdict: `R4_FAILURE_DIAGNOSTIC_ONLY__RENAME_DECLARATION_PLUS_FIVE_USES__BUILD_AND_STATICALLY_REVIEW_FRESH_R5_BEFORE_ANY_TOOL`

This review was read-only with respect to the M518 production sources, the r4
failed result, and `docs/359_DATE终局冻结_20260813.md`. The reviewer did not run
VCS, DC, Formality, PT/PTPX, or an open-source RTL tool.

## What the r4 evidence establishes

- The exact reviewed runner identity was used: `d656d11dc32e...5883ef55b`.
- All 17 positive frozen-input SHA checks matched. The frozen RTL SHA was
  `09b1d976595f...1379a93412a`; SVA, TB, filelist, contract, prior sealed
  specification, prior failure review, and docs/359 also matched.
- The automatic wrong-RTL negative control exited 10 before any tool invocation.
  Its member manifest and outer seal both verify.
- `vcs -full64 -ID` succeeded and records
  `Synopsys VCS V-2023.12-SP1_Full64`.
- The full64 compile reached the production RTL parser, returned 255, and emitted
  one error only: `Error-[SE] Syntax error` at RTL line 279, token `within`.
  The runner correctly converted the unsuccessful compile into exit 20.
- There is no `simv`, simulation log/return code, assertion report, positive
  receipt, `RUN_COMPLETE`, or positive publication seal. The failure marker says
  `FAILED_OR_INCOMPLETE_DO_NOT_CITE`.

Therefore r4 proves neither compile legality nor behavior, V01-V20, numeric
equivalence, the 29/80-cycle schedule, speedup, area, energy, PPA, or any system
claim. DC remains locked.

## Root cause and the exact minimal source repair

`within` was used as a block-local integer identifier in the multiplier-slot
mapping logic, but it is a SystemVerilog assertion-language reserved token. The
source contains **six** word-token occurrences, not five total edits:

1. line 279: declaration;
2. line 281: initialization;
3. line 287: assignment;
4. line 290: use;
5. line 300: assignment;
6. line 303: use.

Equivalently, there are five uses plus one declaration. The only RTL change
justified by this failure is a consistent rename of all six tokens, for example
to `tap_within`. Renaming only five total tokens is incomplete. No arithmetic,
schedule, public-port, SVA, TB, or filelist-content change is currently
justified.

This is a first-parser failure: VCS did not reach the SVA or TB and did not run
the design. Consequently the review cannot promise that the rename is the last
possible compile or behavioral blocker; it says only that no broader source
change is supported by the present evidence.

## Required r5 chain

r5 must have a new RTL SHA, contract identity, canonical result path, runner
identity, and independent static review. The new runner must bind and verify
this review's member manifest and outer seal, retain the wrong-RTL exit-10
negative control and both full64 VCS invocations, and preserve the r4 failure
directory. Only a fresh exact-SHA, one-shot static authorization may permit r5
VCS. DC can be considered only after a complete r5 VCS publication and a
separate independent post-run receipt hammer.

This review authorizes neither direct r5 VCS nor DC/Formality/PT/PTPX.
