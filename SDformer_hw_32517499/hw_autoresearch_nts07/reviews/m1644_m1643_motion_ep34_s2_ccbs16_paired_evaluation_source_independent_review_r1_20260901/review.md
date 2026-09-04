# M1644 independent review of the M1643 S2/CCBS16 paired evaluator source

Status: `PASS_M1644_M1643_S2_CCBS16_PAIRED_EVALUATOR_SOURCE__SOURCE_ONLY_NO_EXECUTION` (98/100; P0=0, P1=0, P2=1).

The frozen M1643 source, test and contract hashes match their committed and pushed identities. Both the contract and author receipt verify through their outer seals. The independent suite passes 18/18 on CPython 3.6.8 and 18/18 on CPython 3.10.16, with 77 mutation or boundary assertions per runtime. No M1624 payload/result file was opened, and no GPU, DSE, RTL, EDA, remote or production run was performed.

The source correctly enforces the intended S2 boundary:

- one exact 16x16 decision block;
- epsilon zero is a literal bypass with identical paired AEE/cycles, no decision rows, no metadata and no savings;
- every positive-epsilon block, whether kept or dropped, costs one two-byte bound;
- a drop is credited only when the decision strictly precedes weight fetch, compute and psum, while saved bytes/operations are derived exclusively from the paired baseline ledger;
- the ordered 40-sample cohort is exact, AEE deltas are paired overall and per sequence, and cycle speedup is ratio-of-sums;
- the four fixed gates are conjunctive at 0.02 overall AEE, 0.03 every-sequence AEE, 2% metadata and 1.15x same-resource local cycle speedup; and
- admitted TSBG forces the admitted-TSBG baseline and forbids multiplying component speedups.

One non-blocking source-stage integration condition remains. Candidate cycle counts are necessarily supplied by the future runner; the accounting kernel does not itself simulate them. Therefore the future runner must generate and seal candidate and baseline cycles under one exact cycle/resource model. Caller-provided candidate cycle values alone can never become performance evidence.

This review accepts the frozen source for future runner authoring only. It does not authorize payload opening, paired execution, DSE, RTL/EDA, performance release or a paper claim.
