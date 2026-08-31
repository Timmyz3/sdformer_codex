# M1325 runtime-projection successor source author receipt

M1325 is an additive, source-only repair for the M1320 pre-GPU failure.  It
keeps exact M1319/M1313/M1314 identity binding and constructs only the four
runtime keys directly consumed by frozen M1227: `contract_path`, `capture`,
`cohort`, and `output`.  `capture.attention_windows_per_call` is exactly 100;
the cohort is a deep copy of exact M1313; all result/attempt/log names are new
M1325 names.

Ten tests passed, including an AST exact-key audit and a mocked traversal of
the real M1249→M1243→M1233 delegation chain up to the M1227 entry.  The new
output propagated through that chain.  This source has no attempt consumer,
no GPU lease and no production CLI.  A fresh different-author hammer and a
separate release remain mandatory.
