# M1323 ep34 decoder capture adapter — author receipt

Status: `PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION`

M1323 is an additive successor; sealed M1321 was not modified.  It closes all
three M1322 false-negative findings before decoder projection:

- exact integer fields reject Python `bool`;
- all 9,880 `global_order` values equal their JSONL file ordinal; and
- all 40×247 rows, including non-retained rows, bind the exact M1313 sample
  identity and exact frozen live module identity once per sample; and
- all 320 retained payload pairs are named by exact sample/order/module identity
  and are unique, so no two calls can alias one captured tensor.

The author regression passed 9/9 tests under pinned Python 3.10.  It attacks
boolean ordinals, duplicate/non-contiguous orders, ignored-row duplicate and
replacement, ignored-row identity/key drift, sample drift, module execution
order drift, cross-call retained-payload aliasing, and CLI production escalation.

This is only a decoder-input source audit.  No GPU, remote capture, payload
normalization, weight export, cycle simulator, VCS, DC, PTPX, system metric, or
Table-A production was run or admitted.
