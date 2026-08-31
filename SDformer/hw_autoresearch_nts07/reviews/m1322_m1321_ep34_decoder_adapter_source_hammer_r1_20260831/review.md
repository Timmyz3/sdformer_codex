# M1322 different-author blind hammer of sealed M1321

Verdict: `FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`.

The sealed M1321 source gets the difficult data semantics right: exact
positive/negative plane extent and order, sign and padding rejection,
D0/D2/D3 exact binary words, D1 dynamic positive-finite theta without
coercion, zlib EOF/trailing checks, raw/compressed/support SHA checks, the
10..39 / 120-call positive projection, strict weight keys/checkpoint SHA, and
an inert source-only CLI.  Author tests pass 8/8 and eighteen independent
positive/negative controls pass.

It is not admissible because three exact-graph attacks remain open:

1. A selected D1 row may duplicate D0's `global_order`; stable input ordering
   still lets the adapter return all 120 calls.
2. An ignored JSONL row may be replaced by a duplicate while the 9880-row
   count stays unchanged.
3. `module_ordinal=true` is accepted as D1 ordinal one due to Python's boolean
   integer equality.

The minimal additive repair is narrow: validate every JSONL row before
projection with `type(global_order) is int` and
`global_order == file_ordinal` for the complete 0..9879 stream, and require
`type(module_ordinal) is int` before exact weight ordinal equality.  M1321 is
sealed and was not modified.

No remote access, GPU work, production capture, payload normalization, replay,
cycle/traffic measurement, or paper-performance admission occurred.
