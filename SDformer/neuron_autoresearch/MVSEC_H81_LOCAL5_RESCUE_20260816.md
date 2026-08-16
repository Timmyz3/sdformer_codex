# MVSEC H81 completion and Local5 rescue

H81 same-protocol full-sequence all-four-better-than-NB0: `False` (fails `indoor_flying1`).

| protocol | OD1 | IF1 | IF2 | IF3 | macro AEE |
|---|---:|---:|---:|---:|---:|
| H81 fixed800 | 0.8154 | 1.6224 | 2.6628 | 2.0586 | 1.7898 |
| H81 full | 0.8205 | 1.6248 | 2.6670 | 2.0581 | 1.7926 |
| H67 full | 0.8201 | 1.5868 | 2.6258 | 2.0357 | 1.7671 |
| NB0 full | 0.8450 | 1.5998 | 2.7536 | 2.1106 | 1.8273 |

Local5 DSEC-ep44 day2 FT: first smoke saw `overlay=210 missing=12 unexpected=0`; the 12 keys are window-8 `positional_encoding`. Audit was relaxed and FT was restarted.
