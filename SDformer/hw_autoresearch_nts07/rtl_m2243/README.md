# M2243: consume a borrowed weight beat, release at its last consumer

This small optional C2 extension retains only four active/sign masks, metadata,
and a pending-consumer bitmap. The existing M803 adapter continues to own and
hold the 128-byte response. Every context gets its own signed weight operation;
only weight delivery is shared. An accepted final consumer retires the response.

It is **not** a replacement for M2018's complete scheduler, descriptor capture,
Acc24 array, or commit path. The caller must associate metadata with the selected
M803 response. Full integration, timing, and energy are still pending.

The ordinary/cache comparison must receive the same streaming/bypass capability.
M2241 finds no additional cycle advantage from dropping that cache. M2244 finds
no additional read advantage against a cache receiving the same union masks.
The remaining claim to test is reduced payload storage/copy energy and area.
Conventional elastic eager-fork acknowledgements are prior art, not our novelty.

## Regression

`tb_m2243/tb_m2243_borrowed_weight_consumers.sv` instantiates the actual M803
adapter. Two logical request slots, unequal bank readiness/response latency,
downstream stalls, empty response drainage, independent context signs, and
INT8 -128 negation are exercised. SVA checks stalled payload stability.
Each transaction's consumers and retirement are scored exactly once.

The two runs use full-bank and active-union request masks. Empty-union beats
send a diagnostic one-bank probe only to test draining an empty response;
a production union scheduler would avoid issuing those beats altogether.
The integer reference checks per-beat signed deltas, not a full G48 Acc24 run.

From the repository directory:

```sh
/opt/anaconda3/bin/python3.12 hw_autoresearch_nts07/dc_handoff/scripts/run_m2243_borrowed_consumers_vcs.py --after-power
```

One such process is already queued behind M2242; do not launch a duplicate.
The runner uses a new temporary result directory, compiles once, and runs both
mask modes. No result is admitted merely because these sources exist.
