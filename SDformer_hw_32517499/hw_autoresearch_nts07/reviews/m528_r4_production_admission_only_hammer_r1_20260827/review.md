# M528 r4 production admission-only independent hammer

## Verdict

PASS, 99/100, P0/P1/P2 = 0/0/1.

This review authorizes root to execute exactly one CPU production invocation of the byte-pinned M528 r4 production runner, and only if its live 48-GiB commit-headroom, 128-GiB MemAvailable, 32-GiB SwapFree, clean-cgroup, and UID-local Synopsys/VCS/simv collision gates pass. It does not authorize a second CPU run, EDA, GPU, RTL, a paper claim, or a system-speedup claim. Any raw result still requires a fresh independent result hammer.

Required invocation identities:

- production runner: `a29827cd099b96f390d766715df584455dc3311f54b5db0487e19a22e3c007ac`
- production admission: `contracts/m528_r4_production_static_admission_r1_20260827.json`
- production admission SHA-256: `5c2043ef49f8cbcc5cbeb1695d0185f841783dc0cdef4940267cb46b3c28de2b`

## What was checked

The production admission is strict JSON and its member and outer sidecars verify. Its schema and status authorize one CPU run and one pre-attempt spawn/schema repetition, with EDA/GPU zero and RTL false. The live runner, analyzer, preflight runner, execution contract, governing contract, strict-JSON tool, pinned Python binary, and docs/359 identities all match the admission.

All upstream evidence seals verify. More importantly, the production runner directly parses the static review PASS/P0/P1/identity/authorization, the preflight admission semantics, all three preflight receipt cases and forbidden-activity fields, the receipt hammer PASS/P0/P1/receipt identity/authorization, and the r2/r3 NO-GO boundaries. It does not accept an outer seal alone as authority.

The runner performs three resource snapshots and the exact one-worker spawn/schema repetition before creating the production attempt sentinel. Production then uses workers=3 and chunksize=2. The exact execution contract and analyzer preserve row64, B8, 128 B/cycle, CAM64, the frozen cycle/traffic/capacity aggregation, and the original decision gates. At review time both the r4 production canonical directory and production attempt sentinel were absent.

## Non-blocking finding

P2-01: the admission predicate does not separately restate every duplicated non-authorizing claim-boundary or boolean field from the exact admission. This is non-blocking because the caller pins the entire reviewed admission SHA; the operative resource/collision controls execute in the runner; the frozen coordinate is enforced by the exact analyzer and execution contract; and no raw result becomes citable without another independent hammer.

No preflight, spawn, analyzer, production, EDA, GPU, or RTL action was executed by this review. Reviewed evidence and docs/359 were not modified.
