# M142 independent hammer review r1

Outcome: **84/100, P0=0, P1=3, P2=5**. GO for the narrow B4 controller RTL/VCS seal and its 3 ns flattened logic-only DC receipt. Independently directed B3 and B4 controller behavior also passes. NO-GO for claiming M143r2 cycle equivalence, measured RTL throughput, macro-complete PPA, energy, physical speedup or system speedup.

## Exact evidence independently reproduced

- All eight sealed VCS inputs, four sealed VCS outputs and 20 sealed DC evidence entries pass SHA verification. The required VCS and DC receipt SHA256 values are respectively `dfd3d8854572a778a70cba207f938a16164eba65d1b1dcc8417624376d586735` and `3e6f6fabc2b4fdd686f54a57a6a724e451e402ede138486a3ed2ff87a6f0fef6`.
- Synopsys VCS V-2023.12-SP1 reproduces the production PASS exactly: 32 windows, 96 accepted rows, 32 zero rows, 1,184 descriptors, 4,576 checked sources, 32 PWP/correction launches and completions, 1,038 II1 descriptor intervals, 120/3/2 descriptor/PWP/correction stall cycles, and 28 bank reuses. All six required covers are nonzero and there is no assertion failure.
- The M143r2 production analyzer independently reruns byte-identically at SHA256 `8b5821d747e653ac9053a4cfe94fe9eb40c78ce0eaaca4c9af4fdf8073b5bd19`.
- The final M142 DC flow was independently rerun with the exact RTL, Tcl, SDC and TSMC28 libraries. It reproduces 2,562.462012 um2, 3,313 cells, 561 sequential cells, setup +0.0291 ns, reported hold +0.0000 ns, 68 logic levels and zero macros. Both sealed and independent logs contain zero TIM-209, OPT-150, ELAB-312, Error or Fatal events.
- `docs/359` remains unchanged at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Independent B3/B4 protocol hammer

The same separately written VCS test passed for both allowed bank parameters. It checks:

- all eight blocks in canonical block-major K1..K4 order, including strictly increasing sources and signed negate bits;
- an all-zero closing row with no descriptor emission;
- dirty negate padding fail-closed;
- four cycles of final/closing-descriptor backpressure with stable output and no early PWP;
- PWP only after the final descriptor is accepted and the bank is fully materialized;
- wrong active PWP/correction bank or tag fail-closed;
- all B3/B4 banks owned, oldest-first PWP/correction launch and normal bank reuse.

The hammer also exposes the following model/interface boundaries rather than treating them as expected-safe behavior.

## P1-1: M143 endpoint recurrence is not M142 cycle behavior

M143r2 contains 1,332 zero-PWP units and 300 zero-correction units. Its recurrence gives zero PWP work zero duration and releases zero-correction units without a correction endpoint. M142 has no empty-work bypass: every closed bank follows `FILLED -> PWP -> WAIT_CORRECTION -> CORRECTION -> FREE` and only a matching correction completion releases it.

Independent VCS confirms that same-cycle PWP launch/completion and same-cycle correction launch/completion both quarantine the controller. It also confirms that, with another bank waiting and ready held high, M142 cannot launch that bank on the prior completion edge; PWP and correction each have a registered completion-to-next-launch edge. The previously published “one-cycle floor for zero work” B4 sensitivity of 135,461,027 cycles only floors zero-duration jobs. It does not model the registered handoff for ordinary nonzero jobs and therefore is not an RTL-cycle closure.

The production M143 values—B4 135,461,009 cycles, 2.594690240x versus compact256 and 1.812225612x versus dualrow512—remain reproducible **same-clock Python module-cycle ratios**, not measured or cycle-equivalent M142 throughput.

## P1-2: ordering and completion identity are externally assumed

M142 stores a 32-bit internal sequence, but endpoint requests/completions expose only bank plus a default 16-bit tag. The frozen model schedules 69,120 units, 3,584 more than the 65,536-value tag space. An independently replayed stale completion with the same reused bank/tag is silently accepted for the current PWP lifetime; the same ambiguity exists at correction completion.

The controller also accepts decreasing row IDs and early window closure without error, so the modeled exact row extent/order is a producer contract, not an RTL check. At the minimum allowed `SEQUENCE_BITS=18`, a reachable wrap boundary exposes another ordering defect: raw unsigned `<` selects the newer sequence zero before the older `max-1` bank. The hammer reaches that cone by backdoor-initializing the otherwise reachable pre-wrap counter state and then using only normal row handshakes.

## P1-3: external SRAM, engines and outer barriers are missing

M143r2 schedules 119,447,791 PWP tokens, 124,730,596 correction tokens and 160 outer flush/commit barriers. M142 contains no PWP/correction arithmetic, no descriptor/result SRAM, no work-length/empty indication and no outer commit/barrier interface. Its endpoint launch carries only bank and tag; completion timing is supplied arbitrarily by the environment.

Consequently an unstated external wrapper must materialize descriptor/result SRAM, track per-bank lengths/emptiness, run both engines, generate truthful completions, enforce the 160 barriers and gate producer/service readiness. Until that wrapper or a cycle-exact driver is implemented and replayed, the M143 recurrence cannot be attributed to the standalone RTL.

## DC and performance boundary

The reproduced DC number is internally clean but narrow: +0.0291 ns is 29.1 ps, only 0.97% of the 3 ns period. The critical path is the 68-level sequence/oldest-bank selection cone ending at `pwp_window_tag`. Clock is ideal, wireload is ZeroWireload, macro count is zero, and there is no routing, power, SRAM or engine timing. Reported hold +0.0000 ns is rounded, not physical margin.

The earlier debug r1/r2 OPT-150 loop runs remain correctly marked DO_NOT CITE. The final sealed and independent reruns contain no TIM-209 or OPT-150.

## Findings

- **P0 (0):** none in the admitted B4 standalone-controller scope.
- **P1 (3):** endpoint/cycle recurrence mismatch; incomplete sequence/completion identity with demonstrated ABA and wrap ordering; absent external SRAM/engine/barrier implementation required by M143r2.
- **P2 (5):** B2 is model-only although M143 lists it; B3 lacks a production seal and matched DC point; M142 has no Formality seal; DC is ideal-clock/zero-wire/zero-macro with only 29.1 ps setup margin; the M143r2 result still marks M142 VCS/DC unsealed and omits correction tokens from top-level `exact_work`.

## Required next actions

1. Freeze cycle semantics for endpoint launch, service, completion and next launch. Recompute M143 with the registered M142 handoffs for every unit, not only zero-work units.
2. Add an explicit empty-work bypass or charge legal PWP and correction handshakes for the 1,332/300 zero-work cases.
3. Carry an epoch/sequence token through PWP and correction request/completion, use wrap-safe oldest comparison, and add stale/duplicate/reorder tests.
4. Specify or enforce row count/order and outer barrier/commit handshakes.
5. Implement the minimal external descriptor/result SRAM plus engine wrapper, then replay heldout service tokens through a cycle simulator driven by the real interface.
6. Seal Formality and macro-inclusive PT/PTPX before any physical or energy claim. Keep the 2.5947x/1.8122x ratios labeled as model-only until then.

Machine-readable findings are in `m142_independent_hammer_review_r1.json`; every review artifact is pinned by `manifest.sha256`.
