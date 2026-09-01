# M1791 independent source hammer of M1790 C3 mapped energy

Verdict: **FAIL-CLOSED, 88/100, P0=0, P1=2, P2=0. No EDA is authorized.**

The frozen evidence chain is sound. I independently reverified the M1454 DC, M1456 PrimeTime, M1457 gate-to-gate Formality, M1479 admission, M518 R11 VCS, and M1790 author-receipt seals. The mapped netlist and SDC hashes are exact, and `docs/359_DATE终局冻结_20260813.md` remains `dedde7ce...bdfc4`. CPython 3.6 and 3.10 both pass the source checker and reject all 10 author mutations. No EDA, license, GPU, or remote action was run.

The proposed testbench is otherwise strong: it is public-port-only, builds an independent signed-INT8 10x10 plus Q24 bias/threshold reference, checks result tag/beat/valid/data and context-retire latency, uses a 3 ns clock, keeps configuration and one warmup tile outside the SAIF window, measures eight dense tiles, and makes raw/result stalls non-vacuous. The UCLI window is DUT-only. The PTPX path requires zero black boxes, zero macros, TT 0.9 V 25 C, ideal clock, ZeroWireload, no SPEF, exact net/leaf annotation, TX=0, and conserved component power.

Two source defects block launch:

1. `tile_done_tag` is not functionally checked. The testbench samples it, rejects X/Z, and increments a count, but never compares it with the expected tile sequence. This contradicts the author receipt's checked-field list and permits a stale/wrong tile-done tag to pass.
2. The future M1792 release is pinned by an external SHA, status, and budget, but the runner does not require release identity fields bound to the exact runner, M1790 contract, and M1791 review; it also does not require the all-false prelaunch claim boundary or verify release double seals. A same-status release from the wrong successor identity could therefore be accepted.

Required correction: add an ordered expected tile-done tag scoreboard plus a mutation that attacks it; make the runner verify the release schema, transitive identity bindings, all-false claim boundary, budget, sidecar, and outer seal; then obtain a fresh different-author review. The temporary C2 use of M1792 was re-homed and its old files were removed, so there is no remaining milestone-number collision.

Even after a corrected campaign passes, the only permissible output is directed Fixed-T10 component power and directed-window energy for the M1454 prelayout logic-only mapped netlist. It is not speedup, system/frame energy, silicon, signoff, paper-PPA-ready, or headline evidence.
