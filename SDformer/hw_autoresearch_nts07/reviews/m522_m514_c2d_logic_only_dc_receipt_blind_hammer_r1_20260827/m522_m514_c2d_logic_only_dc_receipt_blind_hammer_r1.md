# M522/M514 logic-only DC receipt-blind hammer r1

Verdict: **PASS with a deliberately narrow admission.** Score 98/100; P0=0, P1=0, P2=1. The sealed result may support only the additive physical cost of the standalone decoder address/control mapper: **383.670001 µm², 442 standard cells, +1.4266 ns setup slack, and +0.0106 ns hold slack at 3.0 ns**. It does not admit cycle speedup, system speedup, energy, full-decoder execution, SRAM, Formality, paper-ready PPA, or a DATE headline.

This review did not run DC, VCS, PT, Formality, or any open-source EDA. It did not trust the generated receipt values; every reported number below was re-extracted from sealed raw reports or the mapped netlist.

## The physical result is internally consistent

The raw area report gives 383.670001 µm² total cell area, split into 242.045999 µm² combinational and 141.624002 µm² noncombinational area. It reports 442 leaf cells: 372 combinational and 70 sequential, including 180 buffers/inverters. The QoR report independently repeats the same cell counts and area, reports 23 logic levels and a 1.12 ns critical path length, and lists zero setup and hold violating paths.

The mapped Verilog has one module and exactly 442 standard-cell instance lines. Its reference multiset agrees with the 43-reference postcompile report; no memory/macro/black-box token occurs. Both the area and QoR reports give macro count zero and macro area zero.

## Setup, hold, and five constraint classes pass the frozen model

The minimum setup slack re-parsed across `timing_setup.rpt` is +1.4266 ns. The minimum hold slack re-parsed across `timing_hold.rpt` is +0.0106 ns after the Tcl changes final hold uncertainty to 0.090 ns and performs the hold-only incremental mapping step. Neither timing report contains a violated path.

The five explicit report classes—max delay, min delay, max capacitance, max transition, and max fanout—each state that the design has no violated constraints. The three precompile evidence sources contain zero occurrences of TIM-209 and zero of OPT-150; the loop gate consequently records `TIM-209=0`, `OPT-150=0`, and PASS. Postcompile `check_design` is empty of warnings/errors, and postcompile `check_timing` reports no generated-clock, loop, missing-delay, unconstrained-endpoint, pulse-clock, no-driving-cell, or partial-delay problem.

The qualifiers matter: this is a 3.0 ns, `ssg0p9v125c`, ideal-clock, ZeroWireload, 0.25 ns input/output delay, 0.01 pF output-load, standard-cell-only pre-macro point. It is not a routed or memory-inclusive PPA result.

## Launcher, attempt, and sealed inputs are authentic

The canonical outer-seal file SHA is exactly `f5c527b419bc58d65b427418929d0d1e7de5d7980c4678a383bcf1c5a37d570f`. Its inner manifest verifies 38/38 entries. The canonical root has 40 regular files, three subdirectories, and zero symlinks. Its topology JSON exactly matches the three directories and 37 non-seal/non-topology files.

All 28 frozen input hashes currently match. The five input roots each pass both inner and outer seals, exact file inventories, and member hashes. Four review roots have zero symlinks. The historical VCS root has exactly its two declared links; both raw link strings, internal regular-file targets, and target hashes match the frozen inventory.

The persistent one-shot directory exists and contains `ONE_SHOT_ATTEMPT_CONSUMED_DO_NOT_RETRY`, the exact runner SHA, and `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` as launcher. That launcher remains the `snps_shell` symlink; its resolved regular target SHA is `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`. The DC log identifies Design Compiler Graphical/DC Ultra V-2023.12-SP3, loads the frozen RTL and two libraries, reaches `Thank you...`, and `dc.rc` is zero. The isolated wrong-runner-SHA negative preflight returned the expected rc=10.

## One P2 warning is retained, not hidden

The DC log contains 18 warnings. Most are repeated precompile lint messages, the expected no-hierarchy ungroup warning, or DesignWare library-load messages. One source-level warning deserves a P2 note: VER-318 at RTL line 104 reports a signed-to-unsigned part selection for `selected_slot = slot[3:0]`. The loop bounds constrain `slot` to 0–8 and the independently sealed directed VCS review already admits functional completeness, so this does not block the present logic-cost result. If the RTL identity is reopened, an explicit unsigned cast would improve source hygiene and eliminate ambiguity.

## Citation boundary

The only admissible wording is substantially equivalent to:

> The standalone exact M514 ConvTranspose2d polyphase address/control mapper synthesizes to 383.670001 µm² and 442 TSMC28 standard cells, meeting a 3.0 ns ideal-clock ZeroWireload pre-macro point with +1.4266 ns setup and +0.0106 ns hold worst slack.

It must remain labelled as standalone decoder-support logic. It must not be presented as decoder or system speedup, energy, memory-inclusive cost, placed-and-routed PPA, or a paper headline. Optional Formality/SAIF can strengthen completeness, while exact full-decoder trace and memory-inclusive evaluation remain separate gates.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The companion portable report passed schema validation, packaging, exact-payload, runtime-root, and semantic-fallback structural checks. Browser QA is explicitly `structural_only` because no installed Chromium headless-shell was available; no browser was downloaded or installed.
