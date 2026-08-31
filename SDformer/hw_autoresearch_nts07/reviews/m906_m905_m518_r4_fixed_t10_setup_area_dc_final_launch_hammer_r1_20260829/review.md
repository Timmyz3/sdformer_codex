# M906 final-launch hammer

**PASS 100/100, P0/P1/P2 = 0/0/0.** This is a static release decision only. It admits consideration of exactly one `M518_R4_POINT=fixed` setup/area DC attempt after the C2 one-shot is fully terminal, the C1-priority queue is respected, and the frozen runner repeats its live collision/resource gates. The reviewer did not run DC, VCS, any license command, or the released runner.

The identity chain is exact: M905 release, handoff and M906 request; M904; Fixed r11 VCS; Fixed baseline spec; r4 runner/Tcl/filelist/SDC/RTL; source contract, Fixed admission, candidate chain; Synopsys executable and TSMC28 slow/fast libraries. Python 3.6.8 and 3.10.18 independently passed the full closure. Each rejected 43 typed release mutations, 3 duplicate-key attacks and 3 non-finite attacks.

The production predicate is setup/area only: 3.0 ns ideal clock, ZeroWireload, logic-only, zero macros, one `compile_ultra`, no incremental compile or hold fix, and precompile `TIM-209=0`/`OPT-150=0`. Seven raw artifacts are required—area, QoR, setup, mapped Verilog, mapped SDC, DDC and SVF—plus a separate structured postcompile gate. Hold remains diagnostic and is not closed at DC.

At the hammer observation, the C2 attempt and one C2 work population were still present while its canonical result was absent. Therefore the exact command is deliberately **not executable now**. Passing M906 does not waive C2 terminality, the C1 priority, fresh three-sample resource/collision checks, or the later independent Fixed result hammer.

The only exact no-positional-argument command is stored in `review.json`. Rank3, paired comparison, PT, PTPX, SAIF, power, energy, PPA, throughput/mm2, system speedup and headline claims remain unauthorized.
