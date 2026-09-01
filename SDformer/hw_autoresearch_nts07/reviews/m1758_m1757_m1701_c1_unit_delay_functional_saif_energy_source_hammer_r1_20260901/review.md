# M1758 independent review of M1757 C1 energy source

Verdict: **PASS, 99/100, P0/P1/P2 = 0/0/0.** M1759 may be authored, but this review does not launch or independently authorize EDA without that exact-SHA release.

The M1750 failure is fully contained: its sealed budget is one compile, one simulation, zero SAIF, and zero PTPX. The separately disclosed post-failure `simv -help` invocation used the unsealed old binary without UCLI or the SAIF plusarg and produced no result. Therefore every old-build artifact remains permanently unsealed and forbidden for reuse.

M1757 uses a fresh namespace and fresh VCS compile with exactly one foundry-supported `+define+UNIT_DELAY`. The source contains none of `+notimingcheck`, `+no_notifier`, `+nospecify`, `+initreg`, warning suppression, or input-floating suppression. It preserves the exact M1701 mapped top, public-port-only M1739 testbench, DUT-scoped UCLI, and M1750 whole-top PTPX Tcl.

The future claim is deliberately split: VCS supplies **UNIT_DELAY functional mapped-gate SAIF**, while timing comes from the independently sealed M1740 PrimeTime result (setup/hold WNS +0.027871/+0.001827 ns, 16,549 Formality passing points, nine SRAM macros). It is not a timing simulation or timing signoff claim.

Power accounting is also clean: the primary is the whole mapped top including nine linked SRAM Liberty macros. The datasheet SRAM estimate is a separate alternative sensitivity and is never added. The mixed TT-standard-cell/SSG-SRAM corner is explicitly non-signoff.

Both CPython 3.6 and 3.10 independently recomputed all 51,840,000 support rows and rejected 23 mutations each: seven source-policy, seven runtime/conservation, five SAIF, and four power/accounting attacks. No EDA or license query was made.

Remaining boundary: even a successful M1757 run is a 64-row, ep34-density-conditioned directed component estimate with synthetic residual/psum values. An independent result hammer is mandatory before citation, and no frame/system energy or performance conclusion follows from this source review.
