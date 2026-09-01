# M1580 — M1578 C2 RTL/mapped case0 independent source QA

Decision: **PASS source-only QA; authorize exactly one future VCS compile and
one `k8_case0` simulation.** M1580 itself ran no VCS, `simv`, or EDA tool and
consumed no attempt.

The existing Python suite passes 9/9 and the author static check remains
`PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_HAMMER__NO_TOOL_RUN`. An independent
text-and-identity checker also passes 16/16 robustness mutations under both
CPython 3.10.18 and 3.6.8.

The 16-entry filelist binds the frozen RTL wrapper, the exact mapped
`ARCH_MODE1` netlist, the reset-safe memory model, and the M1578 top source.
The top instantiates RTL `ARCH_MODE=1` and the mapped module once each. The two
memory fabrics have distinct RTL/mapped signal namespaces, so neither DUT can
consume the other's requests or responses.

Four-state information is preserved with case equality/inequality,
`$isunknown`, and explicit `0/1/X` rendering. The same-cycle trace precedes the
stop record and reports header, source, endpoint request, memory response,
commit and done, plus top protocol/numeric/stale bits, eight endpoint bits and
six internal taps per DUT. This is sufficient to separate reset/X, protocol,
stale and numeric first-fault classes.

The future one-shot wrapper must explicitly pass
`-top tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault` and the frozen filelist
SHA. The filelist itself contains source paths only. Reusing M1502 `simv`, UCLI,
initreg, SAIF, PTPX, force/release, a second compile, or a second simulation is
not authorized. No RTL/mapped PASS or paper claim exists until that one run is
independently reviewed.
